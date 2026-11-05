"""
PART B with the context-drift diagnostic (extends src/mechanism_synthetic.py without changing it).

Everything in mechanism_synthetic.py (same blobs, defect content, methods, per-method metrics), plus per
synthetic image:
  drift of NORMAL patches = 1 - cosine_similarity(token in the ORIGINAL clean image, token of the same patch in the
  defect-pasted image), averaged in three bands of distance from the blob, measured in patches from the
  nearest defect patch on the 37x37 grid (exact Euclidean distance transform):
      d <= 2,   2 < d <= 5,   d > 5      (plus the number of patches in each band)
  A patch is a defect patch if any pixel of its 14x14 window overlaps the mask (as for A-clean).
Control (once per clean image): the clean image is passed through the model twice; the drift between the
two passes (max and mean over all patches) must be 0, otherwise the pipeline is nondeterministic.

Stages:
  --stage quick : carpet, the same 6 images, all types and sizes -> results/mechanism_synthetic/sanity/quick_drift/
  --stage full  : all textures; resumable; per (texture, type, size) two CSVs in results/mechanism_synthetic/partial/
                  (<unit>.csv = method metrics, <unit>__drift.csv = drift per image);
                  drift_control.csv; combined per_image_synthetic.csv and per_image_drift.csv at the end.
Requires results/mechanism_synthetic/config.json from `mechanism_synthetic.py --stage calibrate`.
"""
import argparse, glob, os, sys, time, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mechanism_common import A_METHODS, GRID, patch_defect_mask  # noqa: E402
from mechanism_synthetic import (Runner, TEXTURES, TYPES, SIZES, map_stats, ring_interior,  # noqa: E402
                                 load_cfg, log)

OUT = os.path.join(REPO, "results", "mechanism_synthetic")
BANDS = [("drift_le2", 0, 2), ("drift_2to5", 2, 5), ("drift_gt5", 5, np.inf)]


class DriftRunner(Runner):
    def __init__(self, noise_sigma):
        super().__init__(noise_sigma)
        self.clean = {}
        self.control = []

    def clean_tokens(self, tex, idx):
        from PIL import Image
        key = (tex, idx)
        if key not in self.clean:
            t1 = self.bb.tokens(Image.fromarray(self.bases[tex][idx]))
            t2 = self.bb.tokens(Image.fromarray(self.bases[tex][idx]))
            dr = (1.0 - self.bb.F.cosine_similarity(t1, t2, dim=1)).cpu().numpy()
            self.control.append({"texture": tex, "image_id": idx, "control_drift_max": float(np.abs(dr).max()),
                                 "control_drift_mean": float(dr.mean()),
                                 "control_tokens_identical": bool((t1 == t2).all().item())})
            self.clean[key] = t1.cpu().half()
        return self.clean[key].to(self.bb.dev).float()

    def evaluate_drift(self, tex, kind, size, idx):
        import cv2
        from PIL import Image
        mask = self.blobs[tex][idx].mask(size)
        img = self.painter.paint(self.bases[tex][idx], mask, kind, tex, idx)
        tok = self.bb.tokens(Image.fromarray(img))
        maps, contam = self.bb.maps(tok, mask, self.banks[tex])
        ring, inter = ring_interior(mask)
        rows = []
        for k, m in maps.items():
            if m is None:
                continue
            method, seed = (k, None) if k in A_METHODS else ("B", int(k[1:]))
            r = {"texture": tex, "type": kind, "size_pct": size, "image_id": idx, "method": method, "seed": seed,
                 "area_frac": float(mask.mean()), "contamination": contam}
            r.update(map_stats(m, mask, ring, inter))
            rows.append(r)
        dp = patch_defect_mask(mask).reshape(GRID, GRID)
        dist = cv2.distanceTransform((~dp).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        drift = (1.0 - self.bb.F.cosine_similarity(self.clean_tokens(tex, idx), tok, dim=1)).cpu().numpy().reshape(GRID, GRID)
        d = {"texture": tex, "type": kind, "size_pct": size, "image_id": idx, "area_frac": float(mask.mean()),
             "n_defect_patches": int(dp.sum())}
        for name, lo, hi in BANDS:
            sel = (~dp) & (dist > lo) & (dist <= hi)
            d[name] = float(drift[sel].mean()) if sel.any() else float("nan")
            d[f"n_{name}"] = int(sel.sum())
        return rows, d


def stage_quick(n_img=6, tex="carpet"):
    import pandas as pd
    cfg = load_cfg(OUT)
    qdir = os.path.join(OUT, "sanity", "quick_drift")
    os.makedirs(qdir, exist_ok=True)
    R = DriftRunner(cfg["noise_sigma"])
    t0 = time.time()
    rows, drows, per = [], [], []
    for kind in TYPES:
        for size in SIZES:
            for i in R.load_texture(tex, n_img):
                t1 = time.time()
                r, d = R.evaluate_drift(tex, kind, size, i)
                per.append(time.time() - t1)
                rows += r
                drows.append(d)
    df, dd, ctl = pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(R.control)
    df.to_csv(os.path.join(qdir, f"quick_{tex}.csv"), index=False)
    dd.to_csv(os.path.join(qdir, f"quick_{tex}_drift.csv"), index=False)
    ctl.to_csv(os.path.join(qdir, f"quick_{tex}_control.csv"), index=False)
    au = df.groupby(["type", "size_pct", "method"])["auroc"].mean().unstack("method")[["A", "B"]]
    tab = dd.groupby(["type", "size_pct"])[[b[0] for b in BANDS] + ["n_drift_gt5"]].mean()
    tab = tab.join(au.rename(columns={"A": "AUROC_A", "B": "AUROC_B"}))
    pd.set_option("display.width", 200)
    log(f"Quick drift test {tex}, {n_img} images (drift = 1 - cos; n_drift_gt5 = mean number of normal patches "
        f"> 5 patches away; AUROC_B = 5-seed mean):\n" + tab.to_string(float_format=lambda v: f"{v:.4f}"))
    log("Control (clean image twice): max drift over all patches and images = "
        f"{ctl['control_drift_max'].max():.3e}; tokens bit-identical in {int(ctl['control_tokens_identical'].sum())}"
        f"/{len(ctl)} images")
    med = float(np.median(per))
    n_eval = sum(len(glob.glob(f"{REPO}/data/mvtec/{t}/test/good/*.png")) for t in TEXTURES) * len(TYPES) * len(SIZES)
    n_imgs = n_eval // (len(TYPES) * len(SIZES))
    log(f"Measured {med:.2f}s per (image, type, size) incl. drift; + 2 clean passes per image ({n_imgs} images) -> "
        f"full run ~ {(med * n_eval + n_imgs * 2 * 0.15) / 60:.0f} min")
    log(f"quick drift stage done in {time.time() - t0:.1f}s")


def stage_full():
    import pandas as pd
    cfg = load_cfg(OUT)
    pdir = os.path.join(OUT, "partial")
    os.makedirs(pdir, exist_ok=True)
    R = DriftRunner(cfg["noise_sigma"])
    units = [(t, k, s) for t in TEXTURES for k in TYPES for s in SIZES]
    stem = lambda t, k, s: os.path.join(pdir, f"{t}__{k}__{int(round(s * 100)):05d}")
    todo = [u for u in units if not (os.path.exists(stem(*u) + ".csv") and os.path.exists(stem(*u) + "__drift.csv"))]
    log(f"{len(units)} units, {len(units) - len(todo)} done, {len(todo)} to run")
    t0, times = time.time(), []
    for n, (t, k, s) in enumerate(todo, 1):
        t1 = time.time()
        rows, drows = [], []
        for i in R.load_texture(t):
            r, d = R.evaluate_drift(t, k, s, i)
            rows += r
            drows.append(d)
        for suffix, data in ((".csv", rows), ("__drift.csv", drows)):
            tmp = stem(t, k, s) + suffix + ".tmp"
            pd.DataFrame(data).to_csv(tmp, index=False)
            os.replace(tmp, stem(t, k, s) + suffix)
        pd.DataFrame(R.control).to_csv(os.path.join(pdir, "control_running.csv"), index=False)
        times.append(time.time() - t1)
        eta = datetime.timedelta(seconds=int(np.mean(times) * (len(todo) - n)))
        a = np.mean([r["auroc"] for r in rows if r["method"] == "A"])
        b = np.mean([r["auroc"] for r in rows if r["method"] == "B"])
        g = np.nanmean([d["drift_gt5"] for d in drows])
        log(f"[{n}/{len(todo)}] {t} {k} {s}%: A {a:.2f} B {b:.2f} drift>5 {g:.4f} ({times[-1]:.0f}s) ETA {eta}")
    m = sorted(f for f in glob.glob(os.path.join(pdir, "*.csv")) if not f.endswith("__drift.csv") and "control" not in f)
    dfiles = sorted(glob.glob(os.path.join(pdir, "*__drift.csv")))
    pd.concat([pd.read_csv(f) for f in m]).to_csv(os.path.join(OUT, "per_image_synthetic.csv"), index=False)
    pd.concat([pd.read_csv(f) for f in dfiles]).to_csv(os.path.join(OUT, "per_image_drift.csv"), index=False)
    pd.DataFrame(R.control).to_csv(os.path.join(OUT, "drift_control.csv"), index=False)
    log(f"full stage done in {time.time() - t0:.1f}s")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["quick", "full"])
    a = ap.parse_args()
    stage_quick() if a.stage == "quick" else stage_full()


if __name__ == "__main__":
    main()
