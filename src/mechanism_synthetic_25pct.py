"""
PART B add-on: the missing 25% size of the synthetic sweep.

Runs ONLY area = 25% for all 5 textures x 4 defect types x 133 base (good test) images, with exactly the code
of src/mechanism_synthetic_drift.py (DriftRunner.evaluate_drift: same blobs - the 25% blob is the same field
thresholded to 25%, so it stays nested with the other sizes -, same defect content, same noise sigma from
results/mechanism_synthetic/config.json, same methods A, A_clean, A_median, A_trim, Route B seeds 42, 123, 2026,
7, 11, same per-method metrics and the same drift columns, float32 cosine).

Output (new folder, nothing existing is modified): results/mechanism_synthetic_25pct/
  partial/<texture>__<type>__02500.csv and __drift.csv   (resumable: finished units are skipped)
  per_image_synthetic.csv, per_image_drift.csv, drift_control.csv  (combined at the end)
"""
import datetime, glob, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mechanism_synthetic import TEXTURES, TYPES, load_cfg, log  # noqa: E402
from mechanism_synthetic_drift import DriftRunner  # noqa: E402

SIZE = 25.0
CFG_DIR = os.path.join(REPO, "results", "mechanism_synthetic")      # config.json (noise sigma), read only
OUT = os.path.join(REPO, "results", "mechanism_synthetic_25pct")


def main():
    import pandas as pd
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_cfg(CFG_DIR)
    pdir = os.path.join(OUT, "partial")
    os.makedirs(pdir, exist_ok=True)
    units = [(t, k) for t in TEXTURES for k in TYPES]
    stem = lambda t, k: os.path.join(pdir, f"{t}__{k}__{int(round(SIZE * 100)):05d}")
    todo = [u for u in units if not (os.path.exists(stem(*u) + ".csv") and os.path.exists(stem(*u) + "__drift.csv"))]
    n_img = {t: len(glob.glob(f"{REPO}/data/mvtec/{t}/test/good/*.png")) for t in TEXTURES}
    n_eval = sum(n_img[t] for t, _ in todo)
    # estimate from the full 9-size run: 4689 s for 179 units = 4,769 synthetic images -> 0.98 s per image
    log(f"ESTIMATE: {len(units)} units ({len(TEXTURES)} textures x {len(TYPES)} types, size {SIZE}%), "
        f"{len(units) - len(todo)} done, {len(todo)} to run = {n_eval} synthetic images; at 0.98 s per image "
        f"(measured in the full run) + model and training-bank loading ~ {(n_eval * 0.98 + 60 + 15 * len(TEXTURES)) / 60:.0f} min")
    R = DriftRunner(cfg["noise_sigma"])
    t0, times = time.time(), []
    for n, (t, k) in enumerate(todo, 1):
        t1 = time.time()
        rows, drows = [], []
        for i in R.load_texture(t):
            r, d = R.evaluate_drift(t, k, SIZE, i)
            rows += r
            drows.append(d)
        for suffix, data in ((".csv", rows), ("__drift.csv", drows)):
            tmp = stem(t, k) + suffix + ".tmp"
            pd.DataFrame(data).to_csv(tmp, index=False)
            os.replace(tmp, stem(t, k) + suffix)
        pd.DataFrame(R.control).to_csv(os.path.join(pdir, "control_running.csv"), index=False)
        times.append(time.time() - t1)
        eta = datetime.timedelta(seconds=int(np.mean(times) * (len(todo) - n)))
        a = np.mean([r["auroc"] for r in rows if r["method"] == "A"])
        b = np.mean([r["auroc"] for r in rows if r["method"] == "B"])
        g = np.nanmean([d["drift_gt5"] for d in drows])
        log(f"[{n}/{len(todo)}] {t} {k} {SIZE}%: A {a:.2f} B {b:.2f} drift>5 {g:.4f} ({times[-1]:.0f}s) ETA {eta}")
    m = sorted(f for f in glob.glob(os.path.join(pdir, "*.csv")) if not f.endswith("__drift.csv") and "control" not in f)
    dfiles = sorted(glob.glob(os.path.join(pdir, "*__drift.csv")))
    pd.concat([pd.read_csv(f) for f in m]).to_csv(os.path.join(OUT, "per_image_synthetic.csv"), index=False)
    pd.concat([pd.read_csv(f) for f in dfiles]).to_csv(os.path.join(OUT, "per_image_drift.csv"), index=False)
    pd.DataFrame(R.control).to_csv(os.path.join(OUT, "drift_control.csv"), index=False)
    log(f"25% stage done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
