"""
PART B: self-contamination mechanism on SYNTHETIC defects.

Defects (abnormal inside, same content across sizes for a given image):
  shape     : one irregular connected blob per image, grown from a fixed field (radial term + smooth noise,
              fixed seed per image), thresholded to the target area, fully inside the image (2 px margin).
              The blobs of one image are nested across sizes and identical across defect types.
  othertexture : fixed crop of a DIFFERENT texture category (centre 259x259 of its first train/good image),
                 resized to the blob's bounding box
  noise     : additive Gaussian noise, fixed strength (calibrated: smallest sigma on a fixed grid giving
              Route B (K=4, 5-seed mean) pixel AUROC > 90 at 4% area on every texture), fixed per-image noise field
  blur      : Gaussian blur, sigma 8 px
  hueshift  : hue +60 degrees and brightness x0.65 (HSV)
  edges     : alpha = GaussianBlur(mask, sigma 1 px) (about 2-3 px of blending); ground truth = blob
Sizes: 0.25, 0.5, 1, 2, 4, 8, 16, 32, 50 % of the 518x518 image. Images: all GOOD test images of the 5 textures.
Methods: A, A_clean, A_median, A_trim, Route B seeds 42, 123, 2026, 7, 11.
Per (image, type, size, method[, seed]): pixel AUROC, AP, mean score inside / outside the mask,
d' = (mean in - mean out) / std out, fraction of boundary-ring (<=14 px from the blob edge) and interior
mask pixels above the 99th percentile of that map's normal pixels, contamination index.

Stages:
  --stage calibrate : noise calibration, Route B per-type AUROC at 4% (4 images per texture), preview grid,
                      time estimate -> <out>/sanity/
  --stage quick     : one texture (default carpet, 6 images), every type and size -> <out>/sanity/quick/
  --stage full      : everything; resumable, one partial CSV per (texture, type, size) -> <out>/partial/
"""
import argparse, glob, json, os, sys, time, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mechanism_common import Backbone, IMG, A_METHODS  # noqa: E402

DATA = os.path.join(REPO, "data", "mvtec")
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
TEX_ID = {t: i for i, t in enumerate(TEXTURES)}
OTHER = {"carpet": "wood", "grid": "leather", "leather": "tile", "tile": "carpet", "wood": "grid"}
TYPES = ["othertexture", "noise", "blur", "hueshift"]
SIZES = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 50]
B_SEEDS = [42, 123, 2026, 7, 11]
NOISE_GRID = [8, 12, 16, 24, 32, 48, 64]
BLUR_SIGMA = 8.0
HUE_DEG, VAL_MULT = 60, 0.65
MARGIN = 2


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


# ---------------------------------------------------------------- shapes
class BlobMaker:
    def __init__(self, tex, idx):
        import cv2
        rng = np.random.default_rng([7, TEX_ID[tex], idx])
        noise = cv2.GaussianBlur(rng.standard_normal((IMG, IMG)).astype(np.float32), (0, 0), 25)
        noise /= noise.std() + 1e-8
        cy, cx = rng.uniform(0.35, 0.65, 2) * IMG
        yy, xx = np.mgrid[0:IMG, 0:IMG]
        f = -np.hypot(yy - cy, xx - cx) / (IMG / 4) + 0.35 * noise
        f[:MARGIN], f[-MARGIN:], f[:, :MARGIN], f[:, -MARGIN:] = -np.inf, -np.inf, -np.inf, -np.inf
        self.f = f
        self.seed_pt = np.unravel_index(np.argmax(np.where(np.isfinite(f), f, -1e9)), f.shape)
        self.cache = {}

    def _comp(self, t):
        import cv2
        n, lab = cv2.connectedComponents((self.f > t).astype(np.uint8), connectivity=8)
        return lab == lab[self.seed_pt]

    def mask(self, frac):
        if frac in self.cache:
            return self.cache[frac]
        target = frac / 100.0 * IMG * IMG
        lo, hi = float(np.nanmin(self.f[np.isfinite(self.f)])), float(self.f[self.seed_pt])
        best = None
        for _ in range(60):
            mid = (lo + hi) / 2
            m = self._comp(mid)
            a = m.sum()
            if best is None or abs(a - target) < abs(best.sum() - target):
                best = m
            if abs(a - target) / target < 0.005:
                break
            if a > target:
                lo = mid
            else:
                hi = mid
        self.cache[frac] = best
        return best


# ---------------------------------------------------------------- defect content
class Painter:
    def __init__(self, noise_sigma):
        self.noise_sigma = noise_sigma
        self.other = {}

    def other_crop(self, tex):
        from PIL import Image
        if tex not in self.other:
            src = OTHER[tex]
            p = sorted(glob.glob(f"{DATA}/{src}/train/good/*.png"))[0]
            im = np.array(Image.open(p).convert("RGB").resize((IMG, IMG), Image.BILINEAR))
            o = (IMG - 259) // 2
            self.other[tex] = im[o:o + 259, o:o + 259]
        return self.other[tex]

    def paint(self, base, mask, kind, tex, idx):
        import cv2
        img = base.astype(np.float32)
        if kind == "othertexture":
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            layer = img.copy()
            layer[y0:y1, x0:x1] = cv2.resize(self.other_crop(tex), (x1 - x0, y1 - y0),
                                             interpolation=cv2.INTER_LINEAR).astype(np.float32)
        elif kind == "noise":
            rng = np.random.default_rng([11, TEX_ID[tex], idx])
            layer = img + rng.standard_normal(img.shape).astype(np.float32) * self.noise_sigma
        elif kind == "blur":
            layer = cv2.GaussianBlur(img, (0, 0), BLUR_SIGMA)
        elif kind == "hueshift":
            hsv = cv2.cvtColor(base, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + HUE_DEG / 2) % 180
            hsv[..., 2] = hsv[..., 2] * VAL_MULT
            layer = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        else:
            raise ValueError(kind)
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None]
        return np.clip(img * (1 - alpha) + layer * alpha + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- metrics
def map_stats(m, mask, ring, interior):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, s = mask.ravel(), m.ravel()
    out = s[~y]
    thr = np.percentile(out, 99)
    sd = out.std()
    return {"auroc": roc_auc_score(y, s) * 100, "ap": average_precision_score(y, s) * 100,
            "mean_in": float(s[y].mean()), "mean_out": float(out.mean()),
            "dprime": float((s[y].mean() - out.mean()) / sd) if sd > 0 else float("nan"),
            "frac_ring_above_p99": float((m[ring] > thr).mean()) if ring.any() else float("nan"),
            "frac_interior_above_p99": float((m[interior] > thr).mean()) if interior.any() else float("nan")}


def ring_interior(mask):
    import cv2
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return mask & (dist <= 14), mask & (dist > 14)


class Runner:
    def __init__(self, noise_sigma):
        self.bb = Backbone()
        self.painter = Painter(noise_sigma)
        self.train, self.banks, self.bases, self.blobs = {}, {}, {}, {}

    def load_texture(self, tex, max_images=None):
        from PIL import Image
        from torchvision import transforms
        if tex not in self.train:
            self.train[tex] = [self.bb.tokens(Image.open(p)) for p in
                               sorted(glob.glob(f"{DATA}/{tex}/train/good/*.png"))]
            self.banks[tex] = {sd: self.bb.bank(self.train[tex], sd) for sd in B_SEEDS}
            paths = sorted(glob.glob(f"{DATA}/{tex}/test/good/*.png"))
            rz = transforms.Resize((IMG, IMG))
            self.bases[tex] = [np.array(rz(Image.open(p).convert("RGB"))) for p in paths]
            self.blobs[tex] = [BlobMaker(tex, i) for i in range(len(paths))]
        n = len(self.bases[tex])
        return range(n if max_images is None else min(n, max_images))

    def evaluate(self, tex, kind, size, idx, banks=None):
        from PIL import Image
        mask = self.blobs[tex][idx].mask(size)
        img = self.painter.paint(self.bases[tex][idx], mask, kind, tex, idx)
        tok = self.bb.tokens(Image.fromarray(img))
        maps, contam = self.bb.maps(tok, mask, self.banks[tex] if banks is None else banks)
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
        return rows, img, mask


# ---------------------------------------------------------------- stages
def stage_calibrate(out):
    import pandas as pd
    from PIL import Image
    sdir = os.path.join(out, "sanity")
    os.makedirs(sdir, exist_ok=True)
    R = Runner(noise_sigma=None)
    t0 = time.time()
    per_img_t = []
    cal = []
    for sig in NOISE_GRID:
        R.painter.noise_sigma = sig
        for tex in TEXTURES:
            for i in R.load_texture(tex, 4):
                t1 = time.time()
                rows, _, _ = R.evaluate(tex, "noise", 4, i)
                per_img_t.append(time.time() - t1)
                b = np.mean([r["auroc"] for r in rows if r["method"] == "B"])
                cal.append({"sigma": sig, "texture": tex, "image_id": i, "B_auroc": b})
    cal = pd.DataFrame(cal)
    tab = cal.groupby(["sigma", "texture"])["B_auroc"].mean().unstack()
    ok = [s for s in NOISE_GRID if (tab.loc[s] > 90).all()]
    sigma = ok[0] if ok else NOISE_GRID[-1]
    log("Noise calibration, Route B (5-seed mean) pixel AUROC at 4%, 4 images per texture:\n" + tab.round(2).to_string())
    log(f"Chosen noise sigma = {sigma} (grey levels, 0-255)" + ("" if ok else "  WARNING: no sigma reached >90 on all textures"))
    R.painter.noise_sigma = sigma
    per_type = []
    for kind in TYPES:
        for tex in TEXTURES:
            for i in R.load_texture(tex, 4):
                rows, _, _ = R.evaluate(tex, kind, 4, i)
                for r in rows:
                    per_type.append(r)
    pt = pd.DataFrame(per_type)
    btab = pt[pt.method == "B"].groupby(["type", "texture"])["auroc"].mean().unstack()
    bmean = pt[pt.method == "B"].groupby("type")["auroc"].mean()
    subtle = [k for k in TYPES if bmean[k] < 80]
    log("Route B (5-seed mean) pixel AUROC at 4% by type and texture:\n" + btab.round(2).to_string())
    log("Route B mean over textures: " + ", ".join(f"{k} {bmean[k]:.2f}" for k in TYPES) +
        (f"  -> TOO SUBTLE: {subtle}" if subtle else "  -> no type below 80"))
    # preview grid: types x sizes (1, 8, 50) x textures (carpet, grid)
    import cv2
    tiles = []
    for tex in ("carpet", "grid"):
        R.load_texture(tex, 1)
        for kind in TYPES:
            row = []
            for size in (1, 8, 50):
                mask = R.blobs[tex][0].mask(size)
                img = R.painter.paint(R.bases[tex][0], mask, kind, tex, 0).copy()
                cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
                small = cv2.resize(img, (259, 259))
                cv2.putText(small, f"{tex} {kind} {size}%", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                row.append(small)
            tiles.append(np.concatenate(row, axis=1))
    Image.fromarray(np.concatenate(tiles, axis=0)).save(os.path.join(sdir, "preview_grid.png"))
    n_eval = sum(len(glob.glob(f"{DATA}/{t}/test/good/*.png")) for t in TEXTURES) * len(TYPES) * len(SIZES)
    est = np.median(per_img_t) * n_eval
    log(f"Measured {np.median(per_img_t):.2f}s per (image, type, size) incl. 9 maps -> full run {n_eval} evaluations "
        f"~ {est / 60:.0f} min")
    cfg = {"noise_sigma": sigma, "blur_sigma": BLUR_SIGMA, "hue_deg": HUE_DEG, "val_mult": VAL_MULT,
           "other_texture_source": OTHER, "sizes": SIZES, "types": TYPES, "b_seeds": B_SEEDS,
           "too_subtle": subtle, "B_auroc_at_4pct_by_type": {k: float(bmean[k]) for k in TYPES},
           "estimate_minutes": round(est / 60, 1)}
    cal.to_csv(os.path.join(sdir, "noise_calibration.csv"), index=False)
    pt.to_csv(os.path.join(sdir, "per_type_at_4pct.csv"), index=False)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    log(f"calibrate stage done in {time.time() - t0:.1f}s")


def load_cfg(out):
    p = os.path.join(out, "config.json")
    if not os.path.exists(p):
        sys.exit("ERROR: run --stage calibrate first")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stage_quick(out, tex, n_img):
    import pandas as pd
    cfg = load_cfg(out)
    qdir = os.path.join(out, "sanity", "quick")
    os.makedirs(qdir, exist_ok=True)
    R = Runner(cfg["noise_sigma"])
    t0 = time.time()
    rows = []
    for kind in TYPES:
        for size in SIZES:
            for i in R.load_texture(tex, n_img):
                rows += R.evaluate(tex, kind, size, i)[0]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(qdir, f"quick_{tex}.csv"), index=False)
    piv = df.groupby(["type", "size_pct", "method"])["auroc"].mean().unstack("method")[["A", "A_clean", "A_median", "A_trim", "B"]]
    log(f"Quick test {tex}, {n_img} images, mean pixel AUROC (B = mean over 5 seeds):\n" + piv.round(2).to_string())
    log(f"quick stage done in {time.time() - t0:.1f}s")


def stage_full(out):
    import pandas as pd
    cfg = load_cfg(out)
    pdir = os.path.join(out, "partial")
    os.makedirs(pdir, exist_ok=True)
    R = Runner(cfg["noise_sigma"])
    units = [(t, k, s) for t in TEXTURES for k in TYPES for s in SIZES]
    path = lambda t, k, s: os.path.join(pdir, f"{t}__{k}__{int(round(s * 100)):05d}.csv")
    todo = [u for u in units if not os.path.exists(path(*u))]
    log(f"{len(units)} units, {len(units) - len(todo)} done, {len(todo)} to run")
    t0, times = time.time(), []
    for n, (t, k, s) in enumerate(todo, 1):
        t1 = time.time()
        rows = []
        for i in R.load_texture(t):
            rows += R.evaluate(t, k, s, i)[0]
        tmp = path(t, k, s) + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, path(t, k, s))
        times.append(time.time() - t1)
        eta = datetime.timedelta(seconds=int(np.mean(times) * (len(todo) - n)))
        b = np.mean([r["auroc"] for r in rows if r["method"] == "B"])
        a = np.mean([r["auroc"] for r in rows if r["method"] == "A"])
        log(f"[{n}/{len(todo)}] {t} {k} {s}%: A {a:.2f}  B {b:.2f}  ({times[-1]:.0f}s) ETA {eta}")
    files = sorted(glob.glob(os.path.join(pdir, "*.csv")))
    pd.concat([pd.read_csv(f) for f in files]).to_csv(os.path.join(out, "per_image_synthetic.csv"), index=False)
    log(f"full stage done in {time.time() - t0:.1f}s; combined {len(files)} partial files")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["calibrate", "quick", "full"])
    ap.add_argument("--out", default=os.path.join(REPO, "results", "mechanism_synthetic"))
    ap.add_argument("--quick-texture", default="carpet")
    ap.add_argument("--quick-images", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == "calibrate":
        stage_calibrate(a.out)
    elif a.stage == "quick":
        stage_quick(a.out, a.quick_texture, a.quick_images)
    else:
        stage_full(a.out)


if __name__ == "__main__":
    main()
