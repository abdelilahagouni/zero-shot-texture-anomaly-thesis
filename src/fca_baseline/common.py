"""
Shared paths and helpers for the FCA / aota baseline (third_party/AnomalyLocalizationFCA, not modified).

Test-image order, ground truth and the 518x518 grid are those of src/eval_router_full.py:
  defects in sorted order, images sorted; mask = PIL resize to 518x518 with NEAREST, then > 128; good = zeros.
Maps of FCA / aota are stored raw (their native resolution, float16 .npy) and resized to 518x518 on load with
cv2.resize INTER_LINEAR (bilinear, half-pixel centres, the same convention as F.interpolate(align_corners=False)
used for Routes A and B).
Border crop for the fairness check: 10% of 518 = 51.8 -> 52 px removed on every side (centre 414x414).
"""
import glob, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
REPO = os.path.dirname(SRC)
FCA_REPO = os.path.join(REPO, "third_party", "AnomalyLocalizationFCA")
DATA = os.path.join(REPO, "data", "mvtec")
RES = os.path.join(REPO, "results", "fca_baseline")
MAPS = os.path.join(RES, "maps")                    # raw maps (not committed: large, regenerable)
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
GRID = 518
BORDER = int(round(0.10 * GRID))                    # 52


def test_items(cat):
    """[(defect, file_name, image_path, mask_path or None)] in eval_router_full.py order."""
    tdir, gdir = os.path.join(DATA, cat, "test"), os.path.join(DATA, cat, "ground_truth")
    out = []
    for d in sorted(os.listdir(tdir)):
        if not os.path.isdir(os.path.join(tdir, d)):
            continue
        for p in sorted(glob.glob(os.path.join(tdir, d, "*.png"))):
            fn = os.path.basename(p)
            mp = None
            if d != "good":
                mp = os.path.join(gdir, d, fn.replace(".png", "_mask.png"))
                if not os.path.exists(mp):
                    mp = os.path.join(gdir, d, fn)
            out.append((d, fn, p, mp))
    return out


def load_gt(mask_path):
    from PIL import Image
    if mask_path is None:
        return np.zeros((GRID, GRID), np.uint8)
    return (np.array(Image.open(mask_path).convert("L").resize((GRID, GRID), Image.NEAREST)) > 128).astype(np.uint8)


def map_path(method, cat, defect, fn):
    return os.path.join(MAPS, method, cat, defect, fn.replace(".png", ".npy"))


def to_grid(m):
    import cv2
    m = np.asarray(m, np.float32)
    if m.shape == (GRID, GRID):
        return m
    return cv2.resize(m, (GRID, GRID), interpolation=cv2.INTER_LINEAR)


def crop(a):
    return a[BORDER:GRID - BORDER, BORDER:GRID - BORDER]
