"""
eval_router_full.py
Full benchmark: adaptive routing on MVTec AD textures.
Categories: carpet, grid, leather, tile, wood
Backbone: DINOv2 ViT-B/14 (frozen), 518x518 -> 37x37 = 1369 patch tokens

Route A : 0-shot intra-image self-similarity  (1 - mean cosine sim per patch)
Route B : 4-shot memory-bank NN distance       (1 - max cosine sim to bank)

Usage:
    python src/eval_router_full.py              # carpet only (sanity check)
    python src/eval_router_full.py --all        # all 5 categories
"""

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0.  CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--all", dest="run_all", action="store_true",
                    help="Run all 5 categories instead of carpet only")
args = parser.parse_args()

CATEGORIES = (["carpet", "grid", "leather", "tile", "wood"]
              if args.run_all else ["carpet"])
SEEDS = [42, 123, 2026]
K_SHOT = 4
MAX_FPR = 0.3
NUM_THRESH = 1000          # linear thresholds for new AU-PRO
NUM_THRESH_OLD = 200       # kept for backward-compatible comparison

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  MODEL
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"\n=== eval_router_full.py ===")
print(f"Device : {device}  ({gpu_name})")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
print(f"Categories : {CATEGORIES}")
print(f"Seeds      : {SEEDS}")

print("Loading DINOv2 ViT-B/14 ...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# 2.  HELPERS
# ---------------------------------------------------------------------------

def postprocess(score_1d: torch.Tensor) -> np.ndarray:
    """Upsample 37x37 -> 518x518 and apply Gaussian blur. Exact reuse."""
    hm = F.interpolate(score_1d.reshape(1, 1, 37, 37),
                       size=(518, 518), mode="bilinear",
                       align_corners=False).squeeze().cpu().numpy()
    return cv2.GaussianBlur(hm, (15, 15), 0)


def extract_tokens_fp16(img_path: str) -> torch.Tensor:
    """Extract L2-normalised patch tokens; store on CPU in fp16 to save RAM."""
    img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.forward_features(img)["x_norm_patchtokens"][0]   # [1369,768]
    feat = F.normalize(feat, p=2, dim=-1)
    return feat.cpu().to(torch.float16)   # fp16 on CPU


def tokens_gpu(tok_cpu: torch.Tensor) -> torch.Tensor:
    """Move fp16 CPU tokens to GPU as fp32 for computation."""
    return tok_cpu.to(device).to(torch.float32)


def score_A(tok: torch.Tensor) -> np.ndarray:
    """Route A: 0-shot self-similarity. tok is [1369,768] on GPU fp32."""
    sim = torch.mm(tok, tok.T)
    return postprocess(1.0 - sim.mean(dim=1))


def score_B(tok: torch.Tensor, bank: torch.Tensor) -> np.ndarray:
    """Route B: 4-shot NN distance. tok [1369,768], bank [4*1369,768] on GPU fp32."""
    sim = torch.mm(tok, bank.T)
    return postprocess(1.0 - sim.max(dim=1)[0])


# ---------------------------------------------------------------------------
# 3.  AU-PRO  (new + legacy)
# ---------------------------------------------------------------------------

def _pre_extract_regions(gt_masks):
    """Return (img_regions, total_normal_px, total_regions) for vectorised PRO."""
    img_regions = []   # per image: list of (bool_mask, int size)
    total_normal = 0
    total_regions = 0
    for gt in gt_masks:
        gt_bin = (gt > 0).astype(np.uint8)
        total_normal += int((gt_bin == 0).sum())
        n_lab, labs = cv2.connectedComponents(gt_bin)
        regs = []
        for lid in range(1, n_lab):
            rm = labs == lid
            rs = int(rm.sum())
            if rs > 0:
                regs.append((rm, rs))
                total_regions += 1
        img_regions.append(regs)
    return img_regions, total_normal, total_regions


def compute_au_pro_new(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    max_fpr: float = MAX_FPR,
    num_thresholds: int = NUM_THRESH,
    verbose: bool = False,
) -> Tuple[float, int]:
    """
    AU-PRO (corrected spec, vectorized for fast execution):
    - good images (empty masks) contribute to FPR pool
    - num_thresholds linear thresholds across [global_min, global_max]
    - linear interpolation at the FPR=0.3 boundary
    - integrate with trapezoid, normalize by max_fpr
    Returns (au_pro, n_thresholds_below_max_fpr).
    """
    # 1. Gather normal pixel scores across all images
    normal_pixels = []
    total_normal = 0
    # 2. Extract per-connected-component scores for anomalous regions
    region_scores_list = []
    total_regions = 0

    for am, gt in zip(anomaly_maps, gt_masks):
        gt_bin = (gt > 0).astype(np.uint8)
        norm_scores = am[gt_bin == 0]
        normal_pixels.append(norm_scores)
        total_normal += len(norm_scores)

        n_lab, labs = cv2.connectedComponents(gt_bin)
        for lid in range(1, n_lab):
            rm = (labs == lid)
            rs = int(rm.sum())
            if rs > 0:
                s_sorted = np.sort(am[rm])
                region_scores_list.append((s_sorted, rs))
                total_regions += 1

    if total_regions == 0 or total_normal == 0:
        return 0.0, 0

    all_normal = np.sort(np.concatenate(normal_pixels))
    all_scores = np.concatenate([am.ravel() for am in anomaly_maps])
    t_min, t_max = float(all_scores.min()), float(all_scores.max())
    if t_min == t_max:
        return 0.0, 0

    # Linearly spaced thresholds from max down to min
    thresholds = np.linspace(t_max, t_min, num_thresholds)

    # Vectorized FPR via searchsorted on sorted normal scores
    # Number of normal pixels >= t
    fp_counts = len(all_normal) - np.searchsorted(all_normal, thresholds, side='left')
    fprs = fp_counts / total_normal

    # Vectorized PRO overlap via searchsorted on per-region sorted scores
    pros = []
    for t in thresholds:
        tot_overlap = 0.0
        for s_sorted, rs in region_scores_list:
            n_ge = len(s_sorted) - np.searchsorted(s_sorted, t, side='left')
            tot_overlap += n_ge / rs
        pros.append(tot_overlap / total_regions)

    fprs = np.array(fprs)
    pros = np.array(pros)

    n_below = int((fprs <= max_fpr).sum())

    # Linear interpolation at boundary
    valid = fprs <= max_fpr
    fpr_v = list(fprs[valid])
    pro_v = list(pros[valid])

    exceed = np.where(fprs > max_fpr)[0]
    if len(exceed) > 0:
        i_hi = exceed[0]
        i_lo = i_hi - 1
        if i_lo >= 0 and fprs[i_hi] > fprs[i_lo]:
            alpha = (max_fpr - fprs[i_lo]) / (fprs[i_hi] - fprs[i_lo])
            fpr_interp = max_fpr
            pro_interp = pros[i_lo] + alpha * (pros[i_hi] - pros[i_lo])
            fpr_v.append(fpr_interp)
            pro_v.append(pro_interp)

    fpr_v = np.array(fpr_v)
    pro_v = np.array(pro_v)

    if len(fpr_v) < 2:
        return 0.0, n_below

    order = np.argsort(fpr_v)
    fpr_v, pro_v = fpr_v[order], pro_v[order]

    au_pro = float(np.trapezoid(pro_v, fpr_v) / max_fpr)
    if verbose:
        print(f"    AU-PRO: {n_below}/{num_thresholds} thresholds below FPR={max_fpr}")
    return au_pro, n_below


def compute_au_pro_old(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    max_fpr: float = MAX_FPR,
    num_thresholds: int = NUM_THRESH_OLD,
) -> float:
    """Legacy AU-PRO kept exactly as in evaluation.py for comparison, accelerated mathematically."""
    all_scores = np.concatenate([am.flatten() for am in anomaly_maps])
    t_min, t_max = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(t_max, t_min, num_thresholds)

    pro_regions = []
    normal_scores = []
    total_normal = 0
    for am, gt in zip(anomaly_maps, gt_masks):
        gt_bin = (gt > 0).astype(np.uint8)
        norm_sc = am[gt_bin == 0]
        normal_scores.append(norm_sc)
        total_normal += len(norm_sc)
        n_lab, labs = cv2.connectedComponents(gt_bin)
        for lid in range(1, n_lab):
            rm = labs == lid
            rs = int(rm.sum())
            if rs > 0:
                pro_regions.append((rm, rs))

    if not pro_regions or total_normal == 0:
        return 0.0

    all_normal = np.sort(np.concatenate(normal_scores))
    fp_counts = len(all_normal) - np.searchsorted(all_normal, thresholds, side='left')
    fpr_list = fp_counts / total_normal

    pair_scores = []
    for am in anomaly_maps:
        for rm, rs in pro_regions:
            pair_scores.append(np.sort(am[rm]))

    pro_list = []
    for t in thresholds:
        tot_ol = 0.0
        for s in pair_scores:
            tot_ol += (len(s) - np.searchsorted(s, t, side='left')) / len(s)
        pro_list.append(tot_ol / len(pair_scores))

    fprs = np.array(fpr_list)
    pros = np.array(pro_list)
    valid = fprs <= max_fpr
    if not np.any(valid):
        return 0.0
    fpr_v, pro_v = fprs[valid], pros[valid]
    order = np.argsort(fpr_v)
    fpr_v, pro_v = fpr_v[order], pro_v[order]
    return float(np.trapezoid(pro_v, fpr_v) / max_fpr)


# ---------------------------------------------------------------------------
# 4.  METRICS
# ---------------------------------------------------------------------------

def compute_metrics(
    anomaly_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
    compute_pro: bool = True,
    verbose_pro: bool = False,
) -> Dict[str, float]:
    """Pixel AUROC, Pixel AP, AU-PRO (new), Image AUROC with fast computation on large datasets."""
    y_pix = np.concatenate([(gt.ravel() > 0).astype(np.uint8) for gt in gt_masks])
    s_pix = np.concatenate([am.ravel() for am in anomaly_maps])

    has_both_pix = (y_pix.sum() > 0) and (y_pix.sum() < len(y_pix))
    if not has_both_pix:
        pixel_auroc = float("nan")
        pixel_ap    = float("nan")
    elif len(s_pix) > 10_000_000:
        # Fast binned AUROC and AP for pooled arrays (~138M pixels in ~2 seconds instead of ~150s)
        n_bins = 200_000
        s_min, s_max = float(s_pix.min()), float(s_pix.max())
        cp, _ = np.histogram(s_pix[y_pix == 1], bins=n_bins, range=(s_min, s_max))
        cn, _ = np.histogram(s_pix[y_pix == 0], bins=n_bins, range=(s_min, s_max))
        
        # AUROC
        cum_neg = np.cumsum(cn)
        pixel_auroc = float(np.sum(cp * (cum_neg - cn / 2.0))) / (cp.sum() * cn.sum()) * 100.0
        
        # Average Precision (AP)
        tp = np.cumsum(cp[::-1])
        fp = np.cumsum(cn[::-1])
        recall = tp / tp[-1]
        precision = tp / (tp + fp)
        delta_recall = np.diff(np.concatenate(([0.0], recall)))
        pixel_ap = float(np.sum(delta_recall * precision)) * 100.0
    else:
        pixel_auroc = roc_auc_score(y_pix, s_pix) * 100.0
        pixel_ap    = average_precision_score(y_pix, s_pix) * 100.0

    au_pro_val = 0.0
    n_below = 0
    if compute_pro:
        au_pro_val, n_below = compute_au_pro_new(anomaly_maps, gt_masks,
                                                  verbose=verbose_pro)
        au_pro_val *= 100.0

    y_img = np.array([(gt.sum() > 0) for gt in gt_masks], dtype=np.uint8)
    s_img = np.array([am.max() for am in anomaly_maps])
    has_both_img = (y_img.sum() > 0) and (y_img.sum() < len(y_img))
    image_auroc = roc_auc_score(y_img, s_img) * 100.0 if has_both_img else float("nan")

    return dict(pixel_auroc=pixel_auroc, pixel_ap=pixel_ap,
                au_pro=au_pro_val, image_auroc=image_auroc,
                _n_below=n_below)


# ---------------------------------------------------------------------------
# 5.  DATA LOADING  (tokens cached in fp16 on CPU)
# ---------------------------------------------------------------------------

print("\n--- Phase A: Loading & caching tokens (fp16 CPU) ---")

train_cache: Dict[str, List[dict]] = {}
test_cache:  Dict[str, List[dict]] = {}

for cat in CATEGORIES:
    # ---- train/good ----
    train_paths = sorted(glob.glob(f"data/mvtec/{cat}/train/good/*.png"))
    rows = []
    for p in tqdm(train_paths, desc=f"train/{cat}", leave=False):
        tok = extract_tokens_fp16(p)
        var = torch.var(tok.float(), dim=0).mean().item()
        rows.append(dict(path=p, tok=tok, sigma2=var))
    train_cache[cat] = rows

    # ---- test ----
    rows = []
    test_dir = f"data/mvtec/{cat}/test"
    gt_dir   = f"data/mvtec/{cat}/ground_truth"
    for defect in sorted(os.listdir(test_dir)):
        dp = os.path.join(test_dir, defect)
        if not os.path.isdir(dp):
            continue
        for img_path in sorted(glob.glob(os.path.join(dp, "*.png"))):
            fn = os.path.basename(img_path)
            if defect == "good":
                gt = np.zeros((518, 518), dtype=np.uint8)
            else:
                mp = os.path.join(gt_dir, defect,
                                  fn.replace(".png", "_mask.png"))
                if not os.path.exists(mp):
                    mp = os.path.join(gt_dir, defect, fn)
                gt = (np.array(Image.open(mp).convert("L")
                               .resize((518, 518), Image.NEAREST)) > 128
                      ).astype(np.uint8)

            tok = extract_tokens_fp16(img_path)
            tok_f = tok.float()           # for descriptors (stays CPU momentarily)
            var = torch.var(tok_f, dim=0).mean().item()

            # off-diagonal mean cosine similarity
            sim_cpu = torch.mm(tok_f, tok_f.T)
            N = sim_cpu.shape[0]
            off_diag = (sim_cpu.sum() - sim_cpu.trace()) / (N * (N - 1))
            off_diag_mean = off_diag.item()

            # top-5 intra-image mean (excl. self)
            top5_mean = torch.topk(sim_cpu, k=5, dim=1).values[:, 1:].mean().item()

            rows.append(dict(
                cat=cat, path=img_path, fn=fn, label=defect,
                is_defect=(defect != "good"), gt=gt, tok=tok,
                sigma2=var, off_diag=off_diag_mean, top5=top5_mean,
            ))
    test_cache[cat] = rows
    print(f"  {cat}: train={len(train_cache[cat])}, test={len(test_cache[cat])}")

# ---------------------------------------------------------------------------
# 6.  PRECOMPUTE ROUTE A MAPS  (all categories, deterministic)
# ---------------------------------------------------------------------------

print("\n--- Phase B: Route A maps ---")
for cat in CATEGORIES:
    for item in tqdm(test_cache[cat], desc=f"RouteA/{cat}", leave=False):
        tok = tokens_gpu(item["tok"])
        sim = torch.mm(tok, tok.T)
        item["map_A"] = score_A(tok)

# ---------------------------------------------------------------------------
# 7.  PRECOMPUTE ROUTE B MAPS  (per seed)
# ---------------------------------------------------------------------------

print("\n--- Phase C: Route B maps (3 seeds ... K=4 refs) ---")
# route_B_maps[cat][seed] = list of np.ndarray matching test_cache[cat]
route_B_maps: Dict[str, Dict[int, List[np.ndarray]]] = {}
# also store reference items used per (cat, seed) for calibration
ref_items_by: Dict[str, Dict[int, list]] = {}

for cat in CATEGORIES:
    route_B_maps[cat] = {}
    ref_items_by[cat] = {}
    n_train = len(train_cache[cat])
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_train, size=K_SHOT, replace=False)
        refs = [train_cache[cat][i] for i in idx]
        ref_items_by[cat][seed] = refs

        bank = torch.cat([tokens_gpu(r["tok"]) for r in refs], dim=0)  # [4*1369,768]
        maps = []
        for item in tqdm(test_cache[cat],
                         desc=f"RouteB/{cat}/seed{seed}", leave=False):
            tok = tokens_gpu(item["tok"])
            maps.append(score_B(tok, bank))
        route_B_maps[cat][seed] = maps

# ---------------------------------------------------------------------------
# 8.  THRESHOLD CALIBRATION from pooled train/good  (no test data used)
# ---------------------------------------------------------------------------

all_train_sigma2 = np.array([r["sigma2"]
                              for cat in CATEGORIES
                              for r in train_cache[cat]])
tau_median = float(np.median(all_train_sigma2))
pct_levels  = list(range(10, 100, 10))
tau_by_pct  = {p: float(np.percentile(all_train_sigma2, p)) for p in pct_levels}

print(f"\n--- Threshold calibration (N={len(all_train_sigma2)} train/good images) ---")
print(f"  tau (50th / median): {tau_median:.7f}")
for p in pct_levels:
    print(f"  tau ({p:2d}th pct)    : {tau_by_pct[p]:.7f}")

# ---------------------------------------------------------------------------
# 9.  CALIBRATION PARAMS for variant (f)
#     Route A on reference images; Route B leave-one-out
# ---------------------------------------------------------------------------

calib: Dict[str, Dict[int, dict]] = {}
for cat in CATEGORIES:
    calib[cat] = {}
    for seed in SEEDS:
        refs = ref_items_by[cat][seed]
        # Route A on each ref
        pix_a = []
        for r in refs:
            t = tokens_gpu(r["tok"])
            pix_a.append(score_A(t).ravel())
        all_a = np.concatenate(pix_a)
        mu_a, std_a = all_a.mean(), all_a.std()

        # Route B LOO
        pix_b = []
        for j in range(K_SHOT):
            t_j  = tokens_gpu(refs[j]["tok"])
            bank = torch.cat([tokens_gpu(refs[k]["tok"])
                              for k in range(K_SHOT) if k != j], dim=0)
            pix_b.append(score_B(t_j, bank).ravel())
        all_b = np.concatenate(pix_b)
        mu_b, std_b = all_b.mean(), all_b.std()

        calib[cat][seed] = dict(mu_a=mu_a, std_a=std_a, mu_b=mu_b, std_b=std_b)

# ---------------------------------------------------------------------------
# 10.  SANITY CHECK: old vs new AU-PRO on carpet Route B seed 42
# ---------------------------------------------------------------------------

print("\n--- Sanity: old vs new AU-PRO on carpet/RouteB/seed42 ---")
_cat_s = "carpet"
_maps_s = route_B_maps[_cat_s][42]
_gts_s  = [it["gt"] for it in test_cache[_cat_s]]

_new_aupro, _n_below_s = compute_au_pro_new(_maps_s, _gts_s, verbose=True)
_old_aupro             = compute_au_pro_old(_maps_s, _gts_s)
print(f"  new AU-PRO : {_new_aupro*100:.4f}%  ({_n_below_s}/{NUM_THRESH} thresholds <= FPR 0.3)")
print(f"  old AU-PRO : {_old_aupro*100:.4f}%")
print(f"  |new - old| : {abs(_new_aupro - _old_aupro):.6f}  (< 1e-3? {abs(_new_aupro - _old_aupro) < 1e-3})")

# ---------------------------------------------------------------------------
# 11.  ROUTE-ONLY BASELINES  (Route A det., Route B mean +/- std over seeds)
# ---------------------------------------------------------------------------

print("\n--- Phase D: Route-only baselines ---")
route_A_metrics: Dict[str, dict] = {}
route_B_metrics: Dict[str, Dict[int, dict]] = {}

for cat in CATEGORIES:
    gts = [it["gt"] for it in test_cache[cat]]

    maps_a = [it["map_A"] for it in test_cache[cat]]
    print(f"  Evaluating Route A on {cat}...")
    route_A_metrics[cat] = compute_metrics(maps_a, gts,
                                           compute_pro=True, verbose_pro=True)
    print(f"  Route A on {cat} done. Evaluating Route B seeds...")

    route_B_metrics[cat] = {}
    for seed in SEEDS:
        print(f"    Evaluating Route B on {cat} seed {seed}...")
        maps_b = route_B_maps[cat][seed]
        route_B_metrics[cat][seed] = compute_metrics(maps_b, gts, compute_pro=True)
        print(f"    Route B on {cat} seed {seed} done.")

# Pooled
if len(CATEGORIES) > 1:
    print("  Evaluating POOLED Route A across all categories...")
    all_gts   = [it["gt"]    for cat in CATEGORIES for it in test_cache[cat]]
    all_maps_a = [it["map_A"] for cat in CATEGORIES for it in test_cache[cat]]
    route_A_metrics["POOLED"] = compute_metrics(all_maps_a, all_gts, compute_pro=True)
    print("  POOLED Route A done.")
    route_B_metrics["POOLED"] = {}
    for seed in SEEDS:
        print(f"  Evaluating POOLED Route B seed {seed}...")
        all_maps_b = [route_B_maps[cat][seed][i]
                      for cat in CATEGORIES
                      for i in range(len(test_cache[cat]))]
        route_B_metrics["POOLED"][seed] = compute_metrics(all_maps_b, all_gts, compute_pro=True)
        print(f"  POOLED Route B seed {seed} done.")
else:
    route_A_metrics["POOLED"] = route_A_metrics[CATEGORIES[0]]
    route_B_metrics["POOLED"] = route_B_metrics[CATEGORIES[0]]

# Aggregate B: mean +/- std
def agg_B(key):
    ms = {m: [route_B_metrics[key][s][m] for s in SEEDS]
          for m in ["pixel_auroc", "pixel_ap", "au_pro", "image_auroc"]}
    return {m: (float(np.nanmean(v)), float(np.nanstd(v))) for m, v in ms.items()}

route_B_agg = {k: agg_B(k) for k in CATEGORIES + ["POOLED"]}

# ---------------------------------------------------------------------------
# 12.  SANITY CHECK vs reference numbers
# ---------------------------------------------------------------------------

SANITY_REF = {
    "carpet": dict(A_auc=96.12, A_ap=37.26, B_auc=98.79, B_ap=78.41),  # B_ap from user spec
    "wood":   dict(A_auc=84.37, A_ap=32.09, B_auc=92.73, B_ap=62.72),
}

print("\n--- Sanity check vs reference numbers ---")
sanity_lines = []
for cat, ref in SANITY_REF.items():
    if cat not in CATEGORIES:
        sanity_lines.append(f"{cat}: not in current run")
        continue
    ra = route_A_metrics[cat]
    rb_seed42 = route_B_metrics[cat][42]
    line = (
        f"{cat.upper()}:\n"
        f"  Route A  AUROC: {ra['pixel_auroc']:.2f}%  (ref {ref['A_auc']:.2f}%, "
        f"diff={ra['pixel_auroc']-ref['A_auc']:+.2f})\n"
        f"  Route A  AP:    {ra['pixel_ap']:.2f}%    (ref {ref['A_ap']:.2f}%, "
        f"diff={ra['pixel_ap']-ref['A_ap']:+.2f})\n"
        f"  Route B  AUROC: {rb_seed42['pixel_auroc']:.2f}%  (ref {ref['B_auc']:.2f}%, "
        f"diff={rb_seed42['pixel_auroc']-ref['B_auc']:+.2f})\n"
        f"  Route B  AP:    {rb_seed42['pixel_ap']:.2f}%    (ref {ref['B_ap']:.2f}%, "
        f"diff={rb_seed42['pixel_ap']-ref['B_ap']:+.2f})"
    )
    print(line)
    sanity_lines.append(line)

# ---------------------------------------------------------------------------
# 13.  PER-IMAGE CSV
# ---------------------------------------------------------------------------

print("\n--- Phase E: Per-image CSV ---")
csv_rows = []
for cat in CATEGORIES:
    maps_b_42 = route_B_maps[cat][42]
    for i, it in enumerate(test_cache[cat]):
        map_a = it["map_A"]
        map_b = maps_b_42[i]
        gt    = it["gt"]

        if it["is_defect"] and gt.sum() > 0:
            gt_flat = (gt.ravel() > 0).astype(np.uint8)
            pxa = roc_auc_score(gt_flat, map_a.ravel()) * 100.0
            pxb = roc_auc_score(gt_flat, map_b.ravel()) * 100.0
            gap = pxb - pxa
        else:
            pxa = pxb = gap = float("nan")

        route = "A" if it["sigma2"] < tau_median else "B"
        csv_rows.append(dict(
            category=cat, path=it["path"], label=it["label"],
            sigma2_spatial=it["sigma2"],
            off_diag_mean=it["off_diag"],
            top5_mean=it["top5"],
            map_A_max=map_a.max(), map_A_mean=map_a.mean(),
            map_B_max=map_b.max(), map_B_mean=map_b.mean(),
            px_auroc_A=pxa, px_auroc_B=pxb,
            gap_B_minus_A=gap,
            route_tau_median=route,
        ))

csv_path = "results/per_image.csv"
try:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Written {len(csv_rows)} rows -> {csv_path}")
except PermissionError:
    alt_csv = f"results/per_image_latest_{int(time.time())}.csv"
    with open(alt_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Warning: {csv_path} was locked (e.g. open in Excel). Written {len(csv_rows)} rows -> {alt_csv}")

# ---------------------------------------------------------------------------
# 14.  SPEARMAN CORRELATIONS
#      Only anomalous images with gt.sum() > 0; NaN for undefined
# ---------------------------------------------------------------------------

print("\n--- Phase F: Spearman correlations ---")
DESCRIPTORS = ["sigma2_spatial", "off_diag_mean", "top5_mean"]

def spearman_or_nan(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan"), float("nan")
    rho, pval = spearmanr(x[mask], y[mask])
    return float(rho), float(pval)

corr_results = {}
for key in CATEGORIES + (["POOLED"] if len(CATEGORIES) > 1 else []):
    if key == "POOLED":
        rows = [r for r in csv_rows if not np.isnan(r["gap_B_minus_A"])]
    else:
        rows = [r for r in csv_rows
                if r["category"] == key and not np.isnan(r["gap_B_minus_A"])]
    n = len(rows)
    gaps = [r["gap_B_minus_A"] for r in rows]
    corr_results[key] = {"n": n}
    for desc in DESCRIPTORS:
        vals = [r[desc] for r in rows]
        rho, pv = spearman_or_nan(vals, gaps)
        corr_results[key][desc] = (rho, pv)

# ---------------------------------------------------------------------------
# 15.  VARIANCE DISTRIBUTION TABLE
# ---------------------------------------------------------------------------

var_table = {}
for cat in CATEGORIES:
    var_table[cat] = {}
    for split, rows in [
        ("train_good",   train_cache[cat]),
        ("test_good",    [it for it in test_cache[cat] if not it["is_defect"]]),
        ("test_defect",  [it for it in test_cache[cat] if it["is_defect"]]),
    ]:
        v = np.array([r["sigma2"] for r in rows])
        if len(v) == 0:
            var_table[cat][split] = dict(n=0, min=float("nan"), median=float("nan"),
                                          mean=float("nan"), max=float("nan"))
        else:
            var_table[cat][split] = dict(n=len(v), min=v.min(), median=np.median(v),
                                          mean=v.mean(), max=v.max())

# ---------------------------------------------------------------------------
# 16.  ROUTER VARIANTS  (a)-(f)
# ---------------------------------------------------------------------------

print("\n--- Phase G: Router variants ---")

def route_decisions(items, tau):
    return [it["sigma2"] < tau for it in items]

def apply_variant(maps_a, maps_b, gts, decisions, cp, compute_pro=True):
    """
    Returns (metrics_d, metrics_e, metrics_f, subs_d, subs_e, subs_f, n_A, n_B).
    (a) always-A  (b) always-B  (c) oracle   computed separately.
    """
    n_A = sum(decisions)
    n_B = len(decisions) - n_A

    # (d) raw concat
    maps_d = [maps_a[i] if decisions[i] else maps_b[i] for i in range(len(decisions))]
    m_d = compute_metrics(maps_d, gts, compute_pro=compute_pro)

    # (e) per-image min-max normalise (negative control)
    maps_e = []
    for i in range(len(decisions)):
        m = maps_a[i] if decisions[i] else maps_b[i]
        lo, hi = m.min(), m.max()
        maps_e.append((m - lo) / (hi - lo + 1e-8))
    m_e = compute_metrics(maps_e, gts, compute_pro=compute_pro)

    # (f) per-route calibration: (m - mu) / std
    maps_f = []
    for i in range(len(decisions)):
        if decisions[i]:
            maps_f.append((maps_a[i] - cp["mu_a"]) / (cp["std_a"] + 1e-8))
        else:
            maps_f.append((maps_b[i] - cp["mu_b"]) / (cp["std_b"] + 1e-8))
    m_f = compute_metrics(maps_f, gts, compute_pro=compute_pro)

    # Subset evaluations for d/e/f (only needed for main evaluation, skipped in sweep)
    if compute_pro:
        def subsets(maps_x):
            iA = [i for i, d in enumerate(decisions) if d]
            iB = [i for i, d in enumerate(decisions) if not d]
            sA = compute_metrics([maps_x[i] for i in iA],
                                  [gts[i] for i in iA], compute_pro=compute_pro) if iA else None
            sB = compute_metrics([maps_x[i] for i in iB],
                                  [gts[i] for i in iB], compute_pro=compute_pro) if iB else None
            return sA, sB
        subs_d, subs_e, subs_f = subsets(maps_d), subsets(maps_e), subsets(maps_f)
    else:
        subs_d = subs_e = subs_f = (None, None)

    return (m_d, m_e, m_f, subs_d, subs_e, subs_f, n_A, n_B)


def evaluate_all_variants(tau: float, primary_seed: int = 42, compute_pro: bool = True, pooled_only: bool = False):
    """Evaluate variants (a)-(f) per category and pooled at given tau."""
    out = {}
    cats_to_eval = [] if (pooled_only and len(CATEGORIES) > 1) else CATEGORIES
    for cat in cats_to_eval:
        items = test_cache[cat]
        gts   = [it["gt"] for it in items]
        maps_a = [it["map_A"] for it in items]
        maps_b = route_B_maps[cat][primary_seed]
        cp     = calib[cat][primary_seed]
        dec    = route_decisions(items, tau)

        m_a = route_A_metrics[cat]
        m_b = route_B_metrics[cat][primary_seed]
        oracle_route = "A" if m_a["pixel_auroc"] >= m_b["pixel_auroc"] else "B"
        m_oracle = m_a if oracle_route == "A" else m_b

        m_d, m_e, m_f, sd, se, sf, nA, nB = apply_variant(
            maps_a, maps_b, gts, dec, cp, compute_pro=compute_pro)

        out[cat] = dict(
            always_A=(m_a, len(items), 0, None, None),
            always_B=(m_b, 0, len(items), None, None),
            oracle=(m_oracle, oracle_route, None, None),
            raw=(m_d, nA, nB, sd[0], sd[1]),
            minmax=(m_e, nA, nB, se[0], se[1]),
            calib=(m_f, nA, nB, sf[0], sf[1]),
        )
        print(f"    Variants done for {cat} (nA={nA}, nB={nB}).")

    # Pooled
    if len(CATEGORIES) > 1:
        all_gts_p   = [it["gt"]    for cat in CATEGORIES for it in test_cache[cat]]
        all_maps_a_p = [it["map_A"] for cat in CATEGORIES for it in test_cache[cat]]
        all_maps_b_p = [route_B_maps[cat][primary_seed][i]
                        for cat in CATEGORIES for i in range(len(test_cache[cat]))]
        all_items_p  = [it for cat in CATEGORIES for it in test_cache[cat]]
        all_dec_p    = route_decisions(all_items_p, tau)
        # pooled calibration: use carpet params as proxy (best we can without per-image cat)
        # Instead: iterate by cat to apply correct params
        maps_dp, maps_ep, maps_fp = [], [], []
        for cat in CATEGORIES:
            items = test_cache[cat]
            maps_a_ = [it["map_A"] for it in items]
            maps_b_ = route_B_maps[cat][primary_seed]
            cp_ = calib[cat][primary_seed]
            dec_ = route_decisions(items, tau)
            for i, d in enumerate(dec_):
                ma, mb = maps_a_[i], maps_b_[i]
                maps_dp.append(ma if d else mb)
                lo, hi = (ma if d else mb).min(), (ma if d else mb).max()
                maps_ep.append(((ma if d else mb) - lo) / (hi - lo + 1e-8))
                if d:
                    maps_fp.append((ma - cp_["mu_a"]) / (cp_["std_a"] + 1e-8))
                else:
                    maps_fp.append((mb - cp_["mu_b"]) / (cp_["std_b"] + 1e-8))

        # Reuse precomputed pooled metrics for A and B
        m_a_p = route_A_metrics["POOLED"] if compute_pro else dict(pixel_auroc=route_A_metrics["POOLED"]["pixel_auroc"], pixel_ap=route_A_metrics["POOLED"]["pixel_ap"])
        m_b_p = route_B_metrics["POOLED"][primary_seed] if compute_pro else dict(pixel_auroc=route_B_metrics["POOLED"][primary_seed]["pixel_auroc"], pixel_ap=route_B_metrics["POOLED"][primary_seed]["pixel_ap"])
        
        m_d_p = compute_metrics(maps_dp, all_gts_p, compute_pro=compute_pro)
        m_e_p = compute_metrics(maps_ep, all_gts_p, compute_pro=compute_pro)
        m_f_p = compute_metrics(maps_fp, all_gts_p, compute_pro=compute_pro)
        oracle_p = m_a_p if m_a_p["pixel_auroc"] >= m_b_p["pixel_auroc"] else m_b_p
        nA_p = sum(all_dec_p); nB_p = len(all_dec_p) - nA_p

        if compute_pro:
            def pool_sub(maps_x):
                iA = [i for i, d in enumerate(all_dec_p) if d]
                iB = [i for i, d in enumerate(all_dec_p) if not d]
                sA = compute_metrics([maps_x[i] for i in iA],
                                      [all_gts_p[i] for i in iA],
                                      compute_pro=compute_pro) if iA else None
                sB = compute_metrics([maps_x[i] for i in iB],
                                      [all_gts_p[i] for i in iB],
                                      compute_pro=compute_pro) if iB else None
                return sA, sB

            sd_p = pool_sub(maps_dp)
            se_p = pool_sub(maps_ep)
            sf_p = pool_sub(maps_fp)
        else:
            sd_p = se_p = sf_p = (None, None)
        print(f"    POOLED done (nA={nA_p}, nB={nB_p}).")

        out["POOLED"] = dict(
            always_A=(m_a_p, len(all_gts_p), 0, None, None),
            always_B=(m_b_p, 0, len(all_gts_p), None, None),
            oracle=(oracle_p, "?", None, None),
            raw=(m_d_p, nA_p, nB_p, sd_p[0], sd_p[1]),
            minmax=(m_e_p, nA_p, nB_p, se_p[0], se_p[1]),
            calib=(m_f_p, nA_p, nB_p, sf_p[0], sf_p[1]),
        )

    return out


print(f"  Evaluating router variants at tau = {tau_median:.7f} (median) ...")
variant_results = evaluate_all_variants(tau_median, compute_pro=True)

# TAU SWEEP (no AU-PRO to keep it fast)
print("  Tau sweep ...")
sweep_results = {}
for p in pct_levels:
    tau_v = tau_by_pct[p]
    r = evaluate_all_variants(tau_v, compute_pro=False, pooled_only=True)
    pool_key = "POOLED" if "POOLED" in r else CATEGORIES[0]
    sweep_results[p] = dict(
        tau=tau_v,
        nA=r[pool_key]["raw"][1],
        nB=r[pool_key]["raw"][2],
        raw_auc=r[pool_key]["raw"][0]["pixel_auroc"],
        raw_ap=r[pool_key]["raw"][0]["pixel_ap"],
        cal_auc=r[pool_key]["calib"][0]["pixel_auroc"],
        cal_ap=r[pool_key]["calib"][0]["pixel_ap"],
    )
    print(f"    Sweep {p}th pct done (tau={tau_v:.7f}, nA={r[pool_key]['raw'][1]}, nB={r[pool_key]['raw'][2]}).")

# ---------------------------------------------------------------------------
# 17.  PRINT SUMMARY TO TERMINAL
# ---------------------------------------------------------------------------

def fmt(v, decimals=2):
    return f"{v:.{decimals}f}" if np.isfinite(v) else "NaN"

SEP = "=" * 78

print(f"\n{SEP}")
print("  ROUTE-ONLY BASELINES")
print(SEP)
print(f"{'Category':<10} {'Route':<10} {'Px AUROC':>10} {'Px AP':>8} {'AU-PRO':>8} {'Img AUROC':>10}")
print("-" * 60)
keys = CATEGORIES + (["POOLED"] if len(CATEGORIES) > 1 else [])
for cat in keys:
    ra = route_A_metrics[cat]
    print(f"{cat:<10} {'RouteA':<10} {fmt(ra['pixel_auroc']):>10} {fmt(ra['pixel_ap']):>8} "
          f"{fmt(ra['au_pro']):>8} {fmt(ra['image_auroc']):>10}")
    rb = route_B_agg[cat]
    print(f"{'':<10} {'RouteB':10} "
          f"{fmt(rb['pixel_auroc'][0])}+/-{fmt(rb['pixel_auroc'][1]):>5} "
          f"{fmt(rb['pixel_ap'][0])}+/-{fmt(rb['pixel_ap'][1]):>5} "
          f"{fmt(rb['au_pro'][0])}+/-{fmt(rb['au_pro'][1]):>5} "
          f"{fmt(rb['image_auroc'][0])}+/-{fmt(rb['image_auroc'][1]):>5}")

# ---------------------------------------------------------------------------
# 18.  WRITE summary.md
# ---------------------------------------------------------------------------

print("\n--- Writing results/summary.md ---")

def md_var(v):
    return f"{v:.6f}" if np.isfinite(v) else "NaN"

def mds(v):
    return f"{v:.2f}" if np.isfinite(v) else "NaN"

with open("results/summary.md", "w", encoding="utf-8") as f:

    f.write("# eval_router_full.py -- Comprehensive Benchmark Summary\n\n")

    # Env
    f.write("## 1  Environment\n\n")
    f.write(f"| Key | Value |\n|---|---|\n")
    f.write(f"| GPU | {gpu_name} |\n")
    f.write(f"| CUDA | {torch.version.cuda} |\n")
    f.write(f"| PyTorch | {torch.__version__} |\n")
    f.write(f"| Seeds | {SEEDS} |\n")
    f.write(f"| K-shot | {K_SHOT} |\n")
    f.write(f"| Categories | {CATEGORIES} |\n")
    f.write(f"| AU-PRO thresholds | {NUM_THRESH} linear |\n\n")

    # Sanity check
    f.write("## 2  Sanity Check vs Reference Numbers\n\n")
    f.write("```\n")
    for line in sanity_lines:
        f.write(line + "\n")
    f.write("```\n\n")

    # AU-PRO old vs new
    f.write("## 3  AU-PRO: Old vs New (carpet / Route B / seed 42)\n\n")
    f.write(f"| | Value |\n|---|---|\n")
    f.write(f"| New AU-PRO | {_new_aupro*100:.4f}% |\n")
    f.write(f"| Old AU-PRO | {_old_aupro*100:.4f}% |\n")
    f.write(f"| Absolute diff | {abs(_new_aupro - _old_aupro):.6f} |\n")
    f.write(f"| Thresholds <= FPR 0.3 | {_n_below_s} / {NUM_THRESH} |\n\n")

    # Variance distribution
    f.write("## 4  Variance (sigma^2_spatial) Distribution\n\n")
    f.write("| Category | Split | n | Min | Median | Mean | Max |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for cat in CATEGORIES:
        for split in ["train_good", "test_good", "test_defect"]:
            d = var_table[cat][split]
            f.write(f"| {cat} | {split} | {d['n']} | {md_var(d['min'])} | "
                    f"{md_var(d['median'])} | {md_var(d['mean'])} | {md_var(d['max'])} |\n")

    # Tau calibration
    f.write("\n## 5  Threshold tau Calibration\n\n")
    f.write(f"Pooled train/good sigma^2 N={len(all_train_sigma2)}\n\n")
    f.write("| Percentile | tau |\n|---|---|\n")
    f.write(f"| 50 (median, **selected**) | `{tau_median:.7f}` |\n")
    for p in pct_levels:
        f.write(f"| {p} | `{tau_by_pct[p]:.7f}` |\n")

    # Route-only baselines
    f.write("\n## 6  Route-Only Baselines\n\n")
    f.write("| Category | Route | Pixel AUROC | Pixel AP | AU-PRO | Image AUROC |\n")
    f.write("|---|---|---|---|---|---|\n")
    for cat in keys:
        ra = route_A_metrics[cat]
        f.write(f"| **{cat}** | Route A | {mds(ra['pixel_auroc'])} | "
                f"{mds(ra['pixel_ap'])} | {mds(ra['au_pro'])} | {mds(ra['image_auroc'])} |\n")
        rb = route_B_agg[cat]
        f.write(f"| | Route B (mean+/-std) | "
                f"{mds(rb['pixel_auroc'][0])}+/-{mds(rb['pixel_auroc'][1])} | "
                f"{mds(rb['pixel_ap'][0])}+/-{mds(rb['pixel_ap'][1])} | "
                f"{mds(rb['au_pro'][0])}+/-{mds(rb['au_pro'][1])} | "
                f"{mds(rb['image_auroc'][0])}+/-{mds(rb['image_auroc'][1])} |\n")

    # Per-seed Route B
    f.write("\n### Per-Seed Route B Details\n\n")
    f.write("| Category | Seed | Pixel AUROC | Pixel AP | AU-PRO | Image AUROC |\n")
    f.write("|---|---|---|---|---|---|\n")
    for cat in keys:
        for seed in SEEDS:
            rb = route_B_metrics[cat][seed]
            f.write(f"| {cat} | {seed} | {mds(rb['pixel_auroc'])} | "
                    f"{mds(rb['pixel_ap'])} | {mds(rb['au_pro'])} | {mds(rb['image_auroc'])} |\n")

    # Router variants
    f.write("\n## 7  Router Variants (tau = median)\n\n")
    f.write("| Variant | Category | Routed A/B | Pixel AUROC | Pixel AP | AU-PRO | Image AUROC |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    variant_labels = [
        ("always_A", "(a) Always-A"),
        ("always_B", "(b) Always-B"),
        ("oracle",   "(c) Oracle"),
        ("raw",      "(d) Raw concat"),
        ("minmax",   "(e) Min-Max"),
        ("calib",    "(f) Calibrated"),
    ]
    for vk, vlabel in variant_labels:
        for cat in (CATEGORIES + (["POOLED"] if len(CATEGORIES) > 1 else [])):
            if cat not in variant_results:
                continue
            e = variant_results[cat][vk]
            m = e[0]
            nA_str = str(e[1]) if e[1] is not None else "-"
            nB_str = str(e[2]) if e[2] is not None else "-"
            cnt = f"{nA_str}/{nB_str}"
            f.write(f"| **{vlabel}** | {cat} | {cnt} | {mds(m['pixel_auroc'])} | "
                    f"{mds(m['pixel_ap'])} | {mds(m['au_pro'])} | {mds(m['image_auroc'])} |\n")

    # Subset A/B rows
    f.write("\n### Subset A / Subset B Breakdown (variants d, e, f)\n\n")
    f.write("| Variant | Category | Subset | Pixel AUROC | Pixel AP | AU-PRO |\n")
    f.write("|---|---|---|---|---|---|\n")
    for vk, vlabel in [("raw", "(d) Raw"), ("minmax", "(e) Min-Max"), ("calib", "(f) Calib")]:
        for cat in (CATEGORIES + (["POOLED"] if len(CATEGORIES) > 1 else [])):
            if cat not in variant_results:
                continue
            e = variant_results[cat][vk]
            for sub_label, sub_m in [("Subset A", e[3]), ("Subset B", e[4])]:
                if sub_m is None:
                    f.write(f"| {vlabel} | {cat} | {sub_label} | - | - | - |\n")
                else:
                    f.write(f"| **{vlabel}** | {cat} | {sub_label} | "
                            f"{mds(sub_m['pixel_auroc'])} | {mds(sub_m['pixel_ap'])} | "
                            f"{mds(sub_m['au_pro'])} |\n")

    # Tau sweep
    f.write("\n## 8  tau Sweep (pooled, variants d/f, AU-PRO omitted for speed)\n\n")
    f.write("| Percentile | tau | A/B | Raw Px AUROC | Raw AP | Calib Px AUROC | Calib AP |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for p in pct_levels:
        sr = sweep_results[p]
        f.write(f"| {p}th | `{sr['tau']:.7f}` | {sr['nA']}/{sr['nB']} | "
                f"{mds(sr['raw_auc'])} | {mds(sr['raw_ap'])} | "
                f"{mds(sr['cal_auc'])} | {mds(sr['cal_ap'])} |\n")

    # Correlations
    f.write("\n## 9  Descriptors vs Gap (Spearman rho; anomalous images only)\n\n")
    f.write("| Category | n | sigma^2 rho (p) | Off-diag rho (p) | Top-5 rho (p) |\n")
    f.write("|---|---|---|---|---|\n")
    corr_keys = CATEGORIES + (["POOLED"] if "POOLED" in corr_results else [])
    for cat in corr_keys:
        c = corr_results[cat]
        def rp(desc):
            rho, pv = c[desc]
            if np.isnan(rho):
                return "NaN"
            return f"{rho:.3f} ({pv:.3f})"
        f.write(f"| {cat} | {c['n']} | {rp('sigma2_spatial')} | "
                f"{rp('off_diag_mean')} | {rp('top5_mean')} |\n")

    f.write("\n## 10  Artifacts\n\n")
    f.write("- `results/per_image.csv` -- per-image metrics, descriptors, gap, routing\n")
    f.write("- `results/summary.md`   -- this file\n")

print("  results/summary.md written.")
print(f"\nAll done. Check results/per_image.csv and results/summary.md.")
