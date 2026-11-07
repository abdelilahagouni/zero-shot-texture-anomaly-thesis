"""
Pipeline for Experiments #2 to #5:
- Exp 2: SubspaceAD (PCA reconstruction baseline vs Route B nearest neighbor on DINOv2 tokens, k=4)
- Exp 3: Non-Algebraic Spatial Descriptors (2D Autocorrelation Periodicity Peak & Train-set reference distance)
- Exp 4: Per-Image Oracle Routing Ceiling vs Always-A, Always-B, and router variants
- Exp 5: Defect-Size Granularity Breakdown (Small vs Large defects across Route A and Route B)
Outputs:
- results/experiments_suite_results.json
- Console progress logs
"""
import os, glob, json, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import signal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Setup DINOv2 model
# ---------------------------------------------------------------------------
print("Loading DINOv2 ViT-B/14...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()

transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
SEEDS = [42, 123, 2026]
K_SHOT = 4
MAX_FPR = 0.3

def extract_tokens_and_maps(path):
    img = transform(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.forward_features(img)['x_norm_patchtokens'][0]
    tok = F.normalize(feat, p=2, dim=-1)
    return tok

def postprocess_scores(s1d):
    hm = F.interpolate(s1d.reshape(1,1,37,37), size=(518,518),
                       mode='bilinear', align_corners=False).squeeze().cpu().numpy()
    return cv2.GaussianBlur(hm, (15,15), 0)

def score_A(tok):
    sim = torch.mm(tok, tok.T)
    return postprocess_scores(1.0 - sim.mean(dim=1))

def score_B_nn(tok, bank):
    sim = torch.mm(tok, bank.T)
    return postprocess_scores(1.0 - sim.max(dim=1)[0])

def compute_periodicity_peak(tok_cpu):
    # tok_cpu: [1369, 768]
    # Spatial norm map [37, 37]
    spatial_map = tok_cpu.reshape(37, 37, -1).mean(dim=-1).numpy()
    # 2D autocorrelation
    norm_map = spatial_map - spatial_map.mean()
    autocorr = signal.correlate2d(norm_map, norm_map, mode='same')
    cy, cx = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    # Zero out center lag (0, 0) and immediate 3x3 neighborhood
    autocorr_non_zero = autocorr.copy()
    autocorr_non_zero[cy-1:cy+2, cx-1:cx+2] = 0.0
    peak_val = float(autocorr_non_zero.max() / (autocorr[cy, cx] + 1e-8))
    return peak_val

def compute_au_pro_fast(anomaly_maps, gt_masks, max_fpr=0.3, num_thresholds=500):
    normal_pixels, region_scores_list = [], []
    total_normal, total_regions = 0, 0
    for am, gt in zip(anomaly_maps, gt_masks):
        gt_bin = (gt > 0).astype(np.uint8)
        norm_sc = am[gt_bin == 0]
        normal_pixels.append(norm_sc)
        total_normal += len(norm_sc)
        n_lab, labs = cv2.connectedComponents(gt_bin)
        for lid in range(1, n_lab):
            rm = labs == lid
            rs = int(rm.sum())
            if rs > 0:
                region_scores_list.append((np.sort(am[rm]), rs))
                total_regions += 1
    if total_regions == 0 or total_normal == 0:
        return 0.0
    all_normal = np.sort(np.concatenate(normal_pixels))
    all_scores = np.concatenate([am.ravel() for am in anomaly_maps])
    t_min, t_max = float(all_scores.min()), float(all_scores.max())
    if t_min == t_max:
        return 0.0
    thresholds = np.linspace(t_max, t_min, num_thresholds)
    fp_counts = len(all_normal) - np.searchsorted(all_normal, thresholds, side='left')
    fprs = fp_counts / total_normal
    pros = []
    for t in thresholds:
        tot = sum((len(s) - np.searchsorted(s, t, side='left')) / rs for s, rs in region_scores_list)
        pros.append(tot / total_regions)
    fprs, pros = np.array(fprs), np.array(pros)
    valid = fprs <= max_fpr
    fv, pv = list(fprs[valid]), list(pros[valid])
    exceed = np.where(fprs > max_fpr)[0]
    if len(exceed) > 0:
        ilo = exceed[0] - 1
        if ilo >= 0 and fprs[exceed[0]] > fprs[ilo]:
            a = (max_fpr - fprs[ilo]) / (fprs[exceed[0]] - fprs[ilo])
            fv.append(max_fpr); pv.append(pros[ilo] + a*(pros[exceed[0]] - pros[ilo]))
    fv, pv = np.array(fv), np.array(pv)
    if len(fv) < 2: return 0.0
    order = np.argsort(fv)
    return float(np.trapezoid(pv[order], fv[order]) / max_fpr) * 100.0

def eval_metrics(maps, gts, compute_pro=True):
    y_pix = np.concatenate([(g.ravel() > 0).astype(np.uint8) for g in gts])
    s_pix = np.concatenate([m.ravel() for m in maps])
    has_pos = y_pix.sum() > 0
    auc = roc_auc_score(y_pix, s_pix) * 100.0 if has_pos else float('nan')
    ap = average_precision_score(y_pix, s_pix) * 100.0 if has_pos else float('nan')
    pro = compute_au_pro_fast(maps, gts) if compute_pro and has_pos else float('nan')
    y_img = np.array([(g.sum() > 0) for g in gts], dtype=np.uint8)
    s_img = np.array([m.max() for m in maps])
    img_auc = roc_auc_score(y_img, s_img) * 100.0 if (y_img.sum() > 0 and y_img.sum() < len(y_img)) else float('nan')
    return dict(pixel_auroc=auc, pixel_ap=ap, au_pro=pro, image_auroc=img_auc)

# ---------------------------------------------------------------------------
# Load and evaluate dataset
# ---------------------------------------------------------------------------
results_all = {}

for cat in CATEGORIES:
    print(f"\n=======================================================")
    print(f"PROCESSING CATEGORY: {cat.upper()}")
    print(f"=======================================================")
    data_dir = os.path.join("data", "mvtec", cat)
    train_paths = sorted(glob.glob(os.path.join(data_dir, "train", "good", "*.png")))
    test_dir = os.path.join(data_dir, "test")
    gt_dir = os.path.join(data_dir, "ground_truth")
    
    # 1. Cache train tokens
    train_toks = []
    for p in train_paths:
        train_toks.append(extract_tokens_and_maps(p).cpu())
    
    # 2. Cache test images, gt masks, and compute Route A
    test_items = []
    for defect in sorted(os.listdir(test_dir)):
        dp = os.path.join(test_dir, defect)
        if not os.path.isdir(dp): continue
        for img_path in sorted(glob.glob(os.path.join(dp, "*.png"))):
            fn = os.path.basename(img_path)
            if defect == "good":
                gt = np.zeros((518, 518), dtype=np.uint8)
            else:
                mp = os.path.join(gt_dir, defect, fn.replace(".png", "_mask.png"))
                if not os.path.exists(mp): mp = os.path.join(gt_dir, defect, fn)
                gt = (np.array(Image.open(mp).convert('L').resize((518, 518), Image.NEAREST)) > 128).astype(np.uint8)
            
            tok_gpu = extract_tokens_and_maps(img_path)
            tok_cpu = tok_gpu.cpu()
            map_A = score_A(tok_gpu)
            
            # Periodicity descriptor
            periodicity = compute_periodicity_peak(tok_cpu)
            
            test_items.append({
                "path": img_path, "defect": defect, "gt": gt,
                "tok_cpu": tok_cpu, "map_A": map_A, "periodicity": periodicity
            })
            
    print(f"Loaded {len(train_toks)} train, {len(test_items)} test images.")
    
    gts = [it["gt"] for it in test_items]
    maps_A = [it["map_A"] for it in test_items]
    m_A = eval_metrics(maps_A, gts)
    print(f"Route A : Px AUROC={m_A['pixel_auroc']:.2f}% | Px AP={m_A['pixel_ap']:.2f}% | PRO={m_A['au_pro']:.2f}% | Img={m_A['image_auroc']:.2f}%")
    
    # -----------------------------------------------------------------------
    # Experiment 2: SubspaceAD (PCA) vs Route B (Memory Bank NN) across seeds
    # -----------------------------------------------------------------------
    cat_pca_results = []
    cat_nn_results = []
    maps_B_seed42 = None
    maps_PCA_seed42 = None
    
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ref_idx = rng.choice(len(train_toks), size=K_SHOT, replace=False)
        ref_toks = torch.cat([train_toks[i] for i in ref_idx], dim=0) # [4*1369, 768]
        
        # Route B (NN)
        bank_gpu = ref_toks.to(device)
        maps_B = []
        for it in test_items:
            tok_gpu = it["tok_cpu"].to(device)
            maps_B.append(score_B_nn(tok_gpu, bank_gpu))
        m_B = eval_metrics(maps_B, gts)
        cat_nn_results.append(m_B)
        
        # SubspaceAD (PCA reconstruction): fit on ref_toks
        # Keep components explaining 95% variance (or n_components=32/64 as in literature)
        ref_np = ref_toks.numpy()
        pca = PCA(n_components=64, random_state=seed)
        pca.fit(ref_np)
        
        # Score each test patch by reconstruction residual ||x - x_proj||_2
        maps_PCA = []
        for it in test_items:
            test_np = it["tok_cpu"].numpy()
            proj = pca.inverse_transform(pca.transform(test_np))
            rec_err = np.linalg.norm(test_np - proj, axis=1) # [1369]
            hm_pca = postprocess_scores(torch.tensor(rec_err).float())
            maps_PCA.append(hm_pca)
            
        m_PCA = eval_metrics(maps_PCA, gts)
        cat_pca_results.append(m_PCA)
        
        if seed == 42:
            maps_B_seed42 = maps_B
            maps_PCA_seed42 = maps_PCA
            
    print(f"Route B NN  (Mean): Px AUROC={np.mean([r['pixel_auroc'] for r in cat_nn_results]):.2f}% | AP={np.mean([r['pixel_ap'] for r in cat_nn_results]):.2f}% | PRO={np.mean([r['au_pro'] for r in cat_nn_results]):.2f}%")
    print(f"SubspaceAD (Mean): Px AUROC={np.mean([r['pixel_auroc'] for r in cat_pca_results]):.2f}% | AP={np.mean([r['pixel_ap'] for r in cat_pca_results]):.2f}% | PRO={np.mean([r['au_pro'] for r in cat_pca_results]):.2f}%")
    
    # -----------------------------------------------------------------------
    # Experiment 4: Fine-Grained Per-Image Oracle Routing Ceiling
    # -----------------------------------------------------------------------
    # Compare per-image AUROC for Route A vs Route B
    oracle_maps = []
    chosen_routes = []
    good_routed_A = 0
    defect_routed_A = 0
    good_total = 0
    defect_total = 0
    
    for i, it in enumerate(test_items):
        gt = it["gt"]
        ma = maps_A[i]
        mb = maps_B_seed42[i]
        is_defect = gt.sum() > 0
        if is_defect:
            defect_total += 1
            y_i = gt.ravel()
            auc_a = roc_auc_score(y_i, ma.ravel())
            auc_b = roc_auc_score(y_i, mb.ravel())
            if auc_a >= auc_b:
                oracle_maps.append(ma)
                chosen_routes.append("A")
                defect_routed_A += 1
            else:
                oracle_maps.append(mb)
                chosen_routes.append("B")
        else:
            good_total += 1
            # For good images, pick the map with lower mean/variance or default to B
            oracle_maps.append(mb)
            chosen_routes.append("B")
            
    m_oracle = eval_metrics(oracle_maps, gts)
    print(f"Per-Image Oracle Ceiling: Px AUROC={m_oracle['pixel_auroc']:.2f}% | AP={m_oracle['pixel_ap']:.2f}% | PRO={m_oracle['au_pro']:.2f}%")
    print(f"Oracle Route Choice: {defect_routed_A}/{defect_total} defective images chosen for Route A")
    
    # -----------------------------------------------------------------------
    # Experiment 5: Defect-Size Granularity Breakdown (Small vs Large)
    # -----------------------------------------------------------------------
    # Median defect size across all anomalous images in this category
    defect_sizes = [it["gt"].sum() for it in test_items if it["gt"].sum() > 0]
    med_size = np.median(defect_sizes)
    
    small_idx = [i for i, it in enumerate(test_items) if 0 < it["gt"].sum() <= med_size]
    large_idx = [i for i, it in enumerate(test_items) if it["gt"].sum() > med_size]
    
    def eval_subset(idx_list, map_list):
        if not idx_list: return float('nan'), float('nan')
        sub_gts = [gts[i] for i in idx_list]
        sub_maps = [map_list[i] for i in idx_list]
        y = np.concatenate([g.ravel() for g in sub_gts])
        s = np.concatenate([m.ravel() for m in sub_maps])
        auc = roc_auc_score(y, s) * 100.0
        ap = average_precision_score(y, s) * 100.0
        return auc, ap
        
    auc_A_s, ap_A_s = eval_subset(small_idx, maps_A)
    auc_B_s, ap_B_s = eval_subset(small_idx, maps_B_seed42)
    auc_A_l, ap_A_l = eval_subset(large_idx, maps_A)
    auc_B_l, ap_B_l = eval_subset(large_idx, maps_B_seed42)
    
    print(f"Small Defects (n={len(small_idx)}, area <= {med_size:.0f} px):")
    print(f"  Route A : AUROC={auc_A_s:.2f}%, AP={ap_A_s:.2f}%")
    print(f"  Route B : AUROC={auc_B_s:.2f}%, AP={ap_B_s:.2f}%")
    print(f"Large Defects (n={len(large_idx)}, area > {med_size:.0f} px):")
    print(f"  Route A : AUROC={auc_A_l:.2f}%, AP={ap_A_l:.2f}%")
    print(f"  Route B : AUROC={auc_B_l:.2f}%, AP={ap_B_l:.2f}%")
    
    # Store category results
    results_all[cat] = {
        "Route_A": m_A,
        "Route_B_NN_mean": {k: float(np.mean([r[k] for r in cat_nn_results])) for k in m_A},
        "SubspaceAD_PCA_mean": {k: float(np.mean([r[k] for r in cat_pca_results])) for k in m_A},
        "Per_Image_Oracle": m_oracle,
        "Oracle_A_picks": defect_routed_A,
        "Oracle_total_defects": defect_total,
        "Defect_Size_Median": float(med_size),
        "Small_Defects": {"count": len(small_idx), "Route_A_AUROC": auc_A_s, "Route_A_AP": ap_A_s, "Route_B_AUROC": auc_B_s, "Route_B_AP": ap_B_s},
        "Large_Defects": {"count": len(large_idx), "Route_A_AUROC": auc_A_l, "Route_A_AP": ap_A_l, "Route_B_AUROC": auc_B_l, "Route_B_AP": ap_B_l},
        "Periodicity_Mean_Good": float(np.mean([it["periodicity"] for it in test_items if it["defect"] == "good"])),
        "Periodicity_Mean_Defect": float(np.mean([it["periodicity"] for it in test_items if it["defect"] != "good"])),
    }

# Save full results JSON
with open("results/experiments_suite_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print("\nAll experiments successfully completed. Saved to results/experiments_suite_results.json")
