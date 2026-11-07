"""
Image-level validation of Route A scores (rule fixed in advance).

Candidate score : mean of the Route A anomaly map over the whole image (higher = more anomalous)
Alternatives    : max of the map (current Route A image score), mean of the top 1% of the map,
                  sigma^2 (spatial feature variance, as in eval_router_full.py)
No other scores are evaluated.

Route A map: exactly as run_full_experiments_suite.py (DINOv2 ViT-B/14, 518x518, fp32 tokens,
L2-normalised, s = 1 - mean_j cos, bilinear 37->518, GaussianBlur 15x15).
sigma^2: torch.var(tokens, dim=0).mean() (unbiased, as eval_router_full.py line 390).

Image-level AUROC (defective vs good), with 95% bootstrap CIs that resample IMAGES
(good and defective resampled separately, 5000 resamples, seed 0).
Subsets: defective images split at the per-category median defect area (ground-truth mask at
518x518): small = <= median, large = > median; each subset is scored against ALL good images.

Writes to results/image_level_validation/ only.
"""
import argparse, datetime, glob, json, os, sys, time, platform
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECTS = ["bottle", "cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
SCORES = ["mean_map", "max_map", "top1pct_map", "sigma2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(REPO, "data", "mvtec"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "image_level_validation"))
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")

    import torch, torch.nn.functional as F, cv2
    from PIL import Image
    from torchvision import transforms
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cats_tex = [c for c in TEXTURES if os.path.isdir(os.path.join(a.data_root, c, "test"))]
    cats_obj = [c for c in OBJECTS if os.path.isdir(os.path.join(a.data_root, c, "test"))]
    missing_obj = [c for c in OBJECTS if c not in cats_obj]
    print(f"Textures found: {cats_tex}")
    print(f"Held-out object categories found: {cats_obj or 'NONE'}; missing: {missing_obj}")

    t0 = time.time()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()
    tf = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    print(f"Model load: {time.time() - t0:.1f}s")

    def per_image(cat):
        rows = []
        tdir = os.path.join(a.data_root, cat, "test")
        gdir = os.path.join(a.data_root, cat, "ground_truth")
        for d in sorted(os.listdir(tdir)):
            dp = os.path.join(tdir, d)
            if not os.path.isdir(dp):
                continue
            for p in sorted(glob.glob(os.path.join(dp, "*.png"))):
                fn = os.path.basename(p)
                if d == "good":
                    area = 0
                else:
                    mp = os.path.join(gdir, d, fn.replace(".png", "_mask.png"))
                    if not os.path.exists(mp):
                        mp = os.path.join(gdir, d, fn)
                    gt = np.array(Image.open(mp).convert("L").resize((518, 518), Image.NEAREST)) > 128
                    area = int(gt.sum())
                img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
                with torch.no_grad():
                    tok = F.normalize(model.forward_features(img)["x_norm_patchtokens"][0], p=2, dim=-1)
                raw = 1.0 - torch.mm(tok, tok.T).mean(dim=1)
                hm = F.interpolate(raw.reshape(1, 1, 37, 37), size=(518, 518), mode="bilinear",
                                   align_corners=False).squeeze().cpu().numpy()
                hm = cv2.GaussianBlur(hm, (15, 15), 0)
                flat = np.sort(hm.ravel())
                k = int(np.ceil(0.01 * flat.size))
                rows.append({"category": cat, "path": os.path.relpath(p, a.data_root), "label": d,
                             "is_defect": int(d != "good"), "defect_area_px": area,
                             "mean_map": float(hm.mean()), "max_map": float(flat[-1]),
                             "top1pct_map": float(flat[-k:].mean()),
                             "sigma2": float(torch.var(tok.float(), dim=0).mean().item()),
                             # equivalence check only (not a candidate score):
                             "_raw_score_mean": float(raw.mean().item()),
                             "_off_diag_mean": float(((torch.mm(tok, tok.T).sum() - torch.mm(tok, tok.T).trace())
                                                      / (1369 * 1368)).item())})
        return rows

    def auc_ci(y, s, rng):
        y, s = np.asarray(y), np.asarray(s)
        auc = roc_auc_score(y, s)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        bs = np.empty(a.n_boot)
        for b in range(a.n_boot):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            bs[b] = roc_auc_score(y[idx], s[idx])
        return float(auc * 100), float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)

    results, all_rows, timings = {}, [], {}
    for cat in cats_tex + cats_obj:
        t1 = time.time()
        rows = per_image(cat)
        all_rows += rows
        y = np.array([r["is_defect"] for r in rows])
        areas = np.array([r["defect_area_px"] for r in rows])
        med = float(np.median(areas[y == 1]))
        good = y == 0
        small = (y == 1) & (areas <= med)
        large = (y == 1) & (areas > med)
        res = {"group": "texture" if cat in TEXTURES else "object", "n_good": int(good.sum()),
               "n_defect": int((y == 1).sum()), "n_small": int(small.sum()), "n_large": int(large.sum()),
               "median_defect_area_px": med, "median_defect_area_pct": med / 518 ** 2 * 100}
        for subset, mask in (("all", np.ones_like(y, bool)), ("small", good | small), ("large", good | large)):
            res[subset] = {}
            for sc in SCORES:
                v = np.array([r[sc] for r in rows])
                rng = np.random.default_rng([0, SCORES.index(sc), ["all", "small", "large"].index(subset)])
                auc, lo, hi = auc_ci(y[mask], v[mask], rng)
                res[subset][sc] = {"auroc": auc, "ci95": [lo, hi]}
        v_mean = np.array([r["mean_map"] for r in rows])
        v_raw = np.array([r["_raw_score_mean"] for r in rows])
        v_s2 = np.array([r["sigma2"] for r in rows])
        v_od = np.array([r["_off_diag_mean"] for r in rows])
        res["equivalence_check"] = {
            "spearman_sigma2_vs_raw_score_mean": float(spearmanr(v_s2, v_raw).correlation),
            "spearman_sigma2_vs_off_diag": float(spearmanr(v_s2, v_od).correlation),
            "spearman_sigma2_vs_mean_map": float(spearmanr(v_s2, v_mean).correlation),
        }
        results[cat] = res
        timings[cat] = round(time.time() - t1, 1)
        print(f"{cat}: {len(rows)} images, {timings[cat]}s")

    import csv
    with open(os.path.join(a.out, "per_image_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Fixed success criterion
    def strictly_better(cat):
        m, x = results[cat]["small"]["mean_map"], results[cat]["small"]["max_map"]
        return m["auroc"] > x["auroc"] and m["ci95"][0] > x["ci95"][1]
    tex_wins = [c for c in cats_tex if strictly_better(c)]
    obj_higher = [c for c in cats_obj if results[c]["all"]["mean_map"]["auroc"] > results[c]["all"]["max_map"]["auroc"]]
    crit1 = len(tex_wins) >= 4
    crit2 = (len(obj_higher) > len(cats_obj) / 2) if cats_obj else None
    verdict = "supported" if crit1 and (crit2 is None or crit2) else "not supported"
    decision = {"textures_mean_beats_max_small_nonoverlapping_CI": tex_wins, "criterion_1_met": crit1,
                "heldout_available": cats_obj, "heldout_missing": missing_obj,
                "heldout_mean_higher_than_max": obj_higher, "criterion_2_met": crit2,
                "verdict": verdict,
                "note": "criterion 2 not evaluated: no held-out categories available" if not cats_obj else ""}
    env = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "argv": sys.argv, "n_boot": a.n_boot, "timings_s": timings,
           "total_s": round(time.time() - t0, 1)}
    with open(os.path.join(a.out, "image_level_auroc.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "decision": decision, "environment": env}, f, indent=1)

    L = []
    for cat in cats_tex + cats_obj:
        r = results[cat]
        L.append(f"{cat.upper()} ({r['group']}): good={r['n_good']} defect={r['n_defect']} "
                 f"(small={r['n_small']}, large={r['n_large']}, median area {r['median_defect_area_pct']:.2f}%)")
        for subset in ("all", "small", "large"):
            L.append(f"  {subset:<5} " + "  ".join(
                f"{sc}={r[subset][sc]['auroc']:6.2f} [{r[subset][sc]['ci95'][0]:6.2f},{r[subset][sc]['ci95'][1]:6.2f}]"
                for sc in SCORES))
        e = r["equivalence_check"]
        L.append(f"  equivalence (Spearman): sigma2~raw-score-mean {e['spearman_sigma2_vs_raw_score_mean']:+.4f}, "
                 f"sigma2~off_diag {e['spearman_sigma2_vs_off_diag']:+.4f}, sigma2~mean_map {e['spearman_sigma2_vs_mean_map']:+.4f}")
    L.append("")
    L.append(f"Criterion 1 (mean > max, non-overlapping 95% CIs, small subset, >= 4 of 5 textures): "
             f"{len(tex_wins)}/5 {tex_wins} -> {'met' if crit1 else 'NOT met'}")
    L.append(f"Criterion 2 (held-out majority): " + ("not evaluable, no held-out categories under data root"
             if not cats_obj else f"{len(obj_higher)}/{len(cats_obj)} -> {'met' if crit2 else 'NOT met'}"))
    L.append(f"VERDICT: {verdict}")
    L.append(f"Total time: {env['total_s']}s")
    txt = "\n".join(L)
    with open(os.path.join(a.out, "image_level_auroc.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
