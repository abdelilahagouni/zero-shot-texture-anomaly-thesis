"""
Metrics with the project's own evaluation code (src/evaluation.py, fixed AU-PRO; not modified) for
FCA, aota, Route A and Route B (seeds 42, 123, 2026 as in src/eval_router_full.py; reported as the 3-seed mean).

Maps on the 518x518 grid of the ground-truth masks:
  FCA / aota : raw maps from results/fca_baseline/maps (float16), resized with common.to_grid (bilinear).
  Route A, B : recomputed with src/mechanism_common.py (DINOv2 ViT-B/14 pipeline of eval_router_full.py).
Per texture, over all its test images (good + defective):
  pixel AP, AU-PRO (FPR 0.3, 1000 thresholds), pixel AUROC   -> src/evaluation.py
  image AUROC = sklearn roc_auc_score(label, max of map)
  5-texture average = mean of the 5 per-texture values; Route B = mean (and std) over the 3 seeds.
Two versions:
  full    : whole 518x518 map and mask
  crop10  : 10% border of every map and mask removed (52 px per side -> 414x414 centre), as the FCA/Aota
            protocol; image labels then come from the CROPPED mask (as in the repo's evaluate_experiment.py),
            and the number of defective images whose defect lies entirely in the border is reported.
Per-image pixel AUROC of the 382 defective images (full and crop10; NaN if the crop has no defect pixels).
Writes results/fca_baseline/metrics/.
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import DATA, RES, TEXTURES, GRID, BORDER, test_items, load_gt, map_path, to_grid, crop  # noqa: E402
from evaluation import compute_pixel_auroc, compute_pixel_ap, compute_au_pro  # noqa: E402

SEEDS = [42, 123, 2026]
OUT = os.path.join(RES, "metrics")
METRICS = ["pixel_ap", "au_pro", "pixel_auroc", "image_auroc"]


def metrics(maps, gts):
    from sklearn.metrics import roc_auc_score
    lab = np.array([int(g.any()) for g in gts])
    img = roc_auc_score(lab, [m.max() for m in maps]) * 100 if 0 < lab.sum() < len(lab) else float("nan")
    return {"pixel_ap": compute_pixel_ap(maps, gts) * 100, "au_pro": compute_au_pro(maps, gts) * 100,
            "pixel_auroc": compute_pixel_auroc(maps, gts) * 100, "image_auroc": img}


def per_image_auroc(m, g):
    from sklearn.metrics import roc_auc_score
    y = g.ravel() > 0
    return roc_auc_score(y, m.ravel()) * 100 if 0 < y.sum() < y.size else float("nan")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from PIL import Image
    from mechanism_common import Backbone
    T0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bb = Backbone()
    rows, per_img, n_border_only = [], [], {}
    torch.cuda.reset_peak_memory_stats()
    for cat in TEXTURES:
        t0 = time.time()
        items = test_items(cat)
        gts = [load_gt(mp) for _, _, _, mp in items]
        train = [bb.tokens(Image.open(p)) for p in sorted(glob.glob(os.path.join(DATA, cat, "train", "good", "*.png")))]
        banks = {sd: bb.bank(train, sd) for sd in SEEDS}
        maps = {"FCA": [], "aota": [], "A": []}
        maps.update({f"B{sd}": [] for sd in SEEDS})
        for d, fn, p, _ in items:
            maps["FCA"].append(to_grid(np.load(map_path("fca", cat, d, fn))))
            maps["aota"].append(to_grid(np.load(map_path("aota", cat, d, fn))))
            tok = bb.tokens(Image.open(p))
            maps["A"].append(bb.post(1.0 - (tok @ tok.T).mean(dim=1)))
            for sd in SEEDS:
                maps[f"B{sd}"].append(bb.post(1.0 - (tok @ banks[sd].T).max(dim=1)[0]))
        t_maps = time.time() - t0
        gts_c = [crop(g) for g in gts]
        n_border_only[cat] = int(sum(g.any() and not gc.any() for g, gc in zip(gts, gts_c)))
        for meth, ms in maps.items():
            ms_c = [crop(m) for m in ms]
            for ver, M, G in (("full", ms, gts), ("crop10", ms_c, gts_c)):
                r = metrics(M, G)
                rows.append({"texture": cat, "method": meth, "version": ver, **r})
        for i, (d, fn, _, _) in enumerate(items):
            if d == "good":
                continue
            r = {"category": cat, "defect_type": d, "file": fn, "area_px": int(gts[i].sum()),
                 "area_px_crop10": int(gts_c[i].sum())}
            for meth, ms in maps.items():
                r[f"{meth}_auroc"] = per_image_auroc(ms[i], gts[i])
                r[f"{meth}_auroc_crop10"] = per_image_auroc(crop(ms[i]), gts_c[i])
            per_img.append(r)
        print(f"{cat}: {len(items)} test images, maps {t_maps:.0f}s, total {time.time() - t0:.0f}s, "
              f"peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated", flush=True)

    df = pd.DataFrame(rows)
    pim = pd.DataFrame(per_img)
    for v in ("", "_crop10"):
        pim[f"B_auroc{v}"] = pim[[f"B{sd}_auroc{v}" for sd in SEEDS]].mean(axis=1)
    pim.to_csv(os.path.join(OUT, "per_image_defective.csv"), index=False)
    df.to_csv(os.path.join(OUT, "per_texture_metrics_all_seeds.csv"), index=False)

    # table: per texture and average; B = 3-seed mean (std over seeds)
    tab = []
    for ver in ("full", "crop10"):
        for cat in TEXTURES + ["average"]:
            sub = df[df["version"] == ver] if cat == "average" else df[(df["version"] == ver) & (df["texture"] == cat)]
            for meth in ["FCA", "aota", "A", "B"]:
                if meth == "B":
                    per_seed = [sub[sub["method"] == f"B{sd}"][METRICS].mean() for sd in SEEDS]
                    vals = pd.concat(per_seed, axis=1)
                    r = {k: float(vals.loc[k].mean()) for k in METRICS}
                    r.update({f"{k}_std_seeds": float(vals.loc[k].std(ddof=1)) for k in METRICS})
                else:
                    r = {k: float(sub[sub["method"] == meth][k].mean()) for k in METRICS}
                tab.append({"version": ver, "texture": cat, "method": meth, **r})
    tab = pd.DataFrame(tab)
    tab.to_csv(os.path.join(OUT, "metrics_table.csv"), index=False)

    # check Route A / B42 per-image AUROC against the earlier size_effect_robustness run
    prev = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "size_effect_robustness",
                                    "per_image_metrics.csv"))
    j = pim.merge(prev[["category", "defect_type", "file", "A_auroc", "B42_auroc"]],
                  on=["category", "defect_type", "file"], suffixes=("", "_prev"))
    check = {"n_matched": int(len(j)),
             "max_abs_diff_A": float((j["A_auroc"] - j["A_auroc_prev"]).abs().max()),
             "max_abs_diff_B42": float((j["B42_auroc"] - j["B42_auroc_prev"]).abs().max())}

    L = ["METRICS with src/evaluation.py (pixel AP, AU-PRO@0.3, pixel AUROC) + image AUROC (max of map); "
         "B = mean of seeds 42/123/2026", f"crop10 = {BORDER}px border removed on each side of the {GRID}x{GRID} "
         f"grid; defective images with the defect only in the border (become normal in crop10): {n_border_only}", "",
         f"Route A / B42 per-image AUROC vs size_effect_robustness: n {check['n_matched']}, max |diff| A "
         f"{check['max_abs_diff_A']:.6f}, B42 {check['max_abs_diff_B42']:.6f}", ""]
    hdr = f"{'texture':<8} {'method':<5} " + "  ".join(f"{k + ' full':>16} {k + ' crop10':>18}" for k in METRICS)
    L.append(hdr)
    for cat in TEXTURES + ["average"]:
        for meth in ["FCA", "aota", "A", "B"]:
            f = tab[(tab.texture == cat) & (tab.method == meth)].set_index("version")
            L.append(f"{cat:<8} {meth:<5} " + "  ".join(f"{f.loc['full', k]:16.2f} {f.loc['crop10', k]:18.2f}"
                                                         for k in METRICS))
        L.append("")
    L.append("Per-image pixel AUROC of the defective images (mean [median]; crop10 excludes NaN):")
    for meth in ["FCA", "aota", "A", "B"]:
        a, c = pim[f"{meth}_auroc"], pim[f"{meth}_auroc_crop10"]
        L.append(f"  {meth:<5} full n {a.notna().sum()} mean {a.mean():.2f} [median {a.median():.2f}]   "
                 f"crop10 n {c.notna().sum()} mean {c.mean():.2f} [median {c.median():.2f}]")
    L.append(f"\nTotal {time.time() - T0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated")
    txt = "\n".join(L)
    open(os.path.join(OUT, "metrics.txt"), "w", encoding="utf-8").write(txt)
    json.dump({"table": tab.to_dict("records"), "border_only_defective_images": n_border_only,
               "route_ab_reproduction_check": check, "n_defective_images": int(len(pim)),
               "seconds": round(time.time() - T0, 1)}, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
