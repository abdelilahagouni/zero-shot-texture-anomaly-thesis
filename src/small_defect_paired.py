"""
Paired per-image test: does Route A beat Route B on SMALL real defects?

Inputs (read only):
  results/per_image_latest_1791743390.csv          per-image pixel AUROC, Route A and Route B (seed 42),
                                                   written by src/eval_router_full.py (5 textures)
  results/image_level_validation/per_image_scores.csv   defect area (ground-truth mask at 518x518)

Per texture and per subset (small = area <= per-texture median of defective images, large = above):
  mean and median paired difference d = B - A (pixel AUROC, points), Wilcoxon signed-rank (two-sided),
  95% bootstrap CI of the mean difference (resampling images, 10,000), fraction of images with A > B.
Holm correction across the 5 textures, separately for small and large.
Difference-of-differences: mean d(large) - mean d(small), bootstrap CI (images resampled within each subset).

Criteria (fixed before running):
  C1 "Route A beats Route B on small defects": Holm p < 0.05 AND A > B (median d < 0) on >= 2 textures.
  C2 "Route B degrades relative to A with size": mean d(large) > mean d(small) with the bootstrap CI of
     d(large) - d(small) excluding zero, on >= 3 textures.
     (As worded in the request: "larger for large than for small".)
Writes to results/small_defect_paired/ only.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "small_defect_paired")
N_BOOT = 10000


def norm(p):
    p = p.replace("\\", "/")
    return p.split("data/mvtec/", 1)[-1]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    pi = pd.read_csv(os.path.join(REPO, "results", "per_image_latest_1791743390.csv"))
    ar = pd.read_csv(os.path.join(REPO, "results", "image_level_validation", "per_image_scores.csv"))
    pi["key"] = pi["path"].map(norm)
    ar["key"] = ar["path"].map(norm)
    d = pi.merge(ar[["key", "defect_area_px"]], on="key", how="left", validate="one_to_one")
    d = d[d["label"] != "good"].copy()
    if d["defect_area_px"].isna().any() or d[["px_auroc_A", "px_auroc_B"]].isna().any().any():
        sys.exit("ERROR: missing defect areas or per-image AUROCs after the join")
    d["diff_B_minus_A"] = d["px_auroc_B"] - d["px_auroc_A"]

    rows, res = [], {}
    for ti, (tex, g) in enumerate(sorted(d.groupby("category"), key=lambda x: x[0])):
        med = float(g["defect_area_px"].median())
        g = g.assign(subset=np.where(g["defect_area_px"] <= med, "small", "large"))
        rows.append(g)
        res[tex] = {"median_area_px": med, "median_area_pct": med / 518 ** 2 * 100}
        boots = {}
        for si, sub in enumerate(("small", "large")):
            x = g.loc[g["subset"] == sub, "diff_B_minus_A"].to_numpy()
            rng = np.random.default_rng([ti, si])
            bm = np.array([rng.choice(x, len(x)).mean() for _ in range(N_BOOT)])
            boots[sub] = bm
            w = wilcoxon(x)
            res[tex][sub] = {"n": int(len(x)), "mean_diff": float(x.mean()), "median_diff": float(np.median(x)),
                             "ci95_mean_diff": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                             "wilcoxon_W": float(w.statistic), "p": float(w.pvalue),
                             "frac_A_gt_B": float((x < 0).mean())}
        dd = boots["large"] - boots["small"]
        res[tex]["diff_large_minus_small"] = {
            "estimate": res[tex]["large"]["mean_diff"] - res[tex]["small"]["mean_diff"],
            "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}

    texs = list(res)
    for sub in ("small", "large"):
        adj = holm([res[t][sub]["p"] for t in texs])
        for t, pa in zip(texs, adj):
            res[t][sub]["p_holm"] = float(pa)

    c1 = [t for t in texs if res[t]["small"]["p_holm"] < 0.05 and res[t]["small"]["median_diff"] < 0]
    c2 = [t for t in texs if res[t]["diff_large_minus_small"]["estimate"] > 0
          and res[t]["diff_large_minus_small"]["ci95"][0] > 0]
    decision = {"C1_textures": c1, "C1": "supported" if len(c1) >= 2 else "not supported",
                "C2_textures": c2, "C2": "supported" if len(c2) >= 3 else "not supported"}

    pd.concat(rows)[["category", "key", "label", "defect_area_px", "subset", "px_auroc_A", "px_auroc_B",
                     "diff_B_minus_A"]].to_csv(os.path.join(OUT, "per_image_paired.csv"), index=False)
    with open(os.path.join(OUT, "paired_results.json"), "w", encoding="utf-8") as f:
        json.dump({"route_B_seed": 42, "source": "results/per_image_latest_1791743390.csv (eval_router_full.py)",
                   "results": res, "decision": decision, "seconds": round(time.time() - t0, 2)}, f, indent=1)

    L = ["Paired per-image pixel AUROC, d = Route B (seed 42) - Route A, in points", ""]
    for t in texs:
        r = res[t]
        L.append(f"{t.upper()}  (median defect area {r['median_area_pct']:.2f}% of image)")
        for sub in ("small", "large"):
            s = r[sub]
            L.append(f"  {sub:<5} n={s['n']:>2}  mean d={s['mean_diff']:+6.2f} [{s['ci95_mean_diff'][0]:+6.2f}, "
                     f"{s['ci95_mean_diff'][1]:+6.2f}]  median d={s['median_diff']:+6.2f}  "
                     f"Wilcoxon p={s['p']:.3g} (Holm {s['p_holm']:.3g})  A>B in {s['frac_A_gt_B'] * 100:.0f}%")
        e = r["diff_large_minus_small"]
        L.append(f"  d(large) - d(small) = {e['estimate']:+6.2f} [{e['ci95'][0]:+6.2f}, {e['ci95'][1]:+6.2f}]")
        L.append("")
    L.append(f"C1 Route A beats Route B on small defects (Holm p<0.05, median d<0, >=2 textures): "
             f"{len(c1)}/5 {c1} -> {decision['C1']}")
    L.append(f"C2 Route B degrades relative to A with size (d(large)>d(small), CI excludes 0, >=3 textures): "
             f"{len(c2)}/5 {c2} -> {decision['C2']}")
    L.append(f"Time: {time.time() - t0:.1f}s")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "paired_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
