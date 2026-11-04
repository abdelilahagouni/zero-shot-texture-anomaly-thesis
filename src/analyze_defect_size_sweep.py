"""
Post-hoc analysis of results/defect_size_sweep/defect_size_sweep.json.

Per texture:
  * table: Route A AP / AU-PRO, Route B AP / AU-PRO mean and sample SD (n-1), chance AP, AP lift
  * Spearman correlation of defect size vs AU-PRO (Route A; Route B on the seed mean)
  * bootstrap 95% CI for each correlation: resample the (size, AU-PRO) pairs with replacement
    (10,000 resamples, fixed seed; resamples where the correlation is undefined are dropped).
    With only 8 sizes this CI is coarse.

Writes analysis.json and analysis.txt next to the sweep results. Reads nothing else.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def boot_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xs, ys = x[idx], y[idx]
        if np.unique(xs).size < 2 or np.unique(ys).size < 2:
            continue
        r = spearmanr(xs, ys).correlation
        if np.isfinite(r):
            vals.append(r)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(REPO, "results", "defect_size_sweep"))
    a = ap.parse_args()
    with open(os.path.join(a.results_dir, "defect_size_sweep.json"), encoding="utf-8") as f:
        data = json.load(f)
    out, lines = {}, []
    for tex, rows in data["aggregate"].items():
        rows = sorted(rows, key=lambda e: e["area_pct"])
        table = []
        for e in rows:
            ch = e["A_chance_ap"]
            table.append({
                "area_pct": e["area_pct"], "chance_ap": ch, "n_seeds_B": e["n_seeds_B"],
                "A_ap": e["A_pixel_ap"], "A_au_pro": e["A_au_pro"],
                "B_ap_mean": e["B_pixel_ap_mean"], "B_ap_sd": e["B_pixel_ap_sd"],
                "B_au_pro_mean": e["B_au_pro_mean"], "B_au_pro_sd": e["B_au_pro_sd"],
                "A_ap_lift": e["A_pixel_ap"] / ch, "B_ap_lift": e["B_pixel_ap_mean"] / ch,
            })
        sz = [t["area_pct"] for t in table]
        rA = spearmanr(sz, [t["A_au_pro"] for t in table])
        rB = spearmanr(sz, [t["B_au_pro_mean"] for t in table])
        ciA = boot_ci(sz, [t["A_au_pro"] for t in table])
        ciB = boot_ci(sz, [t["B_au_pro_mean"] for t in table], seed=1)
        a_ap_wins = [t["area_pct"] for t in table if t["A_ap"] > t["B_ap_mean"]]
        a_pro_wins = [t["area_pct"] for t in table if t["A_au_pro"] > t["B_au_pro_mean"]]
        out[tex] = {"table": table,
                    "spearman_size_vs_au_pro": {
                        "route_A": {"rho": float(rA.correlation), "p": float(rA.pvalue), "boot95": ciA[:2], "n_valid_boot": ciA[2]},
                        "route_B_mean": {"rho": float(rB.correlation), "p": float(rB.pvalue), "boot95": ciB[:2], "n_valid_boot": ciB[2]}},
                    "sizes_where_A_beats_B": {"pixel_ap": a_ap_wins, "au_pro": a_pro_wins}}
        lines.append(f"{tex.upper()}")
        lines.append(f"  {'area%':>6} {'chance':>7} | {'A AP':>6} {'A PRO':>6} {'A lift':>6} | "
                     f"{'B AP':>6} {'±SD':>5} {'B PRO':>6} {'±SD':>5} {'B lift':>6}")
        for t in table:
            lines.append(f"  {t['area_pct']:>6g} {t['chance_ap']:7.2f} | {t['A_ap']:6.2f} {t['A_au_pro']:6.2f} "
                         f"{t['A_ap_lift']:6.2f} | {t['B_ap_mean']:6.2f} {t['B_ap_sd']:5.2f} {t['B_au_pro_mean']:6.2f} "
                         f"{t['B_au_pro_sd']:5.2f} {t['B_ap_lift']:6.2f}")
        s = out[tex]["spearman_size_vs_au_pro"]
        for k, lab in (("route_A", "Route A"), ("route_B_mean", "Route B (mean)")):
            lines.append(f"  Spearman size vs AU-PRO, {lab}: rho={s[k]['rho']:+.2f} (p={s[k]['p']:.3g}), "
                         f"bootstrap 95% CI [{s[k]['boot95'][0]:+.2f}, {s[k]['boot95'][1]:+.2f}]")
        lines.append(f"  A > B on Pixel AP at: {a_ap_wins or 'no size'}; on AU-PRO at: {a_pro_wins or 'no size'}")
        lines.append("")
    with open(os.path.join(a.results_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.results_dir, "analysis.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
