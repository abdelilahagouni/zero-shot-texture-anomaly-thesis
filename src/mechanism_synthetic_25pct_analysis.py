"""
Analysis of the 25% add-on together with the 9-size Part B run (inputs are read only).

Inputs : results/mechanism_synthetic/per_image_synthetic.csv, per_image_drift.csv          (sizes 0.25 ... 50%)
         results/mechanism_synthetic_25pct/per_image_synthetic.csv, per_image_drift.csv,
         drift_control.csv                                                                   (size 25%)
Output : results/mechanism_synthetic_25pct/analysis/

Unit: one synthetic image (texture, base image, defect type, size). Route B = mean of its 5 seeds.
Bootstrap: 10,000 resamples of the 133 BASE images (each keeps all its types and sizes), 95% percentile CIs;
the draws are generated exactly as in mechanism_synthetic_analysis.py (sorted base-image keys,
np.random.default_rng(12345)), so the resampled base images are the same ones.

(a) criterion (iii) as originally written: diff = mean AUROC at 50% minus mean AUROC at 25%, for A and A_clean,
    pooled and per defect type. Holds if A's diff < 0 with a CI excluding 0 (upper bound < 0) AND A_clean's
    lower CI bound > -2.
(b) drift test without collinearity: for EACH size separately, per-image AUROC ~ drift with texture x type fixed
    effects (20 dummies), for A and for B; drift = drift of normal patches > 5 patches from the blob
    (drift_gt5, as in the drift criterion). Reports the coefficient, its CI and sign at each size; images with NaN
    drift are dropped and counted. Plus the correlation (Pearson and Spearman) between drift and log(area):
    pooled over all 10 sizes, per defect type over all sizes, and within each size.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES9 = os.path.join(REPO, "results", "mechanism_synthetic")
RES25 = os.path.join(REPO, "results", "mechanism_synthetic_25pct")
OUT = os.path.join(RES25, "analysis")
N_BOOT = 10000
TYPES = ["othertexture", "noise", "blur", "hueshift"]
KEY = ["texture", "type", "size_pct", "image_id"]


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def load(res):
    raw = pd.read_csv(os.path.join(res, "per_image_synthetic.csv"))
    drift = pd.read_csv(os.path.join(res, "per_image_drift.csv"))
    w = raw.groupby(KEY + ["method"])["auroc"].mean().unstack("method")[["A", "A_clean", "B"]]
    area = raw.groupby(KEY)["area_frac"].first()
    df = w.join(area).reset_index().merge(drift[KEY + ["drift_gt5"]], on=KEY, how="left")
    return df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    T0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.concat([load(RES9), load(RES25)], ignore_index=True)
    df["x"] = np.log(df["area_frac"])
    df["cluster"] = df["texture"] + "/" + df["image_id"].astype(str)
    keys = np.unique(df["cluster"].to_numpy())
    draws = np.random.default_rng(12345).integers(0, len(keys), size=(N_BOOT, len(keys)))
    W = np.stack([np.bincount(d, minlength=len(keys)) for d in draws]).astype(float)   # (n_boot, n_keys) counts
    kidx = {k: i for i, k in enumerate(keys)}
    df["ci"] = df["cluster"].map(kidx)
    control = pd.read_csv(os.path.join(RES25, "drift_control.csv"))
    sizes = sorted(df["size_pct"].unique())
    out = {"n_synthetic_images": int(len(df)), "n_base_images": int(len(keys)), "n_boot": N_BOOT,
           "sizes": [float(s) for s in sizes],
           "n_images_per_size": {str(s): int((df["size_pct"] == s).sum()) for s in sizes},
           "drift_control_25pct": {"max": float(control["control_drift_max"].max()),
                                   "tokens_identical": int(control["control_tokens_identical"].sum()),
                                   "n": int(len(control))}}
    f3 = lambda c, lo, hi: f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    L = [f"25% ADD-ON ANALYSIS: {len(df)} synthetic images ({len(sizes)} sizes) from {len(keys)} base images; "
         f"bootstrap {N_BOOT} over base images", "",
         f"25% run drift control (clean image twice): max {out['drift_control_25pct']['max']:.3e}, tokens identical "
         f"{out['drift_control_25pct']['tokens_identical']}/{out['drift_control_25pct']['n']}", ""]

    # ---------------------------------------------------------------- (a)
    L.append("(a) criterion (iii): diff = mean AUROC at 50% minus at 25%; holds if A diff < 0 with CI excl. 0 "
             "AND A_clean lower CI bound > -2")
    piv = df[df["size_pct"].isin([25.0, 50.0])].pivot_table(index=["ci", "type"], columns="size_pct",
                                                            values=["A", "A_clean"])
    a_res = {}
    for scope in ["pooled"] + TYPES:
        sub = piv if scope == "pooled" else piv.xs(scope, level="type", drop_level=False)
        r = {}
        for m in ["A", "A_clean"]:
            v25, v50 = sub[(m, 25.0)], sub[(m, 50.0)]
            if v25.isna().any() or v50.isna().any():
                sys.exit(f"ERROR: missing AUROC for {m} at 25/50% in {scope}")
            ci_of = sub.index.get_level_values("ci").to_numpy()
            # per base image: sum of diffs over its rows and number of rows, then weighted by bootstrap counts
            d = (v50 - v25).to_numpy()
            s_d = np.bincount(ci_of, weights=d, minlength=len(keys))
            s_25 = np.bincount(ci_of, weights=v25.to_numpy(), minlength=len(keys))
            s_50 = np.bincount(ci_of, weights=v50.to_numpy(), minlength=len(keys))
            n_r = np.bincount(ci_of, minlength=len(keys)).astype(float)
            est = s_d.sum() / n_r.sum()
            bd = (W @ s_d) / (W @ n_r)
            r[m] = {"mean_25": float(s_25.sum() / n_r.sum()), "mean_50": float(s_50.sum() / n_r.sum()),
                    "diff": float(est), "ci95": ci(bd), "n": int(n_r.sum())}
        hA = r["A"]["diff"] < 0 and r["A"]["ci95"][1] < 0
        hC = r["A_clean"]["ci95"][0] > -2
        r["A_lower_ci_excl_0"], r["A_clean_lower_bound_above_minus2"], r["holds"] = hA, hC, bool(hA and hC)
        a_res[scope] = r
        L.append(f"  {scope:<12} A {r['A']['mean_25']:6.2f}->{r['A']['mean_50']:6.2f} diff "
                 f"{f3(r['A']['diff'], *r['A']['ci95'])}   A_clean {r['A_clean']['mean_25']:6.2f}->"
                 f"{r['A_clean']['mean_50']:6.2f} diff {f3(r['A_clean']['diff'], *r['A_clean']['ci95'])}   "
                 f"A part {'met' if hA else 'NOT met'}, A_clean part {'met' if hC else 'NOT met'} -> "
                 f"holds: {r['holds']}")
    n_types = sum(a_res[t]["holds"] for t in TYPES)
    L.append(f"  holds pooled: {a_res['pooled']['holds']}; holds in {n_types}/4 types")
    out["a_criterion_iii_50_vs_25"] = a_res
    out["a_criterion_iii_types_holding"] = int(n_types)
    L.append("")

    # ---------------------------------------------------------------- (b)
    L.append("(b) per size: AUROC ~ drift(>5 patches) + texture x type fixed effects; coefficient [95% CI]")
    b_res, maxdev = {}, 0.0
    for s in sizes:
        sub = df[df["size_pct"] == s]
        n_nan = int(sub["drift_gt5"].isna().sum())
        sub = sub[sub["drift_gt5"].notna()]
        g = (sub["texture"] + "/" + sub["type"]).to_numpy()
        lev, gi = np.unique(g, return_inverse=True)
        G = np.zeros((len(sub), len(lev)))
        G[np.arange(len(sub)), gi] = 1.0
        x = sub["drift_gt5"].to_numpy(float)
        Wr = W[:, sub["ci"].to_numpy()]                      # (n_boot, n_rows) bootstrap weight of each row
        rs = {"n": int(len(sub)), "n_dropped_nan_drift": n_nan}
        for m in ["A", "B"]:
            y = sub[m].to_numpy(float)

            def slope(w):
                sw, swx, swy = w @ G, (w * x) @ G, (w * y) @ G
                with np.errstate(invalid="ignore", divide="ignore"):
                    cxy = (w * x * y).sum(-1) - np.nansum(np.where(sw > 0, swx * swy / sw, 0.0), -1)
                    cxx = (w * x * x).sum(-1) - np.nansum(np.where(sw > 0, swx * swx / sw, 0.0), -1)
                return cxy / cxx

            est = float(slope(np.ones(len(sub))))
            X = np.column_stack([x, G])                       # check against explicit dummy OLS
            maxdev = max(maxdev, abs(est - np.linalg.lstsq(X, y, rcond=None)[0][0]))
            c = ci(slope(Wr))
            rs[m] = {"coef": est, "ci95": c, "sign": "+" if est > 0 else "-",
                     "ci_excludes_0": bool(c[0] > 0 or c[1] < 0)}
        b_res[str(s)] = rs
        L.append(f"  size {s:>5}%  n {rs['n']:3d} (NaN drift dropped {n_nan})   "
                 f"A {f3(rs['A']['coef'], *rs['A']['ci95'])} sign {rs['A']['sign']} "
                 f"{'excl. 0' if rs['A']['ci_excludes_0'] else 'incl. 0'}   "
                 f"B {f3(rs['B']['coef'], *rs['B']['ci95'])} sign {rs['B']['sign']} "
                 f"{'excl. 0' if rs['B']['ci_excludes_0'] else 'incl. 0'}")
    out["b_drift_coef_per_size"] = b_res
    out["b_check_max_abs_dev_vs_dummy_ols"] = float(maxdev)
    L.append(f"  (within-estimator vs explicit dummy OLS, max |diff| {maxdev:.2e})")

    from scipy.stats import pearsonr, spearmanr
    dd = df[df["drift_gt5"].notna()]
    corr = {}

    def cc(sub):
        return {"n": int(len(sub)), "pearson": float(pearsonr(sub["x"], sub["drift_gt5"])[0]),
                "spearman": float(spearmanr(sub["x"], sub["drift_gt5"]).correlation)}

    corr["pooled_all_sizes"] = cc(dd)
    for t in TYPES:
        corr[f"{t}_all_sizes"] = cc(dd[dd["type"] == t])
    for s in sizes:
        corr[f"size_{s}"] = cc(dd[dd["size_pct"] == s])
    out["b_corr_drift_vs_logarea"] = corr
    L.append("  correlation of drift(>5 patches) with log(area):")
    for k, v in corr.items():
        L.append(f"    {k:<24} n {v['n']:4d}  Pearson {v['pearson']:+.3f}  Spearman {v['spearman']:+.3f}")
    out["seconds"] = round(time.time() - T0, 1)
    L.append("")
    L.append(f"Total analysis time {out['seconds']}s")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "mechanism_25pct_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, "mechanism_25pct_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    rows = [{"size_pct": s, "method": m, "n": b_res[str(s)]["n"], "coef": b_res[str(s)][m]["coef"],
             "ci_lo": b_res[str(s)][m]["ci95"][0], "ci_hi": b_res[str(s)][m]["ci95"][1]}
            for s in sizes for m in ["A", "B"]]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "drift_coef_per_size.csv"), index=False)
    print(txt)


if __name__ == "__main__":
    main()
