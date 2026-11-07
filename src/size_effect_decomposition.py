"""
Decomposition of the size effect (no model runs).

Input : results/size_effect_robustness/per_image_metrics.csv (382 defective texture images)
        A = Route A per-image pixel AUROC; B = mean over 10 Route B seeds of per-image pixel AUROC.
x = log(defect area in pixels at 518x518).

1. Per texture: Spearman(x, A), Spearman(x, B), OLS slopes of A and B on x, and slope(A) - slope(B).
   Pooled: OLS with texture x defect-type fixed effects for A and for B, and coef(A) - coef(B).
   Pooled Spearman with fixed effects = Spearman of within-group-demeaned x and y.
   All CIs: 95% percentile bootstrap resampling IMAGES (10,000; paired, so A and B share each resample).
2. Ceiling check (pooled, fixed effects): errA = 100 - A, errB = 100 - B.
   (i) errA - errB on x (note: errA - errB = B - A = d, so this equals the earlier d regression);
   (ii) log(errA / errB) on x.
3. Fixed criterion: supported only if (a) coef_A < 0 with CI excluding 0, (b) coef_A - coef_B < 0 with CI
   excluding 0, (c) coef of log(errA/errB) > 0 with CI excluding 0.
Writes to results/size_effect_decomposition/ only.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "size_effect_decomposition")
SEEDS = [42, 123, 2026, 7, 11, 13, 17, 19, 23, 29]
N_BOOT = 10000


def fe_design(x, groups):
    levels = sorted(set(groups))
    g = np.asarray(groups)
    return np.column_stack([x] + [(g == l).astype(float) for l in levels])


def fe_coef(x, y, groups):
    X = fe_design(np.asarray(x, float), groups)
    beta, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    return float(beta[0])


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc ** 2).sum())


def demean(v, groups):
    s = pd.Series(np.asarray(v, float))
    return (s - s.groupby(np.asarray(groups)).transform("mean")).to_numpy()


def ci(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(REPO, "results", "size_effect_robustness", "per_image_metrics.csv"))
    df["A"] = df["A_auroc"]
    df["B"] = df[[f"B{s}_auroc" for s in SEEDS]].mean(axis=1)
    df["x"] = np.log(df["area_px"])
    df["errA"], df["errB"] = 100 - df["A"], 100 - df["B"]
    df["d_err"] = df["errA"] - df["errB"]
    df["log_ratio"] = np.log(df["errA"] / df["errB"])
    df["grp"] = df["category"] + "/" + df["defect_type"]
    res = {"n": int(len(df)), "per_texture": {}, "pooled": {}}

    # 1. per texture
    for ti, (cat, g) in enumerate(sorted(df.groupby("category"))):
        x, A, B = g["x"].to_numpy(), g["A"].to_numpy(), g["B"].to_numpy()
        rng = np.random.default_rng([1, ti])
        bs = {k: [] for k in ("rhoA", "rhoB", "slA", "slB", "diff")}
        for _ in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            if np.unique(x[i]).size < 2:
                continue
            bs["rhoA"].append(spearmanr(x[i], A[i]).correlation)
            bs["rhoB"].append(spearmanr(x[i], B[i]).correlation)
            sa, sb = slope(x[i], A[i]), slope(x[i], B[i])
            bs["slA"].append(sa); bs["slB"].append(sb); bs["diff"].append(sa - sb)
        sa, sb = slope(x, A), slope(x, B)
        res["per_texture"][cat] = {
            "n": int(len(g)),
            "spearman_A": {"rho": float(spearmanr(x, A).correlation), "ci95": ci(bs["rhoA"])},
            "spearman_B": {"rho": float(spearmanr(x, B).correlation), "ci95": ci(bs["rhoB"])},
            "slope_A": {"coef": sa, "ci95": ci(bs["slA"])},
            "slope_B": {"coef": sb, "ci95": ci(bs["slB"])},
            "slope_A_minus_B": {"coef": sa - sb, "ci95": ci(bs["diff"])}}

    # pooled with fixed effects
    x, grp = df["x"].to_numpy(), df["grp"].to_numpy()
    cols = {k: df[k].to_numpy() for k in ("A", "B", "d_err", "log_ratio")}
    est = {k: fe_coef(x, v, grp) for k, v in cols.items()}
    rhoA = spearmanr(demean(x, grp), demean(cols["A"], grp)).correlation
    rhoB = spearmanr(demean(x, grp), demean(cols["B"], grp)).correlation
    rng = np.random.default_rng(99)
    bs = {k: [] for k in ("A", "B", "diff", "d_err", "log_ratio", "rhoA", "rhoB")}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        xi, gi = x[i], grp[i]
        cA, cB = fe_coef(xi, cols["A"][i], gi), fe_coef(xi, cols["B"][i], gi)
        bs["A"].append(cA); bs["B"].append(cB); bs["diff"].append(cA - cB)
        bs["d_err"].append(fe_coef(xi, cols["d_err"][i], gi))
        bs["log_ratio"].append(fe_coef(xi, cols["log_ratio"][i], gi))
        dx = demean(xi, gi)
        bs["rhoA"].append(spearmanr(dx, demean(cols["A"][i], gi)).correlation)
        bs["rhoB"].append(spearmanr(dx, demean(cols["B"][i], gi)).correlation)
    res["pooled"] = {
        "fixed_effects": int(len(set(grp))),
        "coef_A": {"coef": est["A"], "ci95": ci(bs["A"])},
        "coef_B": {"coef": est["B"], "ci95": ci(bs["B"])},
        "coef_A_minus_B": {"coef": est["A"] - est["B"], "ci95": ci(bs["diff"])},
        "spearman_within_A": {"rho": float(rhoA), "ci95": ci(bs["rhoA"])},
        "spearman_within_B": {"rho": float(rhoB), "ci95": ci(bs["rhoB"])},
        "ceiling_errA_minus_errB": {"coef": est["d_err"], "ci95": ci(bs["d_err"])},
        "ceiling_log_errA_over_errB": {"coef": est["log_ratio"], "ci95": ci(bs["log_ratio"])},
    }
    P = res["pooled"]
    a_ok = P["coef_A"]["coef"] < 0 and P["coef_A"]["ci95"][1] < 0
    b_ok = P["coef_A_minus_B"]["coef"] < 0 and P["coef_A_minus_B"]["ci95"][1] < 0
    c_ok = P["ceiling_log_errA_over_errB"]["coef"] > 0 and P["ceiling_log_errA_over_errB"]["ci95"][0] > 0
    failed = [k for k, ok in (("(a)", a_ok), ("(b)", b_ok), ("(c)", c_ok)) if not ok]
    res["decision"] = {"a": a_ok, "b": b_ok, "c": c_ok,
                       "verdict": "supported" if not failed else "not supported", "failed_parts": failed}
    res["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "decomposition_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    f2 = lambda e: f"{e['coef']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
    fr = lambda e: f"{e['rho']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
    L = ["SIZE-EFFECT DECOMPOSITION (per-image pixel AUROC; x = log defect area; 95% bootstrap CIs over images)", ""]
    L.append("Per texture:")
    for cat, e in res["per_texture"].items():
        L.append(f"  {cat:<8} n={e['n']:>3}  Spearman A {fr(e['spearman_A'])}  B {fr(e['spearman_B'])}")
        L.append(f"  {'':<8}        slope A {f2(e['slope_A'])}  slope B {f2(e['slope_B'])}  A-B {f2(e['slope_A_minus_B'])}")
    L.append("")
    L.append(f"Pooled, {P['fixed_effects']} texture x defect-type fixed effects, n={res['n']}:")
    L.append(f"  coef A on log(area)         {f2(P['coef_A'])}")
    L.append(f"  coef B on log(area)         {f2(P['coef_B'])}")
    L.append(f"  coef A - coef B             {f2(P['coef_A_minus_B'])}")
    L.append(f"  within-group Spearman A     {fr(P['spearman_within_A'])}")
    L.append(f"  within-group Spearman B     {fr(P['spearman_within_B'])}")
    L.append("Ceiling check (pooled, fixed effects):")
    L.append(f"  errA - errB on log(area)    {f2(P['ceiling_errA_minus_errB'])}   (identical to d = B - A)")
    L.append(f"  log(errA/errB) on log(area) {f2(P['ceiling_log_errA_over_errB'])}")
    L.append("")
    L.append(f"Criterion (a) coef A < 0, CI excl. 0: {'met' if a_ok else 'NOT met'}")
    L.append(f"Criterion (b) coef A - coef B < 0, CI excl. 0: {'met' if b_ok else 'NOT met'}")
    L.append(f"Criterion (c) log error-ratio coef > 0, CI excl. 0: {'met' if c_ok else 'NOT met'}")
    L.append(f"VERDICT: {res['decision']['verdict']}" + (f" (failed: {', '.join(failed)})" if failed else ""))
    L.append(f"Time: {res['seconds']}s")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "decomposition_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
