"""
Size analysis for FCA and aota, same method as src/size_effect_decomposition.py (pooled part):
  per-image pixel AUROC (full 518x518 maps) of the 382 defective images ~ log(area_px at 518x518)
  + texture x defect-type fixed effects (OLS with one dummy per group); 95% percentile bootstrap over IMAGES,
  10,000 paired resamples (all methods share each resample), np.random.default_rng(99).
  Methods: FCA, aota, A, B (= mean of seeds 42/123/2026 per image).
Reported: slope(FCA), slope(aota), slope(A), slope(B); slope(FCA)-slope(B), slope(aota)-slope(B), slope(FCA)-slope(A),
  and slope(A)-slope(B) (needed by criterion i).
Criteria (fixed before the results):
  (i)  "size effect specific to simple mean self-similarity" supported only if slope(FCA)-slope(B) has a CI that
       includes 0 or is positive (lower bound <= 0 <= upper, or lower > 0) AND slope(A)-slope(B) < 0 with CI excl. 0.
  (ii) "size effect also applies to a strong single-image method" supported only if slope(FCA)-slope(B) < 0 with CI
       excluding 0.
  (iii) mixed results are reported as mixed: if neither (i) nor (ii) is supported, the verdict is "mixed".
Input : results/fca_baseline/metrics/per_image_defective.csv. Output: results/fca_baseline/size_analysis/.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RES  # noqa: E402

N_BOOT = 10000
METHODS = ["FCA", "aota", "A", "B"]
DIFFS = [("FCA", "B"), ("aota", "B"), ("FCA", "A"), ("A", "B")]


def fe_coefs(x, Y, groups):
    g = np.asarray(groups)
    lev = sorted(set(g))
    X = np.column_stack([x] + [(g == l).astype(float) for l in lev])
    return np.linalg.lstsq(X, Y, rcond=None)[0][0]


def ci(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    out = os.path.join(RES, "size_analysis")
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(os.path.join(RES, "metrics", "per_image_defective.csv"))
    Y = df[[f"{m}_auroc" for m in METHODS]].to_numpy(float)
    if np.isnan(Y).any():
        sys.exit("ERROR: NaN per-image AUROC")
    x = np.log(df["area_px"].to_numpy(float))
    grp = (df["category"] + "/" + df["defect_type"]).to_numpy()
    est = dict(zip(METHODS, fe_coefs(x, Y, grp)))
    rng = np.random.default_rng(99)
    B = np.empty((N_BOOT, len(METHODS)))
    for b in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        B[b] = fe_coefs(x[i], Y[i], grp[i])
    res = {"n": int(len(df)), "fixed_effects": int(len(set(grp))), "n_boot": N_BOOT, "slopes": {}, "differences": {}}
    for k, m in enumerate(METHODS):
        res["slopes"][m] = {"coef": float(est[m]), "ci95": ci(B[:, k])}
    for a, b in DIFFS:
        res["differences"][f"{a}-{b}"] = {"coef": float(est[a] - est[b]),
                                          "ci95": ci(B[:, METHODS.index(a)] - B[:, METHODS.index(b)])}
    dFB, dAB = res["differences"]["FCA-B"], res["differences"]["A-B"]
    i_part1 = dFB["ci95"][1] >= 0                                  # CI includes 0 or is positive
    i_part2 = dAB["coef"] < 0 and dAB["ci95"][1] < 0
    crit_i = bool(i_part1 and i_part2)
    crit_ii = bool(dFB["coef"] < 0 and dFB["ci95"][1] < 0)
    verdict = ("(i) supported" if crit_i else "(ii) supported" if crit_ii else "mixed")
    res["criteria"] = {"i_FCA_minus_B_ci_includes_0_or_positive": bool(i_part1),
                       "i_A_minus_B_negative_ci_excl_0": bool(i_part2), "i_supported": crit_i,
                       "ii_FCA_minus_B_negative_ci_excl_0": crit_ii, "ii_supported": crit_ii, "verdict": verdict}
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(out, "size_analysis_results.json"), "w"), indent=1)
    f2 = lambda e: f"{e['coef']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
    L = [f"SIZE ANALYSIS: per-image pixel AUROC ~ log(area) + {res['fixed_effects']} texture x defect-type fixed "
         f"effects; n={res['n']}; bootstrap {N_BOOT} over images (paired)", "", "Slopes:"]
    L += [f"  {m:<5} {f2(res['slopes'][m])}" for m in METHODS]
    L.append("Differences:")
    L += [f"  {k:<7} {f2(v)}" for k, v in res["differences"].items()]
    L += ["", f"(i)  slope(FCA)-slope(B) CI includes 0 or positive: {'met' if i_part1 else 'NOT met'}; "
              f"slope(A)-slope(B) < 0, CI excl. 0: {'met' if i_part2 else 'NOT met'} -> "
              f"{'supported' if crit_i else 'not supported'}",
          f"(ii) slope(FCA)-slope(B) < 0, CI excl. 0: {'met' if crit_ii else 'NOT met'} -> "
          f"{'supported' if crit_ii else 'not supported'}",
          f"VERDICT: {verdict}", f"({res['seconds']}s)"]
    txt = "\n".join(L)
    open(os.path.join(out, "size_analysis_results.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
