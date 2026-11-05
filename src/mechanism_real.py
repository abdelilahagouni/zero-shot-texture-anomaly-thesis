"""
PART A: self-contamination mechanism on the 382 REAL defective texture images.

Per image: pixel AUROC for A, A-clean, A-median, A-trim and Route B (10 seeds; analysis uses the mean),
area fraction, contamination index. Cached in per_image.csv (re-used if present).

Analysis (as size_effect_decomposition.py): OLS with texture x defect-type fixed effects, 95% percentile
bootstrap over images (10,000 paired resamples):
  slopes of per-image AUROC on log(area) for A, A_clean, A_median, A_trim, B;
  differences A_clean-A, A_median-A, A_clean-B (and A_trim-A, reported as an extra);
  contamination index on log(area); A on log(area) + contamination index.
Fixed criterion: contamination "explains" the size effect only if slope(A_clean)-slope(A) > 0 with CI excluding 0
AND slope(A_clean)-slope(B) has a CI that includes or exceeds 0 (upper bound >= 0). If A_clean is still steeper
than B (whole CI < 0), the remainder is reported as unexplained.
Writes to results/mechanism_real/ only.
"""
import glob, json, os, sys, time, datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mechanism_common import Backbone, fe_solve, ci95, IMG  # noqa: E402

OUT = os.path.join(REPO, "results", "mechanism_real")
DATA = os.path.join(REPO, "data", "mvtec")
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
SEEDS = [42, 123, 2026, 7, 11, 13, 17, 19, 23, 29]
N_BOOT = 10000


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


def run_models(csv_path):
    from PIL import Image
    from sklearn.metrics import roc_auc_score
    bb = Backbone()
    rows = []
    for cat in TEXTURES:
        t0 = time.time()
        train = [bb.tokens(Image.open(p)) for p in sorted(glob.glob(f"{DATA}/{cat}/train/good/*.png"))]
        banks = {sd: bb.bank(train, sd) for sd in SEEDS}
        tdir, gdir = f"{DATA}/{cat}/test", f"{DATA}/{cat}/ground_truth"
        n = 0
        for d in sorted(os.listdir(tdir)):
            if d == "good" or not os.path.isdir(os.path.join(tdir, d)):
                continue
            for p in sorted(glob.glob(os.path.join(tdir, d, "*.png"))):
                fn = os.path.basename(p)
                mp = os.path.join(gdir, d, fn.replace(".png", "_mask.png"))
                if not os.path.exists(mp):
                    mp = os.path.join(gdir, d, fn)
                gt = np.array(Image.open(mp).convert("L").resize((IMG, IMG), Image.NEAREST)) > 128
                maps, contam = bb.maps(bb.tokens(Image.open(p)), gt, banks)
                y = gt.ravel()
                r = {"category": cat, "defect_type": d, "file": fn, "area_px": int(y.sum()),
                     "area_frac": float(y.mean()), "contamination": contam}
                for k, m in maps.items():
                    r[f"auroc_{k}"] = float(roc_auc_score(y, m.ravel()) * 100) if m is not None else float("nan")
                rows.append(r)
                n += 1
        log(f"{cat}: {n} defective images, {time.time() - t0:.1f}s")
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    csv_path = os.path.join(OUT, "per_image.csv")
    log("ESTIMATE: 382 defective + 150 train images, 4 Route A variants + 10 Route B seeds per image -> about 10 min, "
        "plus about 2 min of bootstrap.")
    if not os.path.exists(csv_path):
        t = time.time()
        run_models(csv_path)
        log(f"model run {time.time() - t:.1f}s")
    t = time.time()
    df = pd.read_csv(csv_path)
    df["auroc_B"] = df[[f"auroc_B{s}" for s in SEEDS]].mean(axis=1)
    df["x"] = np.log(df["area_px"])
    grp = (df["category"] + "/" + df["defect_type"]).to_numpy()

    # reproduction check of A and B42 against the size_effect_robustness run
    prev = pd.read_csv(os.path.join(REPO, "results", "size_effect_robustness", "per_image_metrics.csv"))
    j = df.merge(prev[["category", "defect_type", "file", "A_auroc", "B42_auroc"]],
                 on=["category", "defect_type", "file"], how="left")
    repro = {"max_abs_diff_A": float((j["auroc_A"] - j["A_auroc"]).abs().max()),
             "max_abs_diff_B42": float((j["auroc_B42"] - j["B42_auroc"]).abs().max())}

    methods = ["A", "A_clean", "A_median", "A_trim", "B"]
    Y = df[[f"auroc_{m}" for m in methods]].to_numpy()
    if np.isnan(Y).any():
        sys.exit("ERROR: NaN AUROC (an image with no clean patches?)")
    x = df["x"].to_numpy()
    c = df["contamination"].to_numpy()

    def estimates(i):
        b = fe_solve(x[i], Y[i], grp[i])[0]                     # slopes per method
        s = dict(zip(methods, b))
        s["diff_Aclean_minus_A"] = s["A_clean"] - s["A"]
        s["diff_Amedian_minus_A"] = s["A_median"] - s["A"]
        s["diff_Atrim_minus_A"] = s["A_trim"] - s["A"]
        s["diff_Aclean_minus_B"] = s["A_clean"] - s["B"]
        s["contam_on_logarea"] = fe_solve(x[i], c[i], grp[i])[0]
        both = fe_solve(np.column_stack([x[i], c[i]]), Y[i][:, 0], grp[i])
        s["A_on_logarea_with_contam"], s["A_on_contam_with_logarea"] = both[0], both[1]
        return s

    full = estimates(np.arange(len(df)))
    rng = np.random.default_rng(2024)
    boots = {k: [] for k in full}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(df), len(df))
        for k, v in estimates(i).items():
            boots[k].append(v)
    res = {k: {"coef": float(full[k]), "ci95": ci95(boots[k])} for k in full}
    from scipy.stats import spearmanr
    rho = spearmanr(df["x"], df["contamination"]).correlation

    d1, d2 = res["diff_Aclean_minus_A"], res["diff_Aclean_minus_B"]
    p1 = d1["coef"] > 0 and d1["ci95"][0] > 0
    p2 = d2["ci95"][1] >= 0
    verdict = "explains" if (p1 and p2) else "does not explain"
    note = ""
    if p1 and not p2:
        note = "A_clean is still steeper than B: the remainder of the size effect is unexplained."
    out = {"n": int(len(df)), "fixed_effects": int(len(set(grp))), "reproduction_check": repro,
           "estimates": res, "spearman_contamination_vs_logarea_pooled": float(rho),
           "mean_auroc": {m: float(df[f"auroc_{m}"].mean()) for m in methods},
           "criterion": {"part1_Aclean_minus_A_pos_ci_excl_0": p1, "part2_Aclean_not_steeper_than_B": p2,
                         "verdict": verdict, "note": note},
           "seconds_analysis": round(time.time() - t, 1), "seconds_total": round(time.time() - T0, 1)}
    with open(os.path.join(OUT, "mechanism_real_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    f2 = lambda e: f"{e['coef']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
    L = [f"PART A: real defects, n={out['n']}, {out['fixed_effects']} texture x defect-type fixed effects, "
         f"bootstrap {N_BOOT} over images", "",
         f"Reproduction vs size_effect_robustness: max |diff| A {repro['max_abs_diff_A']:.6f}, B42 {repro['max_abs_diff_B42']:.6f}",
         "Mean per-image AUROC: " + ", ".join(f"{m} {v:.2f}" for m, v in out["mean_auroc"].items()), "",
         "Slope of per-image pixel AUROC on log(area):"]
    for m in methods:
        L.append(f"  {m:<9} {f2(res[m])}")
    L.append("Differences of slopes:")
    for k, lab in (("diff_Aclean_minus_A", "A_clean - A"), ("diff_Amedian_minus_A", "A_median - A"),
                   ("diff_Atrim_minus_A", "A_trim - A (extra)"), ("diff_Aclean_minus_B", "A_clean - B")):
        L.append(f"  {lab:<19} {f2(res[k])}")
    L.append("Contamination index:")
    L.append(f"  on log(area) (FE)                 {f2(res['contam_on_logarea'])}   pooled Spearman {rho:+.3f}")
    L.append(f"  A ~ log(area) + contamination: log(area) {f2(res['A_on_logarea_with_contam'])}, "
             f"contamination {f2(res['A_on_contam_with_logarea'])}")
    L.append("")
    L.append(f"Criterion part 1 (A_clean - A > 0, CI excl. 0): {'met' if p1 else 'NOT met'}")
    L.append(f"Criterion part 2 (A_clean no steeper than B, CI upper >= 0): {'met' if p2 else 'NOT met'}")
    L.append(f"VERDICT: contamination {verdict} the real-defect size effect. {note}")
    L.append(f"Analysis {out['seconds_analysis']}s, total {out['seconds_total']}s")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "mechanism_real_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
