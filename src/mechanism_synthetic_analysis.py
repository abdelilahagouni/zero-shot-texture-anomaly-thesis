"""
PART B analysis (synthetic defects), as fixed before the full run.

Inputs : results/mechanism_synthetic/per_image_synthetic.csv, per_image_drift.csv, drift_control.csv
Output : results/mechanism_synthetic/analysis/

Unit of observation : one synthetic image (texture, base image, defect type, size).
x                   : log(achieved area fraction).
Route B             : mean over its 5 seeds per synthetic image.
Bootstrap           : 10,000 resamples of the 133 BASE images (each base image keeps all its sizes and types),
                      95% percentile CIs. Same resample indices are shared by all methods (paired).
Fixed effects       : pooled = texture + defect-type dummies; per type = texture dummies.

a. slopes of per-image pixel AUROC on x for A, A_clean, A_median, A_trim, B; differences
   A_clean-A, A_median-A, A_trim-A, A_clean-B (per type and pooled)
b. 50% prediction: mean AUROC(A) at 50% minus at the reference size has a 95% CI below 0, AND
   mean AUROC(A_clean) at 50% minus at the reference size >= -2 points (point estimate; CI reported).
   The fixed size grid has no 25%, so the prediction is evaluated against TWO reference sizes,
   32% (nearest to 25%) and 0.25% (same digits), and both are reported.
c. mechanism criterion: pooled AND >= 3 of 4 types pass (i) slope(A) < 0, CI excl. 0; (ii) A_clean-A > 0,
   CI excl. 0; (iii) the 50% prediction (reported for each reference size).
d. drift criterion: drift of normal patches > 5 patches from the blob on x: > 0 with CI excl. 0 pooled AND in
   >= 3 of 4 types; AND in AUROC ~ x + drift (pooled fixed effects) the drift coefficient has a CI excluding 0
   for BOTH A and B. Images with NaN drift (no patch > 5 away) are dropped and counted.
e. boundary vs interior: mean fraction of ring (<= 14 px inside the edge) and interior mask pixels above the
   99th percentile of the same map's normal pixels, per method, type and size.
f. Route B slope per type (from a).
g. leave-one-texture-out pooled slopes, and per texture x type slopes, for A, A_clean, B.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(REPO, "results", "mechanism_synthetic")
OUT = os.path.join(RES, "analysis")
N_BOOT = 10000
METHODS = ["A", "A_clean", "A_median", "A_trim", "B"]
TYPES = ["othertexture", "noise", "blur", "hueshift"]
DIFFS = [("A_clean", "A"), ("A_median", "A"), ("A_trim", "A"), ("A_clean", "B")]
REFS = [32.0, 0.25]


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def design(df, xcols, fe_cols):
    parts = [df[xcols].to_numpy(float)]
    for c in fe_cols:
        parts.append(pd.get_dummies(df[c]).to_numpy(float))
    return np.column_stack(parts)


class Boot:
    """Cluster (base image) bootstrap with indices shared across every analysis."""
    def __init__(self, clusters, n_boot, seed=12345):
        self.keys = np.unique(clusters)
        rng = np.random.default_rng(seed)
        self.draws = rng.integers(0, len(self.keys), size=(n_boot, len(self.keys)))

    def rows(self, df, b):
        idx = df.groupby("cluster").indices
        take = [idx[self.keys[k]] for k in self.draws[b] if self.keys[k] in idx]
        return np.concatenate(take)


def fe_coefs(df, xcols, ycols, fe_cols, boot):
    """Point estimates and bootstrap draws of the coefficients on xcols for each y column."""
    X = design(df, xcols, fe_cols)
    Y = df[ycols].to_numpy(float)
    est = np.linalg.lstsq(X, Y, rcond=None)[0][:len(xcols)]
    idx = df.groupby("cluster").indices
    present = np.array(sorted(idx))
    if len(present) == len(boot.keys):
        keys, D = boot.keys, boot.draws          # full set of base images: shared paired draws
    else:                                        # subset (e.g. one texture left out): resample its own base images
        keys = present
        D = np.random.default_rng(777 + len(present)).integers(0, len(present), size=(N_BOOT, len(present)))
    draws = np.empty((N_BOOT, len(xcols), len(ycols)))
    for b in range(N_BOOT):
        r = np.concatenate([idx[keys[k]] for k in D[b]])
        draws[b] = np.linalg.lstsq(X[r], Y[r], rcond=None)[0][:len(xcols)]
    return est, draws


def main():
    global RES, OUT, N_BOOT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES, help="folder with per_image_synthetic.csv, per_image_drift.csv, drift_control.csv")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    RES, OUT, N_BOOT = a.res, os.path.join(a.res, "analysis"), a.n_boot
    sys.stdout.reconfigure(encoding="utf-8")
    T0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    raw = pd.read_csv(os.path.join(RES, "per_image_synthetic.csv"))
    drift = pd.read_csv(os.path.join(RES, "per_image_drift.csv"))
    control = pd.read_csv(os.path.join(RES, "drift_control.csv"))
    key = ["texture", "type", "size_pct", "image_id"]
    raw["m"] = raw["method"]
    agg = raw.groupby(key + ["m"])[["auroc", "frac_ring_above_p99", "frac_interior_above_p99", "area_frac"]].mean()
    wide = agg["auroc"].unstack("m")[METHODS]
    area = agg["area_frac"].groupby(level=key).first()
    df = wide.join(area).reset_index()
    df = df.merge(drift[key + ["drift_gt5", "n_drift_gt5"]], on=key, how="left")
    df["x"] = np.log(df["area_frac"])
    df["cluster"] = df["texture"] + "/" + df["image_id"].astype(str)
    boot = Boot(df["cluster"].to_numpy(), N_BOOT)
    out = {"n_synthetic_images": int(len(df)), "n_base_images": int(len(boot.keys)), "n_boot": N_BOOT,
           "drift_control": {"max": float(control["control_drift_max"].max()),
                             "tokens_identical": int(control["control_tokens_identical"].sum()),
                             "n": int(len(control))}}
    L = [f"PART B ANALYSIS: {len(df)} synthetic images from {len(boot.keys)} base images; "
         f"bootstrap {N_BOOT} over base images", ""]
    f3 = lambda c, lo, hi: f"{c:+.3f} [{lo:+.3f}, {hi:+.3f}]"

    # ---- a / f: slopes
    t = time.time()
    slopes = {}
    for scope in ["pooled"] + TYPES:
        sub = df if scope == "pooled" else df[df["type"] == scope]
        fe = ["texture", "type"] if scope == "pooled" else ["texture"]
        est, dr = fe_coefs(sub, ["x"], METHODS, fe, boot)
        s = {m: {"coef": float(est[0, j]), "ci95": ci(dr[:, 0, j])} for j, m in enumerate(METHODS)}
        for a, b in DIFFS:
            ja, jb = METHODS.index(a), METHODS.index(b)
            s[f"{a}-{b}"] = {"coef": float(est[0, ja] - est[0, jb]), "ci95": ci(dr[:, 0, ja] - dr[:, 0, jb])}
        slopes[scope] = s
    out["slopes"] = slopes
    L.append("a/f. Slope of per-image pixel AUROC on log(area) [95% CI]")
    for scope, s in slopes.items():
        L.append(f"  {scope}")
        for k in METHODS + [f"{a}-{b}" for a, b in DIFFS]:
            L.append(f"    {k:<17} {f3(s[k]['coef'], *s[k]['ci95'])}")
    L.append(f"  (time {time.time() - t:.0f}s)")
    L.append("")

    # ---- b: 50% prediction
    pred = {}
    for scope in ["pooled"] + TYPES:
        sub = df if scope == "pooled" else df[df["type"] == scope]
        pred[scope] = {}
        for ref in REFS:
            r50, rr = sub[sub.size_pct == 50], sub[sub.size_pct == ref]
            e = {}
            for m in ("A", "A_clean"):
                diff = r50[m].mean() - rr[m].mean()
                bs = []
                g50, gr = r50.groupby("cluster")[m].apply(list), rr.groupby("cluster")[m].apply(list)
                for b in range(N_BOOT):
                    ks = boot.keys[boot.draws[b]]
                    v50 = np.concatenate([g50[k] for k in ks if k in g50.index])
                    vr = np.concatenate([gr[k] for k in ks if k in gr.index])
                    bs.append(v50.mean() - vr.mean())
                e[m] = {"mean_at_50": float(r50[m].mean()), f"mean_at_ref": float(rr[m].mean()),
                        "diff": float(diff), "ci95": ci(bs)}
            e["A_lower"] = e["A"]["ci95"][1] < 0
            e["A_clean_not_lower_by_more_than_2"] = e["A_clean"]["diff"] >= -2
            e["holds"] = bool(e["A_lower"] and e["A_clean_not_lower_by_more_than_2"])
            pred[scope][str(ref)] = e
    out["prediction_50pct"] = pred
    L.append("b. 50% prediction (diff = mean AUROC at 50% minus at reference size)")
    for scope, p in pred.items():
        for ref in REFS:
            e = p[str(ref)]
            L.append(f"  {scope:<12} ref {ref:>5}%  A {e['A']['mean_at_ref']:6.2f}->{e['A']['mean_at_50']:6.2f} "
                     f"diff {f3(e['A']['diff'], *e['A']['ci95'])}   A_clean {e['A_clean']['mean_at_ref']:6.2f}->"
                     f"{e['A_clean']['mean_at_50']:6.2f} diff {f3(e['A_clean']['diff'], *e['A_clean']['ci95'])}   "
                     f"holds: {e['holds']}")
    L.append("")

    # ---- c: mechanism criterion
    mech = {}
    for ref in REFS:
        rows = {}
        for scope in ["pooled"] + TYPES:
            s = slopes[scope]
            i_ = s["A"]["coef"] < 0 and s["A"]["ci95"][1] < 0
            ii = s["A_clean-A"]["coef"] > 0 and s["A_clean-A"]["ci95"][0] > 0
            iii = pred[scope][str(ref)]["holds"]
            rows[scope] = {"i": bool(i_), "ii": bool(ii), "iii": bool(iii), "all": bool(i_ and ii and iii)}
        n_types = sum(rows[t]["all"] for t in TYPES)
        rows["verdict"] = "supported" if rows["pooled"]["all"] and n_types >= 3 else "not supported"
        rows["types_passing_all"] = int(n_types)
        mech[str(ref)] = rows
    out["mechanism_criterion"] = mech
    L.append("c. Self-contamination criterion: (i) slope A<0; (ii) A_clean-A>0; (iii) 50% prediction")
    for ref in REFS:
        r = mech[str(ref)]
        L.append(f"  reference size for (iii) = {ref}%")
        for scope in ["pooled"] + TYPES:
            v = r[scope]
            L.append(f"    {scope:<12} (i) {'pass' if v['i'] else 'FAIL'}  (ii) {'pass' if v['ii'] else 'FAIL'}  "
                     f"(iii) {'pass' if v['iii'] else 'FAIL'}")
        L.append(f"    types passing all three: {r['types_passing_all']}/4 -> VERDICT: {r['verdict']}")
    L.append("")

    # ---- d: drift criterion
    t = time.time()
    dd = df[np.isfinite(df["drift_gt5"])].copy()
    dropped = df[~np.isfinite(df["drift_gt5"])]
    drift_res = {"n_used": int(len(dd)), "n_dropped_nan": int(len(dropped)),
                 "dropped_by_size": {str(k): int(v) for k, v in dropped.groupby("size_pct").size().items()}}
    for scope in ["pooled"] + TYPES:
        sub = dd if scope == "pooled" else dd[dd["type"] == scope]
        fe = ["texture", "type"] if scope == "pooled" else ["texture"]
        est, dr = fe_coefs(sub, ["x"], ["drift_gt5"], fe, boot)
        drift_res[f"drift_on_logarea_{scope}"] = {"coef": float(est[0, 0]), "ci95": ci(dr[:, 0, 0])}
    est, dr = fe_coefs(dd, ["x", "drift_gt5"], ["A", "B"], ["texture", "type"], boot)
    for j, m in enumerate(["A", "B"]):
        drift_res[f"auroc_{m}_on_drift_given_logarea"] = {"coef": float(est[1, j]), "ci95": ci(dr[:, 1, j])}
        drift_res[f"auroc_{m}_on_logarea_given_drift"] = {"coef": float(est[0, j]), "ci95": ci(dr[:, 0, j])}
    pos = lambda e: e["coef"] > 0 and e["ci95"][0] > 0
    excl = lambda e: e["ci95"][0] > 0 or e["ci95"][1] < 0
    p1_pooled = pos(drift_res["drift_on_logarea_pooled"])
    p1_types = [t_ for t_ in TYPES if pos(drift_res[f"drift_on_logarea_{t_}"])]
    p2 = excl(drift_res["auroc_A_on_drift_given_logarea"]) and excl(drift_res["auroc_B_on_drift_given_logarea"])
    drift_res["part1_pooled"], drift_res["part1_types"] = bool(p1_pooled), p1_types
    drift_res["part2_both_A_and_B"] = bool(p2)
    drift_res["verdict"] = "supported" if (p1_pooled and len(p1_types) >= 3 and p2) else "not supported"
    out["drift_criterion"] = drift_res
    L.append(f"d. Drift criterion (drift of normal patches > 5 patches from the blob)")
    L.append(f"  images used {drift_res['n_used']}, dropped for NaN drift {drift_res['n_dropped_nan']} "
             f"(by size: {drift_res['dropped_by_size']})")
    L.append(f"  control: max drift clean-vs-clean {out['drift_control']['max']:.3e}, tokens identical "
             f"{out['drift_control']['tokens_identical']}/{out['drift_control']['n']}")
    for scope in ["pooled"] + TYPES:
        e = drift_res[f"drift_on_logarea_{scope}"]
        L.append(f"  drift on log(area), {scope:<12} {f3(e['coef'], *e['ci95'])}")
    for m in ("A", "B"):
        e1, e2 = drift_res[f"auroc_{m}_on_drift_given_logarea"], drift_res[f"auroc_{m}_on_logarea_given_drift"]
        L.append(f"  AUROC {m} ~ log(area) + drift: drift {f3(e1['coef'], *e1['ci95'])}, "
                 f"log(area) {f3(e2['coef'], *e2['ci95'])}")
    L.append(f"  part 1 pooled {'pass' if p1_pooled else 'FAIL'}; types passing {len(p1_types)}/4 {p1_types}; "
             f"part 2 (A and B) {'pass' if p2 else 'FAIL'} -> VERDICT: {drift_res['verdict']}")
    L.append(f"  (time {time.time() - t:.0f}s)")
    L.append("")

    # ---- e: boundary vs interior
    bi = raw.copy()
    bi.loc[bi["method"] == "B", "m"] = "B"
    tab = bi.groupby(["m", "type", "size_pct"]).agg(
        ring=("frac_ring_above_p99", "mean"), interior=("frac_interior_above_p99", "mean"),
        n_interior=("frac_interior_above_p99", lambda s: int(s.notna().sum()))).reset_index()
    tab.to_csv(os.path.join(OUT, "boundary_vs_interior.csv"), index=False)
    L.append("e. Boundary (<=14 px) vs interior: mean fraction of mask pixels above the 99th percentile of the same "
             "map's normal pixels (B: all 5 seeds pooled); interior is NaN when the blob has no interior")
    piv = tab.pivot_table(index=["type", "size_pct"], columns="m", values=["ring", "interior"])
    for (ty, sz), r in piv.iterrows():
        L.append(f"  {ty:<12} {sz:>5}%  " + "  ".join(
            f"{m}: ring {r[('ring', m)]:.3f} int {r[('interior', m)]:.3f}" for m in METHODS))
    L.append("")

    # ---- g: leave-one-texture-out and texture x type slopes
    t = time.time()
    g3 = ["A", "A_clean", "B"]
    loto = {}
    for tex in sorted(df["texture"].unique()):
        sub = df[df["texture"] != tex]
        if sub.empty:
            continue
        est, dr = fe_coefs(sub, ["x"], g3, ["texture", "type"], boot)
        loto[f"without_{tex}"] = {m: {"coef": float(est[0, j]), "ci95": ci(dr[:, 0, j])} for j, m in enumerate(g3)}
    combos = {}
    for (tex, ty), sub in df.groupby(["texture", "type"]):
        est, dr = fe_coefs(sub.assign(one=1), ["x"], g3, ["one"], boot)
        combos[f"{tex}/{ty}"] = {m: {"coef": float(est[0, j]), "ci95": ci(dr[:, 0, j])} for j, m in enumerate(g3)}
    out["leave_one_texture_out"], out["texture_x_type_slopes"] = loto, combos
    L.append("g. Leave-one-texture-out pooled slopes (texture + type fixed effects)")
    for k, v in loto.items():
        L.append(f"  {k:<16} " + "  ".join(f"{m} {f3(v[m]['coef'], *v[m]['ci95'])}" for m in g3))
    L.append("   Texture x type slopes")
    for k, v in combos.items():
        L.append(f"  {k:<22} " + "  ".join(f"{m} {f3(v[m]['coef'], *v[m]['ci95'])}" for m in g3))
    L.append(f"  (time {time.time() - t:.0f}s)")

    # ---- figure: mean AUROC by size per type, and drift > 5 by size
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for j, ty in enumerate(TYPES):
        sub = df[df["type"] == ty].groupby("size_pct")
        ax = axes[0, j]
        for m in METHODS:
            ax.plot(sub[m].mean().index, sub[m].mean().values, marker="o", label=m)
        ax.set_xscale("log"); ax.minorticks_off(); ax.set_title(ty); ax.grid(alpha=0.3)
        ax2 = axes[1, j]
        ax2.plot(sub["drift_gt5"].mean().index, sub["drift_gt5"].mean().values, marker="s", color="k")
        ax2.set_xscale("log"); ax2.minorticks_off(); ax2.grid(alpha=0.3)
        ax2.set_xticks(SIZES_PLOT); ax2.set_xticklabels([f"{v:g}" for v in SIZES_PLOT], fontsize=7)
        ax2.set_xlabel("defect area (% of image)")
    axes[0, 0].set_ylabel("mean per-image pixel AUROC"); axes[1, 0].set_ylabel("drift, normal patches > 5 away")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "auroc_and_drift_by_size.png"), dpi=130)

    out["seconds_total"] = round(time.time() - T0, 1)
    L.append("")
    L.append(f"Total analysis time {out['seconds_total']}s")
    with open(os.path.join(OUT, "mechanism_synthetic_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    df.to_csv(os.path.join(OUT, "per_synthetic_image_wide.csv"), index=False)
    txt = "\n".join(L)
    with open(os.path.join(OUT, "mechanism_synthetic_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


SIZES_PLOT = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 50]

if __name__ == "__main__":
    main()
