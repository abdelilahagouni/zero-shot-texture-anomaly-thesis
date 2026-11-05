"""
Feature-layer sweep for Route A and Route B (criteria fixed before running). Writes results/feature_layer_sweep/ only.

Setup (as src/mechanism_common.py / src/eval_router_full.py): 515 MVTec texture test images, DINOv2 ViT-B/14 at
518x518, ImageNet normalisation, tokens L2-normalised, fp16 round trip, fp32 scoring, post-processing bilinear
37->518 + GaussianBlur 15x15; metrics from src/evaluation.py via src/fca_baseline/evaluate.py (pixel AP, AU-PRO@0.3,
pixel AUROC, image AUROC = max of map); Route B: 4 training images chosen by np.random.default_rng(seed).choice,
seeds 42, 123, 2026; reported as the 3-seed mean (and std).

Factors
  layer   : L6, L8, L10, L12 = output of block 6/8/10/12 (indices 5/7/9/11) from
            model.get_intermediate_layers(x, n=[i], norm=True) (final LayerNorm applied); M789 = mean of the normed
            outputs of blocks 7, 8, 9, then L2-normalised. L12 equals the current forward_features tokens.
  pooling : p1 = none; p3 = 3x3 average pooling of the tokens on the 37x37 grid (stride 1, padding 1,
            count_include_pad=False), then L2-normalised again; applied identically to test tokens and bank tokens.
  Route A : mean = 1 - mean cos to all 1369 patches (current); k25 / k75 = 1 - mean of the K largest cos
            (K = 342 / 1027 = round(0.25 / 0.75 * 1369); the patch itself included, as in Aota).
  Route B : 1 - max cos to the 4-image bank, same layer and pooling.
Protocols, always both:
  full   : 518x518 maps and masks.
  crop10 : 52 px border removed from maps and masks; test images whose defect lies entirely in the border are
           DROPPED (pixel and image metrics) and counted. For the reproduction check only, "crop10_keep" (those
           images kept as normal, the convention of results/fca_baseline) is computed as well.

Stages
  --stage repro  : units (texture, L12, p1) for all 5 textures; checks against the existing numbers (5-texture
                   averages to 2 decimals; crop10_keep vs results/fca_baseline/metrics/metrics_table.csv; per-image
                   pixel AUROC of A and B42/B123/B2026, full and crop10, EXACTLY equal to
                   results/fca_baseline/metrics/per_image_defective.csv). Prints PASS/FAIL and an estimate.
  --stage full   : all remaining units (resumable: one metrics CSV + one per-image CSV per (texture, layer, pooling)).
  --stage analyze: tables (next to FCA), size analysis, criteria.

Size analysis (as src/fca_baseline/size_analysis.py): per-image pixel AUROC of the defective images ~ log(area) +
texture x defect-type fixed effects, 10,000 paired bootstrap resamples over images, np.random.default_rng(99);
full: 382 images, x = log(area_px); crop10: 380 images, x = log(area in the crop). B per image = mean of 3 seeds.
For every Route A variant: slope(A), slope(B of the same layer and pooling), slope(A) - slope(B).

Criteria (fixed now)
  (i)  supported only if some Route A configuration (layer x scoring x pooling) improves Route A's 5-texture average
       AU-PRO by >= 3 points over the last layer (L12, mean, p1), the SAME configuration on BOTH protocols.
  (ii) supported only if slope(A) - slope(B) < 0 with CI excluding 0 for >= 4 of the 5 layer configurations
       (A = mean scoring, p1; B = p1, same layer), on BOTH protocols.
  (iii) for each protocol: mean over the 10 (layer, pooling) pairs of |AU-PRO(k) - AU-PRO(mean)| (5-texture average)
       for k = k25 and k75; holds on a protocol if both means are > 1 point; supported only if it holds on both.
No configuration is selected as best; all are reported.
"""
import argparse, datetime, glob, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "fca_baseline"))
from common import DATA, TEXTURES, GRID, BORDER, test_items, load_gt, crop, map_path, to_grid  # noqa: E402

OUT = os.path.join(REPO, "results", "feature_layer_sweep")
PART = os.path.join(OUT, "partial")
FCA_RES = os.path.join(REPO, "results", "fca_baseline", "metrics")
LAYERS = {"L6": [5], "L8": [7], "L10": [9], "L12": [11], "M789": [6, 7, 8]}
POOLS = ["p1", "p3"]
SCORINGS = {"mean": None, "k25": int(round(0.25 * 1369)), "k75": int(round(0.75 * 1369))}
SEEDS = [42, 123, 2026]
METRICS = ["pixel_ap", "au_pro", "pixel_auroc", "image_auroc"]
N_BOOT = 10000
EXISTING = {"A": {"pixel_ap": 32.09, "au_pro": 86.60, "pixel_auroc": 92.08, "image_auroc": 87.05},
            "B": {"pixel_ap": 50.04, "au_pro": 94.85, "pixel_auroc": 97.00, "image_auroc": 99.79}}


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", flush=True)


class Tok:
    def __init__(self):
        from mechanism_common import Backbone
        self.bb = Backbone()
        self.torch, self.F = self.bb.torch, self.bb.F

    def tokens(self, pil, idx):
        img = self.bb.tf(pil.convert("RGB")).unsqueeze(0).to(self.bb.dev)
        with self.torch.no_grad():
            outs = self.bb.model.get_intermediate_layers(img, n=idx, norm=True)
        f = outs[0][0] if len(outs) == 1 else self.torch.stack([o[0] for o in outs]).mean(0)
        f = self.F.normalize(f, p=2, dim=-1).cpu().to(self.torch.float16)
        return f.to(self.bb.dev).float()

    def pool(self, t, pool):
        if pool == "p1":
            return t
        g = t.T.reshape(1, -1, 37, 37)
        g = self.F.avg_pool2d(g, 3, stride=1, padding=1, count_include_pad=False)
        return self.F.normalize(g.reshape(t.shape[1], -1).T, p=2, dim=-1)


def run_unit(T, cat, layer, pool, extra_keep=False):
    from PIL import Image
    from evaluate import metrics, per_image_auroc
    items = test_items(cat)
    gts = [load_gt(mp) for _, _, _, mp in items]
    idx = LAYERS[layer]
    train = [T.pool(T.tokens(Image.open(p), idx), pool)
             for p in sorted(glob.glob(os.path.join(DATA, cat, "train", "good", "*.png")))]
    banks = {sd: T.bb.bank(train, sd) for sd in SEEDS}
    names = [f"A_{s}" for s in SCORINGS] + [f"B{sd}" for sd in SEEDS]
    maps = {k: [] for k in names}
    for _, _, p, _ in items:
        tok = T.pool(T.tokens(Image.open(p), idx), pool)
        sim = tok @ tok.T
        for s, K in SCORINGS.items():
            sc = sim.mean(dim=1) if K is None else T.torch.topk(sim, K, dim=1).values.mean(dim=1)
            maps[f"A_{s}"].append(T.bb.post(1.0 - sc))
        for sd in SEEDS:
            maps[f"B{sd}"].append(T.bb.post(1.0 - (tok @ banks[sd].T).max(dim=1)[0]))
    gts_c = [crop(g) for g in gts]
    keep = [not (g.any() and not gc.any()) for g, gc in zip(gts, gts_c)]
    n_drop = int(len(keep) - sum(keep))
    rows = []
    for k, ms in maps.items():
        ms_c = [crop(m) for m in ms]
        prots = [("full", ms, gts), ("crop10", [m for m, w in zip(ms_c, keep) if w], [g for g, w in zip(gts_c, keep) if w])]
        if extra_keep:
            prots.append(("crop10_keep", ms_c, gts_c))
        for prot, M, G in prots:
            rows.append({"texture": cat, "layer": layer, "pool": pool, "method": k, "protocol": prot,
                         "n_images": len(M), "n_dropped_border_only": n_drop if prot == "crop10" else 0, **metrics(M, G)})
    pim = []
    for i, (d, fn, _, _) in enumerate(items):
        if d == "good":
            continue
        r = {"category": cat, "defect_type": d, "file": fn, "layer": layer, "pool": pool,
             "area_px": int(gts[i].sum()), "area_px_crop10": int(gts_c[i].sum())}
        for k, ms in maps.items():
            r[f"{k}_auroc"] = per_image_auroc(ms[i], gts[i])
            r[f"{k}_auroc_crop10"] = per_image_auroc(crop(ms[i]), gts_c[i])
        pim.append(r)
    return pd.DataFrame(rows), pd.DataFrame(pim)


def unit_files(cat, layer, pool):
    s = os.path.join(PART, f"{cat}__{layer}__{pool}")
    return s + "__metrics.csv", s + "__per_image.csv"


def run_units(units, extra_keep=False):
    T = Tok()
    todo = [u for u in units if not all(os.path.exists(f) for f in unit_files(*u))]
    log(f"{len(units)} units, {len(units) - len(todo)} done, {len(todo)} to run")
    times = []
    T.torch.cuda.reset_peak_memory_stats()
    for n, (cat, layer, pool) in enumerate(todo, 1):
        t = time.time()
        m, p = run_unit(T, cat, layer, pool, extra_keep)
        for df, f in zip((m, p), unit_files(cat, layer, pool)):
            df.to_csv(f + ".tmp", index=False)
            os.replace(f + ".tmp", f)
        times.append(time.time() - t)
        eta = datetime.timedelta(seconds=int(np.mean(times) * (len(todo) - n)))
        a = m[(m.method == "A_mean") & (m.protocol == "full")]["au_pro"].iloc[0]
        log(f"[{n}/{len(todo)}] {cat} {layer} {pool}: A_mean AU-PRO full {a:.2f} ({times[-1]:.0f}s) ETA {eta}  "
            f"peak GPU {T.torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated, "
            f"{T.torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved")
    return times


def load_all():
    m = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PART, "*__metrics.csv")))], ignore_index=True)
    p = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PART, "*__per_image.csv")))], ignore_index=True)
    return m, p


def stage_repro():
    os.makedirs(PART, exist_ok=True)
    t0 = time.time()
    times = run_units([(c, "L12", "p1") for c in TEXTURES], extra_keep=True)
    m, p = load_all()
    m = m[(m.layer == "L12") & (m.pool == "p1")]
    p = p[(p.layer == "L12") & (p.pool == "p1")]
    L, ok = ["REPRODUCTION CHECK: L12 (get_intermediate_layers n=[11], norm=True), mean scoring, p1", ""], True
    avg = lambda prot, meth: m[(m.protocol == prot) & (m.method == meth)].groupby("texture")[METRICS].mean().mean()
    a_full = avg("full", "A_mean")
    b_full = pd.concat([avg("full", f"B{sd}") for sd in SEEDS], axis=1).mean(axis=1)
    L.append("5-texture averages (full) vs existing:")
    for meth, v in (("A", a_full), ("B", b_full)):
        for k in METRICS:
            e = EXISTING[meth][k]
            good = round(v[k], 2) == e
            ok &= good
            L.append(f"  {meth} {k:<12} {v[k]:8.4f} (rounded {v[k]:.2f}) existing {e:.2f}  {'match' if good else 'MISMATCH'}")
    prev = pd.read_csv(os.path.join(FCA_RES, "metrics_table.csv"))
    L.append("crop10_keep (border-only images kept as normal, as results/fca_baseline) vs metrics_table.csv:")
    a_c = avg("crop10_keep", "A_mean")
    b_c = pd.concat([avg("crop10_keep", f"B{sd}") for sd in SEEDS], axis=1).mean(axis=1)
    for meth, v in (("A", a_c), ("B", b_c)):
        e = prev[(prev.version == "crop10") & (prev.texture == "average") & (prev.method == meth)].iloc[0]
        for k in METRICS:
            d = abs(v[k] - e[k])
            good = d < 1e-6
            ok &= good
            L.append(f"  {meth} {k:<12} {v[k]:8.4f} existing {e[k]:8.4f}  |diff| {d:.2e}  {'match' if good else 'MISMATCH'}")
    pv = pd.read_csv(os.path.join(FCA_RES, "per_image_defective.csv"))
    key = ["category", "defect_type", "file"]
    pv = pv.rename(columns={c: f"{c}_prev" for c in pv.columns if c not in key})
    j = p.merge(pv, on=key)
    L.append(f"per-image pixel AUROC vs per_image_defective.csv (n matched {len(j)} of {len(p)}):")
    for new, old in [("A_mean", "A")] + [(f"B{sd}", f"B{sd}") for sd in SEEDS]:
        for v in ("", "_crop10"):
            a, b = j[f"{new}_auroc{v}"], j[f"{old}_auroc{v}_prev"]
            same_nan = bool((a.isna() == b.isna()).all())
            d = float((a - b).abs().max())
            good = same_nan and d == 0.0 and len(j) == 382
            ok &= good
            L.append(f"  {new}{v or ' full'}: max |diff| {d:.3e}, NaN pattern equal {same_nan}  {'exact' if good else 'NOT exact'}")
    nd = m[(m.protocol == "crop10") & (m.method == "A_mean")].set_index("texture")["n_dropped_border_only"].to_dict()
    L.append(f"crop10: border-only defective images dropped per texture {nd} (total {sum(nd.values())})")
    per_unit = float(np.mean(times)) if times else float("nan")
    n_rest = len(TEXTURES) * len(LAYERS) * len(POOLS) - len(TEXTURES)
    L.append("")
    L.append(f"REPRODUCTION: {'PASS' if ok else 'FAIL'}   ({time.time() - t0:.0f}s)")
    if times:
        L.append(f"ESTIMATE for the full stage: {n_rest} remaining units x {per_unit:.0f} s (mean of the 5 repro units) "
                 f"~ {n_rest * per_unit / 3600:.1f} h")
    txt = "\n".join(L)
    open(os.path.join(OUT, "reproduction_check.txt"), "w", encoding="utf-8").write(txt)
    json.dump({"pass": bool(ok), "seconds_per_unit": per_unit}, open(os.path.join(OUT, "reproduction_check.json"), "w"))
    print(txt)
    sys.exit(0 if ok else 3)


def stage_full():
    rc = json.load(open(os.path.join(OUT, "reproduction_check.json")))
    if not rc["pass"]:
        sys.exit("reproduction check did not pass; not running the sweep")
    run_units([(c, l, p) for c in TEXTURES for l in LAYERS for p in POOLS])
    log("full stage done")


def stage_analyze():
    from size_analysis import fe_coefs, ci
    t0 = time.time()
    m, p = load_all()
    m = m[m.protocol.isin(["full", "crop10"])]
    assert len(m.groupby(["texture", "layer", "pool"])) == len(TEXTURES) * len(LAYERS) * len(POOLS)
    # ---------------- tables
    rows = []
    for prot in ("full", "crop10"):
        for (layer, pool), g in m[m.protocol == prot].groupby(["layer", "pool"]):
            for tex in TEXTURES + ["average"]:
                sub = g if tex == "average" else g[g.texture == tex]
                for s in SCORINGS:
                    v = sub[sub.method == f"A_{s}"].groupby("texture")[METRICS].mean().mean()
                    rows.append({"protocol": prot, "texture": tex, "route": "A", "layer": layer, "pool": pool,
                                 "scoring": s, **v.to_dict()})
                per = pd.concat([sub[sub.method == f"B{sd}"].groupby("texture")[METRICS].mean().mean() for sd in SEEDS], axis=1)
                rows.append({"protocol": prot, "texture": tex, "route": "B", "layer": layer, "pool": pool, "scoring": "max",
                             **per.mean(axis=1).to_dict(), **{f"{k}_std_seeds": per.loc[k].std(ddof=1) for k in METRICS}})
    tab = pd.DataFrame(rows)
    # FCA next to it: results/fca_baseline (crop10 there keeps border-only images as normal) + crop10 with them dropped
    fprev = pd.read_csv(os.path.join(FCA_RES, "metrics_table.csv"))
    fca = []
    for prot, ver in (("full", "full"), ("crop10_keep", "crop10")):
        for tex in TEXTURES + ["average"]:
            r = fprev[(fprev.version == ver) & (fprev.texture == tex) & (fprev.method == "FCA")].iloc[0]
            fca.append({"protocol": prot, "texture": tex, "route": "FCA", "layer": "-", "pool": "-",
                        "scoring": "results/fca_baseline", **{k: r[k] for k in METRICS}})
    from evaluate import metrics
    per_tex = {}
    for tex in TEXTURES:
        M, G = [], []
        for d, fn, _, mp in test_items(tex):
            g = crop(load_gt(mp))
            if mp is not None and not g.any():
                continue
            M.append(crop(to_grid(np.load(map_path("fca", tex, d, fn)))))
            G.append(g)
        per_tex[tex] = metrics(M, G)
    for tex in TEXTURES + ["average"]:
        v = pd.DataFrame(per_tex).T.mean() if tex == "average" else pd.Series(per_tex[tex])
        fca.append({"protocol": "crop10", "texture": tex, "route": "FCA", "layer": "-", "pool": "-",
                    "scoring": "saved maps, border-only dropped", **v.to_dict()})
    tab = pd.concat([tab, pd.DataFrame(fca)], ignore_index=True)
    tab.to_csv(os.path.join(OUT, "metrics_table.csv"), index=False)

    # ---------------- size analysis
    a_names = [(l, pl, s) for l in LAYERS for pl in POOLS for s in SCORINGS]
    b_names = [(l, pl) for l in LAYERS for pl in POOLS]
    size = {}
    for prot in ("full", "crop10"):
        v = "" if prot == "full" else "_crop10"
        base = p[(p.layer == "L12") & (p.pool == "p1")][["category", "defect_type", "file", "area_px", "area_px_crop10"]]
        W = base.copy()
        for l, pl in b_names:
            q = p[(p.layer == l) & (p.pool == pl)].set_index(["category", "defect_type", "file"])
            for s in SCORINGS:
                W[f"A|{l}|{pl}|{s}"] = q.loc[list(zip(base.category, base.defect_type, base.file)), f"A_{s}_auroc{v}"].to_numpy()
            W[f"B|{l}|{pl}"] = q.loc[list(zip(base.category, base.defect_type, base.file)),
                                     [f"B{sd}_auroc{v}" for sd in SEEDS]].mean(axis=1).to_numpy()
        area = "area_px" if prot == "full" else "area_px_crop10"
        n_drop = int((W[area] == 0).sum())
        W = W[W[area] > 0].reset_index(drop=True)
        cols = [f"A|{l}|{pl}|{s}" for l, pl, s in a_names] + [f"B|{l}|{pl}" for l, pl in b_names]
        Y = W[cols].to_numpy(float)
        if np.isnan(Y).any():
            sys.exit(f"ERROR: NaN per-image AUROC ({prot})")
        x = np.log(W[area].to_numpy(float))
        grp = (W["category"] + "/" + W["defect_type"]).to_numpy()
        est = fe_coefs(x, Y, grp)
        rng = np.random.default_rng(99)
        Bt = np.empty((N_BOOT, len(cols)))
        for b in range(N_BOOT):
            i = rng.integers(0, len(x), len(x))
            Bt[b] = fe_coefs(x[i], Y[i], grp[i])
        ci_of = lambda k: ci(Bt[:, k])
        res = {"n": int(len(W)), "n_dropped_border_only": n_drop, "fixed_effects": int(len(set(grp))), "rows": []}
        for l, pl, s in a_names:
            ia, ib = cols.index(f"A|{l}|{pl}|{s}"), cols.index(f"B|{l}|{pl}")
            res["rows"].append({"layer": l, "pool": pl, "scoring": s,
                                "slope_A": float(est[ia]), "slope_A_ci": ci_of(ia),
                                "slope_B": float(est[ib]), "slope_B_ci": ci_of(ib),
                                "diff": float(est[ia] - est[ib]), "diff_ci": ci(Bt[:, ia] - Bt[:, ib])})
        size[prot] = res

    # ---------------- criteria
    A = tab[(tab.route == "A") & (tab.texture == "average")]
    base_pro = {prot: float(A[(A.protocol == prot) & (A.layer == "L12") & (A.pool == "p1") & (A.scoring == "mean")]["au_pro"].iloc[0])
                for prot in ("full", "crop10")}
    gains = []
    for l, pl, s in a_names:
        g = {prot: float(A[(A.protocol == prot) & (A.layer == l) & (A.pool == pl) & (A.scoring == s)]["au_pro"].iloc[0]) - base_pro[prot]
             for prot in ("full", "crop10")}
        gains.append({"layer": l, "pool": pl, "scoring": s, "gain_full": g["full"], "gain_crop10": g["crop10"],
                      "both_ge_3": bool(g["full"] >= 3 and g["crop10"] >= 3)})
    crit_i = any(r["both_ge_3"] for r in gains)
    ii = {}
    for prot in ("full", "crop10"):
        neg = [r for r in size[prot]["rows"] if r["pool"] == "p1" and r["scoring"] == "mean"]
        ii[prot] = {r["layer"]: bool(r["diff"] < 0 and r["diff_ci"][1] < 0) for r in neg}
    crit_ii = all(sum(ii[prot].values()) >= 4 for prot in ii)
    iii = {}
    for prot in ("full", "crop10"):
        d = {}
        for k in ("k25", "k75"):
            diffs = [float(A[(A.protocol == prot) & (A.layer == l) & (A.pool == pl) & (A.scoring == k)]["au_pro"].iloc[0]
                           - A[(A.protocol == prot) & (A.layer == l) & (A.pool == pl) & (A.scoring == "mean")]["au_pro"].iloc[0])
                     for l, pl in b_names]
            d[k] = {"mean_abs_diff": float(np.mean(np.abs(diffs))), "mean_signed_diff": float(np.mean(diffs)),
                    "diffs": diffs}
        d["holds"] = bool(d["k25"]["mean_abs_diff"] > 1 and d["k75"]["mean_abs_diff"] > 1)
        iii[prot] = d
    crit_iii = all(iii[p_]["holds"] for p_ in iii)
    out = {"criteria": {"i": {"supported": crit_i, "baseline_au_pro": base_pro, "gains": gains},
                        "ii": {"supported": crit_ii, "per_protocol": ii},
                        "iii": {"supported": crit_iii, "per_protocol": iii}},
           "size_analysis": size, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(os.path.join(OUT, "analysis_results.json"), "w"), indent=1)
    pd.DataFrame([dict(protocol=pr, **r) for pr in size for r in size[pr]["rows"]]).to_csv(
        os.path.join(OUT, "size_analysis.csv"), index=False)

    # ---------------- text
    L = ["FEATURE-LAYER SWEEP (all configurations reported; none selected)", ""]
    for prot in ("full", "crop10"):
        nd = int(m[(m.protocol == "crop10") & (m.method == "A_mean") & (m.layer == "L12") & (m.pool == "p1")]["n_dropped_border_only"].sum())
        L.append(f"=== {prot}" + (f" (52 px border removed; {nd} border-only defective images dropped)" if prot == "crop10" else ""))
        for tex in TEXTURES + ["average"]:
            L.append(f"  {tex}:  {'config':<22} " + " ".join(f"{k:>11}" for k in METRICS))
            sub = tab[(tab.protocol == prot) & (tab.texture == tex)]
            for _, r in sub[sub.route != "FCA"].iterrows():
                lab = f"{r.route} {r.layer} {r.pool} {r.scoring}"
                L.append(f"     {lab:<24} " + " ".join(f"{r[k]:11.2f}" for k in METRICS))
            for _, r in tab[(tab.route == "FCA") & (tab.texture == tex) &
                            (tab.protocol.isin([prot] + (["crop10_keep"] if prot == "crop10" else [])))].iterrows():
                L.append(f"     {'FCA (' + r.scoring + (', kept' if r.protocol == 'crop10_keep' else '') + ')':<24} "
                         + " ".join(f"{r[k]:11.2f}" for k in METRICS))
        L.append("")
    for prot in ("full", "crop10"):
        s = size[prot]
        L.append(f"SIZE ANALYSIS {prot}: n={s['n']} (dropped {s['n_dropped_border_only']}), {s['fixed_effects']} FE, "
                 f"bootstrap {N_BOOT}")
        for r in s["rows"]:
            L.append(f"  A {r['layer']:<4} {r['pool']} {r['scoring']:<4}  slope A {r['slope_A']:+.3f} [{r['slope_A_ci'][0]:+.3f}, "
                     f"{r['slope_A_ci'][1]:+.3f}]  slope B {r['slope_B']:+.3f} [{r['slope_B_ci'][0]:+.3f}, {r['slope_B_ci'][1]:+.3f}]"
                     f"  A-B {r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")
        L.append("")
    L.append(f"(i)  baseline AU-PRO (L12 mean p1): full {base_pro['full']:.2f}, crop10 {base_pro['crop10']:.2f}")
    for r in gains:
        L.append(f"     {r['layer']:<4} {r['pool']} {r['scoring']:<4} gain full {r['gain_full']:+6.2f}  crop10 {r['gain_crop10']:+6.2f}"
                 f"  both >= 3: {r['both_ge_3']}")
    L.append(f"     -> (i) {'supported' if crit_i else 'not supported'}")
    for prot in ("full", "crop10"):
        L.append(f"(ii) {prot}: A-B < 0 with CI excl. 0 per layer (A mean p1, B p1): {ii[prot]} -> "
                 f"{sum(ii[prot].values())}/5")
    L.append(f"     -> (ii) {'supported' if crit_ii else 'not supported'}")
    for prot in ("full", "crop10"):
        d = iii[prot]
        L.append(f"(iii) {prot}: mean |AU-PRO(k) - AU-PRO(mean)| over 10 layer x pooling pairs: k25 "
                 f"{d['k25']['mean_abs_diff']:.2f} (signed {d['k25']['mean_signed_diff']:+.2f}), k75 "
                 f"{d['k75']['mean_abs_diff']:.2f} (signed {d['k75']['mean_signed_diff']:+.2f}) -> holds {d['holds']}")
    L.append(f"     -> (iii) {'supported' if crit_iii else 'not supported'}")
    L.append(f"\nanalysis {out['seconds']}s")
    txt = "\n".join(L)
    open(os.path.join(OUT, "analysis_results.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["repro", "full", "analyze"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    {"repro": stage_repro, "full": stage_full, "analyze": stage_analyze}[a.stage]()


if __name__ == "__main__":
    main()
