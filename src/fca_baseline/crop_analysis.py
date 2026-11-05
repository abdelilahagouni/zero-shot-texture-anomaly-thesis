"""
Crop10 follow-up for the FCA / aota baseline (criteria fixed before running). Writes results/fca_baseline_crop/ only.

1. Size analysis on the crop10 evaluation: per-image pixel AUROC of the 382 defective images computed on the 518x518
   grid with a 52 px border removed from map and mask (already in results/fca_baseline/metrics/per_image_defective.csv,
   *_auroc_crop10). Images whose defect lies entirely in the border (no defect pixel in the crop) are dropped and
   counted. x = log(defect area in the crop, px). Same method as src/fca_baseline/size_analysis.py: OLS with
   texture x defect-type fixed effects, 10,000 paired bootstrap resamples over images, np.random.default_rng(99).
   B = mean of seeds 42/123/2026 per image.
   Criteria: (i) supported only if slope(FCA)-slope(B) has a CI that includes 0 or is positive (upper bound >= 0)
   AND slope(A)-slope(B) < 0 with CI excluding 0; (ii) supported only if slope(FCA)-slope(B) < 0 with CI excluding 0;
   otherwise "mixed".
2. Border diagnostic on the GOOD test images (133), maps on the 518x518 grid (FCA/aota resized as in evaluate.py;
   A and B recomputed with src/mechanism_common.py, the pipeline verified in evaluate.py): mean score of all pixels in
   the 52 px border and of all interior pixels, pooled over the good images (per texture and over all 5 textures);
   reported as border mean / interior mean (interior = 1). B: ratio per seed, then the mean of the 3 seeds.
3. Aota protocol check: the raw 320x320 aota maps; masks resized to 320x320 (PIL NEAREST, > 128); 256x256 centre
   crop (32 px per side) of maps and masks; pixel AUROC per texture over all its test images (src/evaluation.py
   compute_pixel_auroc), next to Aota's published pixel AUROC (carpet 99.1, grid 99.1, leather 99.5, tile 93.2,
   wood 96.5); within 1.0 point or not. No aota setting is changed.
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import DATA, RES, TEXTURES, GRID, BORDER, test_items, map_path, to_grid  # noqa: E402
from size_analysis import fe_coefs, ci  # noqa: E402

OUT = os.path.join(os.path.dirname(RES), "fca_baseline_crop")
N_BOOT = 10000
METHODS = ["FCA", "aota", "A", "B"]
DIFFS = [("FCA", "B"), ("aota", "B"), ("FCA", "A"), ("A", "B")]
SEEDS = [42, 123, 2026]
AOTA_PUBLISHED = {"carpet": 99.1, "grid": 99.1, "leather": 99.5, "tile": 93.2, "wood": 96.5}
f2 = lambda e: f"{e['coef']:+.3f} [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"


def size_crop(L, res):
    t0 = time.time()
    df = pd.read_csv(os.path.join(RES, "metrics", "per_image_defective.csv"))
    n_all = len(df)
    drop = df["area_px_crop10"] == 0
    dropped = df.loc[drop, ["category", "defect_type", "file", "area_px"]].to_dict("records")
    df = df[~drop].reset_index(drop=True)
    Y = df[[f"{m}_auroc_crop10" for m in METHODS]].to_numpy(float)
    if np.isnan(Y).any():
        sys.exit("ERROR: NaN crop10 AUROC after dropping border-only images")
    x = np.log(df["area_px_crop10"].to_numpy(float))
    grp = (df["category"] + "/" + df["defect_type"]).to_numpy()
    est = dict(zip(METHODS, fe_coefs(x, Y, grp)))
    rng = np.random.default_rng(99)
    B = np.empty((N_BOOT, len(METHODS)))
    for b in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        B[b] = fe_coefs(x[i], Y[i], grp[i])
    r = {"n_all": int(n_all), "n_dropped_border_only": int(drop.sum()), "dropped": dropped, "n": int(len(df)),
         "fixed_effects": int(len(set(grp))), "slopes": {}, "differences": {}}
    for k, m in enumerate(METHODS):
        r["slopes"][m] = {"coef": float(est[m]), "ci95": ci(B[:, k])}
    for a, b in DIFFS:
        r["differences"][f"{a}-{b}"] = {"coef": float(est[a] - est[b]),
                                        "ci95": ci(B[:, METHODS.index(a)] - B[:, METHODS.index(b)])}
    dFB, dAB = r["differences"]["FCA-B"], r["differences"]["A-B"]
    p1, p2 = dFB["ci95"][1] >= 0, dAB["coef"] < 0 and dAB["ci95"][1] < 0
    ci_, cii = bool(p1 and p2), bool(dFB["coef"] < 0 and dFB["ci95"][1] < 0)
    r["criteria"] = {"i_FCA_minus_B_ci_includes_0_or_positive": bool(p1), "i_A_minus_B_negative_ci_excl_0": bool(p2),
                     "i_supported": ci_, "ii_supported": cii,
                     "verdict": "(i) supported" if ci_ else "(ii) supported" if cii else "mixed"}
    r["seconds"] = round(time.time() - t0, 1)
    res["size_crop10"] = r
    L += [f"1. SIZE ANALYSIS, crop10 ({BORDER} px border of the {GRID} grid removed from maps and masks): "
          f"{n_all} defective images, dropped {r['n_dropped_border_only']} with the defect entirely in the border "
          f"({', '.join(d['category'] + '/' + d['defect_type'] + '/' + d['file'] for d in dropped)}); n={r['n']}; "
          f"x = log(defect area in the crop); {r['fixed_effects']} fixed effects; bootstrap {N_BOOT} over images",
          "  Slopes:"] + [f"    {m:<5} {f2(r['slopes'][m])}" for m in METHODS] + ["  Differences:"] + \
         [f"    {k:<7} {f2(v)}" for k, v in r["differences"].items()] + \
         [f"  (i)  FCA-B CI includes 0 or positive: {'met' if p1 else 'NOT met'}; A-B < 0, CI excl. 0: "
          f"{'met' if p2 else 'NOT met'} -> {'supported' if ci_ else 'not supported'}",
          f"  (ii) FCA-B < 0, CI excl. 0: {'met' if cii else 'NOT met'} -> {'supported' if cii else 'not supported'}",
          f"  VERDICT: {r['criteria']['verdict']}   ({r['seconds']}s)", ""]


def border_diag(L, res):
    import torch
    from PIL import Image
    from mechanism_common import Backbone
    t0 = time.time()
    bb = Backbone()
    torch.cuda.reset_peak_memory_stats()
    inner = np.zeros((GRID, GRID), bool)
    inner[BORDER:GRID - BORDER, BORDER:GRID - BORDER] = True
    keys = ["FCA", "aota", "A"] + [f"B{s}" for s in SEEDS]
    sums = {c: {k: [0.0, 0.0] for k in keys} for c in TEXTURES}           # [border sum, interior sum]
    n_good = {}
    for cat in TEXTURES:
        good = [it for it in test_items(cat) if it[0] == "good"]
        n_good[cat] = len(good)
        train = [bb.tokens(Image.open(p)) for p in sorted(glob.glob(os.path.join(DATA, cat, "train", "good", "*.png")))]
        banks = {sd: bb.bank(train, sd) for sd in SEEDS}
        for d, fn, p, _ in good:
            maps = {"FCA": to_grid(np.load(map_path("fca", cat, d, fn))),
                    "aota": to_grid(np.load(map_path("aota", cat, d, fn)))}
            tok = bb.tokens(Image.open(p))
            maps["A"] = bb.post(1.0 - (tok @ tok.T).mean(dim=1))
            for sd in SEEDS:
                maps[f"B{sd}"] = bb.post(1.0 - (tok @ banks[sd].T).max(dim=1)[0])
            for k, m in maps.items():
                sums[cat][k][0] += float(m[~inner].sum())
                sums[cat][k][1] += float(m[inner].sum())
    nb, ni = int((~inner).sum()), int(inner.sum())
    out = {"n_good": n_good, "pixels_per_image": {"border": nb, "interior": ni}, "per_texture": {}, "pooled": {}}

    def ratios(s, n_img):
        r = {}
        for k in keys:
            mb, mi = s[k][0] / (nb * n_img), s[k][1] / (ni * n_img)
            r[k] = {"border_mean": mb, "interior_mean": mi, "border_over_interior": mb / mi}
        r["B"] = {"border_over_interior": float(np.mean([r[f"B{sd}"]["border_over_interior"] for sd in SEEDS])),
                  "per_seed": [r[f"B{sd}"]["border_over_interior"] for sd in SEEDS]}
        return r

    for cat in TEXTURES:
        out["per_texture"][cat] = ratios(sums[cat], n_good[cat])
    tot = {k: [sum(sums[c][k][j] for c in TEXTURES) for j in (0, 1)] for k in keys}
    out["pooled"] = ratios(tot, sum(n_good.values()))
    out["seconds"] = round(time.time() - t0, 1)
    out["peak_gpu_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20)
    res["border_diagnostic_good"] = out
    L += [f"2. BORDER DIAGNOSTIC, good test images ({sum(n_good.values())}: {n_good}); mean score in the {BORDER} px "
          f"border / mean interior score (interior = 1.000); B = mean of seeds {SEEDS}"]
    L.append(f"  {'':<8} " + " ".join(f"{m:>7}" for m in METHODS))
    for cat in TEXTURES + ["pooled"]:
        r = out["pooled"] if cat == "pooled" else out["per_texture"][cat]
        L.append(f"  {cat:<8} " + " ".join(f"{r[m]['border_over_interior']:7.3f}" for m in METHODS))
    L.append(f"  ({out['seconds']}s, peak GPU {out['peak_gpu_MiB']} MiB allocated)")
    L.append("")


def aota_protocol(L, res):
    from PIL import Image
    from evaluation import compute_pixel_auroc
    t0 = time.time()
    c0, c1 = 32, 288                                                       # 256x256 centre of 320x320
    out = {}
    L.append("3. AOTA PROTOCOL CHECK: raw 320x320 aota maps, masks NEAREST to 320 (>128), 256x256 centre crop of maps "
             "and masks, pixel AUROC over all test images of the texture")
    for cat in TEXTURES:
        maps, gts = [], []
        for d, fn, _, mp in test_items(cat):
            m = np.load(map_path("aota", cat, d, fn)).astype(np.float32)
            assert m.shape == (320, 320)
            g = np.zeros((320, 320), np.uint8) if mp is None else \
                (np.array(Image.open(mp).convert("L").resize((320, 320), Image.NEAREST)) > 128).astype(np.uint8)
            maps.append(m[c0:c1, c0:c1])
            gts.append(g[c0:c1, c0:c1])
        auc = compute_pixel_auroc(maps, gts) * 100
        pub = AOTA_PUBLISHED[cat]
        out[cat] = {"pixel_auroc": auc, "published": pub, "diff": auc - pub, "within_1": bool(abs(auc - pub) <= 1.0)}
        L.append(f"  {cat:<8} ours {auc:6.2f}   published {pub:5.1f}   diff {auc - pub:+6.2f}   within 1.0: "
                 f"{out[cat]['within_1']}")
    mean = float(np.mean([out[c]["pixel_auroc"] for c in TEXTURES]))
    pmean = float(np.mean(list(AOTA_PUBLISHED.values())))
    out["mean"] = {"pixel_auroc": mean, "published": pmean, "diff": mean - pmean}
    L.append(f"  {'mean':<8} ours {mean:6.2f}   published {pmean:5.2f}   diff {mean - pmean:+6.2f}")
    L.append(f"  textures within 1.0: {sum(out[c]['within_1'] for c in TEXTURES)}/5   ({time.time() - t0:.1f}s)")
    res["aota_protocol_check"] = out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    T0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    res, L = {}, []
    size_crop(L, res)
    border_diag(L, res)
    aota_protocol(L, res)
    res["seconds_total"] = round(time.time() - T0, 1)
    L.append(f"\nTotal {res['seconds_total']}s")
    txt = "\n".join(L)
    json.dump(res, open(os.path.join(OUT, "crop_analysis_results.json"), "w"), indent=1)
    open(os.path.join(OUT, "crop_analysis_results.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
