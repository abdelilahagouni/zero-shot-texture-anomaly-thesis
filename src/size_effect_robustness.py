"""
Robustness of the defect-size effect: does Route B's advantage over Route A grow with defect size?

Phase 1 (model run, cached in per_image_metrics.csv):
  Route A and Route B (seeds 42, 123, 2026, 7, 11, 13, 17, 19, 23, 29) on all test images of the
  5 textures, replicating src/eval_router_full.py exactly (fp16-cached L2-normalised DINOv2 ViT-B/14
  tokens at 518x518, Route A = 1 - mean cos, Route B = 1 - max cos to a 4-image bank chosen by
  np.random.default_rng(seed).choice(n_train, 4, replace=False), bilinear 37->518 + GaussianBlur 15x15,
  PyTorch default TF32 settings). Per defective image: pixel AUROC and AP for A and every B seed.
Phase 2 (analysis):
  d = mean over seeds of Route B - Route A (pixel AUROC and per-image AP).
  Spearman(log area, d) per texture and within defect types with >= 10 images, bootstrap 95% CI (images).
  Pooled OLS: d ~ log(area) + texture/defect-type fixed effects, SEs clustered by image (CR1).
  Seed variability of the per-texture mean d (small / large).
Fixed criterion: supported only if Spearman(log area, d_AUROC) > 0 with CI excluding 0 on >= 4/5 textures
AND pooled log(area) coefficient for pixel AUROC > 0 with CI excluding 0.
Writes to results/size_effect_robustness/ only.
"""
import glob, json, os, sys, time, datetime
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "size_effect_robustness")
DATA = os.path.join(REPO, "data", "mvtec")
TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
SEEDS = [42, 123, 2026, 7, 11, 13, 17, 19, 23, 29]
N_BOOT = 10000


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def phase1(path_csv):
    import torch, torch.nn.functional as F, cv2
    from PIL import Image
    from torchvision import transforms
    from sklearn.metrics import roc_auc_score, average_precision_score
    import pandas as pd
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(dev).eval()
    tf = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def tok16(p):
        img = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            f = model.forward_features(img)["x_norm_patchtokens"][0]
        return F.normalize(f, p=2, dim=-1).cpu().to(torch.float16)

    def post(s):
        hm = F.interpolate(s.reshape(1, 1, 37, 37), size=(518, 518), mode="bilinear",
                           align_corners=False).squeeze().cpu().numpy()
        return cv2.GaussianBlur(hm, (15, 15), 0)

    rows = []
    for cat in TEXTURES:
        t0 = time.time()
        train = [tok16(p) for p in sorted(glob.glob(f"{DATA}/{cat}/train/good/*.png"))]
        banks = {}
        for sd in SEEDS:
            idx = np.random.default_rng(sd).choice(len(train), size=4, replace=False)
            banks[sd] = torch.cat([train[k].to(dev).float() for k in idx], dim=0)
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
                gt = (np.array(Image.open(mp).convert("L").resize((518, 518), Image.NEAREST)) > 128).ravel()
                tk = tok16(p).to(dev).float()
                r = {"category": cat, "defect_type": d, "file": fn, "key": f"{cat}/test/{d}/{fn}",
                     "area_px": int(gt.sum())}
                ma = post(1.0 - torch.mm(tk, tk.T).mean(dim=1)).ravel()
                r["A_auroc"] = roc_auc_score(gt, ma) * 100
                r["A_ap"] = average_precision_score(gt, ma) * 100
                for sd in SEEDS:
                    mb = post(1.0 - torch.mm(tk, banks[sd].T).max(dim=1)[0]).ravel()
                    r[f"B{sd}_auroc"] = roc_auc_score(gt, mb) * 100
                    r[f"B{sd}_ap"] = average_precision_score(gt, mb) * 100
                rows.append(r)
                n += 1
        log(f"{cat}: {len(train)} train, {n} defective test images, {time.time() - t0:.1f}s")
    pd.DataFrame(rows).to_csv(path_csv, index=False)


def spearman_ci(x, y, rng):
    from scipy.stats import spearmanr
    x, y = np.asarray(x), np.asarray(y)
    r = spearmanr(x, y).correlation
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            bs.append(spearmanr(x[i], y[i]).correlation)
    return float(r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def ols_cluster(y, logarea, groups_fe, clusters):
    """OLS with fixed effects (dummy per group, no intercept) and CR1 cluster-robust SE."""
    from scipy.stats import t as tdist
    levels = sorted(set(groups_fe))
    X = np.column_stack([np.asarray(logarea, float)] +
                        [np.array([g == l for g in groups_fe], float) for l in levels])
    y = np.asarray(y, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    N, K = X.shape
    cl = np.asarray(clusters)
    G = len(set(cl))
    meat = np.zeros((K, K))
    for c in set(cl):
        m = cl == c
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
    se = float(np.sqrt(V[0, 0]))
    q = tdist.ppf(0.975, G - 1)
    return {"coef": float(beta[0]), "se": se, "ci95": [float(beta[0] - q * se), float(beta[0] + q * se)],
            "n": int(N), "clusters": int(G), "n_fixed_effects": len(levels)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    T0 = time.time()
    csv_path = os.path.join(OUT, "per_image_metrics.csv")
    n_def = sum(len([p for p in glob.glob(f"{DATA}/{c}/test/*/*.png") if "/good/" not in p.replace("\\", "/")])
                for c in TEXTURES)
    n_all = sum(len(glob.glob(f"{DATA}/{c}/test/*/*.png")) for c in TEXTURES)
    log(f"ESTIMATE: {n_all} test images ({n_def} defective) + 150 train images; ~0.12 s per forward pass and "
        f"~0.1 s per (image, route) for map + per-image AUROC/AP -> about "
        f"{(n_def + 150) * 0.12 / 60 + n_def * 11 * 0.1 / 60:.0f} min (well under 90 min). Proceeding.")
    if not os.path.exists(csv_path):
        t = time.time()
        phase1(csv_path)
        log(f"Phase 1 (model run) done in {time.time() - t:.1f}s")
    else:
        log("Phase 1 output exists; reusing it")
    t = time.time()
    df = pd.read_csv(csv_path)
    for m in ("auroc", "ap"):
        df[f"Bmean_{m}"] = df[[f"B{s}_{m}" for s in SEEDS]].mean(axis=1)
        df[f"d_{m}"] = df[f"Bmean_{m}"] - df[f"A_{m}"]
    df["log_area"] = np.log(df["area_px"])

    # reproduction check vs the old per-image CSV (Route A and Route B seed 42, pixel AUROC)
    old = pd.read_csv(os.path.join(REPO, "results", "per_image_latest_1791743390.csv"))
    old["key"] = old["path"].str.replace("\\", "/", regex=False).str.split("data/mvtec/").str[-1]
    j = df.merge(old[["key", "px_auroc_A", "px_auroc_B"]], on="key", how="left")
    repro = {"matched_images": int(j["px_auroc_A"].notna().sum()), "n": int(len(j)),
             "max_abs_diff_A": float((j["A_auroc"] - j["px_auroc_A"]).abs().max()),
             "max_abs_diff_B42": float((j["B42_auroc"] - j["px_auroc_B"]).abs().max()),
             "mean_abs_diff_A": float((j["A_auroc"] - j["px_auroc_A"]).abs().mean()),
             "mean_abs_diff_B42": float((j["B42_auroc"] - j["px_auroc_B"]).abs().mean())}

    res = {"reproduction_check": repro, "per_texture": {}, "within_type": {}, "pooled": {}, "seed_variability": {}}
    for ti, cat in enumerate(TEXTURES):
        g = df[df.category == cat]
        med = g["area_px"].median()
        e = {"n": int(len(g)), "median_area_px": float(med)}
        for mi, m in enumerate(("auroc", "ap")):
            r, lo, hi = spearman_ci(g["log_area"], g[f"d_{m}"], np.random.default_rng([ti, mi]))
            e[f"spearman_{m}"] = {"rho": r, "ci95": [lo, hi]}
        res["per_texture"][cat] = e
        for di, (dt, h) in enumerate(sorted(g.groupby("defect_type"))):
            if len(h) < 10:
                res["within_type"][f"{cat}/{dt}"] = {"n": int(len(h)), "skipped": "fewer than 10 images"}
                continue
            w = {"n": int(len(h))}
            for mi, m in enumerate(("auroc", "ap")):
                r, lo, hi = spearman_ci(h["log_area"], h[f"d_{m}"], np.random.default_rng([ti, 100 + di, mi]))
                w[f"spearman_{m}"] = {"rho": r, "ci95": [lo, hi]}
            res["within_type"][f"{cat}/{dt}"] = w
        sv = {}
        for sub, mask in (("small", g["area_px"] <= med), ("large", g["area_px"] > med)):
            per_seed = [float((g.loc[mask, f"B{s}_auroc"] - g.loc[mask, "A_auroc"]).mean()) for s in SEEDS]
            sv[sub] = {"seed42": per_seed[0], "mean_over_seeds": float(np.mean(per_seed)),
                       "sd_over_seeds": float(np.std(per_seed, ddof=1)),
                       "min": float(min(per_seed)), "max": float(max(per_seed))}
        res["seed_variability"][cat] = sv
    fe = (df["category"] + "/" + df["defect_type"]).tolist()
    for m in ("auroc", "ap"):
        res["pooled"][m] = ols_cluster(df[f"d_{m}"], df["log_area"], fe, df["key"])

    s_ok = [c for c in TEXTURES if res["per_texture"][c]["spearman_auroc"]["rho"] > 0
            and res["per_texture"][c]["spearman_auroc"]["ci95"][0] > 0]
    p_ok = res["pooled"]["auroc"]["coef"] > 0 and res["pooled"]["auroc"]["ci95"][0] > 0
    verdict = "supported" if len(s_ok) >= 4 and p_ok else "not supported"
    wt = {k: v for k, v in res["within_type"].items() if "skipped" not in v}
    wt_pos = [k for k, v in wt.items() if v["spearman_auroc"]["rho"] > 0 and v["spearman_auroc"]["ci95"][0] > 0]
    res["decision"] = {"textures_spearman_pos_ci_excl_0": s_ok, "pooled_auroc_pos_ci_excl_0": p_ok,
                       "verdict": verdict, "within_type_tested": list(wt),
                       "within_type_pos_ci_excl_0": wt_pos}
    res["seconds_analysis"] = round(time.time() - t, 1)
    res["seconds_total"] = round(time.time() - T0, 1)
    with open(os.path.join(OUT, "size_effect_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    L = ["SIZE-EFFECT ROBUSTNESS (d = mean of 10 Route B seeds - Route A; per defective image)", ""]
    L.append(f"Reproduction vs per_image_latest_1791743390.csv ({repro['matched_images']}/{repro['n']} images matched): "
             f"Route A max |diff| {repro['max_abs_diff_A']:.6f}, Route B seed 42 max |diff| {repro['max_abs_diff_B42']:.6f} "
             f"(AUROC points)")
    L.append("")
    L.append("Per texture, Spearman(log area, d) [95% bootstrap CI]:")
    for c in TEXTURES:
        e = res["per_texture"][c]
        L.append(f"  {c:<8} n={e['n']:>3}  AUROC rho={e['spearman_auroc']['rho']:+.3f} "
                 f"[{e['spearman_auroc']['ci95'][0]:+.3f}, {e['spearman_auroc']['ci95'][1]:+.3f}]   "
                 f"AP rho={e['spearman_ap']['rho']:+.3f} [{e['spearman_ap']['ci95'][0]:+.3f}, {e['spearman_ap']['ci95'][1]:+.3f}]")
    L.append("")
    L.append("Within defect type (>= 10 images), Spearman(log area, d):")
    for k, v in res["within_type"].items():
        if "skipped" in v:
            L.append(f"  {k:<26} n={v['n']:>2}  skipped (<10)")
        else:
            L.append(f"  {k:<26} n={v['n']:>2}  AUROC rho={v['spearman_auroc']['rho']:+.3f} "
                     f"[{v['spearman_auroc']['ci95'][0]:+.3f}, {v['spearman_auroc']['ci95'][1]:+.3f}]   "
                     f"AP rho={v['spearman_ap']['rho']:+.3f} [{v['spearman_ap']['ci95'][0]:+.3f}, {v['spearman_ap']['ci95'][1]:+.3f}]")
    L.append("")
    L.append("Pooled OLS d ~ log(area) + texture x defect-type fixed effects, SE clustered by image (CR1):")
    for m in ("auroc", "ap"):
        p = res["pooled"][m]
        L.append(f"  {m.upper():<5} coef on log(area) = {p['coef']:+.3f} (SE {p['se']:.3f}), 95% CI "
                 f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  n={p['n']}, {p['n_fixed_effects']} fixed effects")
    L.append("")
    L.append("Seed variability of per-texture mean d (pixel AUROC), small / large:")
    for c in TEXTURES:
        for sub in ("small", "large"):
            v = res["seed_variability"][c][sub]
            L.append(f"  {c:<8} {sub:<5} seed42 {v['seed42']:+6.2f}  10-seed mean {v['mean_over_seeds']:+6.2f}  "
                     f"SD {v['sd_over_seeds']:.2f}  range [{v['min']:+.2f}, {v['max']:+.2f}]")
    L.append("")
    L.append(f"Criterion: Spearman>0 with CI excl. 0 on >=4/5 textures: {len(s_ok)}/5 {s_ok}; "
             f"pooled AUROC coef >0 with CI excl. 0: {p_ok}")
    L.append(f"VERDICT: {verdict}")
    L.append(f"Within defect types: {len(wt_pos)}/{len(wt)} tested types positive with CI excluding 0: {wt_pos}")
    L.append(f"Analysis time {res['seconds_analysis']}s, total {res['seconds_total']}s")
    txt = "\n".join(L)
    with open(os.path.join(OUT, "size_effect_results.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


if __name__ == "__main__":
    main()
