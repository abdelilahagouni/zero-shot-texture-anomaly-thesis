"""
Reproduction check of the official FCA run against the published numbers (fixed before the run).

Full resolution (dataset=mvtec fe=wide sc=fca image_size=original tile_size=[9,9]), paper Table 6, PRO / pixel AUROC:
  carpet 95.44/98.30, grid 98.07/99.46, leather 98.90/99.45, tile 96.33/98.22, wood 97.18/98.22
  PASS only if EVERY texture is within 1.0 point on BOTH metrics.
Low-resolution fallback (image_size=[320,320] tile_size=[3,3]; a deviation, used only if full resolution does not fit):
  paper Table 3 average PRO 95.46, pixel AUROC 97.74; PASS only if both averages are within 1.0 point.
Input: the repo's metrics_100.json (au_pro and au_segroc are fractions; x100 here).
Writes results/fca_baseline/reproduction/reproduction_check.json and .txt.
"""
import argparse, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import RES, TEXTURES  # noqa: E402

PUBLISHED = {"carpet": (95.44, 98.30), "grid": (98.07, 99.46), "leather": (98.90, 99.45),
             "tile": (96.33, 98.22), "wood": (97.18, 98.22)}
PUBLISHED_320_MEAN = (95.46, 97.74)
TOL = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="official outputs/<run> folder")
    ap.add_argument("--variant", choices=["full", "320"], default="full")
    a = ap.parse_args()
    out = os.path.join(RES, "reproduction")
    os.makedirs(out, exist_ok=True)
    src = os.path.join(a.run, "metrics_100.json")
    shutil.copy(src, os.path.join(out, f"official_metrics_100_{a.variant}.json"))
    m = json.load(open(src))
    rows, ok_all = [], True
    L = [f"REPRODUCTION CHECK ({a.variant}), tolerance {TOL} point, metrics from {src}", ""]
    if a.variant == "full":
        L.append(f"{'texture':<8} {'PRO ours':>9} {'PRO paper':>10} {'diff':>6}   {'AUROC ours':>10} {'paper':>6} {'diff':>6}  ok")
        for t in TEXTURES:
            pro, auc = m[t]["au_pro"] * 100, m[t]["au_segroc"] * 100
            pp, pa = PUBLISHED[t]
            ok = abs(pro - pp) <= TOL and abs(auc - pa) <= TOL
            ok_all &= ok
            rows.append({"texture": t, "pro": pro, "pro_paper": pp, "auroc": auc, "auroc_paper": pa, "ok": ok})
            L.append(f"{t:<8} {pro:9.2f} {pp:10.2f} {pro - pp:+6.2f}   {auc:10.2f} {pa:6.2f} {auc - pa:+6.2f}  {ok}")
        mp, ma = m["mean_au_pro"] * 100, m["mean_au_segroc"] * 100
        L.append(f"{'mean':<8} {mp:9.2f} {97.18:10.2f} {mp - 97.18:+6.2f}   {ma:10.2f} {98.73:6.2f} {ma - 98.73:+6.2f}")
    else:
        mp, ma = m["mean_au_pro"] * 100, m["mean_au_segroc"] * 100
        pp, pa = PUBLISHED_320_MEAN
        ok_all = abs(mp - pp) <= TOL and abs(ma - pa) <= TOL
        rows.append({"texture": "mean", "pro": mp, "pro_paper": pp, "auroc": ma, "auroc_paper": pa, "ok": ok_all})
        L.append(f"mean PRO {mp:.2f} (paper {pp}, diff {mp - pp:+.2f}); pixel AUROC {ma:.2f} (paper {pa}, diff {ma - pa:+.2f})")
        L.append("NOTE: low-resolution variant = deviation from the full-resolution reproduction")
    L.append("")
    L.append(f"REPRODUCTION: {'PASS' if ok_all else 'FAIL'}")
    txt = "\n".join(L)
    json.dump({"variant": a.variant, "run": a.run, "tolerance": TOL, "rows": rows, "pass": bool(ok_all),
               "official_metrics": m}, open(os.path.join(out, f"reproduction_check_{a.variant}.json"), "w"), indent=1)
    open(os.path.join(out, f"reproduction_check_{a.variant}.txt"), "w").write(txt)
    print(txt)
    sys.exit(0 if ok_all else 3)


if __name__ == "__main__":
    main()
