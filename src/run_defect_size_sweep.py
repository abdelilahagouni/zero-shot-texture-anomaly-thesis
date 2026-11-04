"""
Defect-size sweep: does Route A (zero-shot self-similarity) get worse as the
defect grows, while Route B (4-shot memory bank) does not?

For every texture, synthetic square defects of a controlled area are pasted
into the GOOD test images. Route A and Route B are then scored exactly as in
src/run_full_experiments_suite.py (DINOv2 ViT-B/14, 518x518, same scoring and
post-processing). AU-PRO uses the verified implementation in src/evaluation.py.

Synthetic defect (per image, fixed seed):
  * a square patch is cut from a DIFFERENT good test image of the same texture,
  * rotated by a random multiple of 90 degrees (unless --no-rotate),
  * colour-shifted by a random per-channel gain of about +/- --color-shift,
  * pasted at a random location with lightly feathered edges.
  The square itself is the ground-truth mask. Without the rotation and colour
  shift the pasted patch would be ordinary normal texture and not a defect.

The sweep is resumable: every (texture, size, method, seed) result is written
to <results-dir>/partial/ as soon as it is computed, and is skipped on re-run.

Examples
  python src/run_defect_size_sweep.py --quick --textures carpet --sizes 1 8 --seeds 42
  python src/run_defect_size_sweep.py                      # full sweep
  python src/run_defect_size_sweep.py --data-root /content/drive/MyDrive/x/data/mvtec \
         --results-dir /content/drive/MyDrive/x/results/defect_size_sweep
"""
import argparse
import csv
import datetime
import glob
import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ALL_TEXTURES = ["carpet", "grid", "leather", "tile", "wood"]
TEX_ID = {t: i for i, t in enumerate(ALL_TEXTURES)}  # stable ids for the synthetic RNG
DEFAULT_SEEDS = [42, 123, 2026, 7, 11, 13, 17, 19, 23, 29]
DEFAULT_SIZES = [0.25, 0.5, 1, 2, 4, 8, 16, 32]  # % of the 518x518 image area
IMG = 518
GRID = 37


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.path.join(REPO, "data", "mvtec"),
                    help="folder containing <texture>/train/good and <texture>/test/good")
    ap.add_argument("--results-dir", default=None,
                    help="default: results/defect_size_sweep (or results/defect_size_sweep_quick with --quick)")
    ap.add_argument("--textures", nargs="+", default=ALL_TEXTURES, choices=ALL_TEXTURES)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Route B reference seeds")
    ap.add_argument("--sizes", nargs="+", type=float, default=DEFAULT_SIZES, help="defect area in %% of the image")
    ap.add_argument("--quick", action="store_true", help="smoke test: only the first 4 good test images per texture")
    ap.add_argument("--max-images", type=int, default=None, help="limit good test images per texture")
    ap.add_argument("--k-shot", type=int, default=4)
    ap.add_argument("--synth-seed", type=int, default=0, help="seed for defect locations / donors / shifts")
    ap.add_argument("--color-shift", type=float, default=0.15, help="per-channel gain change of the pasted patch")
    ap.add_argument("--no-rotate", action="store_true", help="do not rotate the pasted patch")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    a = ap.parse_args()
    if a.quick and a.max_images is None:
        a.max_images = 4
    if a.results_dir is None:
        a.results_dir = os.path.join(REPO, "results", "defect_size_sweep_quick" if a.quick else "defect_size_sweep")
    for s in a.sizes:
        if not (0 < s < 100):
            ap.error(f"size {s} must be between 0 and 100 (percent)")
    return a


def size_key(s):
    """0.25 -> 25, 1 -> 100, 32 -> 3200 (hundredths of a percent)."""
    return int(round(s * 100))


def fmt_size(s):
    return f"{s:g}%"


# ---------------------------------------------------------------------------
# Environment log
# ---------------------------------------------------------------------------
def git_commit():
    try:
        out = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def log_environment(results_dir, args, torch):
    import sklearn, scipy, cv2, torchvision
    env = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": None, "gpu_memory_gb": None,
        "numpy": np.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "tf32_disabled": True,
        "git_commit": git_commit(),
        "argv": sys.argv,
        "data_root": os.path.abspath(args.data_root),
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        env["gpu"] = prop.name
        env["gpu_memory_gb"] = round(prop.total_memory / 1024 ** 3, 2)
    path = os.path.join(results_dir, "environment.json")
    runs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            runs = json.load(f)
    runs.append(env)  # one entry per run, so a sweep split across machines stays traceable
    with open(path, "w", encoding="utf-8") as f:
        json.dump(runs, f, indent=2)
    return env


def check_config(results_dir, args):
    """Settings that change the numbers must match across resumed runs."""
    cfg = {"k_shot": args.k_shot, "synth_seed": args.synth_seed, "color_shift": args.color_shift,
           "rotate": not args.no_rotate, "max_images": args.max_images, "image_size": IMG,
           "backbone": "dinov2_vitb14", "au_pro": "evaluation.compute_au_pro(max_fpr=0.3, num_thresholds=1000)"}
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old != cfg:
            sys.exit(f"ERROR: {path} was created with different settings:\n  old={old}\n  new={cfg}\n"
                     f"Use a new --results-dir for different settings.")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    return cfg


# ---------------------------------------------------------------------------
# Synthetic defects
# ---------------------------------------------------------------------------
def make_synthetic(base_imgs, tex, s, args, out_dir):
    """Returns lists of uint8 RGB images (518x518x3) and binary masks (518x518)."""
    import cv2
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    n = len(base_imgs)
    if n < 2:
        sys.exit(f"ERROR: {tex} needs at least 2 good test images to take donor patches from another image.")
    side = max(2, int(round(IMG * np.sqrt(s / 100.0))))
    imgs, masks = [], []
    for i in range(n):
        ip = os.path.join(out_dir, f"{i:03d}.png")
        mp = os.path.join(out_dir, f"{i:03d}_mask.png")
        if os.path.exists(ip) and os.path.exists(mp):
            imgs.append(np.array(Image.open(ip).convert("RGB")))
            masks.append((np.array(Image.open(mp)) > 127).astype(np.uint8))
            continue
        rng = np.random.default_rng([args.synth_seed, TEX_ID[tex], size_key(s), i])
        j = (i + 1 + int(rng.integers(n - 1))) % n          # donor: a different image
        donor = base_imgs[j]
        dy, dx = rng.integers(0, IMG - side + 1, size=2)
        patch = donor[dy:dy + side, dx:dx + side].astype(np.float32)
        if not args.no_rotate:
            patch = np.rot90(patch, k=int(rng.integers(1, 4))).copy()
        sign = 1.0 if rng.random() < 0.5 else -1.0
        gains = 1.0 + sign * args.color_shift * (0.5 + 0.5 * rng.random(3))
        patch = np.clip(patch * gains[None, None, :], 0, 255)
        ty, tx = rng.integers(0, IMG - side + 1, size=2)
        canvas = base_imgs[i].astype(np.float32).copy()
        layer = canvas.copy()
        layer[ty:ty + side, tx:tx + side] = patch
        mask = np.zeros((IMG, IMG), np.uint8)
        mask[ty:ty + side, tx:tx + side] = 1
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=max(1.0, 0.04 * side))
        alpha = np.maximum(alpha, 0.0)[..., None]
        out = np.clip(canvas * (1 - alpha) + layer * alpha + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(out).save(ip)
        Image.fromarray(mask * 255).save(mp)
        imgs.append(out)
        masks.append(mask)
    return imgs, masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    import torch
    import torch.nn.functional as F
    import cv2
    from PIL import Image
    from torchvision import transforms
    from sklearn.metrics import roc_auc_score, average_precision_score
    from evaluation import compute_au_pro

    # Same numerics on every GPU (laptop Ampere vs Colab T4/A100): no TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    res = args.results_dir
    part_dir = os.path.join(res, "partial")
    os.makedirs(part_dir, exist_ok=True)
    env = log_environment(res, args, torch)
    check_config(res, args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | GPU: {env['gpu']} ({env['gpu_memory_gb']} GB) | torch {env['torch']} | CUDA {env['cuda']}")
    print(f"Results: {os.path.abspath(res)}")

    sizes = sorted(set(args.sizes))
    units = [(t, s, "routeA", None) for t in args.textures for s in sizes] + \
            [(t, s, "routeB", sd) for t in args.textures for s in sizes for sd in args.seeds]
    units.sort(key=lambda u: (args.textures.index(u[0]), sizes.index(u[1]), u[2], -1 if u[3] is None else u[3]))

    def part_path(t, s, m, sd):
        return os.path.join(part_dir, f"{t}__area{size_key(s):05d}__{m}__seed{'none' if sd is None else sd}.json")

    todo = [u for u in units if not os.path.exists(part_path(*u))]
    print(f"Units: {len(units)} total, {len(units) - len(todo)} already done, {len(todo)} to run")

    model = None
    transform = transforms.Compose([
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def tokens(pil):
        img = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.forward_features(img)["x_norm_patchtokens"][0]
        return F.normalize(feat, p=2, dim=-1)

    def post(s1d):
        hm = F.interpolate(s1d.reshape(1, 1, GRID, GRID), size=(IMG, IMG),
                           mode="bilinear", align_corners=False).squeeze().cpu().numpy()
        return cv2.GaussianBlur(hm, (15, 15), 0)

    def score_A(tok):
        return post(1.0 - torch.mm(tok, tok.T).mean(dim=1))

    def score_B(tok, bank):
        return post(1.0 - torch.mm(tok, bank.T).max(dim=1)[0])

    def metrics(maps, gts):
        y = np.concatenate([g.ravel() for g in gts]).astype(np.uint8)
        sc = np.concatenate([m.ravel() for m in maps])
        return {"pixel_ap": float(average_precision_score(y, sc) * 100),
                "au_pro": float(compute_au_pro(maps, gts) * 100),
                "pixel_auroc": float(roc_auc_score(y, sc) * 100),
                "chance_ap": float(y.mean() * 100)}

    t_start = time.time()
    unit_times = []
    done_now = 0
    for tex in args.textures:
        tex_todo = [u for u in todo if u[0] == tex]
        if not tex_todo:
            continue
        if model is None:
            print("Loading DINOv2 ViT-B/14 ...")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
        tdir = os.path.join(args.data_root, tex)
        train_paths = sorted(glob.glob(os.path.join(tdir, "train", "good", "*.png")))
        good_paths = sorted(glob.glob(os.path.join(tdir, "test", "good", "*.png")))
        if args.max_images:
            good_paths = good_paths[:args.max_images]
        if not train_paths or not good_paths:
            sys.exit(f"ERROR: no images under {tdir}/train/good or {tdir}/test/good")
        t0 = time.time()
        train_toks = [tokens(Image.open(p).convert("RGB")).cpu() for p in train_paths]
        base_imgs = [np.array(transforms.Resize((IMG, IMG))(Image.open(p).convert("RGB"))) for p in good_paths]
        print(f"\n=== {tex.upper()}: {len(train_paths)} train images (reference pool), "
              f"{len(good_paths)} good test images | features {time.time() - t0:.1f}s")

        for s in sizes:
            s_todo = [u for u in tex_todo if u[1] == s]
            if not s_todo:
                continue
            t1 = time.time()
            sdir = os.path.join(res, "synthetic", tex, f"area_{size_key(s):05d}")
            imgs, gts = make_synthetic(base_imgs, tex, s, args, sdir)
            toks = [tokens(Image.fromarray(im)) for im in imgs]
            prep = time.time() - t1
            for (_, _, method, sd) in s_todo:
                t2 = time.time()
                if method == "routeA":
                    maps = [score_A(tk) for tk in toks]
                    ref_idx = None
                else:
                    rng = np.random.default_rng(sd)
                    ref_idx = rng.choice(len(train_toks), size=args.k_shot, replace=False)
                    bank = torch.cat([train_toks[k] for k in ref_idx], dim=0).to(device)
                    maps = [score_B(tk, bank) for tk in toks]
                m = metrics(maps, gts)
                secs = time.time() - t2 + prep
                prep = 0.0  # count the shared preparation once
                rec = {"texture": tex, "area_pct": s, "method": method, "seed": sd,
                       "metrics": m, "n_images": len(imgs), "n_train": len(train_paths),
                       "reference_indices": None if ref_idx is None else [int(k) for k in ref_idx],
                       "seconds": round(secs, 2), "host": env["host"], "gpu": env["gpu"],
                       "finished": datetime.datetime.now().isoformat(timespec="seconds")}
                tmp = part_path(tex, s, method, sd) + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=1)
                os.replace(tmp, part_path(tex, s, method, sd))  # atomic: no half-written results
                done_now += 1
                unit_times.append(secs)
                left = len(todo) - done_now
                eta = datetime.timedelta(seconds=int(np.mean(unit_times) * left))
                label = "Route A" if method == "routeA" else f"Route B seed {sd}"
                print(f"[{len(units) - len(todo) + done_now}/{len(units)}] {tex} {fmt_size(s):>6} {label:<17} "
                      f"AP={m['pixel_ap']:6.2f}  AU-PRO={m['au_pro']:6.2f}  AUROC={m['pixel_auroc']:6.2f}  "
                      f"(chance AP {m['chance_ap']:.2f}) {secs:5.1f}s | ETA {eta}")
            del toks
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\nComputed {done_now} units in {datetime.timedelta(seconds=int(time.time() - t_start))}")
    combine(res, args)


# ---------------------------------------------------------------------------
# Combine, plot, summarise (uses every partial result present in the folder)
# ---------------------------------------------------------------------------
def combine(res, args):
    recs = []
    for p in sorted(glob.glob(os.path.join(res, "partial", "*.json"))):
        with open(p, encoding="utf-8") as f:
            recs.append(json.load(f))
    if not recs:
        print("No results yet.")
        return
    keys = ["pixel_ap", "au_pro", "pixel_auroc", "chance_ap"]
    with open(os.path.join(res, "defect_size_sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texture", "area_pct", "method", "seed"] + keys + ["n_images", "n_train", "host", "gpu"])
        for r in sorted(recs, key=lambda r: (r["texture"], r["area_pct"], r["method"], r["seed"] or -1)):
            w.writerow([r["texture"], r["area_pct"], r["method"], r["seed"]] +
                       [round(r["metrics"][k], 4) for k in keys] + [r["n_images"], r["n_train"], r["host"], r["gpu"]])

    textures = [t for t in ALL_TEXTURES if any(r["texture"] == t for r in recs)]
    agg = {}
    for t in textures:
        for s in sorted({r["area_pct"] for r in recs if r["texture"] == t}):
            A = [r for r in recs if r["texture"] == t and r["area_pct"] == s and r["method"] == "routeA"]
            B = [r for r in recs if r["texture"] == t and r["area_pct"] == s and r["method"] == "routeB"]
            e = {"texture": t, "area_pct": s, "n_seeds_B": len(B)}
            for k in keys:
                e[f"A_{k}"] = A[0]["metrics"][k] if A else None
                vals = np.array([r["metrics"][k] for r in B])
                e[f"B_{k}_mean"] = float(vals.mean()) if len(vals) else None
                e[f"B_{k}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else None  # sample SD (n-1)
            agg.setdefault(t, []).append(e)
    hosts = sorted({(r["host"], r["gpu"]) for r in recs}, key=str)
    with open(os.path.join(res, "defect_size_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"note": "B_*_sd is the sample SD (n-1) over Route B seeds; chance_ap = fraction of defect pixels.",
                   "machines": [{"host": h, "gpu": g} for h, g in hosts],
                   "aggregate": agg, "runs": recs}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(textures), figsize=(3.6 * len(textures), 3.4), squeeze=False)
    for ax, t in zip(axes[0], textures):
        rows = agg[t]
        x = [e["area_pct"] for e in rows]
        a = [e["A_pixel_ap"] for e in rows]
        bm = np.array([np.nan if e["B_pixel_ap_mean"] is None else e["B_pixel_ap_mean"] for e in rows])
        bs = np.array([0.0 if e["B_pixel_ap_sd"] is None else e["B_pixel_ap_sd"] for e in rows])
        ax.plot(x, a, "o-", color="#0891b2", label="Route A (zero-shot)")
        ax.plot(x, bm, "s-", color="#2563eb", label="Route B (4-shot), mean")
        ax.fill_between(x, bm - bs, bm + bs, color="#2563eb", alpha=0.2, label="Route B ± sample SD")
        ax.plot(x, [e["A_chance_ap"] or e["B_chance_ap_mean"] for e in rows], ":", color="grey", label="chance AP")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:g}" for v in x], fontsize=7)
        ax.set_title(t.capitalize())
        ax.set_xlabel("Defect area (% of image)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Pixel AP (%)")
    axes[0][0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(res, "defect_size_sweep.png"), dpi=150)
    plt.close(fig)

    lines = ["DEFECT-SIZE SWEEP SUMMARY",
             f"Results: {os.path.abspath(res)}",
             f"Machines: {', '.join(f'{h} ({g})' for h, g in hosts)}",
             "Pixel AP rises with defect area partly because its chance level (= defect fraction) rises,",
             "so the trend test below uses AU-PRO, which does not depend on the defect fraction.", ""]
    from scipy.stats import spearmanr
    for t in textures:
        rows = [e for e in agg[t] if e["A_pixel_ap"] is not None and e["B_pixel_ap_mean"] is not None]
        if not rows:
            lines.append(f"{t}: incomplete (needs both Route A and Route B results)")
            continue
        n_seeds = min(e["n_seeds_B"] for e in rows)
        lines.append(f"{t.upper()} ({len(rows)} sizes, Route B over {n_seeds} seed(s)):")
        for e in rows:
            sd = e["B_pixel_ap_sd"]
            lines.append(f"  {fmt_size(e['area_pct']):>6}  AP  A={e['A_pixel_ap']:6.2f}  B={e['B_pixel_ap_mean']:6.2f}"
                         f"{'' if sd is None else f' ± {sd:.2f}'}   AU-PRO  A={e['A_au_pro']:6.2f}  "
                         f"B={e['B_au_pro_mean']:6.2f}   chance AP {e['A_chance_ap']:.2f}")
        a_wins = [e["area_pct"] for e in rows if e["A_pixel_ap"] > e["B_pixel_ap_mean"]]
        if not a_wins:
            lines.append("  Crossover: none. Route B has the higher Pixel AP at every size.")
        elif len(a_wins) == len(rows):
            lines.append("  Crossover: none. Route A has the higher Pixel AP at every size.")
        else:
            lines.append(f"  Route A has the higher Pixel AP at: {', '.join(fmt_size(v) for v in a_wins)}; "
                         f"Route B at the other sizes.")
        if len(rows) >= 3:
            rA = spearmanr([e["area_pct"] for e in rows], [e["A_au_pro"] for e in rows]).correlation
            rB = spearmanr([e["area_pct"] for e in rows], [e["B_au_pro_mean"] for e in rows]).correlation
            lines.append(f"  AU-PRO trend vs size (Spearman rho): Route A {rA:+.2f}, Route B {rB:+.2f}")
            if rA >= 0:
                lines.append("  -> Route A does NOT get worse with defect size here (AU-PRO does not fall).")
            else:
                lines.append("  -> Route A AU-PRO falls as defects grow.")
        else:
            dA = rows[-1]["A_au_pro"] - rows[0]["A_au_pro"]
            lines.append(f"  Only {len(rows)} sizes: Route A AU-PRO changes by {dA:+.2f} from smallest to largest "
                         f"(too few sizes for a trend test).")
            if dA >= 0:
                lines.append("  -> Route A does NOT get worse between these sizes.")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(res, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved: defect_size_sweep.json, defect_size_sweep.csv, defect_size_sweep.png, summary.txt in {res}")


if __name__ == "__main__":
    main()
