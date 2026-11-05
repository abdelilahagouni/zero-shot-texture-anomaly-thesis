"""
Saves the raw anomaly map of every one of the 515 texture test images as float16 .npy:
  results/fca_baseline/maps/<method>/<texture>/<defect>/<stem>.npy        (native map resolution)

  --method fca  --official-run <outputs/... folder of the passed reproduction run>
        converts the official run's float32 .tiff maps (fe=wide sc=fca image_size=original tile_size=[9,9]),
        i.e. exactly the maps that were evaluated in the reproduction check; no recomputation.
  --method aota
        computes the repo's aota method at 320x320 with the repo's own code (main.compute_anomaly_map,
        WideResnetExtractor(post_scale=True), sc_aota with tile_size [3,3], sigma 100, k 400), i.e. the same as
        `main.py dataset=mvtec fe=wide sc=aota image_size=[320,320] tile_size=[3,3]` (reasons: results/fca_baseline/config.json). Resumable (existing .npy
        files are skipped); prints an estimate, progress, time remaining and peak GPU memory.
Resizing to the 518x518 grid is done on load (common.to_grid).
"""
import argparse, datetime, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FCA_REPO, TEXTURES, test_items, map_path  # noqa: E402


def from_official(run_dir):
    import tifffile
    n, shapes = 0, set()
    for cat in TEXTURES:
        for d, fn, _, _ in test_items(cat):
            src = os.path.join(run_dir, cat, "test", d, fn.replace(".png", ".tiff"))
            m = tifffile.imread(src)
            dst = map_path("fca", cat, d, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, m.astype(np.float16))
            shapes.add((cat, m.shape, str(m.dtype)))
            n += 1
    print(f"fca: converted {n} maps from {run_dir}; shapes {sorted(shapes)}")


def compute_aota():
    import torch
    from functools import partial
    sys.path.insert(0, FCA_REPO)
    from feature_extractors import WideResnetExtractor
    from sc_methods import sc_aota
    from op_utils import load_image_tensor
    from main import compute_anomaly_map
    fe = WideResnetExtractor(post_scale=True, device="cuda")
    sc = partial(sc_aota, tile_size=[3, 3])
    items = [(c,) + it for c in TEXTURES for it in test_items(c)]
    todo = [it for it in items if not os.path.exists(map_path("aota", it[0], it[1], it[2]))]
    print(f"aota: {len(items)} test images, {len(items) - len(todo)} done, {len(todo)} to run", flush=True)
    t0, times = time.time(), []
    torch.cuda.reset_peak_memory_stats()
    for k, (cat, d, fn, p, _) in enumerate(todo, 1):
        t = time.time()
        m = compute_anomaly_map(load_image_tensor(p, [320, 320], device="cuda"), fe, sc).cpu().numpy()
        dst = map_path("aota", cat, d, fn)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.save(dst + ".tmp.npy", m.astype(np.float16))
        os.replace(dst + ".tmp.npy", dst)
        times.append(time.time() - t)
        if k == 3 or k % 25 == 0 or k == len(todo):
            eta = datetime.timedelta(seconds=int(np.mean(times) * (len(todo) - k)))
            print(f"[{k}/{len(todo)}] {cat}/{d}/{fn} map {m.shape} {np.mean(times):.2f}s/image ETA {eta}  "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved", flush=True)
    print(f"aota done in {time.time() - t0:.1f}s")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fca", "aota"])
    ap.add_argument("--official-run")
    a = ap.parse_args()
    if a.method == "fca":
        if not a.official_run:
            sys.exit("--official-run is required for fca")
        from_official(a.official_run)
    else:
        compute_aota()


if __name__ == "__main__":
    main()
