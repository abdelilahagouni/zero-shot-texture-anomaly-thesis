"""
Timing and GPU-memory estimate before the real runs (uses the repo's own classes, nothing is saved).
FCA  : WideResnetExtractor(post_scale=True) + ScFCA(tile 9x9, chunk 8, sigma_p 3, median reference, sigma_s 1)
       at original resolution  (= conf fe=wide sc=fca image_size=original tile_size=[9,9])
aota : WideResnetExtractor + sc_aota(tile 9x9, sigma 100, k 400) at 320x320 (= sc=aota image_size=[320,320], repo
       default tile_size [9,9])
Times n images per method on carpet (1024 px) and tile (840 px), after one warm-up image.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FCA_REPO, TEXTURES, test_items  # noqa: E402


def main(n=3):
    import torch
    from functools import partial
    sys.path.insert(0, FCA_REPO)
    from feature_extractors import WideResnetExtractor
    from sc_methods import ScFCA, sc_aota
    from op_utils import load_image_tensor
    from main import compute_anomaly_map
    dev = "cuda"
    fe = WideResnetExtractor(post_scale=True, device=dev)
    methods = {"fca": (ScFCA(tile_size=[9, 9], chunk_size=8, sigma_p=3.0, reference_selection="median", sigma_s=1.0),
                       "original"),
               "aota": (partial(sc_aota, tile_size=[9, 9]), [320, 320])}
    n_img = {c: len(test_items(c)) for c in TEXTURES}
    total = sum(n_img.values())
    est = {}
    for name, (sc, size) in methods.items():
        per = []
        for cat in ["carpet", "tile"]:
            items = test_items(cat)[: n + 1]
            torch.cuda.reset_peak_memory_stats()
            for j, (_, _, p, _) in enumerate(items):
                torch.cuda.synchronize(); t = time.time()
                m = compute_anomaly_map(load_image_tensor(p, size, device=dev), fe, sc)
                torch.cuda.synchronize()
                if j > 0:
                    per.append(time.time() - t)
            print(f"{name:<5} {cat:<7} map {tuple(m.shape)}  {np.mean(per[-n:]):.2f}s/image  peak GPU "
                  f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB allocated, "
                  f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB reserved", flush=True)
        est[name] = float(np.mean(per))
    print(f"\nESTIMATE for {total} test images (5 textures):")
    for name, s in est.items():
        print(f"  {name:<5} {s:.2f}s/image -> {s * total / 60:.1f} min of map computation")
    print("  official run (fca) also writes tiff+jpg per image and runs the repo evaluation "
          "(PRO with skimage at 1024 px): add roughly 5-15 min")


if __name__ == "__main__":
    main()
