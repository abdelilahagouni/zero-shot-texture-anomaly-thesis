"""
Runs the OFFICIAL entry point third_party/AnomalyLocalizationFCA/main.py unchanged, with the given Hydra overrides,
e.g.   <fca venv python> src/fca_baseline/run_official.py dataset=mvtec fe=wide sc=fca image_size=original tile_size=[9,9]

The only additions (no repo file is edited):
  * working directory = the repo folder (as when calling `python main.py` there), so datasets/ and outputs/ resolve
    as in the README;
  * if the installed tifffile no longer has `imsave` (removed in recent tifffile; the repo pins 2023.2.28),
    `tifffile.imsave` is aliased to `tifffile.imwrite` (same function, the old name);
  * wall time and peak GPU memory (torch.cuda.max_memory_allocated / max_memory_reserved) are printed at the end.
The repo writes per-image maps (float32 .tiff) to outputs/<run>/<object>/test/<defect>/<stem>.tiff and
the metrics to outputs/<run>/metrics_100.json.
"""
import atexit, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FCA_REPO  # noqa: E402

T0 = time.time()


def report():
    import torch
    mb = 1024 ** 2
    if torch.cuda.is_available():
        print(f"[run_official] peak GPU memory allocated {torch.cuda.max_memory_allocated() / mb:.0f} MiB, "
              f"reserved {torch.cuda.max_memory_reserved() / mb:.0f} MiB "
              f"({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / mb:.0f} MiB)")
    print(f"[run_official] wall time {time.time() - T0:.1f}s")


def main():
    import tifffile
    if not hasattr(tifffile, "imsave"):
        tifffile.imsave = tifffile.imwrite
        print(f"[run_official] tifffile {tifffile.__version__}: aliased imsave -> imwrite")
    os.chdir(FCA_REPO)
    sys.path.insert(0, FCA_REPO)
    sys.argv = [os.path.join(FCA_REPO, "main.py")] + sys.argv[1:]
    atexit.register(report)
    import runpy                      # run main.py as a script (as `python main.py`), so Hydra finds conf/ next to it
    runpy.run_path(os.path.join(FCA_REPO, "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
