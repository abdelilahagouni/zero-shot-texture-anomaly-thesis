"""
Runs the repo's OWN evaluation step (data_loaders.StandardFormatDataset.run_evaluation -> evaluation/
evaluate_experiment.py, the call main.py makes after computing the maps) on an existing official output folder.

Why: on Windows the repo's evaluation/generic_util.read_tiff fails. It checks the extensions .tif, .tiff, .TIF, .TIFF
with os.path.exists; Windows paths are case-insensitive, so 000.tiff is also found as 000.TIFF and it raises
"Found multiple files with a TIFF extension". The only change here: read_tiff is called with exts=('.tiff',), the
only extension the repo writes (data_loaders.save_output). No repo file is edited; the maps are not recomputed.
Arguments are those of conf/dataset/mvtec.yaml: 5 textures, paddings [100], pro_integration_limit 0.3, resize off.

usage: <fca venv python> src/fca_baseline/run_official_eval.py <outputs/run folder>
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FCA_REPO, TEXTURES  # noqa: E402


def main():
    run = os.path.abspath(sys.argv[1])
    t0 = time.time()
    os.chdir(FCA_REPO)
    sys.path.insert(0, FCA_REPO)
    from evaluation import generic_util
    orig = generic_util.read_tiff
    generic_util.read_tiff = lambda p, exts=None: orig(p, exts=(".tiff",))
    from data_loaders import StandardFormatDataset
    dm = StandardFormatDataset(data_dir=os.path.join(FCA_REPO, "datasets", "mvtec_anomaly_detection"), out_dir=run,
                               objects=TEXTURES, paddings=(100,))
    dm.run_evaluation()
    print(f"[run_official_eval] wall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
