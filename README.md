# Zero-shot vs few-shot texture anomaly detection with DINOv2

Code and results of an MSc thesis comparing a **zero-shot** and a **few-shot** way of finding and localising
defects in industrial textures, using frozen DINOv2 ViT-B/14 patch features, on the five texture categories of
MVTec AD (carpet, grid, leather, tile, wood; 515 test images, 382 of them defective).

- **Route A (zero-shot):** each patch is scored by how dissimilar it is to all other patches of the *same* image
  (1 − mean cosine similarity). No normal training images are needed.
- **Route B (4-shot):** each patch is scored by its distance to the nearest patch in a memory bank built from
  **4 normal training images** (1 − max cosine similarity). Results are averaged over seeds 42, 123, 2026.

The thesis asks how much the 4 reference images buy, and **why** the zero-shot route falls behind, in particular
how both routes depend on the **size of the defect**.

## Main results

Five-texture average; full image / with a 10 % image border ignored (from
`results/fca_baseline/metrics/metrics.txt`):

| Method | Pixel AP | AU-PRO | Pixel AUROC | Image AUROC |
|---|---|---|---|---|
| Route A (zero-shot) | 32.09 / 43.77 | 86.60 / 89.14 | 92.08 / 94.08 | 87.05 / 91.27 |
| Route B (4-shot) | 50.04 / 51.29 | 94.85 / 93.20 | 97.00 / 97.06 | 99.79 / 99.89 |
| FCA (zero-shot reference, WACV 2024) | 70.40 / 76.76 | 95.88 / 97.13 | 98.07 / 98.75 | 97.06 / 99.54 |

Route B − Route A: **+17.94 pixel AP, +8.26 AU-PRO** (`results/experiments_suite_results.json`).

**Defect size.** On real defects, Route A's per-image pixel AUROC falls with log defect area much faster than
Route B's: slope A −3.887 [−5.484, −2.531] vs B −1.177 [−2.141, −0.472]; difference −2.710 [−3.487, −1.918]
(382 images, texture × defect-type fixed effects, 95 % bootstrap CIs; `results/size_effect_decomposition/`).
For the zero-shot method FCA the difference to Route B has a CI that includes zero (FCA − B +0.372
[−0.188, +1.080]; `results/fca_baseline/size_analysis/`).

All experiments, with their commands, pre-set criteria, verdicts (including the ones that were **not**
supported) and reproduction checks, are listed in **[REPRODUCE.md](REPRODUCE.md)**.

## Repository layout

```
src/                      analysis and experiment scripts
  eval_router_full.py       main Route A / Route B benchmark
  run_full_experiments_suite.py  Route A / B / SubspaceAD / oracle routing / size breakdown
  evaluation.py             metrics: pixel AUROC, pixel AP, AU-PRO (FPR 0.3)
  mechanism_*.py            self-contamination and context-drift experiments
  feature_layer_sweep.py    DINOv2 layer / scoring / pooling sweep
  fca_baseline/             wrappers around the official FCA code (not modified)
results/                  one folder per experiment: results (.txt, .json, .csv), configs, environment files
paper/                    the paper: Overleaf project, Part 1 PDF/Word, figures, verified references, audit notes (see paper/README.md)
presentation/             defence slides and HTML explainers
REPRODUCE.md              how to reproduce every number
```

## Quick start

1. Python 3.13, PyTorch 2.6 with CUDA 12.4 (tested on an RTX 3050 Laptop GPU, 4 GB); plus numpy, scipy,
   scikit-learn, opencv-python, pillow, pandas. Exact versions: `results/feature_layer_sweep/environment.json`.
2. Download MVTec AD and place the five textures in `data/mvtec/<texture>/` (the dataset is not included).
3. DINOv2 is loaded automatically through `torch.hub` (`facebookresearch/dinov2`, `dinov2_vitb14`).
4. Run the experiment you need, e.g. `python src/size_effect_decomposition.py`; see [REPRODUCE.md](REPRODUCE.md).

## Not included

The MVTec AD images, the external FCA code (`third_party/`), model weights, token caches, raw anomaly maps and
synthetic defect images. REPRODUCE.md explains how each is obtained or regenerated.

## Data, code and licence

- MVTec AD: Bergmann et al., licence CC BY-NC-SA 4.0 (download from MVTec).
- FCA: Ardelean & Weyrich, WACV 2024, https://github.com/TArdelean/AnomalyLocalizationFCA (MIT), used unmodified at
  commit `afe8326`.
- **Licence of this repository: not yet chosen** (all rights reserved until a licence is added).
