# Reproducing the results

MSc thesis: **zero-shot vs few-shot texture anomaly detection with DINOv2** on the five MVTec AD textures
(carpet, grid, leather, tile, wood; 515 test images, 382 defective).

- **Route A** (zero-shot): anomaly score of a patch = 1 − mean cosine similarity to all patches of the same image.
- **Route B** (4-shot): 1 − max cosine similarity to a memory bank of 4 normal training images
  (bank chosen by `np.random.default_rng(seed).choice(n_train, 4)`).
- Backbone: DINOv2 ViT-B/14 (`torch.hub facebookresearch/dinov2`, `dinov2_vitb14`), 518×518 input, 37×37 patch
  tokens, L2-normalised; maps upsampled bilinearly to 518×518 and smoothed with a 15×15 Gaussian.
- Metrics: `src/evaluation.py` (pixel AUROC, pixel AP, AU-PRO up to FPR 0.3 with 1000 thresholds, per-image
  connected components); image AUROC = max of the map.

Every number below is copied from the committed result file named in its section. Each experiment's
criteria are written in the header of its script (or its `config.json`) and are applied mechanically by the
analysis code.

---

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB (driver 621.14) |
| Main environment | Python 3.13.2, torch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1), torchvision 0.21.0, numpy 2.2.6, scikit-learn 1.8.0, scipy 1.17.1, opencv 4.13.0 — see `results/feature_layer_sweep/environment.json` |
| FCA / aota environment | separate venv, Python 3.13 + torch 2.6.0+cu124, hydra-core 1.3.2 — see `results/fca_baseline/environment.json` and `fca_venv_pip_freeze.txt` |
| Dataset | MVTec AD (not included; licence CC BY-NC-SA 4.0). Download from mvtec.com and place the five textures as `data/mvtec/<texture>/{train,test,ground_truth}` |
| FCA code | `git clone https://github.com/TArdelean/AnomalyLocalizationFCA third_party/AnomalyLocalizationFCA`, commit `afe8326790de30f33cf0e91db7c099bdc6608a84` (MIT), files unmodified; link `third_party/AnomalyLocalizationFCA/datasets/mvtec_anomaly_detection` → `data/mvtec` |

Run all commands from the repository root.

## Reproduction checks

| Check | Result | File |
|---|---|---|
| FCA (official command, full resolution) vs paper Table 6, per texture PRO / pixel AUROC | **PASS**: carpet 95.44/98.30, grid 98.07/99.46, leather 98.90/99.45, tile 96.33/98.22, wood 97.18/98.22 (all equal to the paper to 2 decimals) | `results/fca_baseline/reproduction/reproduction_check_full.txt` |
| Route A / Route B recomputed in later experiments vs the original per-image values | **exact** (max abs. difference 0 for all 382 images) | `results/size_effect_robustness/size_effect_results.txt`, `results/mechanism_real/mechanism_real_results.txt`, `results/feature_layer_sweep/reproduction_check.txt` |
| Feature-layer sweep, block 12 via `get_intermediate_layers` vs existing 5-texture averages | **PASS** (to 2 decimals; crop10 to 1.4e-14; per-image exact) | `results/feature_layer_sweep/reproduction_check.txt` |
| aota (repo re-implementation) vs Aota et al. published pixel AUROC, 256×256 centre crop | within 1.0 point on 4/5 textures; **tile +3.24 (not within)** | `results/fca_baseline_crop/crop_analysis_results.txt` |

## Main numbers (5-texture average, full image / 10 % border crop)

From `results/fca_baseline/metrics/metrics.txt` (Route B = mean of seeds 42, 123, 2026):

| Method | Pixel AP | AU-PRO | Pixel AUROC | Image AUROC |
|---|---|---|---|---|
| Route A | 32.09 / 43.77 | 86.60 / 89.14 | 92.08 / 94.08 | 87.05 / 91.27 |
| Route B | 50.04 / 51.29 | 94.85 / 93.20 | 97.00 / 97.06 | 99.79 / 99.89 |
| FCA (zero-shot) | 70.40 / 76.76 | 95.88 / 97.13 | 98.07 / 98.75 | 97.06 / 99.54 |
| aota (zero-shot) | 26.34 / 66.49 | 83.77 / 95.94 | 93.38 / 98.23 | 38.95 / 99.33 |

---

## Experiments

### 0. Main Route A / Route B benchmark
- **Script:** `src/eval_router_full.py` (`python src/eval_router_full.py`)
- **Results:** `results/per_image_latest_1791743390.csv` (per-image values of the 382 defective images, used as
  the reference by later experiments); `results/summary.md` is the summary of a **carpet-only** run of the same script.

### 1. Defect-size sweep (synthetic defects)
- **Question:** how do Route A and Route B change as a pasted defect grows from 0.25 % to 32 % of the image?
- **Command:** `python src/run_defect_size_sweep.py` then `python src/analyze_defect_size_sweep.py`
  (Colab: `colab_defect_size_sweep.ipynb`)
- **Results:** `results/defect_size_sweep/` (`summary.txt`, `analysis.txt`); synthetic images are not committed
  (regenerated deterministically from `synth_seed` 0 in `config.json`)
- **Key numbers (descriptive):** Spearman(size, AU-PRO), Route A: carpet −0.98, grid −1.00, leather −1.00,
  tile −0.98, wood −1.00; Route B: −0.86, −1.00, −1.00, −1.00, −1.00. Route A beats Route B on Pixel AP or
  AU-PRO at no size in any texture.

### 2. Image-level validation
- **Question:** is the mean of the Route A map a better image-level score than its max?
- **Command:** `python src/validate_image_level.py`
- **Results:** `results/image_level_validation/image_level_auroc.txt`
- **Criterion 1** (mean > max, non-overlapping 95 % CIs, small-defect subset, ≥ 4 of 5 textures): 2/5
  (leather, tile) → not met. **Criterion 2** (held-out categories): not evaluable (none available).
- **Verdict: not supported.**

### 3. Small-defect paired test
- **Question:** does Route A beat Route B on small defects? d = Route B (seed 42) − Route A, per image.
- **Command:** `python src/small_defect_paired.py`
- **Results:** `results/small_defect_paired/paired_results.txt`
- **C1** (A beats B on small defects in ≥ 2 textures): 1/5 (leather) → **not supported**.
- **C2** (d(large) > d(small), CI excludes 0, ≥ 3 textures): 5/5 → condition met. *Note:* the label printed
  for C2 ("Route B degrades relative to A with size") is inverted relative to its condition: d = B − A growing
  with size means **Route A** falls behind Route B as defects grow.

### 4. Size-effect robustness
- **Question:** does d = (mean of 10 Route B seeds) − Route A grow with defect area?
- **Command:** `python src/size_effect_robustness.py`
- **Results:** `results/size_effect_robustness/size_effect_results.txt`
- **Criterion:** Spearman > 0 with CI excluding 0 on ≥ 4/5 textures (5/5) and pooled fixed-effects coefficient
  > 0 with CI excluding 0 (+2.710, 95 % CI [+1.918, +3.502], clustered SE, 25 fixed effects).
- **Verdict: supported.**

### 5. Size-effect decomposition
- **Question:** is the size effect Route A getting worse, Route B getting better, or a ceiling artefact?
- **Command:** `python src/size_effect_decomposition.py`
- **Results:** `results/size_effect_decomposition/decomposition_results.txt`
- **Pooled (n = 382, 25 fixed effects):** slope A −3.887 [−5.484, −2.531]; slope B −1.177 [−2.141, −0.472];
  A − B −2.710 [−3.487, −1.918]; log(errA/errB) +0.159 [+0.025, +0.292]. Criteria (a), (b), (c) met.
- **Verdict: supported.**

### 6. Mechanism, Part A (real defects): self-contamination
- **Question:** does excluding the defect's own patches from Route A's mean (A-clean, uses ground truth,
  diagnostic only) remove the size effect?
- **Command:** `python src/mechanism_real.py`
- **Results:** `results/mechanism_real/mechanism_real_results.txt`
- **Key numbers:** mean per-image AUROC A 95.43, A-clean 97.35, A-median 94.88, A-trim 95.48, B 98.70.
  Part 1 (A-clean − A > 0, CI excl. 0) met; part 2 (A-clean no steeper than B) not met.
- **Verdict: contamination does not explain the real-defect size effect** (the remainder is unexplained).

### 7. Mechanism, Part B (synthetic defects) and context drift
- **Question:** the same test with controlled synthetic blobs (4 defect types, 9 sizes, 133 base images), plus
  whether "drift" of normal patches far from the defect explains the AUROC loss.
- **Commands:** `python src/mechanism_synthetic.py --stage calibrate`, `python -u src/mechanism_synthetic_drift.py --stage full`,
  `python src/mechanism_synthetic_analysis.py`
- **Results:** `results/mechanism_synthetic/analysis/mechanism_synthetic_results.txt`
- **Pooled slopes:** A −8.561 [−8.757, −8.362], A-clean −2.879 [−3.011, −2.750], B −4.226 [−4.349, −4.103].
- **Self-contamination criterion:** the fixed size grid had no 25 % size, so the 50 % prediction was evaluated
  against 32 % (supported, 4/4 types) and 0.25 % (not supported, 1/4 types).
- **Drift criterion: supported** (drift rises with size pooled and in 4/4 types; drift predicts AUROC beyond
  size for both A and B; 0 images dropped; control drift ≤ 2.4e-7).

### 8. 25 % add-on
- **Question:** criterion (iii) exactly as written (50 % vs 25 %), and drift without collinearity (per size).
- **Commands:** `python -u src/mechanism_synthetic_25pct.py`, `python src/mechanism_synthetic_25pct_analysis.py`
- **Results:** `results/mechanism_synthetic_25pct/analysis/mechanism_25pct_results.txt`
- **Criterion (iii) vs 25 %:** holds pooled and in 4/4 types (pooled A 70.33 → 44.26, diff −26.075
  [−26.794, −25.363]; A-clean −0.193 [−0.553, +0.175]).
- **Per-size drift coefficient:** CI excludes 0 for A at 9/10 sizes and for B at 5/10 sizes.

### 9. FCA and aota zero-shot baselines
- **Question:** is the size effect specific to Route A's simple mean self-similarity, or does it also affect a
  strong single-image method (FCA, WACV 2024)?
- **Commands (FCA venv `V`):** `V src/fca_baseline/run_official.py dataset=mvtec fe=wide sc=fca image_size=original tile_size=[9,9]`,
  `V src/fca_baseline/run_official_eval.py <outputs run folder>`, `V src/fca_baseline/compare_reproduction.py --run <folder> --variant full`,
  `V src/fca_baseline/save_maps.py --method fca --official-run <folder>`, `V src/fca_baseline/save_maps.py --method aota`;
  then (main environment) `python src/fca_baseline/evaluate.py`, `python src/fca_baseline/size_analysis.py`.
  Exact sequence and timings: `results/fca_baseline/commands_and_timings.txt`; aota settings and reasons:
  `results/fca_baseline/config.json`.
- **Size analysis (n = 382):** slope FCA −0.798, aota −1.878, A −3.887, B −1.169; FCA − B +0.372 [−0.188, +1.080];
  A − B −2.717 [−3.488, −1.929].
- **Verdict:** (i) "size effect specific to simple mean self-similarity" **supported**; (ii) **not supported**.

### 10. crop10 follow-up
- **Command:** `python src/fca_baseline/crop_analysis.py`
- **Results:** `results/fca_baseline_crop/crop_analysis_results.txt`
- **Size analysis on the 10 % border crop (n = 380, 2 border-only images dropped):** FCA − B +0.720
  [+0.267, +1.245]; A − B −1.468 [−2.296, −0.848] → (i) **supported**, (ii) not supported.
- **Border diagnostic (good images, border / interior mean score, pooled):** FCA 1.077, aota 1.471, A 1.077, B 1.087.
- **aota vs Aota et al.:** within 1.0 on 4/5 textures (tile +3.24).

### 11. Feature-layer sweep
- **Question:** do intermediate DINOv2 layers, kNN scoring or 3×3 token pooling change Route A, and is the size
  effect robust to the layer?
- **Commands:** `python -u src/feature_layer_sweep.py --stage repro`, `--stage full` (resumable; `resume_layer_sweep.bat`),
  `--stage analyze`
- **Results:** `results/feature_layer_sweep/analysis_results.txt`, `metrics_table.csv`, `size_analysis.csv`
- **(i) intermediate layers improve Route A (≥ +3 AU-PRO on both protocols):** supported (23/30 configurations).
- **(ii) size effect robust to the layer (A − B < 0, CI excl. 0, ≥ 4/5 layers, both protocols):** supported
  (full 5/5, crop10 4/5; at blocks 7–9 mean the crop10 CI includes 0).
- **(iii) K matters (> 1 AU-PRO on average):** not supported (mean |difference| 0.37–0.52).
- No configuration was selected as best; adopting one later needs validation on data not used to choose it.

---

## Not in this repository (and how to regenerate)

| Item | Why | How to regenerate |
|---|---|---|
| MVTec AD images | licence / size | download, place under `data/mvtec/` |
| `third_party/AnomalyLocalizationFCA` | external repo | clone at the commit above |
| FCA / aota raw maps (`results/fca_baseline/maps/`, 120 MB) | size | `save_maps.py` (see section 9) |
| synthetic defect images | size, deterministic | re-run the sweep / mechanism scripts (fixed seeds) |
| token caches (`*.pt`) | size (2.8 GB) | recomputed by the scripts |

## A note on the order of criteria and results

For every experiment the criteria were written into the script header or `config.json` before the run, and the
analysis code applies them mechanically; failures are reported as failures. However, scripts and their results
were committed **together**, so git history alone does not prove that a criterion predates its result.
