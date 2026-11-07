# eval_router_full.py -- Comprehensive Benchmark Summary

## 1  Environment

| Key | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Seeds | [42, 123, 2026] |
| K-shot | 4 |
| Categories | ['carpet'] |
| AU-PRO thresholds | 1000 linear |

## 2  Sanity Check vs Reference Numbers

```
CARPET:
  Route A  AUROC: 96.12%  (ref 96.12%, diff=+0.00)
  Route A  AP:    37.26%    (ref 37.26%, diff=+0.00)
  Route B  AUROC: 98.82%  (ref 98.79%, diff=+0.03)
  Route B  AP:    47.16%    (ref 78.41%, diff=-31.25)
wood: not in current run
```

## 3  AU-PRO: Old vs New (carpet / Route B / seed 42)

| | Value |
|---|---|
| New AU-PRO | 96.9479% |
| Old AU-PRO | 21.6816% |
| Absolute diff | 0.752663 |
| Thresholds <= FPR 0.3 | 890 / 1000 |

## 4  Variance (sigma^2_spatial) Distribution

| Category | Split | n | Min | Median | Mean | Max |
|---|---|---|---|---|---|---|
| carpet | train_good | 30 | 0.000572 | 0.000592 | 0.000596 | 0.000635 |
| carpet | test_good | 28 | 0.000573 | 0.000598 | 0.000597 | 0.000618 |
| carpet | test_defect | 89 | 0.000604 | 0.000659 | 0.000657 | 0.000710 |

## 5  Threshold tau Calibration

Pooled train/good sigma^2 N=30

| Percentile | tau |
|---|---|
| 50 (median, **selected**) | `0.0005921` |
| 10 | `0.0005766` |
| 20 | `0.0005811` |
| 30 | `0.0005849` |
| 40 | `0.0005902` |
| 50 | `0.0005921` |
| 60 | `0.0005962` |
| 70 | `0.0006027` |
| 80 | `0.0006081` |
| 90 | `0.0006169` |

## 6  Route-Only Baselines

| Category | Route | Pixel AUROC | Pixel AP | AU-PRO | Image AUROC |
|---|---|---|---|---|---|
| **carpet** | Route A | 96.12 | 37.26 | 92.50 | 91.65 |
| | Route B (mean+/-std) | 98.81+/-0.01 | 46.76+/-0.34 | 96.91+/-0.05 | 99.79+/-0.08 |

### Per-Seed Route B Details

| Category | Seed | Pixel AUROC | Pixel AP | AU-PRO | Image AUROC |
|---|---|---|---|---|---|
| carpet | 42 | 98.82 | 47.16 | 96.95 | 99.88 |
| carpet | 123 | 98.80 | 46.78 | 96.93 | 99.80 |
| carpet | 2026 | 98.81 | 46.33 | 96.84 | 99.68 |

## 7  Router Variants (tau = median)

| Variant | Category | Routed A/B | Pixel AUROC | Pixel AP | AU-PRO | Image AUROC |
|---|---|---|---|---|---|---|
| **(a) Always-A** | carpet | 117/0 | 96.12 | 37.26 | 92.50 | 91.65 |
| **(b) Always-B** | carpet | 0/117 | 98.82 | 47.16 | 96.95 | 99.88 |
| **(c) Oracle** | carpet | B/- | 98.82 | 47.16 | 96.95 | 99.88 |
| **(d) Raw concat** | carpet | 10/107 | 95.06 | 18.67 | 85.63 | 65.73 |
| **(e) Min-Max** | carpet | 10/107 | 98.34 | 50.26 | 96.74 | 73.21 |
| **(f) Calibrated** | carpet | 10/107 | 98.82 | 47.16 | 96.95 | 99.88 |

### Subset A / Subset B Breakdown (variants d, e, f)

| Variant | Category | Subset | Pixel AUROC | Pixel AP | AU-PRO |
|---|---|---|---|---|---|
| **(d) Raw** | carpet | Subset A | NaN | NaN | 0.00 |
| **(d) Raw** | carpet | Subset B | 98.71 | 47.16 | 96.67 |
| **(e) Min-Max** | carpet | Subset A | NaN | NaN | 0.00 |
| **(e) Min-Max** | carpet | Subset B | 98.51 | 53.48 | 97.02 |
| **(f) Calib** | carpet | Subset A | NaN | NaN | 0.00 |
| **(f) Calib** | carpet | Subset B | 98.71 | 47.16 | 96.67 |

## 8  tau Sweep (pooled, variants d/f, AU-PRO omitted for speed)

| Percentile | tau | A/B | Raw Px AUROC | Raw AP | Calib Px AUROC | Calib AP |
|---|---|---|---|---|---|---|
| 10th | `0.0005766` | 2/115 | 98.11 | 36.51 | 98.82 | 47.16 |
| 20th | `0.0005811` | 4/113 | 97.38 | 29.75 | 98.82 | 47.16 |
| 30th | `0.0005849` | 5/112 | 97.00 | 27.12 | 98.82 | 47.16 |
| 40th | `0.0005902` | 8/109 | 95.84 | 21.49 | 98.82 | 47.16 |
| 50th | `0.0005921` | 10/107 | 95.06 | 18.67 | 98.82 | 47.16 |
| 60th | `0.0005962` | 12/105 | 94.25 | 16.54 | 98.82 | 47.16 |
| 70th | `0.0006027` | 16/101 | 92.61 | 13.44 | 98.82 | 47.16 |
| 80th | `0.0006081` | 26/91 | 88.77 | 10.13 | 98.64 | 46.36 |
| 90th | `0.0006169` | 34/83 | 85.64 | 8.32 | 98.58 | 46.16 |

## 9  Descriptors vs Gap (Spearman rho; anomalous images only)

| Category | n | sigma^2 rho (p) | Off-diag rho (p) | Top-5 rho (p) |
|---|---|---|---|---|
| carpet | 89 | -0.123 (0.249) | 0.123 (0.249) | 0.051 (0.634) |

## 10  Artifacts

- `results/per_image.csv` -- per-image metrics, descriptors, gap, routing
- `results/summary.md`   -- this file
