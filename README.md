# Zero-Shot Texture Anomaly Detection

A research codebase investigating zero-shot anomaly detection on industrial textures using visual foundation models (DINOv2) and self-similarity analysis.

## Project Structure

```
zero_shot_texture_anomaly/
├── data/
│   └── mvtec/             # MVTec AD dataset (carpet, grid, leather, tile, wood)
├── models/                # Pretrained model checkpoints and caches
├── src/                   # Core research modules
│   ├── features.py        # DINOv2 / ViT patch feature extraction
│   ├── similarity.py      # Self-similarity & multi-image similarity matrices
│   ├── anomaly_map.py     # Anomaly scoring and spatial heatmap generation
│   └── evaluation.py      # Evaluation metrics (Pixel AUROC, Pixel AP, AU-PRO)
├── experiments/           # Experiment scripts (Exp 001, 002, etc.)
├── results/               # Metrics and quantitative logs
├── figures/               # Qualitative visualizations (Anomaly maps vs GT)
├── notebooks/             # Exploratory analysis and prototyping
└── README.md
```

## Research Roadmap

1. **Step 1 - Environment & Setup**: PyTorch with CUDA, DINOv2 ViT-B/14.
2. **Step 2 - Baseline Single-Image Self-Similarity (Exp 001)**:
   - Extract patch features from a single test image using frozen DINOv2.
   - Compute pairwise cosine similarity between patches.
   - Estimate anomaly score based on similarity to internal normal patches.
   - Generate pixel-level anomaly maps.
3. **Step 3 - Quantitative Evaluation**:
   - Pixel AUROC, Pixel Average Precision (AP), and Area Under Per-Region Overlap (AU-PRO).
4. **Step 4 - Multi-Image Normality Reference**:
   - Compare test patches against a pool of reference normal patches.
   - Evaluate scaling performance across sample sizes (1, 2, 4, 8, 16 reference images).
5. **Step 5 - Text-Prompt Guidance & Routing**:
   - Integrate CLIP / text embeddings ("normal texture" vs defective).
   - Evaluate performance on homogeneous vs non-homogeneous textures.
   - Reliability scoring and adaptive routing.
