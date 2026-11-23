# Paper: Part 1 (Introduction, Literature Review, Methodology)

Status on 2026-11-23: Part 1 drafted for Dr. Bilal Yousfi. Every number checked
against the result files in `results/`, and every reference checked against the
published paper.

> **Note on authorship.** Parts of this draft were written with an AI assistant
> (Claude, Anthropic) for structure and language. The authors are rewriting each
> section in their own words, and AI use will be declared as Dr. Yousfi advises.
> Do not submit this text as it is.

## Folder layout

| Folder | What it holds |
| --- | --- |
| `overleaf/` | The LaTeX project (IEEEtran, biblatex + biber). Upload as a zip with Overleaf, New Project, Upload Project. Only Part 1 is switched on in `main.tex`; results, discussion and conclusion are ready inside `sections/` |
| `part1/` | Part 1 compiled: PDF, Word, and the authors' own Introduction with tracked corrections |
| `figures/` | Fig. 1 (methodology overview): PDF (vector), PNG 300 and 600 dpi, SVG. Source: `overleaf/figures/method_overview.tex` |
| `references/` | Verified references (`.bib`, `.ris`), the literature matrix and course checklist, the experiment results workbook |
| `../presentation/` | Defence slides (8-slide version, professor version without notes, 19-slide backup with notes) and the HTML explainers |

## Audit of 2026-11-20: corrections made

Each claim in Part 1 was compared with `results/` and the experiment reports.
The main numbers all match. These nine points were corrected:

1. **Seeds.** Size analysis uses Route B seed 42; real-defect interventions use the mean of ten seeds; synthetic experiments use five seeds (the draft said "ten seeds" for the size analysis).
2. **Aota-style is not official code.** It is the reimplementation inside the FCA repository; figure and caption fixed.
3. **Border protocol citation.** Aota et al. evaluate on a centre crop (about 10% per side); the draft wrongly cited them for "border artefacts".
4. **FCA claim softened.** FCA's slope "cannot be distinguished" from Route B's (FCA−B +0.37 [−0.19, +1.08]); FCA also uses different features, so the size dependence is "not shared by every single-image method", not "belongs to the mean reference".
5. **Real-defect criterion not met.** A-clean recovers about half the slope, but stays steeper than Route B (−0.75); the rest is reported as unexplained.
6. **50% test, first sweep.** It held against 32% but not against 0.25%; both are reported.
7. **Drift.** It met its pre-set criterion, but with the opposite sign to the hypothesis, so it is not treated as a mechanism.
8. **Repository availability.** "Will be made public on publication" (the repository is private).
9. **Missing details added.** FCA environment (Python 3.13 / PyTorch 2.6.0 vs pinned 3.9 / 2.1.2, two wrapper fixes), fp16 tokens, synthetic defect settings, PatchCore uses the full training set.

Also corrected: Villanueva López et al. report 1.8 AUROC points on average over four datasets, **1.3 on MVTec AD**.

## Key numbers (all verified)

| Quantity | Value |
| --- | --- |
| Test images | 515 = 133 defect-free + 382 defective (carpet 117, grid 78, leather 124, tile 117, wood 79) |
| Pixel AP, Route A → B | 32.1 → 50.0 |
| AU-PRO, Route A → B | 86.6 → 94.9 |
| Image AUROC, Route A → B | 87.0 → 99.8 |
| Size slopes (AUROC points per unit of ln area) | A −3.89, Aota-style −1.88, B −1.17, FCA −0.80 |
| A − B | −2.72 [−3.49, −1.93]; ten-seed check 2.71 [1.92, 3.50] |
| A-clean recovery | about 50% (real, +1.96 of −3.89), about 66% (synthetic, +5.68 of −8.56) |
| 25% → 50% coverage | A −26.1, A-clean −0.2 |

## Reference check (all 14 verified)

| No. | Reference | Checked against |
| --- | --- | --- |
| 1 | Bergmann et al., MVTec AD, IJCV 2021 | dataset paper |
| 2 | Aota et al., WACV 2023, pp. 5553–5561 | IEEE Xplore PDF (CVF version: 5564–5572) |
| 3 | Ardelean and Weyrich, FCA, WACV 2024 | paper and supplement |
| 4 | Yao et al., GRNR, IEEE TSMC 2025 | paper |
| 5 | Zhang, Song, Yang, Wang (Chencheng), NLPA-AD, Applied Intelligence 55:861, 2025 | full paper |
| 6 | Roth et al., PatchCore, CVPR 2022, pp. 14298–14308 | IEEE Xplore PDF (CVF version: 14318–14328) |
| 7 | Damm et al., AnomalyDINO, WACV 2025 | paper |
| 8 | Lendering et al., SubspaceAD, CVPR 2026 | paper |
| 9 | Li et al., MuSc, ICLR 2024 | paper |
| 10 | Le-Gia and Ahn, arXiv 2025 | paper |
| 11 | Oquab et al., DINOv2, TMLR 2024 | paper |
| 12 | Lehr et al., SPIE ICMV 2023 | abstract (one method, 11 steps, 0.03%) |
| 13 | Villanueva López et al., Electronics 2026 | full paper (1.8 average; 1.3 on MVTec AD) |
| 14 | Luo et al., INP-Former, CVPR 2025 | full paper (needs training) |

## Points to add when rewriting (from reading on 2026-11-20)

- **NLPA-AD**, p. 16: removing large anomalous areas "enhances the purity of the constructed reference library"; Section 5.6: with large anomalies, normal labels spread into the anomaly. Both support the contamination idea.
- **INP-Former**: premise that most anomalies are local; limitation that the anomaly is sometimes taken as a normal prototype.
- **Cao et al.** (CVPRW 2026, arXiv 2411.14515), Section 5.4, Table 4: image-level scores of trained models are biased toward larger anomalies. Different from this work (localisation, training-free methods). Do not copy their "consistently exceed 80%" claim: two Table 4 values are below 80.

## Novelty (course slide 8)

New problem (size dependence of single-image vs memory-bank references on the same features) plus a tested explanation (reference contamination, tested with A-clean), supported by validation (FCA reproduction, two protocols, layer sweep, seeds, synthetic defects). Not a new method and not a state-of-the-art claim.

## Still open

- Dr. Yousfi: AI-use policy, target venue, deadline, final title.
- CRediT roles, Byan's and Dr. Yousfi's ORCID iDs, faculty name, funding statement.
- Google Scholar search before keeping "to our knowledge" in the gap sentence.
- Before making this repository public: decide on a licence, and remove the presenter-notes slides if wanted. MVTec AD thumbnails in `overleaf/figures/` are CC BY-NC-SA 4.0 (attribution, non-commercial).
