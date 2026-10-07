# Pre-registration: an additive-model estimator of the fusion ceiling for natural data

Written 2026-10-01 before any run of these scripts. SHA-256 in `preregistration_additive_est.sha256` (with
`additive_estimator.py`, `liang_bench.py`, `analyze_additive_est.py`).

## Estimator
On data without label tables, the additive ceiling is estimated by the accuracy of the best *jointly trained* additive
model: two unimodal networks (encoder + MLP head each) whose class logits are summed. Same architecture blocks, budget
(30 epochs, Adam 1e-3) and data as the full model (two-branch encoders + joint MLP head). Fusion gap = full - additive.

## E1 (primary, validation): CLEVR, where the exact ceiling is known
103 units, step-2 data and split (700 / 300 images, 50k / 20k cells), seeds 23-25. PASS iff, with symbolic features,
MAE(additive accuracy, exact additive ceiling on the test cells) <= 0.03 and Spearman >= 0.8. CLIP reported.

## E2 (descriptive): seven frozen-feature benchmarks
avmnist, cifar2v, clevr, cremad, food101, n24news, vqa (all classes; their train / test split; seeds 23-25): full,
additive, and unimodal accuracies, fusion gap, and Liang et al.'s PID estimate at their default settings (PCA 5,
k = 30, CVX via ECOS) on each test split. Reported with Spearman across datasets (n = 7; no pass / fail).
Known limitation (from S2-B): small frozen-feature full models under-learn fusion, so the gaps are lower bounds; a
stronger full model (the VLM) is planned for CLEVR after the forgetting test.
- Addendum (2026-10-01): ECOS did not finish the 24-101-class benchmark problems after 6.5 h; all seven benchmarks are re-solved with SCS (liang_bench2.py, same clustering). Cross-check on CREMA-D: SCS vs ECOS synergy 0.861 vs 0.852, other atoms within 0.004. hash: f7c2c39aa91abe32028aac07cdf26869f5be5794e67f839f1e1addc003b7f85b  liang_bench2.py
