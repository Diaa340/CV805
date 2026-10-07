# Pre-registration P1: semi-synthetic bridge (real features, exact interaction types)

Written after v3 passed on synthetic data and failed on real data with probe-estimated types
(preregistration_v3.md, results_v3.txt), and before any P1 run. SHA-256 in `preregistration_p1_bridge.sha256`.

## Question
Does the v3 drift-composition law hold on REAL learned features once per-sample interaction types are exact?

## Data construction
Frozen features (extract_features.py); each paired sample takes modality 1 from one real item and modality 2 from a
DIFFERENT real item:
- `cifar2v` (left half of image A / right half of image B), K = 10;
- `food101` (image of item A / title of item B), K = 10;
- `n24news` (image of item A / headline of item B), K = 3.

AV-MNIST (10 classes) and CREMA-D (6 classes) are excluded: they have too few classes for disjoint pools.

Classes are split (fixed permutation, seed 900) into disjoint pools of K classes: for task 1, one pool per (type, modality);
task 2 uses fresh pools for the shifted modality/modalities. With label y in {0..K-1}:
- R: A from pool R_A class y, B from pool R_B class y;
- U1: A from U1_A class y, B from U1_B random class;
- U2: A from U2_A random class, B from U2_B class y;
- S: A from S_A class a (uniform), B from S_B class (y - a) mod K.

Items are drawn from the dataset's train split for training pairs and from the test split for test pairs.
Type mix [R, U1, U2, S] = [0.4, 0.15, 0.15, 0.3]; 50k train pairs, 20k test pairs per task.

## Runs
dataset {cifar2v, food101, n24news} x shift {both, m1, m2} x architecture {(2,256), (3,512)}, seeds 13–17 (90 units).
Training, replay buffer, task-1 synergy up-weighting (by the exact types), swaps and retention: exactly as v3
(route_exps.py / v3_drift.py).

## Criteria (exactly v3's synthetic thresholds; types exact)
- (a) composition sb_t = s1_t * s2_t: R^2 >= 0.8 and MAE <= 0.06 for S, U1, U2; R^2 >= 0.5 for R.
- (b) U1: s2 >= 0.9 in >= 80% of units, and U2: s1 >= 0.9 in >= 80% of units; mean min-factor of R minus that of S
  >= 0.10.
- (c) replay regime: actual vs swap for S, R^2 >= 0.7 and MAE <= 0.08.
- Judged on all units pooled; per-dataset results are reported.
- **Learnability rule:** if synergy is learned in task 1 (accuracy >= chance + 0.3) in fewer than 50% of units, P1 is
  INCONCLUSIVE rather than pass/fail.
- **Kill criterion:** if (a)–(c) fail with learnable synergy, the law is a property of the toy generator, not of learned
  multimodal representations.
