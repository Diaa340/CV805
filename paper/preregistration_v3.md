# Pre-registration v3: forgetting as composed per-modality encoder drift at a fixed interface

Written after the exploratory Phase-1b finding (not pre-registered there): at the task-1 fusion layer, synergy retention
with both encoders drifted equalled the product of the single-encoder-drift retentions (R^2 0.97 / 0.99, MAE 0.025 /
0.012), and swapping drifted encoders into the task-1 fusion layer reproduced actual forgetting (S 0.60 vs 0.59 with
replay). Written before any v3 run. SHA-256 in `preregistration_v3.sha256`.

## Definitions (per unit, per regime, per PID type t)
- Task-1 model (e1, e2, f), model after task 2 (e1', e2', f'). Retention ret_t(model) = chance-corrected task-1 test
  accuracy on type-t samples relative to the task-1 model (masked to task-1 classes in class-incremental runs).
- Swaps (no training): s1_t = ret_t(e1', e2, f), s2_t = ret_t(e1, e2', f), sb_t = ret_t(e1', e2', f).
- Actual: a_t = ret_t(e1', e2', f').
- A type counts in a unit if it has >= 30 task-1 test samples and task-1 accuracy >= chance + 0.3.

## Claims and criteria
(a) Composition: sb_t = s1_t * s2_t (factors clipped to [0, 1.2]).
    Synthetic: R^2 >= 0.8 and MAE <= 0.06 for S, U1, U2; R^2 >= 0.5 for R.
    Real data: R^2 >= 0.6 and MAE <= 0.08 for S, U1, U2; R reported.
    Also reported on the non-trivial subset where both s1_t <= 0.95 and s2_t <= 0.95.
(b) Two-factor signature (both regimes pooled):
    b1. U1: s2_U1 >= 0.9 in >= 80% of units; U2: s1_U2 >= 0.9 in >= 80% of units (the other modality does not damage it).
    b2. mean min(s1_S, s2_S) is lower than mean min(s1_R, s2_R) by >= 0.10 (synergy is damaged by the less-damaging
        modality more than redundancy is).
(c) Reproduction (replay regime primary; no-replay reported): a_S vs sb_S R^2 >= 0.7 and MAE <= 0.08 (synthetic);
    R^2 >= 0.5 and MAE <= 0.10 (real). Other types reported.
Synthetic and real data are judged separately. Human types (CREMA-D) are a reported cross-check, not a criterion.

## Runs
- Synthetic (ground-truth types): SNR {4.5/3.0, 4.5/4.5, 3.5/3.5} x shift {both, m1, m2} x architecture {(2,256),
  (1,128), (3,512)}, seeds 13–17 (new).
- Real (frozen features; probe-estimated types): two-view CIFAR-100, AV-MNIST, CREMA-D, Food-101, N24News x class
  split {0, 1} x task order {normal, swap}, architecture (2,256), seeds 3–7.
- Training exactly as in route_exps.py (task 1: 30 epochs; task 2: 10 epochs; replay buffer as there).
- Exploratory, no criteria: fusion-only swap ret_t(e1, e2, f') (damage from the fusion layer's own change).
