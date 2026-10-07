# Pre-registration: kinds of cross-modal knowledge, v2 (mismatch control, independent task 2, threshold test)

Written 2026-09-30. SHA-256 in `preregistration_kinds_v2.sha256`. Scripts `kinds_v2.py`, `analyze_kinds_v2.py`
(hashed in an addendum before the real run). Fresh seeds 28-32.

Supersedes `preregistration_kinds_2x2.md` + addendum. That run was stopped after 11 of 15 units, before any of its results
were viewed, because it lacked the checks below; its partial files are in `runs/superseded/`, unread and unused.
Already seen (exploratory, stated here so nothing is hidden): kill-test nesting per seed (comparative 0.2 / 0.6 / 0.9 at
levels 1 / 3 / 10; conjunctive -0.1 / 0.05 / 0.14) and the face-voice label split (C about 0 within each label; N +0.14
on "same" pairs, about 0 on "different" pairs).

## Two stories
- Drifts cancel: coordinated drift keeps the two codes in agreement, so some pairs fail under one drift but survive
  both.
- Same weak pairs fail: drift in either encoder weakens every match a little, so pairs near the decision line fail
  first, whichever encoder drifted. A variant: drift shifts the effective threshold.

## Tasks, task-1 models, checks
As in `preregistration_kinds_kill.md` (two-view CIFAR, view 1 and view 2 from different items; comparative = same
class?, conjunctive = balanced random class-pair table, XOR reported only; task-1 accuracy >= 0.75 and linear readout <=
0.60, else excluded).

## Task-2 conditions (all start from the same task-1 checkpoint; task-1 head frozen and unused during task 2)
- A: coupled (cross-modal InfoNCE) on task-2 pool A (perm[10:20]), seed s_A. Levels 1, 3, 10 epochs (50,000 samples each).
- B1: coupled, pool A, different seed only. B2: coupled, pool B (perm[20:30]), different seed.
- Matched = (A.e1, A.e2). Mismatched = (A.e1, B.e2) and (B.e1, A.e2), averaged.
- Mismatch check: task-2 matching accuracy = top-1 cross-modal retrieval (cosine) among 1,000 held-out pool-A test items.
  B1 is used if, averaged over seeds, mismatched accuracy <= within-run accuracy (A) - 0.10 at every level; otherwise B2.
  If B2 also fails the check, T1 is INCONCLUSIVE.
- I: independent (e1 and e2 each trained with cross-entropy through its own new linear head on pool A), levels 1, 3,
  10, 30, 100.

## Measures (task-1 test pairs, through the frozen task-1 head)
- k = (acc - 0.5) / (acc0 - 0.5) for new e1 + old e2 (k1), old e1 + new e2 (k2), both new (k12). E = k12 - k1*k2.
- Nesting = (k12 - k1*k2) / (min(k1, k2) - k1*k2): 0 = damages multiply, 1 = the same pairs fail, > 1 = drifts cancel.
  Computed per seed and level; a level enters nesting tests only if its seed-mean denominator >= 0.05.
- Raw on task-1-correct pairs: P1, P2, r12, C, N, pooled and split by label (y = 1 / y = 0).
- Threshold test on the both-drifted model: refit only the head's bias difference (one number, added to logit1 -
  logit0). (i) Supervised: grid search maximising accuracy on 500 labelled task-1 train pairs. (ii) Label-free: set so
  that 50% of task-1 test pairs are predicted 1. Recovery = (k12_refit - k12) / (1 - k12).

## Informative levels
Levels where both comparative and conjunctive have seed-mean min(k1, k2) in [0.1, 0.9]. Statistics are per-seed means
over levels; one-sided 95% t-intervals over 5 seeds ("lower"); equivalence uses two-sided 90% intervals.

## Tests and predictions
- T0 (replication, matched): comparative E >= 0.05 and comparative E - conjunctive E >= 0.05, lower bounds > 0.
- T1 (mismatch; comparative; levels informative and nesting-eligible in both matched and mismatched):
  dC = C_matched - C_mismatched, dNest = nesting_matched - nesting_mismatched.
  - Drifts cancel: dNest >= 0.2 (lower > 0) or dC >= 0.03 (lower > 0).
  - Same weak pairs fail: 90% interval of dNest inside [-0.2, 0.2] and of dC inside [-0.03, 0.03].
  - Otherwise mixed.
- T2 (independent, damage-matched): reference = seed-mean comparative k1*k2 in matched coupled level 3; choose the
  independent level with the closest seed-mean comparative k1*k2 (if |difference| > 0.15, T2 is INCONCLUSIVE and reported).
  - Drifts cancel: comparative nesting (independent) < 0.1 (upper 95% bound < 0.2).
  - Same weak pairs fail: comparative nesting (independent) >= 0.2 (lower > 0) and comparative - conjunctive nesting
    (independent) >= 0.2 (lower > 0).
  - Otherwise mixed.
- T3 (threshold, matched coupled, levels 3 and 10, supervised refit): recovery comparative >= 0.25 and comparative -
  conjunctive recovery >= 0.15 (lower > 0) supports the threshold version of "same weak pairs fail". Label-free refit is
  reported.
- Reported: label split (pooled vs within-label C and N), XOR, retrieval accuracies, drift, and nesting after bias refit.
