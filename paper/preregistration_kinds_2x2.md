# Pre-registration: mismatched-drift control + independent task 2 (follow-up to the kinds kill test)

Written 2026-09-30 after the kill test (PASS; comparative excess E = +0.086, conjunctive about 0) and before any run of
this design. SHA-256 in `preregistration_kinds_2x2.sha256`. Scripts: `kinds_2x2.py`, `analyze_kinds_2x2.py` (fixed before
the real run). Fresh seeds 28-32.

## Why
In the kill test the comparative excess sat in C (the same pairs survive or fail under either single drift), not in N.
Two stories fit:
- Agreement-feature story: comparative knowledge hangs on one "do they agree?" feature per pair; it breaks once, so
  damage from the two encoders overlaps. It does not need coordinated drift.
- Co-drift story: the two encoders drift the same way, so their agreement is preserved. It needs coordinated drift.

## Tasks and task-1 models
Identical to `preregistration_kinds_kill.md` (two-view CIFAR, pairs from different items, 10 task-1 classes;
comparative = same class?, conjunctive = balanced random class-pair table, XOR = reported only). Same manipulation
checks (task-1 accuracy >= 0.75, linear readout <= 0.60).

## Task-2 conditions (task-1 head frozen and unused; drift levels 1, 3, 10 epochs of 50,000 samples; snapshots of one
continuing run per condition)
- Coupled A: cross-modal InfoNCE (as in the kill test) on task-2 pool A (classes perm[10:20]).
- Coupled B: same procedure, different seed, task-2 pool B (classes perm[20:30]).
- Matched = (A.e1, A.e2). Mismatched = (A.e1, B.e2) and (B.e1, A.e2), E averaged over the two. Each encoder drifts by
  a coupled run, but the two drifts are not coordinated with each other.
- Independent: on pool A, e1 trained with cross-entropy through its own new linear head on view 1, e2 likewise on
  view 2 (separate heads, separate losses, same items), Adam 1e-3, batch 256.

## Measures
k = (acc - 0.5) / (acc0 - 0.5) for new e1 + old e2, old e1 + new e2, both new; E = k12 - k1*k2. Also raw C, N and drift
(1 - cosine to the task-1 codes). Reported: cosine similarity between A's and B's code changes (how coordinated).

## Informative levels
Per condition, levels where both comparative and conjunctive have seed-mean min(k1, k2) in [0.1, 0.9]. A comparison of
two conditions uses the levels informative in both. Statistics: per-seed mean of E over those levels; one-sided 95%
t-intervals over 5 seeds.

## Tests
- T0 (replication): matched comparative E >= 0.05 and comparative - conjunctive >= 0.05 (lower bounds > 0).
- T1 (mismatch), D1 = E_comp(matched) - E_comp(mismatched):
  - co-drift supported iff D1 >= 0.05 (lower bound > 0);
  - agreement-feature supported iff E_comp(mismatched) >= 0.05 (lower bound > 0) and D1 < 0.05;
  - otherwise mixed.
- T2 (2x2 interaction), G(cond) = E_comp(cond) - E_conj(cond):
  - "only the coupled comparative cell differs" supported iff G(coupled) - G(independent) >= 0.05 (lower bound > 0) and
    G(independent) < 0.03;
  - "comparative advantage without coupling" iff G(independent) >= 0.05 (lower bound > 0);
  - otherwise mixed.
- XOR reported only.
