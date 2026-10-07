# Pre-registration: step 1, consequences of removable vs non-removable interaction in the table framework

Written 2026-09-30, before any step-1 outcome was computed. SHA-256 in `preregistration_step1.sha256`, with hashes of
`step1_build.py`, `tables_pid2.py`, `step1_pairs.py`, `tables_step1.py`, `analyze_step1.py`. Seen before writing: table
set sizes (fresh 62 tables; 60 pair candidates) and one smoke run (seed 99, tables 0 and 60 of the original set;
discarded). Thresholds for D were set after seeing the exploratory P2 of the label-table test (median gaps 0.06 vs 0.43).

## Background
In the label-table test (`preregistration_tables.md`) the linear-readout gap followed XOR structure (partial rho +0.91),
not PID synergy (+0.22). Theory: a table is threshold-additive exactly when it has no 2x2 checkerboard (Ferrers
relations). Removable interaction = what a threshold of an additive model can do; non-removable = the rest.
Structure measures: XOR-square density; additive distance d_add (fewest flips to the nearest threshold table F, exact
MILP). For A and B, d_add is the primary structure measure, because it is the exact quantity and did at least as well as
density in P3.

## Table sets
- selected: the 63 tables of the label-table test.
- fresh: 62 new tables, from a new pool (seed 2027) with the same selection rule (MMI synergy band 0.40-0.75, 4 x 4
  synergy-by-density cells).
- pairs: up to 12 matched pairs by construction: a zero-density table and a table with density >= 0.06, both in the
  band, |BROJA difference| <= 0.03 (greedy nearest, no reuse; candidates matched on MMI first).
- PID (BROJA primary, I_ccs reported) with dit, uniform independent class IDs.

## Per table and seed (23, 24, 25), `tables_step1.py`
Task-1 model as before (two-view CIFAR, view 1 from class i, view 2 from a different item of class j, label T[i, j]).
- Readout gap = (acc_before - acc_linear) / (acc_before - acc_majority).
- 1a repair: task 2 = a random balanced table on 10 new classes; the whole model is trained on it for 10 epochs. Then,
  on the frozen drifted encoders, a linear head (logistic, 500 steps) and an MLP head (256 hidden units, 2,000 steps)
  are retrained on 20,000 task-1 train pairs.
  Shortfall = (acc_MLP_repair - acc_linear_repair) / (acc_before - acc_majority): the share of task-1 knowledge that
  only a non-additive head can restore.
- 1c: survival events under new e1 + old e2, old e1 + new e2, both new (old head), in removable cells (T = F) and
  non-removable cells (T != F).
- 1b OOD: a separate model trained on 80 of the 100 (i, j) cells (20 held out, at most 3 per row / column). Drop =
  balanced accuracy on seen cells - balanced accuracy on held-out cells (test split).
Tables with acc_before - acc_majority < 0.10 are excluded.

## Tests (partial Spearman across tables, per-table seed means; 95% bootstrap intervals reported)
- R1 (replication): on fresh tables, P1's rule: a = rho(gap, density | BROJA), b = rho(gap, BROJA | density);
  structure wins iff a >= 0.5 and a - |b| >= 0.3; synergy wins iff b >= a; else mixed.
- A (1a, selected + fresh): the same rule with y = shortfall and structure = d_add.
- B (1b, selected + fresh): the same rule with y = OOD drop and structure = d_add.
- D (1d): gap(dense) - gap(zero) >= 0.20 in >= 80% of pairs and mean >= 0.25 -> PASS; fewer than 8 pairs ->
  INCONCLUSIVE.
- Reported (1c, with an expectation, not decisive): in non-removable cells, joint-drift survival is close to the product
  of the single-drift survivals (|E| <= 0.05); removable cells may deviate.
- Also reported: density and I_ccs versions, recovery fractions, linear-readout OOD, and ideal one-hot additive
  completion of the held-out cells.

## What happens next
If R1 passes and at least one of A / B passes, step 2 (CLEVR exact structure) follows without further discussion,
pre-registered separately.
