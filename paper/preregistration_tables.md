# Pre-registration: does the linear-readout gap follow XOR structure or PID synergy? (label-table kill test)

Written 2026-09-30, before any PID computation or training on the selected tables. SHA-256 in
`preregistration_tables.sha256`, with hashes of `tables_build.py`, `tables_pid.py`, `tables_train.py`, `analyze_tables.py`.
Seen before writing (outcome-free): the table pool and selection statistics (below), one smoke-test training run
(seed 99, table 0, discarded), and dit's PID of AND / XOR / one threshold / one random table (unrelated to the selection).

## Claim under test
PID synergy (as used for multimodal benchmarks) measures information available only jointly, but part of it (e.g., AND)
can be computed additively. Theory: a binary table over (class_i, class_j) can be written as [a_i + b_j > 0] exactly
when it contains no 2x2 checkerboard ("XOR square"); these are Ferrers relations / biorders (Ducamp & Falmagne 1969;
Doignon, Ducamp & Falmagne 1984). So what an additive (linear) readout cannot do should follow XOR structure, not synergy.

## Tables (tables_build.py)
- Pool of 1,368 binary 10x10 tables over class IDs (ones fraction 0.25-0.75): threshold tables [a_i + b_j > t], AND/OR of
  class attributes, threshold tables with 1-40 flipped cells, random tables, XOR-of-attributes on a sub-block.
- Predictors: XOR-square density (share of the 2,025 2x2 submatrices that are checkerboards); additive distance d_add
  (fewest flips to reach a threshold table, exact MILP); synergy: BROJA (primary), I_ccs, MMI (dit, uniform independent
  class IDs).
- Selection: MMI synergy band 0.40-0.75 (where both zero-density and XOR-dense tables exist), 4 synergy bins x 4
  density bins (0 and terciles), up to 4 tables per cell: 63 tables, 16/16 cells, Spearman(synergy, density) +0.15.

## Training (tables_train.py)
Two-view CIFAR features, view 1 from an item of class i and view 2 from a different item of class j (10 task-1 classes),
label T[i, j]; 50,000 train / 20,000 test pairs. Task-1 model as in all earlier runs (30 epochs). Linear readout on the
frozen codes [z1, z2] (logistic, 500 steps, 20,000 train pairs). Seeds 23, 24, 25.
Outcome per table (seed-mean): readout gap = (acc_full - acc_linear) / (acc_full - acc_majority). Tables with
acc_full - acc_majority < 0.10 are excluded (not learned).

## Tests
- P1 (primary, decides): partial Spearman across tables. a = rho(gap, XOR density | BROJA synergy),
  b = rho(gap, BROJA synergy | XOR density).
  - Structure wins (keep the idea): a >= 0.5 and a - |b| >= 0.3.
  - Synergy wins (drop the idea): b >= a.
  - Otherwise mixed.
- P2 (secondary): matched pairs, a zero-density table and a table with density >= 0.10, |BROJA difference| <= 0.05
  (greedy nearest match, no reuse). PASS if >= 80% of pairs have gap_zero <= 0.10 and gap_dense >= 0.40 (needs >= 5
  pairs, else INCONCLUSIVE).
- P3 (secondary, reported with an expectation): d_add predicts the gap at least as well as XOR density.
- Reported: P1 with I_ccs and MMI synergy; the ideal one-hot additive readout vs d_add; per-family means.
