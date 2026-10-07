# Pre-registration: identical PID, different degree of non-additivity

Written 2026-10-01 before any training on these tables. SHA-256 in `preregistration_samepid.sha256` (with hashes of
`samepid_build.py`, `tables_train2.py`, `analyze_samepid.py`).

## Why
For complete, uniformly sampled tables, additivity (yes / no) is decided by the row and column label counts (Ryser;
Ferrers), and BROJA is computed from those counts. So no two tables with identical PID can differ in *whether* they are
additive. They can differ in *how far* from additive they are. Margin-preserving interchanges change the arrangement but
not a single row or column count, so every BROJA / MMI atom is identical within a family.

## Tables (`samepid_build.py`, seen before writing)
12 random 10x10 tables (label share 0.30-0.70). For each, a search over interchanges minimises and maximises the exact
additive distance d_add: 3 arrangements per family (min, orig, max). Within-family d_add spans 0.03-0.07 (e.g.
0.12 / 0.14 / 0.17); MMI synergy is identical to 3 decimals. BROJA and I_ccs are computed with dit and reported.

## Training
As in the label-table test (`tables_train2.py`, two-view CIFAR, 30 epochs, linear readout on frozen codes), seeds 23-27.
Gap = (acc_full - acc_linear) / (acc_full - acc_majority), seed mean per table.

## Test (PASS iff all three)
- within-family Spearman(d_add, gap) after centring both within family >= 0.4;
- gap(max) - gap(min) > 0 (one-sided 95% t lower bound over families > 0);
- gap(max) - gap(min) >= 0.05 in >= 75% of families.
