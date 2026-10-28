# Pre-registration: null-scorer control for the VLM structure study

Written 2026-10-17 before computing any null-scorer or interaction-strength value. SHA-256 in
`preregistration_vlm3_null.sha256` (with `analyze_vlm3_null.py`).

## Why
A critique: a null scorer reproduces most of the relationship between exception share and EMAP drop, so that
correlation does not show VLM-specific cross-modal reasoning. Our floor (Delta >= s - e) predicts exactly this for any
accurate scorer. The question is whether VLMs show interaction beyond what accuracy on non-additive tables forces.

## Seen before writing
All earlier VLM results (studies A-C, accuracy, EMAP drops). No null scorer or interaction-strength value had been
computed. The analysis script was checked on random fake scores (printing only that it runs).

## Data and measures (`analyze_vlm3_null.py`; saved log-odds, no new model runs)
- The 99 full answer tables (10 x 100) and the four VLMs' zero-shot log-odds (clipped to [-10, 10]).
- Interaction strength IS: share of log-odds variance outside the additive projection (row + column effects).
- EMAP drop Delta, as before.
- Answer-matched null scorer (primary): true answer +/- kappa_y plus N(0, 1) noise per cell, kappa_y set so that the
  expected accuracy on yes cells and on no cells equals the VLM's on that table; 200 draws, values averaged. A simple
  null with one kappa from overall accuracy is reported.
- Excess = VLM value minus answer-matched null value.

## Tests (within-benchmark ranks, accuracy as control, 95% bootstrap over tables)
- N1 (primary): partial rho(excess IS, s | accuracy) has lower bound > 0, for >= 3 of the 4 models.
- N2: partial rho(excess Delta, s | accuracy) has lower bound > 0, for >= 3 of the 4 models.
- Reported: mean IS and Delta for VLMs and both nulls; how much of rho(Delta, s) the null reproduces; per-benchmark
  correlations.

## Interpretation fixed in advance
- N1 passes: VLMs carry more interaction than accuracy requires, and more so where tables have more exceptions.
- N1 fails: the structure-drop relation is what any accurate scorer must show (the floor). The paper then presents the
  floor as the result, the VLMs as matching it, and makes no claim of VLM-specific interaction beyond accuracy.
