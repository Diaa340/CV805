# Pre-registration: EMAP (Hessel & Lee 2020) against exact ground truth, on benchmarks, and on label tables

Written 2026-10-01 before any EMAP output. SHA-256 in `preregistration_emap.sha256` (with `emap.py`, `analyze_emap.py`).
EMAP implementation checked against brute force (max error 1e-6) and on an exactly additive function (unchanged).

## Why
EMAP asks whether a *trained model* uses cross-modal interactions (its additive projection over cross pairs); our trained
additive model estimates the *task's* additive ceiling. The paper needs both.

## Parts (`emap.py`)
- VLM (primary): Qwen2.5-VL-7B, zero-shot, on the 24 CLEVR units whose questions are well-posed on every image (count
  comparisons and exist families; native answer = exact table on every cell). Per unit, a complete 10-question x
  100-image table (100 held-out images of the step-2 split). EMAP on the log-odds. Compared with the VLM's accuracy, the
  best additive fit of the same sub-table (logistic, as in step 2), and EMAP of the truth table.
- Benchmarks: frozen-feature full models (same training as `additive_estimator.py`, seeds 23-25) on the 7 benchmarks;
  EMAP on 2,000 random test items; compared with the trained-additive gap already measured.
- Tables: full models on the 63 label tables (same data and model as the label-table test, seeds 23-25); EMAP on 2,000
  test pairs.

## Tests (`analyze_emap.py`)
- M1 (primary): EMAP of a strong model recovers the exact ceiling: MAE(EMAP(VLM), ceiling) <= 0.05 and Spearman >= 0.6.
- M2 (primary): the VLM uses interactions: mean(VLM - EMAP(VLM)) >= 0.05 with bootstrap (units) lower bound > 0.
- B1: on >= 6 of 7 benchmarks, |full - EMAP| <= 0.02 and within 0.02 of the trained-additive gap.
- T1: on the tables, the normalised EMAP gap follows structure: partial rho(gap, d_add | BROJA) >= 0.5 and exceeds
  |partial rho(gap, BROJA | d_add)| by >= 0.3.
- Reported: equal_integer separately; EMAP gap vs linear-readout gap on the tables.
