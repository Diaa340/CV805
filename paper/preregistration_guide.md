# Pre-registration: (1) VLM on a six-task CLEVR sequence; (2) leave-one-benchmark-out check of a method guide

Written 2026-10-03. SHA-256 in `preregistration_guide.sha256` (with `vlm_seq_clevr.py`, `vlm_seq.py`,
`analyze_clevr_vlm.py`, `analyze_guide.py`). Seen before writing: every earlier result (two-task study, CLOVE small and
VLM studies, same-answer re-analysis, buffer control) and a CLEVR VLM smoke run (6 steps per task; not used).
`analyze_guide.py` was run once to check that it executes, printing only "RUNS OK" (no numbers seen).

## Positioning
Analytical: which CL method to use for which data and model, judged on the samples that need fusion (exceptions).

## Part 1: VLM on CLEVR (`vlm_seq_clevr.py`)
- Qwen2.5-VL-3B-Instruct + LoRA, same learner and budget as the CLOVE VLM study. Six tasks = the 40 CLEVR units grouped
  by question family; two task orders (seeds 60, 61), seed 31. Natural cells only (question well-posed on the image
  and its natural answer equal to the table's); per task 500 exception + 500 rule test cells.
- Methods: finetune; EWC (100, 10000; matched plasticity); LwF (1); ER; DER++; class-balanced ER; LoRA-expansion
  (architectural analogue of DER-expansion: a new LoRA per task, earlier ones frozen but active, ER memory).
- Same-answer metrics (`analyze_clevr_vlm.py`; bootstrap over the 40 units):
  - VC1: ER and DER++ dBWT_s <= -0.03 with upper bound < 0 (both) — does the small-model CLEVR finding hold for a VLM?
  - VC2: DER++ dFWT_s <= -0.03 with upper bound < 0.
  - VC3: LoRA-expansion's dBWT_s exceeds ER's (lower bound of the difference > 0) — does the architectural
    recommendation transfer?
  - Reported: everything by type and answer, zero-shot accuracy by type.

## Part 2: method guide, leave-one-benchmark-out (`analyze_guide.py`; saved results only)
- Observations: units of the two-task study (tables, CLEVR, GQA) and (run, unit) pairs of CLOVE (tasks 1-5); twelve
  standard methods. Target: final accuracy on the unit's exception cells (secondary: type-balanced final accuracy).
- Descriptors (data only): exception share, answer skew (yes share of exceptions minus rule), yes share, exception
  accuracy right after learning. Guide: per-method ridge regression on the descriptors, fit on three benchmarks,
  predicting the fourth. Baselines: each method's mean target on the three (no descriptors); the usual ranking by
  overall BWT.
- G1: the guide's per-observation Spearman with the actual method ordering exceeds the mean-ranking baseline's
  (bootstrap lower bound > 0) on >= 3 of 4 held-out benchmarks.
- G2: the exception-aware mean ranking exceeds the overall-BWT ranking (lower bound > 0) on >= 3 of 4.
- Reported: regret of the picked method, picks per benchmark, coefficients of the guide fit on all four.

## Status at hashing (2026-10-03)
Part 1 (VLM on CLEVR) is ON HOLD and not run: the smoke run showed zero-shot accuracy at ceiling on the natural CLEVR
cells (rule 0.975, exceptions 0.983 on 240 cells), so the planned sequence would teach the model nothing new. A
zero-shot check on the full test sample is being run (descriptive); a revised design will get its own addendum.
Part 2 (guide) is hashed and run now.
