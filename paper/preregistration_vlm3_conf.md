# Pre-registration: confirmatory test of "structure, not PID" with a new VLM and answer balance held fixed

Written 2026-10-04 (evening). SHA-256 in `preregistration_vlm3_conf.sha256` (with `vlm3_select2.py`, `vlm3_score.py`,
`analyze_vlm3_conf.py`).

## Why
On decorrelated sub-tables (preregistration_vlm3_decor.md) all pre-registered tests failed: structure predicted the EMAP
drop independently of PID in all three models (partial rho +0.32 to +0.37), but PID kept a signal of its own. A
post-hoc control for the yes share removed PID's signal (partial rho -0.07 / +0.19 / +0.11, intervals containing 0)
and strengthened structure's (+0.37 to +0.48). Theory predicts this (BROJA depends only on each modality's answer
statistics, so it tracks answer balance). This test confirms or refutes that post-hoc finding on data not used to find it.

## Seen before writing
Everything for Qwen2.5-VL-7B, InternVL3.5-8B and LLaVA-OV-1.5-8B. The new selection (`vlm3_select2.py`, outcome-free,
answer tables only) was run before writing: within-benchmark Spearman(s, BROJA) / (s, yes) = CLEVR +0.02 / -0.00, CLOVE
-0.00 / +0.00, GQA +0.20 / -0.00, POPE +0.02 / +0.00; spread of s kept (>= half the candidate pool's). Idefics3: one
smoke unit (CLEVR 0_0, accuracy 0.685); the analysis was checked on random fake Idefics scores (printing only that it
runs). `vlm3_score.py` changed only by adding the Idefics3 entry (one image tile, like InternVL).

## Design
- Confirmatory model: Idefics3-8B-Llama3 (Llama-3 language model, SigLIP vision; a family not used before), zero-shot,
  same input and score as the other models. Scored on all 99 grids; analysed on the 99 selected 10 x 50 sub-tables.
- Units: one sub-table per grid with structure uncorrelated with both PID (BROJA) and the yes share within benchmark.

## Tests (`analyze_vlm3_conf.py`; within-benchmark ranks; 95% bootstrap over units)
- C1 (primary): partial rho(Delta, s | PID, accuracy, yes share) >= 0.3 with lower bound > 0.
- C2 (primary): partial rho(Delta, PID | s, accuracy, yes share) has an interval containing 0 and |rho| <= 0.15.
- CONFIRMED iff C1 and C2 pass for Idefics3.
- C3 (secondary): regression Delta ~ s + PID + yes + accuracy + benchmark effects: beta_s lower bound > 0, R2 added by
  s >= 0.05, R2 added by PID <= 0.01 (same thresholds as before).
- Reported: within-benchmark correlations, the bound check, and the same statistics for the three earlier models
  (exploratory).
