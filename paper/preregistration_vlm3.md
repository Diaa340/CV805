# Pre-registration: does task structure predict the interaction dependence of modern VLMs? (three families x four benchmarks)

Written 2026-10-04 before the full scoring runs. SHA-256 in `preregistration_vlm3.sha256` (with `vlm3_build.py`,
`vlm3_refine.py`, `vlm3_pid.py`, `vlm3_score.py`, `analyze_vlm3.py`).

## Claim tested
Structural non-additivity of the answer table predicts the interaction dependence required to sustain a VLM's task
performance; PID synergy does not. (Not: "predicts whether a model uses interactions"; a model may use interactions on
an additive task.)

## Quantities (computed on the identical, fully labelled cells)
- Structural score s_u = exception share of the exact additive rule = the minimum error of any additive decision rule
  on unit u's answer table (model-free).
- EMAP drop Delta_{m,u} = Acc(f_m) - Acc(EMAP(f_m)) on the same cells (Hessel & Lee 2020; EMAP of the yes / no log-odds).
- PID synergy: BROJA (Liang et al.'s convex solver, SCS; matches dit to <= 0.0003 bits on six 10 x 20 checks); MMI
  reported.
- Bound (theory, not a test): EMAP's hard prediction is additive, so its error >= s_u; hence Delta >= s_u - e_{m,u}.
  For accurate models part of the s-Delta relation is guaranteed; the tests below target what is not.

## Units (`vlm3_build.py`, outcome-free)
99 complete 10-question x 100-image tables, every question well-posed on every image (no mismatched pairs):
CLEVR 24 (well-posed families), GQA 24 (existence), CLOVE-scene 43 (existence), POPE 8 (10 COCO objects each, POPE's
question). Exact rules: 94 proven at 600 s; the 5 others are re-solved with a 1-hour limit (`vlm3_refine.py`); any
still unproven use the solver's proven lower bound in the bound check. POPE's s range is narrow (0.021-0.047), so
within-POPE tests are expected to be weak.

## Models (`vlm3_score.py`, zero-shot)
Qwen2.5-VL-7B-Instruct; InternVL3.5-8B (HF port, one 448 tile); LLaVA-OneVision-1.5-8B-Instruct (publisher code, pinned
revision e1379105, run with transformers 4.53 in `.venv_llava15`; the code was read: only torch / transformers
imports). Images scaled to a longest side of 448; prompt "<question> Answer with yes or no."; score = log-odds of yes
vs no at the next token.

## Seen before writing
Unit structure summaries (s by benchmark); BROJA values; smoke runs on one unit (CLEVR 0_0, an equal-integer count unit):
accuracy Qwen 0.550, InternVL 0.991 (13 tiles) / 0.973 (one tile), LLaVA 0.988. No EMAP value was computed. The analysis
was checked on random fake scores (printing only that it runs).

## Tests (`analyze_vlm3.py`; 95% bootstrap over units)
- T1 (within benchmark, each VLM): Spearman rho(Delta, s) with lower bound > 0 on >= 3 of 4 benchmarks; PASS iff for all
  three models.
- T2 (partial, within benchmark): per model, ranks within benchmark, partial rho(Delta, s | PID, accuracy) >= 0.3 with
  lower bound > 0, and exceeding |partial rho(Delta, PID | s, accuracy)| by >= 0.2; PASS iff for all three models.
- T3 (regression, all models): Delta ~ s + PID + accuracy + model + benchmark effects (standardised): beta_s lower
  bound > 0; beta_PID's interval contains 0 or |beta_PID| < |beta_s| / 3; R^2 added by s after PID >= 0.05; R^2 added by
  PID after s <= 0.01.
- T4 (leave one benchmark out, each VLM): fit Delta ~ s + accuracy on three benchmarks; within the held-out one,
  Spearman(predicted, actual) > 0 and greater than with PID in place of s, on >= 3 of 4; PASS iff for all three models.
- Reported: the bound check (must hold for every model-unit pair), accuracy on exceptions vs rule cells, the cell-level
  AUC for EMAP-flipped cells (exception label, graded misfit, low confidence), and the scatter figure.
