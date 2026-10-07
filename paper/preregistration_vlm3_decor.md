# Pre-registration: structure vs PID on DECORRELATED natural sub-tables (three VLMs)

Written 2026-10-04 (afternoon), before the full three-model analysis of preregistration_vlm3.md. SHA-256 in
`preregistration_vlm3_decor.sha256` (with `vlm3_subsets.py`, `vlm3_select.py`, `analyze_vlm3_decor.py`).

## Why
An interim look at InternVL3.5 (analysis of preregistration_vlm3.md run on that one model) showed structure predicting
the EMAP drop within CLEVR (+0.91) and CLOVE (+0.76), but PID almost as well (+0.90, +0.70): within each benchmark the
exception share and BROJA synergy are nearly collinear (Spearman +0.95 CLEVR, +0.90 CLOVE, +0.88 GQA, +0.48 POPE). The
structure-vs-PID question cannot be answered on those units. The earlier label-table study answered it by choosing
tables where the two are uncorrelated; this does the same on natural data.

## Seen before writing
InternVL3.5's full-grid results (all of them). Not seen: any Qwen2.5-VL or LLaVA-OV-1.5 EMAP value or accuracy beyond
the one-unit smoke runs. No candidate sub-table or selection result has been computed yet.

## Design
- Candidates (`vlm3_subsets.py`, outcome-free): from each of the 99 grids, 27 sub-tables of 50 images, drawn with
  weights tilted toward or away from columns rich in exceptions or in "yes" answers; exact rule (MILP 300 s), BROJA
  (convex solver, SCS), MMI for each.
- Selection (`vlm3_select.py`, outcome-free): one sub-table per grid, chosen per benchmark by coordinate descent to
  minimise |Spearman(s, BROJA)| while keeping the spread of s at >= half the candidate pool's. The achieved within-
  benchmark correlations are reported; the target is |rho| <= 0.3.
- Analysis (`analyze_vlm3_decor.py`): the tests T1-T4 of preregistration_vlm3.md, unchanged, on the selected sub-tables;
  EMAP recomputed on each sub-table from the saved log-odds (no new model runs).
- Confirmatory models: Qwen2.5-VL-7B and LLaVA-OneVision-1.5-8B. A test passes iff it passes for both (T3: fitted on
  these two). InternVL3.5 is reported as exploratory.
- If the selection cannot reach |rho| <= 0.3 in a benchmark, the tests are still run and the achieved correlation is
  stated next to every result from that benchmark.
