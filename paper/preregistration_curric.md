# Pre-registration: do exception curricula help models learn exceptions?

Written 2026-10-23 before any result. SHA-256 in `preregistration_curric.sha256` (with `curric.py`, `vlm_curric.py`,
`analyze_curric.py`). Seen before writing: one-unit smoke runs of both scripts (output shapes only) and the analysis
run on those smoke files (printing only that it runs). Motivation: exceptions are learned worse than same-answer,
same-difficulty rule cells (difficulty-matched forward gaps on CLEVR), and a VLM with LoRA failed to learn the
exceptions of new-word tasks. Earlier, PID-ordered curricula failed and synergy up-weighting equalled generic
hard-sample weighting, so an exception curriculum must beat model-based hard-example methods.

## Step 1: small fusion network, single task (`curric.py`)
Constructed tasks (93), CLEVR (40 units), GQA (24 units x 3 seeds): same data, network and budget as the first task of
the CL study. Conditions from the same initialisation: random; misfit-weighted sampling (weight 1 + 4 q, q = percentile
of the cell's graded misfit in the training answer table; model-free); easy-first and hard-first pacing by misfit;
focal loss (gamma 2); loss-based selection (1,024 candidates, 256 kept in proportion to loss). Test cells on held-out
images; exception labels as saved.

## Step 2: VLM (`vlm_curric.py`)
Qwen2.5-VL-3B + LoRA trained on each nonce-word CLEVR task alone: uniform_300, misfit_300, loss_300 (16 candidates, 8
kept in proportion to loss), uniform_600, misfit_600; 500 exception + 500 rule test cells per task; zero-shot reported.

## Measures and tests (`analyze_curric.py`; bootstrap over units)
Exception accuracy and rule-cell accuracy, each averaged over answers with the exceptions' answer mix; overall
accuracy; differences in points.
- C1: misfit vs random: exceptions >= +3 points with lower bound > 0, and overall accuracy lower bound > -1 point; on
  >= 2 of 3 benchmarks.
- C2: misfit vs focal AND vs loss-based selection: exceptions >= +2 points with lower bound > 0 against both; on >= 2 of
  3 benchmarks.
- V1: misfit_300 vs uniform_300: exceptions >= +3 points with lower bound > 0, and rule-cell lower bound > -2 points.
- V2: misfit_300 vs loss_300: exception difference lower bound > 0.
- Reported: pacing conditions, 600-step conditions, zero-shot.
