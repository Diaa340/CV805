# Pre-registration: VLM scale test (do a large VLM's errors concentrate on non-additive cells?)

Written 2026-10-01 before any VLM output. SHA-256 in `preregistration_vlm.sha256` with hashes of `vlm_test.py`,
`analyze_vlm.py`. Runs after step 2 (queued). GPU 2 only.

## Motivation
Small additive readouts fail on XOR structure by construction. Large VLMs can represent any interaction, so the open
question is whether they still *fail specifically* where the task is non-additive, or whether errors are unrelated to
structure.

## Design
- Model: Qwen2.5-VL-7B-Instruct (bf16), zero-shot, no training. Prompt: "<image> {CLEVR question} Answer with yes or no."
  Answer = sign of log P(Yes-tokens) - log P(No-tokens) for the next token.
- Cells: the 103 step-2 CLEVR units (10 questions x 1,000 val images each, exact set-lifted answer tables). A cell is
  **non-additive** if the best additive fit [u_q + w_img + b > 0] of the full table (logistic, as in step 2) gets it
  wrong. Only cells where the native CLEVR question is well-posed and its answer equals the set-lifted answer are
  asked (so the natural wording is correct). Up to 250 such cells per unit, sampled at random (seed 31).
- Smoke test before the real run: at most 1 unit, reported but not analysed.

## Tests
- V0 (validity): balanced accuracy >= 0.60, else INCONCLUSIVE.
- V1 (primary): logistic regression of error on the non-additive indicator with question and image fixed effects
  (C = 10); odds ratio with 95% bootstrap interval over images (200 resamples).
  - "Errors concentrate on non-additive cells" iff OR >= 1.5 and lower bound > 1.0.
  - "No specific failure" iff the interval lies within [0.67, 1.5].
  - Otherwise mixed.
- Reported: raw error rates, by final function, and confidence (|log-odds|) on the two cell types.

## Known limitation (stated in advance)
Non-additive cells are the cells that the question's and the image's main effects predict wrongly. A model that answers
by row/column priors (an additive shortcut) errs there, but so might a model with perception noise near decision
boundaries (e.g. count ties). V1 cannot fully separate these; a positive result means "errors concentrate where the task
needs fusion", not a proof of the mechanism.
