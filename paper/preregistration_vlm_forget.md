# Pre-registration: do big models forget fusion first, and is it what cheap repair can't fix?

Written 2026-10-01 before the real run. SHA-256 in `preregistration_vlm_forget.sha256` (with `vlm_forget.py`,
`analyze_forget.py`). Seen before writing: one smoke run (64 cells, 20 OCR steps, 10 repair steps; discarded): base
accuracy 1.00 on those cells, no forgetting after 20 steps; this is why a heavy level (8,000 steps) was added.

## Design (`vlm_forget.py`)
- Model: Qwen2.5-VL-3B-Instruct, bf16; LoRA (r = 16, alpha = 32) on all attention and MLP projections of the language
  model.
- Cells: all 7,630 non-additive cells of the VLM scale test plus 7,630 random additive cells (seed 41), CLEVR val,
  native wording, exact answers; "non-additive" = the best additive fit of the full answer table gets the cell wrong
  (model-free).
- Drift: fine-tuning on synthetic OCR (random 1-3 strings rendered in system fonts; "What text is written in the
  image?"), lr 2e-4, batch 8; CLEVR cells evaluated after 250, 1,000, 3,000 and 8,000 steps.
- Repair (at 8,000 steps): 250 steps (batch 8, lr 1e-4) on 2,000 CLEVR *train* yes/no questions with their own images,
  training only (a) the vision-language projector (merger) or (b) the LoRA weights of the last 4 language layers.
- Answer = sign of log P(Yes tokens) - log P(No tokens).

## Analysis (`analyze_forget.py`)
- Cells correct at base; strata = base-confidence decile (|log-odds|) x question type. Matched gap = weighted mean over
  strata (>= 5 cells of each kind) of retention(non-additive) - retention(additive), weights = non-additive counts.
  95% bootstrap over images (500).
- Primary: excluding equal_integer, at the drift level whose overall retention is closest to 0.80 (among levels with
  retention in [0.50, 0.95]; none -> INCONCLUSIVE):
  - FUSION FORGOTTEN FIRST iff gap <= -0.03 and upper bound < 0;
  - KILL iff upper bound >= 0 and gap > -0.03 (then the forgetting story does not transfer to scale);
  - otherwise MIXED.
- Repair (secondary): among cells lost at 8,000 steps, matched share regained (additive - non-additive) >= 0.05 with
  lower bound > 0 -> "additive recovers more", per arm.
- Reported: equal_integer separately; raw rates by type and level.
