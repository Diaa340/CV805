# Pre-registration: GQA (exact tables on real photos with real scene graphs)

Written 2026-10-01 before any model output on GQA. SHA-256 in `preregistration_gqa.sha256` (with `gqa_exec.py`,
`gqa_build.py`, `gqa_vlm.py`, `gqa_train.py`, `exact_additive.py`, `analyze_gqa.py`).

## Tables (outcome-free; `gqa_build.py`, seen before writing)
- Executor for GQA existence programs (select / filter / exist / and / or), object ids ignored so a question runs on
  any scene graph. Families kept if it reproduces GQA's answer on the question's own image for >= 99% of questions
  (14 families; existAnd 91% and the "not" variants 0% are excluded). Questions kept if it agrees on their own image and
  no selected name is a superordinate category (hypernym list in the script).
- Units: up to 60 distinct programs per family in groups of 10; per unit 200 val images sampled from those containing
  at least one object named by the unit's questions (as POPE samples images where objects plausibly occur). Kept if
  yes-share in [0.10, 0.90]: 24 units (exist, existC, existOr, existOrC; 6 each), yes 11-12%, exact MMI synergy
  0.38-0.50 bits. Relational families are excluded because they presuppose a definite referent.

## Ceiling
The exact additive ceiling is computed by MILP with identical columns merged (exact; replaces the logistic fit, which is
only a lower bound). Balanced-accuracy ceilings use class-balanced cell weights.

## Tests (`analyze_gqa.py`)
- G1: trained additive model on CLIP features (images 150 / 50, seeds 23-25) vs exact ceiling on the test cells:
  PASS iff MAE <= 0.03.
- G2 (primary): the VLM uses interactions: balanced VLM - EMAP(VLM) >= 0.05 with bootstrap lower bound > 0 ("VLM
  fuses"); upper bound < 0.02 -> "no fusion"; else mixed.
- G3: EMAP of the VLM recovers the exact balanced ceiling: MAE <= 0.05 and Spearman >= 0.6.
- Resume (2026-10-02): gqa_vlm.py ran out of GPU memory after 6 units while sharing GPU 2; restarted, skipping completed units (no other change). New hash: 82de0b84608ae78905b2ce8075eafdfec1166704cef1680fccf5f734ee7dd2ec  gqa_vlm.py
