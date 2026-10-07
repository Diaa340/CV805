# Pre-registration: MMBench and MMStar, head-to-head with published "synergy-driven" claims

Written 2026-10-01 before any model output on these datasets. SHA-256 in `preregistration_mm.sha256` (with
`mm_build.py`, `xiu/mm_scores.py`, `xiu/batch_vlm_final_mmbench_drop.py`, `analyze_mm.py`).

## Data
MMBench en-dev (4,329 items; random 3:1 split = 3,246 / 1,083, identical in size to Xiu et al.'s Table 1) and MMStar
(1,493 of 1,500 items whose options parse; 1,119 / 374). Model: Qwen2.5-VL-7B (Xiu et al.'s processor settings; sdpa
attention).

## Pitfall 1 (letters) and how it is handled
With letter answers and noise masking of the text, the "image-only" prediction cannot see the option contents, so
every visual question looks synergistic by construction. Our scorer keeps the option contents on the image side:
- full: image + [hint] + question + lettered options;
- txt: the same text without the image;
- img: image + lettered options only (no hint, no question).

## Tests (`analyze_mm.py`)
- MM-1 (primary, per dataset): additive recombination = conditional logit over options with
  score(o) = a * log p_img(o) + b * log p_txt(o) + letter bias, fitted on the train split, accuracy on the val split.
  Normalised fusion gap = (acc_full - acc_additive) / (acc_full - chance). PASS iff <= 0.30, i.e. the additive
  combination of the model's own unimodal predictions recovers >= 70% of its above-chance accuracy, while Xiu et al. /
  Fang et al. call these benchmarks synergy-driven / synergy-dominant.
- MM-2 (pitfall 1, reported): the same recombination using Xiu et al.'s Gaussian-noise unimodal probabilities.
- MM-3 (reported): their decision-level PID (BATCH) reproduced with their code.
