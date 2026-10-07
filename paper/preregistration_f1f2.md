# Pre-registration: confirmatory tests of the two post-hoc findings (F1, F2)

Written 2026-10-02 before any run. SHA-256 in `preregistration_f1f2.sha256` (with `clx2_structure.py`,
`cl_exceptions2.py`, `unrepair2.py`, `analyze_f1f2.py`). Both findings were seen in earlier runs (results_clx.txt,
results_unrepair.txt); the thresholds below were set after seeing them, so the tests use fresh data only.

## F1. DER++ with a rule + exceptions buffer keeps exceptions without losing the rule
Seen before: CLEVR (40 units, seed 23), buffer 1,000: exceptions +0.120, overall -0.003; buffer 250: +0.214, -0.013.
Constructed tasks: 1,000: +0.063, -0.014; 250: +0.116, -0.053.
Fresh data: seed 24 for all 93 constructed tasks with exceptions; 40 NEW CLEVR units (not used before; stratified,
seed 6) with seed 24. Arms: finetune, ER random, DER++ with random / herding / forgetting-events / rule+exceptions
buffers at 250 and 1,000 (`cl_exceptions2.py`, otherwise identical to the first study).
- Primary (buffer 1,000, per benchmark): exception retention gain over DER++ random >= 0.05 with 95% unit-bootstrap
  lower bound > 0, overall retention non-inferior (lower bound of the difference > -0.02), task-2 accuracy within 0.02.
  F1 PASS iff it holds on both benchmarks.
- Reported: buffer 250; comparison with the best other DER++ buffer.

## F2. The graded additive misfit predicts which forgotten samples cheap repair cannot recover
Seen before (first run): AUC 0.79 / 0.95 / 0.99 / 0.69 (tables / CLEVR / GQA / POPE) vs confidence 0.70 / 0.77 / 0.63
/ 0.66. Fresh data: the same protocol (`unrepair2.py`) with new seeds: tables and CLEVR 24, GQA 26-28, POPE 28-32.
Misfit = local synergy-minus-redundancy of the cell = -log of the additive (row + column) model's score for the true
answer.
- F2a: AUC >= 0.75 on >= 3 of 4 benchmarks.
- F2b: AUC exceeds the low-confidence AUC by >= 0.05 with unit-bootstrap lower bound > 0 on >= 3 of 4 benchmarks.
- Reported: AUC within confidence quintiles.
