# Pre-registration: six-task continual learning on CLOVE-scene, by sample type

Written 2026-10-03 before the full runs. SHA-256 in `preregistration_clove_cl.sha256` (with `cl_seq.py`,
`analyze_clove_cl.py`). Seen before writing: all results of the two-task study (results_clm.txt: constructed tasks,
CLEVR, GQA, POPE), the CLOVE build (43 existence units, 6-8 per scene task, exact rules for all 86 tables; ~6%
exception cells; ~13% yes), and two smoke runs with 2 / 1 epochs (not used).

## Design (`cl_seq.py`)
- Tasks: CLOVE-scene's six scene tasks (ShopAndDining, Workplace, HomeOrHotel, Transportation, SportAndLeisure,
  Outdoors), each the union of its units' cells: CLOVE train images for training (50,000 draws), CLOVE val images for
  testing (all cells). Inputs: CLIP image and question features. Answer: yes / no.
- Runs: CLOVE's six task orders (abcdef and the five sampled orders it releases) x seeds 31, 32 = 12 runs.
- Task 1: 30 epochs, shared by all methods; tasks 2-6: 10 epochs each. Methods: finetune; online EWC (1, 100, 10,000);
  SI (0.1, 1, 10, 100); LwF (0.3, 3); feature distillation (0.3, 3); ER, A-GEM, GEM, DER++ (random memory of 1,000
  split equally over past tasks); DER++ rule+exceptions; iCaRL (herding, distillation 0.1 / 1, nearest-mean);
  DER-expansion (new encoders per task, older ones frozen, ER memory); multi-head (task identity given at test).
  Matched plasticity for EWC / SI / LwF / distillation / iCaRL as before, per run, on own-task accuracy of tasks 2-6.
- Per test sample: correct right after its own task (own), after task 6 (final), and for a model trained on that
  task alone for 10 epochs (scratch).

## Measures (pooled over samples; R = rule cells, E = exception cells)
- BWT_g = mean over tasks 1-5 of (final - own) on type g; dBWT = BWT_E - BWT_R (primary; the kept share is not
  used because answers are ~87% "no", so 0.5 is not chance).
- FWT_g = mean over tasks 2-6 of (own - scratch) on type g; dFWT = FWT_E - FWT_R.
- 95% intervals: bootstrap over the 43 units (all runs kept together).

## Tests (confirmatory: each repeats a pattern seen on constructed tasks / CLEVR / GQA)
- C1: for LwF, ER and DER++, dBWT <= -0.03 with upper bound < 0, for >= 2 of the 3.
- C2: DER++ dFWT <= -0.03 with upper bound < 0.
- C3: DER++ rule+exceptions vs DER++: exception BWT gain >= 0.03 with lower bound > 0, and overall BWT difference
  lower bound > -0.02.
- C4: DER-expansion's dBWT exceeds ER's (lower bound of the difference > 0).
- Reported: the full table, family means, accuracy by type after task 6.
