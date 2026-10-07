# Pre-registration: six-task CLOVE-scene continual learning with a VLM (Qwen2.5-VL-3B + LoRA)

Written 2026-10-03 before the full VLM runs. SHA-256 in `preregistration_clove_vlm.sha256` (with `vlm_seq.py`,
`analyze_clove_vlm.py`). Seen before writing: all small-model results, including the CLOVE small-model study
(results_clove_cl.txt: C1, C2, C4 PASS; C3 FAIL on its overall-BWT bound), and a VLM smoke run (6 steps per task,
40 test cells per task; not used).

## Why
To check that the rule / exception pattern is not specific to small two-branch networks on frozen features.

## Design (`vlm_seq.py`)
- Model: Qwen2.5-VL-3B-Instruct, LoRA r 16 (alpha 32) on all language-model projections; vision tower frozen.
  Images 336 x 336. Prompt: "<question> Answer with yes or no." Score: log-odds of yes vs no at the next token;
  logistic loss.
- Tasks and cells: the same 43 CLOVE-scene units and exact labels as the small-model study. Test cells per task: all
  exception cells plus 600 rule cells (fixed sample, seed 51).
- Training: each task 300 steps x 8 examples, lr 1e-4 (AdamW). Task 1 shared; methods for tasks 2-6: finetune;
  online EWC (100, 10,000; matched plasticity as before); LwF (1; teacher scores on the examples each task trains on, computed before the task); ER,
  DER++ (alpha 0.5 on stored scores), DER++ rule+exceptions: memory of 1,000 split equally over past tasks, 4 replayed
  examples per step. Task-only LoRA (300 steps) per task for forward transfer. Zero-shot accuracy also recorded.
- Runs: CLOVE task orders 0 (abcdef) and 1 (bdfcae), seed 31.

## Tests (bootstrap over the 43 units)
- V1: for LwF, ER and DER++, dBWT = BWT_E - BWT_R <= -0.03 with upper bound < 0, for >= 2 of 3.
- V2: DER++ dFWT <= -0.03 with upper bound < 0.
- V3: DER++ rule+exceptions vs DER++: exception BWT gain >= 0.03 with lower bound > 0, and overall BWT difference
  lower bound > -0.02 (the same bound the small-model study narrowly failed).

Smoke run (seen): pipeline works; LwF teacher scoring of all training cells took ~45 min, so the teacher now scores
only the examples drawn for training (drawn up front). Zero-shot on the 40 smoke cells per task (all exception cells,
91% yes): 0.82 correct.
