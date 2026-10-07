# Pre-registration: limitations of CL methods for non-additive (XOR) cross-modal knowledge, and a fix

Written 2026-10-02 before the real run. SHA-256 in `preregistration_clx.sha256` (with `clx_structure.py`,
`cl_exceptions.py`, `exact_additive.py`, `analyze_clx.py`).

## Idea
A task's answer table = an additive "rule" (row + column effects, about 2K numbers) + "exceptions" (non-additive /
XOR cells). Any data touching each row and column pins the rule; an exception is pinned only where it is covered.
Prediction: CL methods protect the rule almost for free and the exceptions only selectively (LwF distils on new-task
inputs, which never visit old exceptions; replay protects an exception only if its cell is buffered). Fix: store the
exceptions plus a few rule samples.

## Seen before writing (smoke test, 2 tables, discarded; `runs/clx/smoke_*`)
Heavy forgetting under finetune / EWC / L2 (retention about 0.5, chance level); LwF at lambda 3 kills plasticity
(task-2 accuracy 0.51); a pure exception buffer kept 88% of non-additive cells but cut additive retention to about
0.46 (random buffer about 0.66), so the theory's actual prescription, rule + exceptions, is pre-registered as the fix;
at buffer 250 a random buffer already covers every table cell, so a 50-sample buffer is added for the coverage test.

## Design (`cl_exceptions.py`)
Two-branch model on frozen features; task 1 for 30 epochs (per-epoch evaluation of a 10,000-sample candidate pool to
count forgetting events); from the same checkpoint, task 2 for 10 epochs with each method:
finetune; EWC (lambda 1, 100, 10,000); LwF (0.3, 3); feature distillation L2 (0.3, 3); ER (32 buffer samples per step)
with buffers random, herding (iCaRL), low-confidence, forgetting-events (Toneva-style; ties by low confidence),
exception-cell, rule+exceptions (half exception-cell round-robin over cells, half random), each at 50 / 250 / 1,000;
DER++ (alpha 0.5) with random and rule+exceptions at 250 / 1,000. Exception labels come from the exact best additive
rule of the *training* cells (tables: nearest threshold table; CLEVR: MILP on 10 questions x 700 train images,
`clx_structure.py`). Evaluated on task-1 test samples (test-cell exact labels).
- tables: all constructed tables (63 + 62) with d_add > 0; two-view CIFAR; task 2 = random balanced table on other
  classes; seed 23; 5,000 test pairs.
- clevr: 40 units stratified by final function (seed 5); symbolic image + CLIP question features; task 2 = a unit of
  another family; test = 10 x 300 cells.
EWC / LwF / L2 strength per unit: the best-retention lambda with task-2 accuracy >= finetune - 0.02 (else the smallest).

## Tests (`analyze_clx.py`; task-1-correct test samples)
- L1 (limitation): matched gap = retention(non-additive) - retention(additive) within unit x confidence-decile strata.
  A method shows the limitation iff gap <= -0.03 and its 95% unit-bootstrap upper bound < 0. Methods: finetune, EWC,
  LwF, L2, ER-random-1000, DER++-random-1000. PASS iff >= 4 of 6 on both benchmarks; KILL iff 0 on both; else MIXED.
- L2a (mechanism, tables): ER-random-50: (covered - uncovered retention) on non-additive minus the same on additive
  samples >= 0.07, lower bound > 0.
- L2b (mechanism, both): LwF's gain over finetune on additive minus its gain on non-additive samples >= 0.05, lower
  bound > 0.
- L3 (fix, both, buffer 250): ER rule+exceptions vs the best of {random, herding, low-confidence, forgetting-events}:
  overall retention >= +0.02 and non-additive retention >= +0.05 (lower bounds > 0), task-2 accuracy within 0.02.
- Reported: buffers 50 / 1,000, pure exception buffer, DER++ rule+exceptions vs random.
