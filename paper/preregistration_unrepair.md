# Pre-registration: per-sample unrepairability across benchmarks

Written 2026-10-02 before any run. SHA-256 in `preregistration_unrepair.sha256` (with `unrepair_structure.py`,
`unrepair.py`, `exact_additive.py`, `analyze_unrepair.py`).

## Claim
Before any forgetting happens, the task's structure tells which forgotten samples a cheap (additive) repair will not
bring back: samples in non-additive cells. Already shown per task on constructed tables (test A) and per cell for one
VLM repair arm; here per sample, on four benchmarks with exact labels.

## Labels (computed before training; `unrepair_structure.py`)
- non-additive cell: the exact best additive rule of the evaluated cells (MILP, identical columns merged; POPE weighted
  by question multiplicity, 30-minute limit) gets the cell wrong. Tables: cells where T differs from its nearest
  threshold table (exact).
- local synergy-minus-redundancy of the cell (the PID-style per-sample competitor):
  log p(y) - log p(y | row) - log p(y | column) under the same cell distribution.
- low confidence: minus the task-1 model's |logit margin| (difficulty; covers the hard-sample confound).

## Protocol (`unrepair.py`)
Two-branch model on frozen features: task 1 for 30 epochs; task 2 for 10 epochs (whole model); then, on the frozen
drifted encoders, a linear head (logistic, 500 steps) and an MLP head (256 units, 2,000 steps) retrained on 20,000
task-1 training samples. Benchmarks:
- tables: 125 constructed tables (63 + 62), two-view CIFAR, task 2 = random balanced table on other classes; seed 23;
  5,000 test pairs per table.
- clevr: 103 units; symbolic image + CLIP question features; task 2 = a unit of another family; test = 10 x 300 cells.
- gqa: 24 units; CLIP; task 2 = a unit of another family; 150 / 50 images; seeds 23-25.
- pope: CLIP; task 1 = POPE questions on 375 images; task 2 = COCO cells POPE does not ask (50% yes); seeds 23-27.

## Tests (`analyze_unrepair.py`; among task-1-correct test samples lost after drift; target = not recovered by the
linear repair)
- U1: AUC of the non-additive label >= 0.70 on every benchmark.
- U2: it beats both local syn - red and low confidence by >= 0.05 AUC on >= 3 of the 4 benchmarks.
- A benchmark with < 30 lost non-additive samples is INCONCLUSIVE (then U1 fails as stated).
- Reported: recovery rates by cell type for the linear and MLP repairs; unit-bootstrap intervals; AUC within confidence
  quintiles.
- Expectation stated in advance: on constructed tables U1 is close to guaranteed by Prop. 2 (a linear head is
  additive); CLEVR, GQA and POPE are the real tests.
