# Pre-registration: continual-learning metrics by sample type, across method families and four benchmarks

Version 2, written 2026-10-02 before any continual-learning run. Version 1 (hash c1b0f00f, same day,
`preregistration_clm_v1_stopped.sha256`) was stopped before any CL run started (only the cell labels from
`clm_structure.py` had been computed). Version 2 adds methods at the user's request. A smoke test on two constructed
tasks found a bug (the task-2-only model was trained without gradients) and fixed it; that smoke output is not used.
SHA-256 in `preregistration_clm.sha256` (with `clm_structure.py`, `cl_metrics.py`, `analyze_clm.py`). Seen before
writing: the first CL study (constructed tasks, CLEVR; results_clx.txt), F1 / F2, and the smoke output (one table).

## Metrics (per sample type g: R = rule cells, E = exception cells)
- BWT_g = task-1 accuracy on type g after task 2 minus before.
- Kept share KS_g = (Acc_after - 0.5) / (Acc_before - 0.5): the share of above-chance task-1 knowledge kept (undefined
  when Acc_before < 0.55). KS gap = KS_E - KS_R (< 0: exceptions lost more).
- FWT_g = task-2 accuracy on type g after sequential training with the method minus that of a model trained on task 2
  alone with the same budget (10 epochs from scratch). FWT gap = FWT_E - FWT_R.
- Type-balanced ACC = mean over the two tasks of (Acc_R + Acc_E) / 2.

## Methods, by family (`cl_metrics.py`)
- none: finetune.
- regularisation, parameter: EWC (1, 100, 10,000); SI (c = 0.1, 1, 10, 100; damping 0.1; negative importances set to 0).
- regularisation, distillation: LwF (0.3, 3); feature distillation (0.3, 3).
- memory: ER, A-GEM (reference batch 256), GEM (whole memory; with one previous task GEM's QP is a single projection),
  DER++ (alpha 0.5), all with a random 1,000-sample buffer; iCaRL adapted to this setting: herding buffer of 1,000,
  classification + distillation (0.1, 1) on new data and exemplars, nearest-mean-of-exemplars classifier on the fused
  hidden layer, with 250 exemplars per label from each task (the network-head prediction is saved as a secondary column).
- architectural: DER-expansion (Yan et al. 2021): task-1 encoders frozen, new encoders added, head widened (starting as
  the task-1 head), trained with a random 1,000 buffer; multi-head: shared encoders, a new head for task 2, task
  identity given at test. (Progressive networks / PackNet with task identity cannot forget task 1 by construction, so
  their BWT gap is 0 by design; they are not run.)
- our fix: DER++ with the rule+exceptions buffer.
Strengths for EWC, SI, LwF, feature distillation and iCaRL are chosen per unit by matched plasticity (best task-1
accuracy among settings with task-2 accuracy within 0.02 of finetune; else the weakest setting).

## Benchmarks
- tables: 93 constructed tasks with exceptions; task 2 = random balanced table on other classes; seed 23.
- clevr: the 40 units of the first CL study; task 2 = a unit of another family; seed 23.
- gqa: 24 units, seeds 23-25; images split 150 / 50; task 2 = a unit of another family.
- pope: seeds 23-27; task 1 = POPE questions on 375 images, test on the other 125; task 2 = COCO cells POPE does not
  ask (balanced 50% yes; cell types from the exact rule of those cells, class-balanced weights).

## Tests (`analyze_clm.py`; 95% unit-bootstrap intervals)
- M1: for LwF, ER and DER++ (random), KS gap <= -0.05 with upper bound < 0; a benchmark shows it iff >= 2 of the 3
  do; PASS iff >= 3 of 4 benchmarks.
- M2: FWT gap <= -0.03 with upper bound < 0 for >= 2 of those 3 methods; PASS iff >= 3 of 4 benchmarks.
- M3: DER++ rule+exceptions vs DER++ random on GQA and POPE: KS_E gain >= 0.05 (lower bound > 0) and overall BWT
  difference lower bound > -0.02; PASS iff both.
- M4 (exploratory, criterion fixed now): a method "keeps exceptions as well as the rule" on a benchmark iff KS_R >= 0.5
  (it keeps the rule at all) and the KS gap lower bound > -0.05. Reported for every method, with family means. No
  family-level hypothesis is made in advance.

## Note on labels (known before launch)
`clm_structure.py` finished before this version was hashed. POPE task-1 training-cell labels reached the 30-minute
limit (best rule 0.774, upper bound 0.886), so they come from the best rule found, not the proven optimum. They are
used only to fill the rule+exceptions buffer on POPE (M3). Test-cell labels are those of the unrepairability study;
POPE task-2 test labels: 0.947, bound 0.9475.

## Addendum 1 (2026-10-02, at launch, no results seen)
CLEVR, GQA and POPE crashed at the first unit: the exception and rule+exceptions buffers need each training sample's
cell (question, image), which these benchmarks did not pass (`tr1[:3]`). Fixed by passing the cell indices; nothing
else changed. The constructed-task runs were unaffected and kept running. New hash of `cl_metrics.py` below.
