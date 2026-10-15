# Data behind the revised paper's tables and figures

All files are CSV, exported from the saved results by `mm_pid/make_rev_data.py`. **s** = exception share (the minimum
error of any additive rule on the table). **EMAP drop** = accuracy lost when a model's scores are replaced by their
closest additive version. Units: files ending in `_points` are in accuracy points (0–100); the others use fractions
(0–1). Intervals are 95% bootstrap intervals over tables or units.

---

### 01_controlled_tables.csv: Sec. 4.1 and supplement Tab. `sup_controlled`
One row per constructed 10×10 table: 63 `selected` tables and 62 `fresh_replication` tables. Columns: s, XOR density,
synergy (BROJA, I_ccs, MMI), accuracy of the full model, the linear readout and the majority baseline, and the readout
gap (share of the full model's above-majority accuracy that the linear readout misses).
- **Result:** the readout gap follows s, not synergy (partial ρ +0.91 vs +0.22; replication +0.86 vs +0.11).
- **Meaning:** in a controlled setting, how much a task needs non-additive fusion is captured by its exceptions,
  not its synergy. Caveat: the readout is itself bounded by the ceiling, so this shows trained readouts approach
  the bound.

### 02_vlm_studyA_full_tables.csv: supplement Study A table
3 VLMs × 99 full 10×100 tables. s and its proven lower bound, synergy (BROJA, MMI), accuracy, EMAP accuracy, EMAP
drop, accuracy on exception and rule cells.
- **Result:** s predicts the EMAP drop within CLEVR and CLOVE, but synergy does almost as well, because the two
  measures move together on natural tables (ρ +0.88 to +0.95).
- **Meaning:** on natural tables structure and synergy cannot be told apart, which motivates Studies B and C.

### 03_vlm_studyB_subtables.csv: Study B (Sec. 4.2)
3 VLMs × 99 sub-tables of 50 images, chosen (from answer tables only) so that s and synergy are uncorrelated within
each benchmark.
- **Result:** s predicts the EMAP drop with synergy held fixed (+0.32 to +0.37), but synergy keeps a signal of its own,
  which tracks the yes-share.
- **Meaning:** separating the two works, but answer balance is a confound, hence Study C.

### 04_vlm_studyC_subtables_with_null.csv: Study C, Tab. 1 and Tab. `null`, Fig. 1 VLM panel
4 VLMs × 99 sub-tables where s is uncorrelated with both synergy and yes-share. Adds the accuracy-matched null scorer's
EMAP drop (`emap_drop_null`), the excess over it, and the slack above the floor (Δ − (s − error)).
- **Result:** the null scorer, which has each model's accuracy but randomly placed errors, has almost the same EMAP
  drop as the real models (e.g. Idefics3 0.064 vs 0.060). Only InternVL3.5 is clearly above it (0.086 vs 0.044).
- **Meaning:** VLMs depend on interactions roughly as much as their accuracy requires; InternVL3.5 more.

### 05_vlm_partial_correlations.csv: Tab. `null`, Fig. 1 VLM panel, supplement Tab. `partial_corr`
All partial rank correlations of the VLM study: Study B, Study C, and the null analysis (observed, null, excess,
adjusted, slack, yes-share).
- **Result:** in Study C, s predicts the EMAP drop (+0.26 to +0.43) and synergy adds nothing for 3 of 4 models
  (LLaVA keeps +0.24). But the null reproduces most of s's correlation: the excess is significant only for InternVL3.5
  (+0.26). The pre-registered null test on Idefics3 failed. Yes-share has its own signal (+0.23 to +0.34) for every model.
- **Meaning:** the exception share predicts the interaction dependence an accurate model *needs*; the claim that it
  predicts dependence *beyond* that holds for one model only.

### 06_vlm_cell_auc.csv: Sec. 4.2 "Which cells require interactions?", supplement
AUC for predicting which cells' answers change when interactions are removed, using the exception label, the stable
exception label, graded misfit, or the model's low confidence.
- **Result:** graded misfit 0.78–0.88 beats confidence 0.63–0.74; the binary exception label (0.60–0.71) does not.
- **Meaning:** at the cell level use the graded misfit; binary exception labels are too few and not unique.

### 07_cl_main_table_points.csv: Tab. `main`
For each benchmark and method: ACC, BWT, exception accuracy (E_ACC) and same-answer gaps ΔB, ΔF, in points. `probe` is
DER++ with exception cells in its buffer; `zeroshot` appears only for the VLM.
- **Result:** the method with the best standard BWT is not the one that keeps exceptions. On CLEVR, DER++ has BWT +0.1
  but E-ACC 34.9; DER-expansion has the best ACC on 3 of 4 benchmarks.
- **Meaning:** standard forgetting metrics hide the loss of exceptions.

### 08_cl_same_answer_gaps.csv: Fig. `gaps`
Same-answer gaps (exceptions minus rule cells with the same answer) for backward (`dBWT_s`) and forward (`dFWT_s`)
transfer, with intervals; also the unmatched (`_raw`) values and BWT by answer.
- **Result:** on CLEVR every replay method forgets exceptions faster (−8.7 to −11.0 points); on GQA and CLOVE the gap
  mostly disappears once answers are matched.
- **Meaning:** the raw exception gap on existence tables was mostly answer imbalance; the effect is real on
  compositional CLEVR.

### 09_cl_difficulty_matched_points.csv: Tab. `difficulty`, supplement Tabs. `sup_diff_tables` and `sup_diff_gqa`
New runs (6 methods, 3 benchmarks). Gaps matched on answer only, on answer **and** difficulty (margin decile), and
against the hardest quarter of rule cells; with the coverage (share of exceptions that could be matched).
- **Result:** on CLEVR, forgetting survives difficulty matching (ER −12.5, DER++ −7.8) and every method learns
  exceptions worse (−10.6 to −27.0), with 99% coverage. On GQA only 18–19% of exceptions can be matched and the effects
  are marginal. The large positive finetune and LwF gaps vanish (difficulty artefacts).
- **Meaning:** on CLEVR, exceptions are not just "hard examples"; elsewhere the evidence is too thin to claim.

### 10a_exception_uniqueness_per_table.csv: supplement Tab. `unique`
For every table re-solved with random tie-breaking: number of solves, how many reached the minimum error, how many are
proven optimal, and the share of cells that are ambiguous, stable exceptions, or exceptions in any optimal rule.
- **Result:** 3–6% of cells are ambiguous (a lower bound). The constructed tasks' second table (`tables__task2`) is
  ambiguous everywhere.
- **Meaning:** the exception share is unique, but individual exception cells are not; cell-level results must be
  checked on stable labels.

### 10b_cl_gaps_saved_vs_stable_labels_points.csv: supplement Tab. `unique_cl`
Key same-answer gaps with the solver's labels vs stable labels only.
- **Result:** CLEVR forgetting holds for all 4 replay methods (−9.3 to −12.0); the CLEVR learning gap holds (DER++
  −27.8); GQA's learning gap loses significance; constructed tasks have no stable labels.
- **Meaning:** the CLEVR findings do not depend on which optimal rule the solver picked.

### 11_cl_buffer_control.csv: supplement Tab. `der_buffers`
DER++ with four 1,000-sample buffers: random, class-balanced, exception-focused, exception-focused and balanced.
Exception BWT, overall BWT, BWT by answer for each cell type.
- **Result:** at equal answer balance, filling half the buffer with exception cells keeps 6.8–8.5 points more old
  exceptions (constructed, CLEVR, CLOVE; not GQA), at a cost of 1–3 points of overall BWT.
- **Meaning:** replaying exceptions helps beyond class balance, with a real trade-off.

### 12_vlm_cl_clove.csv: Tab. `vlm_cl`
Qwen2.5-VL-3B + LoRA on six CLOVE tasks: ACC, exception accuracy and gaps per method, plus zero-shot.
- **Result:** almost no forgetting by standard BWT (within ±0.5), but exception accuracy falls from 82.0 (zero-shot)
  to 66–72 after any fine-tuning while overall accuracy rises.
- **Meaning:** fine-tuning trades exceptions the pretrained model already knew for gains on the rule, which standard
  metrics do not show (post hoc).

### 13a_stability_subtables.csv / 13b_stability_summary.csv: supplement Tab. `sup_stability`
s recomputed on column subsets of 25–1,000 images with 10 or 5 questions (13a: every sub-table; 13b: mean and SD by
size).
- **Result:** s grows slowly with the number of images (CLEVR 0.079 → 0.099) and is lower with fewer questions, but the
  ranking of tables is stable (Spearman ≥ 0.97; 84–97% of variance between tables).
- **Meaning:** compare s at a fixed table shape; as a relative measure it is reliable.

### 14a / 14b_decorrelated_selection_study{B,C}.csv: supplement Fig. `decor`
The 99 sub-tables chosen for Studies B and C: source table, image columns, s, synergy, yes-share.
- **Result:** within each benchmark ρ(s, BROJA) = 0.00 (B); ρ(s, BROJA) ≤ 0.02 (GQA 0.20) and ρ(s, yes) ≈ 0 (C).
- **Meaning:** the selection achieved the separation the tests need, using answer tables only.
