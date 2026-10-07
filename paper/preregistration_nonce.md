# Pre-registration: VLM continual learning on nonce-word CLEVR

Written 2026-10-03 before the main runs. SHA-256 in `preregistration_nonce.sha256` (with `nonce.py`, `vlm_seq_nonce.py`,
`vlm_seq_clevr.py`, `vlm_seq.py`, `analyze_nonce_vlm.py`).

## Why
On plain CLEVR, Qwen2.5-VL-3B is at ceiling zero-shot (0.92-0.98 on the 6,000 test cells, exceptions as well as rule
cells), so a CL sequence teaches it nothing (preregistration_guide.md, part 1 on hold). Here every task renames the
15 CLEVR attribute concepts (8 colours, 3 shapes, 2 materials, 2 sizes, with their synonyms and plurals) with its own
made-up words (`nonce.py`: pronounceable, not in the system English word lists, offensive look-alikes blocked,
disjoint across tasks). Answers, tables and exception labels are unchanged; the model must learn new word-image
bindings in each task, and can forget earlier tasks' words.

## Seen before writing (pilot, disclosed)
All training questions contain at least one renamed word. Zero-shot on 400 test cells: task 0 0.458, task 3 0.610.
Task-only LoRA: 300 steps -> 0.715 / 0.767; 600 steps -> 0.777 / 0.767. Steps per task fixed at 300 (as in the
other VLM studies). No CL run has been made. The run script differs from the smoke-tested `vlm_seq_clevr.py` only in
the question text, so no separate smoke run.

## Design
`vlm_seq_clevr.py` unchanged (Qwen2.5-VL-3B-Instruct + LoRA; six tasks = the 40 CLEVR units grouped by question family;
natural cells; per task 500 exception + 500 rule test cells; 300 steps x 8 per task; orders from seeds 60, 61; seed
31). Methods: finetune; EWC (100, 10000; matched plasticity); LwF (1); ER; DER++; class-balanced ER; LoRA-expansion.

## Gates (checked first; if not met, the tests are reported but not interpreted)
- Zero-shot accuracy over all test cells <= 0.80.
- Task-only accuracy >= zero-shot + 0.10.

## Tests (`analyze_nonce_vlm.py`; same-answer metrics; bootstrap over the 40 units)
- VC1: ER and DER++ dBWT_s <= -0.03 with upper bound < 0 (both).
- VC2: DER++ dFWT_s <= -0.03 with upper bound < 0.
- VC3: LoRA-expansion dBWT_s minus ER's dBWT_s has lower bound > 0.
- Reported: all methods by type and answer, zero-shot and task-only accuracy by type.

## Addendum 1 (2026-10-03, 21:40; no results looked at)
Both orders ran in parallel and crashed with CUDA out-of-memory when the replay methods started (two processes of
~24 GB on the 47 GB GPU). Saved and kept: zero-shot, task-only, finetune, EWC (100, 10000), LwF. The remaining
methods (ER, DER++, class-balanced ER, LoRA-expansion) are run with `--resume`, one order at a time. Resume reuses the
saved columns and retrains task 1 with the same seed (its model state was not saved); the agreement between the
retrained and the saved task-1 model on task-1 test cells is logged and will be reported. The analysis that ran
automatically after the crash failed (missing columns) and printed no results.
