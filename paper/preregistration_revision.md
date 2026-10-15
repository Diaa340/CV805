# Pre-registration: analyses for the CVPR review (R1-R5)

Written 2026-10-12 before any of these analyses' outcomes were computed. SHA-256 in `preregistration_revision.sha256`
(with `rev_null.py`, `rev_unique.py`, `rev_unique_analysis.py`, `cl_difficulty.py`, `analyze_difficulty.py`,
`rev_stability.py`).

## Seen before writing
The review. A check of the tie-breaking solver on four VLM sub-tables (same minimum error count every time; 4-6
distinct exception sets in 6 solves). One smoke run of `cl_difficulty.py` (one constructed task; output not analysed).
Every analysis script was run on smoke output or on fake random labels, printing only that it runs. The R3 and R5
solvers (answer tables only, no outcomes) started before this file was hashed.

## Environment note
The machine's NVIDIA driver changed (now CUDA 12.9 max); the main environment's PyTorch 2.14.0+cu130 no longer starts.
GPU work uses `.venv_cu128`: the same PyTorch version, 2.14.0, built for CUDA 12.6, with all other packages shared with
the main environment. R4's runs are new runs (same seeds and code paths), not the saved ones.

## R1 (W2 / Q1): is the Study C effect arithmetic? (`rev_null.py`)
Null scorer: accuracy- and answer-matched "oracle + Gaussian logit noise", 20 draws, identical EMAP pipeline on the
Study C sub-tables. Within-benchmark partial rank correlations, controls PID, accuracy, yes share.
- PRIMARY: for Idefics3 (Study C's confirmatory model), rho(Delta - Delta_null, s | PID, acc, yes) has 95% lower
  bound > 0 ("structure predicts interaction use beyond what an accuracy-matched scorer with unstructured errors
  needs"). Reported for all four models, with rho_null, rho_adj (Delta_null as an extra control), the share of the
  original rho that survives, and the slack Delta - (s - e).
## R2 (Q5): rho(Delta, yes | s, PID, acc), all four models (descriptive).
## R3 (W4 / Q2): non-uniqueness of exceptions (`rev_unique.py`, `rev_unique_analysis.py`)
Random tie-breaking re-solves (20 per table; 10 for the 10 x 300 CLEVR tables) of the 99 Study C sub-tables and all
continual-learning test tables. Reported: ambiguous share (a lower bound), agreement of saved labels with stable labels.
- U1: with stable labels only, CLEVR backward same-answer gap <= -3 points with upper bound < 0 for >= 3 of the 4
  replay methods (ER, A-GEM, GEM, DER++).
- U2: with stable labels only, DER++ forward same-answer gap <= -3 points with upper bound < 0 on >= 2 of 3
  benchmarks (constructed, CLEVR, GQA).
- Reported: feature distillation's forward gaps; VLM cell-level AUC with stable labels.
## R4 (W5 / Q3): difficulty-matched comparison (`cl_difficulty.py`, `analyze_difficulty.py`)
New runs (constructed tasks, CLEVR, GQA; finetune, LwF, feature distillation, ER, DER++, DER-expansion) saving each
cell's signed margin. Strata: unit x answer x margin decile (task-1 model before task 2 for backward; task-2-only model
for forward); exception-weighted gaps.
- D1: DER++ forward gap, difficulty-matched, <= -3 points with upper bound < 0 on >= 2 of 3 benchmarks.
- D2: CLEVR backward gap, difficulty-matched, <= -3 points with upper bound < 0 for both ER and DER++.
- If the unmatched same-answer gaps replicate but D1 / D2 fail, we report that difficulty explains the effect.
- Reported: coverage (share of exceptions with matched rule cells), and the comparison with the hardest quarter of rule
  cells of the same answer.
## R5 (Q4): size dependence of s(T) (`rev_stability.py`): descriptive.

## Addendum 1 (2026-10-13; R4 GQA crash)
The GQA runs of `cl_difficulty.py` crashed at the first unit: CLIP feature extraction imports torchvision, and the main
environment's torchvision is built for CUDA 13.0. Fixed by installing the matching torchvision 0.29.0 (CUDA 12.6) into
`.venv_cu128`; no code changed. The R4 analysis had already run once on constructed tasks and CLEVR only (D1 counted
2 of 3 with GQA missing); it is re-run, unchanged, once GQA finishes, and that run is the result.
The first re-run then failed on torchaudio (also CUDA 13.0 in the main environment); fixed by installing the main
environment's version, torchaudio 2.11.0, built for CUDA 12.6 (no dependencies, so PyTorch stays 2.14.0). Also found (R3 output, reported as a result): the constructed tasks' task-2 table has every cell ambiguous across 19
optimal rules, so forward gaps on constructed tasks depend on an arbitrary choice of rule.
