# Pre-registration: robustness of "the PID estimator used in practice misses CLEVR's structure"

Written 2026-10-01 before the sweep. SHA-256 in `preregistration_estimator.sha256` (with `liang_cvx.py`,
`liang_pid.py`, `clevr_liang_sweep.py`, `analyze_liang_sweep.py`).

Seen before writing: the step-2 reported result (our dit BROJA with k = 10 clusters, no PCA: ~0.04 bits vs exact
0.57; Spearman with d_add +0.19 / +0.46); that Liang et al.'s released estimator (run verbatim via ECOS) reproduces
AND = 0.50 and XOR = 1.00 bits; and that, run *without clustering* on three 10 x 100 CLEVR sub-tables, it equals the
exact MMI synergy within 0.004 bits (so exact MMI is the exact BROJA reference here).

## Sweep
Their pipeline (normalize -> PCA(5) -> KMeans per modality -> joint distribution -> CVX), on all 103 CLEVR units (full
10 x 1,000 tables), with k in {5, 10, 30 (their default), 100} image clusters (questions: min(k, 10)), PCA on / off,
symbolic or CLIP features: 16 settings.

## Claims and rule
- Claim 1: the estimate is far below the exact synergy (median ratio < 0.5).
- Claim 2: it ranks units weakly by exact structure (Spearman with additive distance < 0.5).
- ROBUST iff both hold in all 16 settings; SETTING-DEPENDENT iff both hold in some but not all (then the paper must
  state the dependence and give the best setting); NOT SUPPORTED iff in none.
- Bug fix before any sweep output was read: clevr_liang_sweep.py now loads data.npz into memory before forking (the shared zip handle crashed the workers). New hash: 0f8baddb037ad09e65d7047778066c93dacad5cd55263785602724add157410e  clevr_liang_sweep.py
- Second bug fix (worker deadlock: k-means in the parent before fork). Clustering now happens in the parent; workers (spawn) only solve. No sweep output had been produced. New hash: d251c92372c4d30ee398f4b8057c798c648e27f464cd6360a1608e619bfb7125  clevr_liang_sweep.py
- Third fix: ECOS failed on some clustered problems and crashed the pool. Each solve now falls back to SCS (max_iters 50000), else is recorded as failed (NaN); the solver used is stored per row; the run resumes from completed rows. New hash: 770f4c75dfec1535c0b80a8edf8cfc726486fc379ff32b4d203a343dec6e0217  clevr_liang_sweep.py
