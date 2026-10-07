# Pre-registration P3: the readout limit for synergy on natural data

Written after P2 and before any P3 run. SHA-256 in `preregistration_p3_readout.sha256`.

## Claim under test (contrast with unimodal "accessibility" results)
For unimodal networks, much forgotten accuracy comes back by retraining only the readout (Davari et al. 2022;
arXiv 2606.06032). In multimodal late-fusion networks, retraining the readout (the fusion head) should recover unique
and redundant knowledge but NOT synergistic knowledge. Re-aligning the encoders to the task-1 fusion head should recover
synergy better.

## Setup
- Natural data, frozen features, class-incremental: two-view CIFAR-100, AV-MNIST, CREMA-D, Food-101, N24News x class
  split {0, 1} x task order {normal, swap}, architecture (2,256), NEW seeds 18–22 (100 units). Training as in v3.
- Types: model-functional, exactly as in P2.
- Regimes: no replay [primary], replay [reported].
- Repairs after task 2 use the FULL task-1 training set (the most favourable case for readout recovery): 3000 Adam steps,
  lr 1e-3, batch 256, CE loss:
  - `readout_mlp` [primary readout]: drifted encoders frozen + a fresh fusion head (same architecture: can represent
    synergy);
  - `ceiling`: task-1 encoders frozen + a fresh fusion head (the same readout training on intact encoders);
  - `encoder_realign`: the task-1 fusion head frozen, the drifted encoders trained;
  - `readout_linear` [reported only]: a linear readout on the concatenated drifted embeddings. It is additive, so it
    cannot represent synergy by construction.
- Recovery of type t by an arm: (ret_arm - ret_no_repair) / (ret_ceiling - ret_no_repair), only where
  ret_ceiling - ret_no_repair >= 0.05 and the type has >= 30 samples.

## Criteria (primary regime: no replay; all units pooled)
- C1 (readout limit): for `readout_mlp`, median over units of [min(recovery of R, U1, U2) - recovery of S] >= 0.15,
  AND S has the lowest recovery of the four types in >= 70% of units.
- C2 (re-alignment beats readout for synergy): median over units of
  [recovery_S(encoder_realign) - recovery_S(readout_mlp)] >= 0.20.

The claim is supported if C1 and C2 both pass. Per-dataset and replay-regime results are reported.

## Exploratory (no criteria; same runs)
Per-sample graded reliance on each modality (drop in the task-1 model's correct-class probability when that modality is
replaced by a random real sample), and per-sample correctness under each drift swap after task 2. Used afterwards to
explore a graded version of the drift-composition law.
