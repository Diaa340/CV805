# Pre-registration v4 + FIA: covariance-corrected composition law and Fusion-Interface Anchoring

Written after P4 and before any v4 or FIA run. SHA-256 in `preregistration_v4_fia.sha256`.

## Operational definitions (frozen)
- Margin s(z1, z2) = logit of the true class minus the largest other task-1-class logit, through a given head.
- Replacement features z̄_m: encoder-m outputs of randomly chosen other task-1 test samples. There are 3 draws; each
  quantity below is computed per draw and averaged.
- Per-sample evidence profile through the task-1 head g⁰:
  - b = s(z̄1, z̄2);
  - α1 = s(z1, z̄2) - b;
  - α2 = s(z̄1, z2) - b;
  - β = s(z1, z2) - s(z1, z̄2) - s(z̄1, z2) + b.
  - Interaction share = |β| / (|α1| + |α2| + |β|).
- Fusion-sensitive subspace P_m: the top-16 right singular vectors of the matrix of per-sample gradients ∂s/∂z_m (task-1
  model, task-1 test samples).
- Per-sample route retention ρ_m (feature geometry only; no accuracy is used):
  ρ_m = clip(⟨P_m(z'_m - μ_m), P_m(z_m - μ_m)⟩ / ‖P_m(z_m - μ_m)‖², 0, 1), where z'_m = new encoder output, z_m = old
  encoder output, and μ_m = old mean over task-1 train samples.
- Survival events for task-1-correct samples, all through the task-1 head:
  - S1 = still correct with the new encoder 1 only;
  - S2 = still correct with the new encoder 2 only;
  - S12 = still correct with both new encoders.
- Decomposition per unit and sample group: r12 = P(S1)P(S2) + C + N, where C = Cov(1_S1, 1_S2) and
  N = P(S12) - P(S1 ∩ S2).
- Interface maps: Q̂_m = ridge least-squares linear map from old to new centred encoder-m features (task-1 train).
  W = mean over task-1 test samples of the mixed Jacobian ∂²s/∂z1∂z2 through g⁰.
  - κ = ‖Q̂1ᵀ W Q̂2 - W‖_F / ‖W‖_F;
  - compensation index κ_c = ½(‖Q̂1ᵀ W - W‖_F + ‖W Q̂2 - W‖_F)/‖W‖_F - κ.

## Units for v4 (NEW seeds 23–25)
- Synthetic (exact types): SNR 4.5/3.0, shift both.
- P1 bridge (exact types): cifar2v, food101; shift both.
- Natural, model-defined types (P2 procedure): cifar2v, avmnist, cremad, food101, n24news.
- Also clevr, and fv_cremad (all samples synergistic).
- Class split 0, order normal and swap (bridge / synthetic: shift both, 2 architectures (2,256) and (3,512) instead of
  orders).
- Regimes: replay and no replay.
- "Needs-both" group = type S: model-defined, or exact where available.

## Hypotheses (v4)
- H1 (decomposition separates the regimes), needs-both group, both regimes pooled:
  - synthetic + P1 bridge units: median |N| <= 0.05;
  - fv_cremad units: median N >= 0.05.
- H2 (a-priori prediction of N): across all units and both regimes, Spearman(κ_c, N) >= 0.6. κ is reported.
- H3 (graded evidence law), natural datasets (cifar2v, avmnist, cremad, food101, n24news):
  - target: per unit, regime and type, the share of task-1-correct samples surviving S12;
  - graded prediction: mean over the type of 1[ρ1 α1 + ρ2 α2 + ρ1 ρ2 β + b > 0];
  - discrete v3 prediction: P(S1)·P(S2) of the type;
  - success: R²(graded) - R²(discrete) >= 0.15, pooled over units, regimes and types.
- H4 (recoverability), natural datasets, no replay:
  - among task-1-correct samples that are wrong after task 2, the share made correct again by a linear readout (retrained
    on the full task-1 data on frozen new encoders), by quintile of interaction share (pooled);
  - success: Spearman(quintile index, recovered share) <= -0.9 AND recovered share(Q5) - recovered share(Q1) <= -0.15.

## Theory (written alongside, not a criterion)
Formal proofs of Props 1–3 for the bilinear head under the stated drift assumptions, and the threshold bound for
AND-structured additive evidence with N(μ, σ²) margins: `paper/theory_v4.md`.

## FIA and baselines (task 2 training; feature-level models as in v3; seeds 23–25)
- FIA loss = task-2 CE + λ_swap Σ_m KL(g⁰(z⁰) || g⁰(z_m^new, z_{-m}⁰)) + λ_joint KL(g⁰(z⁰) || g⁰(z^new)) + λ_β E(β^new - β⁰)².
  - Anchor data: task-2 inputs by default.
  - β uses replacement features from shuffled pairings within the batch.
  - Default FIA: λ_swap = λ_joint = λ, λ_β = λ.
- Baselines:
  - fine-tuning; joint training (task 1 + task 2 data);
  - EWC (diagonal Fisher from task 1); LwF (KL between the old model's and the new model's outputs on task-2 inputs,
    task-1 classes);
  - L2 feature distillation (both encoders); Jacobian-weighted feature distillation (first-order swap-KL:
    ‖J_m(z_m^new - z_m⁰)‖² with J_m = ∂g⁰/∂z_m at the old features);
  - random replay; DER++; oracle post-hoc encoder re-alignment (Phase 1b).
- Ablations:
  - anchor location: L2 -> Jacobian -> swap-KL;
  - factorisation: joint-only / swap-only / both;
  - with / without L_β;
  - anchor data: task-2 inputs / cross-modal mixed pairs;
  - replay priority for FIA+buffer: random / needs-both / β-weighted.
- λ sweep {0.1, 0.3, 1, 3, 10} for EWC, LwF, L2, Jacobian, FIA and its ablations; replay buffer 2% of task-1 train
  (min 200).
- Datasets: P1 bridge (cifar2v, food101); natural (cifar2v, cremad, food101, n24news, avmnist); fv_cremad; clevr.
  Class split 0, orders normal/swap (bridge: shift both).
- Reported:
  - Pareto fronts (task-2 accuracy vs needs-both retention);
  - retention by type and by interaction-share quintile;
  - average accuracy; BWT; time per step.

## Method predictions (pre-registered; paired over seeds)
- P-a (P1 bridge): at matched plasticity (the best λ whose task-2 accuracy is >= fine-tuning - 0.02), FIA's S retention
  exceeds L2 feature distillation's by >= 0.05.
- P-b (natural datasets), at matched plasticity:
  - FIA's gain over L2 is larger on needs-both (S) than on redundant (R) knowledge by >= 0.03;
  - FIA with vs without L_β differ by < 0.02 in S retention.
- P-c (fv_cremad): at the same λ, FIA-joint-only has task-2 accuracy >= FIA-full's + 0.01, with S retention within ±0.02.

## Deviations declared in advance
- The external audio-visual CL benchmarks (AV-CIL, ICCV 2023) and VQACL are NOT run in this round: they need large
  video downloads and their own architectures. This is a stated limitation.
- H2 uses κ_c instead of κ as the primary statistic, because κ alone is small both with no drift (N ≈ 0) and with
  compensated drift (N > 0), so it cannot predict N monotonically.
