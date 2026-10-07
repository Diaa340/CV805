# Pre-registration, Phase 1: is synergy erased or disconnected?

Written after v2 failed (encoders keep 93–97% of their task-1 content while task-1 accuracy drops; synergy drops most),
and before any Phase-1 run. SHA-256 in `preregistration_phase1.sha256`.

## Hypothesis
After task 2, synergistic knowledge is not erased from the encoders; it is disconnected: the fusion layer no longer
matches the drifted encoder outputs. Re-fitting only the fusion layer should restore it; re-fitting a head that cannot
represent interactions, or re-fitting on encoders that never learned task 1, should not.

## Runs (synthetic, ground-truth types; same generator, training and replay buffer as v2)
- SNR {4.5/3.0, 4.5/4.5, 3.5/3.5} x shift {both, m1, m2}, architecture (2, 256), seeds 3–7 (45 units).
- Task 2 in two regimes: replay (control) and no replay (none).
- After task 2, each repair uses the same budget: 2000 Adam steps (lr 1e-3), batches of 64 replay-buffer + 64 task-2
  samples. The replay buffer is the same 1000 task-1 samples as in task 2.
  - A `no_repair`
  - B `fusion_refit`: encoders frozen, fusion layer trained further (hypothesis arm)
  - C `encoder_refit`: fusion layer frozen, encoders trained (repair from the encoder side)
  - D `additive_refit`: encoders frozen, new additive head (one linear readout per modality, summed; cannot represent
    synergy)
  - E `fresh_encoders`: randomly initialised encoders (never saw task 1) frozen, new fusion layer trained (what the
    buffer alone teaches)
  - F `task1_encoders`: the task-1 encoders frozen, new fusion layer trained (ceiling)
- Measured: chance-corrected task-1 retention per type (R, U1, U2, S) relative to the task-1 model; task-2 accuracy.

## Success criteria (replay regime primary; mean over the 45 units)
1. `fusion_refit` restores synergy: S retention >= 0.8.
2. It is reconnection, not re-learning: S(`fusion_refit`) - S(`fresh_encoders`) >= 0.3.
3. The interaction lives in the fusion layer: S(`fusion_refit`) - S(`additive_refit`) >= 0.3, while
   R(`additive_refit`) >= 0.8 (additive heads recover non-synergistic knowledge).
4. Task 2 is kept: task-2 accuracy of `fusion_refit` >= task-2 accuracy of `no_repair` - 0.05.
All four pass -> "synergy is disconnected, not erased" is supported; continue to real data (frozen features, probe types).
The no-replay regime is reported with the same statistics.

## Exploratory (Phase 2, no criteria)
Linear CKA before/after task 2 of: the fusion layer's hidden representation (task-1 test inputs), each modality's
embedding; relative weight drift of the fusion layer's first-layer weights, split into the modality-1 and modality-2
input blocks; CKA between the two modalities' embeddings before and after (cross-modal alignment).
