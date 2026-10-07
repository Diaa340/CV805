# Pre-registration P4: the synergy law on the least redundant real tasks

Written after P3 showed that on standard benchmarks the "needs both modalities" knowledge is mostly additive (a linear
readout recovered 81% of it). Written before any P4 run, with the datasets chosen from published PID estimates
(Liang et al., NeurIPS 2023: CLEVR ~83% synergy, VQA 2.0 ~75%) plus one task that is synergistic by construction.
SHA-256 in `preregistration_p4_synergy_rich.sha256`.

## Tasks (frozen features: CLIP image/text, wav2vec2 audio; training as in v3/P3)
1. `fv_cremad`: CREMA-D face-voice verification. Pair = (voice of clip i, face of clip j); label 1 iff same actor
   (50/50). Every sample is synergistic by construction. Task 1 / task 2 = two halves of the actors
   (partition by class_split in {0, 1}; order normal/swap).
2. `vqa`: VQA v2 validation subset, top-100 answers, image + question; class-incremental over two answer halves
   (class_split {0, 1}; order normal/swap). Model-functional types (P2 procedure).
3. `clevr`: CLEVR v1.0, image + question, 28 answers; task 1 / task 2 = 'query' questions vs count/exist/compare
   (disjoint answer sets; order normal/swap). Model-functional types (P2 procedure).
Seeds 13–17; architecture (2,256), plus (3,512) for CLEVR only (it has one class split). 20 units per task.

## Measurements
Exactly those of P3: repairs on the full task-1 training set (readout_mlp, ceiling, encoder_realign, readout_linear);
per-sample correctness under the actual model and the drift swaps s1, s2, sb; retention = chance-corrected accuracy on
the type's samples relative to the task-1 model. New: `task1_linear` = the share of the task-1 model's accuracy that a
linear readout on the frozen task-1 encoders reaches (retrained on the full task-1 data).

## Manipulation check (per task; decides whether the task counts as synergy-rich FOR THE MODEL)
Median task1_linear retention on S samples <= 0.6. Tasks failing the check are reported but not tested.

## Criteria (on the S samples of tasks passing the check; units pooled)
- H1 composition: sb_S = s1_S * s2_S, R^2 >= 0.8 and MAE <= 0.06 (both regimes).
- H2 reproduction (replay): actual_S vs sb_S, R^2 >= 0.7 and MAE <= 0.08.
- H3 readout limit (no replay):
  - median recovery_S(readout_mlp) <= 0.6, AND
  - median [recovery_S(encoder_realign) - recovery_S(readout_mlp)] >= 0.2,
  where recovery is computed only where the ceiling-to-no-repair gap is >= 0.05.
- Reported, not criteria: forgetting magnitude (S retention without replay) compared with P3's benchmarks; the other
  types where they have >= 30 samples.
