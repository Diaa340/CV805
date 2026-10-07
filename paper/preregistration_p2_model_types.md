# Pre-registration P2: model-functional interaction types on natural data

Written after P1 passed (exact types on real features) and v3 failed on natural data with probe-estimated PID types.
Written before any P2 run. SHA-256 in `preregistration_p2_model_types.sha256`.

## Question
On natural multimodal data (no constructed labels), does the drift-composition law hold when a sample's interaction
type is defined by what the task-1 model needs, instead of by an estimated information decomposition?

## Model-functional types (fixed on the task-1 model, before task 2)
- Only task-1 test samples the task-1 model classifies correctly with both modalities are typed; the rest are excluded.
- "Modality m removed" = modality m's input replaced by the same modality of a random other task-1 test sample.
  Three random draws; a sample counts as correct if at least 2 of 3 draws are correct. Logits are masked to task-1
  classes, as in all class-incremental runs.
- Types:
  - R: correct with modality 1 alone AND with modality 2 alone;
  - U1: correct with modality 1 alone only;
  - U2: correct with modality 2 alone only;
  - S: correct only with both.
- Types are fixed before task 2. The drift swaps come afterwards (not circular).

## Runs
Natural data with frozen features (exactly the v3 real-data protocol): two-view CIFAR-100, AV-MNIST, CREMA-D, Food-101,
N24News x class split {0, 1} x task order {normal, swap}, architecture (2,256), NEW seeds 13–17 (100 units).
Training, replay buffer, swaps and retention exactly as v3. Retention of a type = chance-corrected accuracy after task 2
relative to the task-1 model (which is 1 by construction).

## Criteria (P1 / synthetic-v3 thresholds, since types are exact with respect to the model)
- (a) composition sb_t = s1_t * s2_t: R^2 >= 0.8 and MAE <= 0.06 for S, U1, U2; R^2 >= 0.5 for R.
- (b) U1: s2 >= 0.9 in >= 80% of units, and U2: s1 >= 0.9 in >= 80% of units; mean min-factor of R minus S >= 0.10.
- (c) replay regime: actual vs swap for S, R^2 >= 0.7 and MAE <= 0.08.
- A type counts in a unit if it has >= 30 samples.
- Judged on all units pooled; per dataset reported.
- **Learnability rule:** if fewer than 50% of units have >= 30 model-synergistic samples, P2 is INCONCLUSIVE.
- Reported, not criteria: agreement between model-functional types and the v3 probe-estimated PID types.
