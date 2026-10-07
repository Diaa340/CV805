# Pre-registration v2: content-specific routes (synthetic test, then real-data gate)

Written after v1 (preregistration.md) was refuted on its pre-registered criteria (held-out R^2 0.39 / 0.32; synergy
under-predicted by +0.13; synergy above min(p1, p2) in 40–60% of replay runs), and before any v2 run. SHA-256 is recorded
in `preregistration_v2.sha256`.

## Hypothesis v2
A route is a (modality m, content) pair, not a whole modality. Type t uses the content that modality m carries for t.
With q(m, t) = survival of that content in encoder m:

    R_U1 = q(1,U1)    R_U2 = q(2,U2)    R_R = 1 - (1 - q(1,R))(1 - q(2,R))    R_S = q(1,S) * q(2,S)

No fitted parameters; no shared-bottleneck term (h = 1).

## Data and runs (synthetic, 2 modalities, K = 10 classes)
- Generator of stage0 (type-specific code tables), extended to return the codes c1, c2 of every sample.
- Task 1 trained with the same synergy up-weighting as v1. Task 2 = new code tables for the modalities in `shift`.
- Grid: SNR {4.5/3.0, 4.5/4.5, 3.5/3.5} x shift {both, m1, m2} x architecture {(2,256), (1,128), (3,512)};
  held-out seeds 3–7; discovery seeds 0–2 on (4.5/3.0, both, (2,256)) only, used to fit baselines.
- Task-2 conditions: none (no replay), control (replay), and single-route interventions in the replay regime:
  freeze_m, distill1_m, replayonly_m for m = 1, 2.

## Measurements (frozen)
- Observed retention of type t: (acc_after - c) / (acc_before - c), c = 1/K, on task-1 test samples of type t.
  Only types with acc_before >= c + 0.3 count.
- **q primary (code decodability, re-fit):** for each (m, t), a linear readout from encoder-m embeddings to code c_m,
  fitted on task-1 train samples of type t and scored on task-1 test samples of type t, before task 2 and again
  (re-fitted) after task 2. q = chance-corrected retention of that accuracy (chance 1/K), clipped to [0, 1].
- **q secondary (code decodability, fixed):** the readout fitted before task 2 is applied unchanged after task 2.
- **q tertiary (representation recoverability, label- and code-free):** ridge regression from the new encoder-m
  embedding to the old one, fitted on task-1 train samples of type t, R^2 on task-1 test samples of type t, clipped to
  [0, 1]. This is the measure that can be computed on real data.

## Success criteria (held-out replay-regime runs: control + interventions)
1. Primary q: for each of R, S, U1, U2, R^2 >= 0.6 and MAE <= 0.08.
2. Primary q, synergy: MAE below v1's synergy MAE on synthetic held-out replay runs (0.140) and below an additive
   baseline a + b q(1,S) + c q(2,S) fitted on the discovery runs.
3. Primary q, synergy stays inside the AND bounds [max(0, q1S + q2S - 1) - 0.05, min(q1S, q2S) + 0.05] in >= 80% of runs.
The no-replay regime is reported separately (known shared-bottleneck failure).

## Gate to real data
Continue to real data only if criteria 1–3 pass AND the tertiary q reaches R^2 >= 0.5 for both R and S on the same runs.
On real data the tertiary q is computed per (modality, estimated type), with the v1 real-data protocol otherwise unchanged.
