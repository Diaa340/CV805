# Pre-registration: Route Law of Multimodal Forgetting (experiments 3–5)

Written before any of the runs below are executed. The file's SHA-256 is recorded in `preregistration.sha256`.

## Frozen law (two modalities, no fitted parameters)
For task-1 knowledge typed by pointwise PID (I_ccs), with route survivals p1, p2:

    R_U1 = p1      R_U2 = p2      R_R = 1 - (1 - p1)(1 - p2)      R_S = p1 * p2

No shared-bottleneck term. The bottleneck extension (every retention multiplied by h) is **exploratory** only.

## Measurements (frozen)
- **Retention of type t:** R_t = (acc_after_t - c) / (acc_before_t - c), where acc is task-1 test accuracy on samples of
  type t (logits masked to task-1 classes in class-incremental runs) and c = 1 / |task-1 classes|. A type counts in a
  run only if it has >= 30 task-1 test samples and acc_before_t >= c + 0.3.
- **Types:** synthetic = ground truth. Real = I_ccs atoms from calibrated probes trained on task-1 train
  (80% fit / 20% temperature calibration); type = argmax(red, u1, u2, syn); samples with i12 <= 0 excluded.
- **p (primary):** p_m = R_Um, clipped to [0, 1] before entering the law. Predicts R_R and R_S.
- **p (secondary, label-free):** after task 1, fit a linear readout on encoder-m embeddings (task-1 train, frozen encoder).
  After task 2, p_m^probe = chance-corrected retention of that fixed readout's task-1 test accuracy using the NEW
  encoder m. Predicts all four atoms.

## Experiment 3: held-out prediction
- Every run in this study is held out for the law (the law was fixed on earlier runs).
- Axes: data (synthetic; frozen features of two-view CIFAR-100, AV-MNIST, CREMA-D, and Food-101 / N24News if available),
  architecture (encoder depth 1/2/3, width 128/512), task order (task 1 <-> task 2 swapped), modality strength (synthetic
  SNR grid), >= 5 seeds.
- Metrics per atom (R, S; plus U1, U2 for the secondary p): Pearson r, Spearman rho, R^2 = 1 - SSE/SST using the
  predictions as-is (no refitting), MAE, RMSE, predicted-vs-observed scatter.
- Baselines (fitted by linear regression on a separate discovery batch: seeds 0–2, one config per data source; scored on
  all other runs): (a) initial PID amount of the atom, (b) accuracy before task 2, (c) per-modality accuracy after task 2
  (other modality zeroed), (d) per-encoder mean gradient norm during task 2, (e) per-encoder relative parameter drift,
  (f) additive model a + b*p1 + c*p2.
- Success: law R^2 >= 0.6 for both R_R and R_S on held-out replay runs, and law MAE lower than every baseline's.

## Experiment 4: causal interventions on one route (during task 2)
- Interventions on modality m: freeze encoder m; lower lr for encoder m (x0.1); feature distillation of encoder m to its
  task-1 copy (lambda in {1, 10}); replay with only modality m present. Same seeds as the matching no-intervention run.
- Pre-registered directions: intervening on m raises p_m and leaves p_other within +-0.1; therefore R_Um and R_S rise,
  R_R rises by at most the law's amount, R_U(other) is unchanged.
- Quantitative: the observed change dR_t (intervention minus control, same seed) is compared with the law's predicted
  change computed from the measured p's (sign agreement and MAE).
- Asymmetric target: an intervention setting with p1 ~ 0.9, p2 ~ 0.4 (replay regime). Prediction: R_S ~ p1*p2 (~0.36);
  R_U1 ~ 0.9.

## Experiment 5: kill / corrupt one modality
- After continual learning (and, as a control, after task 1 only), evaluate task-1 test samples with modality m replaced
  by x_m' = (1 - a) x_m + a * noise (standardised features; noise ~ N(0, 1)), a in {0, 0.25, 0.5, 0.75, 1}, for m = 1,
  m = 2, and a grid over both.
- p_m(a) = chance-corrected U_m accuracy under corruption (relative to uncorrupted). Law predicts R_R and R_S from
  (p1(a1), p2(a2)).
- Pre-registered pattern at a = 1: remove m1 -> U1 and S collapse, U2 stable, R partial; symmetric for m2; both -> all collapse.

## Known failure regime (reported, not hidden)
- No-replay runs: the shared fusion layer is overwritten (h ~ 0.2–0.3), so R_R is expected to be over-predicted. These
  are analysed separately and not used for the success criterion.
