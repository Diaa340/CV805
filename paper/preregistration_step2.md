# Pre-registration: step 2, exact interaction structure on a real benchmark (CLEVR)

Written 2026-09-30. SHA-256 in `preregistration_step2.sha256` with hashes of `clevr_exec.py`, `step2_build.py`,
`step2_pid.py`, `step2_train.py`, `analyze_step2.py`. Runs only if step 1 passes (R1 and at least one of A / B).

## Why CLEVR, and what changed while building (all before any pre-registered run)
- CLEVR ships scene graphs and question programs, so any question can be executed on any image: the answer table
  (question x image) is known exactly. Our executor reproduces all 149,991 val answers.
- Most CLEVR questions refer to "the X", which is ill-posed on other images (with native semantics only 6 families kept
  enough valid cells). We therefore use **set-lifted (existential) semantics**: "the X" -> "some X", relations and
  "same" act on sets, and equal_attribute = "share a value". On each question's own image this equals the original
  answer for all 60,835 yes/no questions; on other images it is always well-posed. The share of cells that needed
  lifting is reported per unit (0.34-0.96 by family).
- Units: the 37 yes/no families, up to 60 distinct programs each, split into groups of 10 questions, x 1,000 fixed val
  images; kept if the yes-share is in [0.10, 0.90] -> 103 units from 28 families.
- Structure: XOR density (sampled 2x2 submatrices), additive distance (error of the best logistic fit u_q + w_img),
  exact-table MMI synergy. Across units, density and synergy are almost collinear (Spearman +0.98), so CLEVR cannot
  separate them the way the constructed tables did. Step 2 therefore tests *calibration* and the *P4 explanation*.
- Features: symbolic (object slots sorted by x plus attribute-combination counts; questions as step-indexed one-hot
  program values) and CLIP ViT-B/32 (image, question text). Counts were added to the symbolic image code after a smoke
  test (seed 99, 1 unit) showed that slots alone could not learn counting from 700 images.
- Smoke results seen before writing (seed 99, 6 units, discarded): the linear readout was within 0.00-0.02 of the
  additive ceiling on 5/6 units; CLIP full models exceeded the ceiling on 4/6 units, which suggests S2-B may fail. The
  S2-B hypothesis and `analyze_step2.py` were written before this smoke test and are kept unchanged.

## Training (`step2_train.py`)
Per unit, feature type and seed (23, 24, 25): two-branch model (as in all earlier runs) on 50,000 (image, question)
cells from 700 train images, tested on 20,000 cells from the 300 held-out images. Linear readout on frozen codes.
Additive ceiling = accuracy on the test cells of the best additive fit of the whole table.

## Tests (`analyze_step2.py`)
- S2-A (primary; symbolic): the answer table's additive ceiling predicts the linear readout.
  (i) Spearman(full - linear, 1 - ceiling) >= 0.6; (ii) among units with full accuracy >= 0.90 (at least 5),
  MAE(linear, ceiling) <= 0.05.
- S2-B (primary): the tasks need fusion, but frozen CLIP models learn only the additive part. The symbolic full model is
  >= ceiling + 0.03 in >= 70% of units, and among learnable CLIP units (full - majority >= 0.05, at least 5) the CLIP
  full model is <= ceiling + 0.02 in >= 70%.
- S2-C (reported): clustered PID synergy (BROJA / I_ccs, k = 10 clusters per modality, symbolic and CLIP features, as in
  Liang et al.) and exact MMI vs exact structure and vs the gap; partial correlations (not decisive, given collinearity).
