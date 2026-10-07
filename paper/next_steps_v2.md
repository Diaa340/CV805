# Next steps, anchored on the label-table result (2026-09-30)

## The result we build on
Across 63 label tables with PID synergy held in one band (0.40–0.75 bits), the part of a two-modality task that an
additive (linear) readout can't do follows **XOR structure** (ρ = +0.91 after controlling for synergy), **not PID
synergy** (+0.22 after controlling for structure). Tables with no XOR squares carry up to 0.67 bits of synergy and are
solved additively (gap ≤ 0.09).

## Thesis (one line)
**PID synergy is not the same as the need to fuse.** Part of synergy is *removable* (AND-like): an additive model
followed by a threshold captures it. Only the *non-removable* (crossover / XOR-like) part needs real fusion, and that
part is what matters for forgetting, repair and compositional generalisation.

## Where it sits in the literature (what is old, what is new)
- **Old: the maths.**
  - A binary table is threshold-additive exactly when it has no 2×2 checkerboard (Ferrers relations / biorders;
    Ducamp & Falmagne 1969; Doignon, Ducamp & Falmagne 1984).
  - In psychology, "removable" (non-crossover) and "non-removable" (crossover) interactions are the same distinction
    (Loftus 1978; Wagenmakers et al. 2012; additive conjoint measurement, Krantz et al. 1971).
- **Old: the observations.**
  - Many multimodal benchmarks are solved additively (EMAP; Hessel & Lee 2020).
  - PID synergy appears even in linear Gaussian systems (Barrett 2015).
- **New: the link and its consequences.**
  - Multimodal ML uses PID synergy as "fusion demand" (Liang et al. 2023; the ICML 2026 PID-for-MLLMs paper), steers
    learning with it (SynCo, Sept 2026) and builds curricula on it (SPICE, June 2026). None of these separates the
    removable part from the non-removable part.
  - We show which of the two predicts model behaviour, and give a quantity that can be measured on natural data.

## Plan (in priority order; every step pre-registered)

### Step 1: consequences in the table framework (≈ 1 day, cheap and decisive)
Same 63 tables plus a fresh replicate set. The rule each time: structure predicts the outcome, synergy doesn't.
- **1a, repair after forgetting (continual learning).** Train on the table, drift the encoders with task 2, then retrain
  a *linear* head vs an MLP head on the drifted encoders.
  - Prediction: linear-head recovery falls with additive distance, not synergy.
  - This retests P3's "recoverable synergy" and v4 H4 (which failed on off-manifold β), now with exact structure.
- **1b, compositional OOD.** Hold out 20% of the (class i, class j) cells during training and test on the unseen
  combinations.
  - Prediction: unseen-cell accuracy falls with additive distance / XOR density, not synergy. Threshold tables should
    generalise almost fully, because the row and column effects are enough.
- **1c, the multiplicative forgetting law restricted to the non-removable part.** Split each table into its nearest
  threshold table plus the flipped cells.
  - Prediction: the v3 two-factor composition holds for the flipped (XOR) cells. For removable cells, damage is
    repairable by the readout.
- **1d, redo P2 properly.** Construct matched-synergy pairs on purpose (a threshold table and an XOR-block table tuned
  to the same BROJA synergy), rather than hoping the random selection contains them.

### Step 2: a real benchmark with exact structure: CLEVR (≈ 2–3 days)
- **Why CLEVR:** the archive contains scene graphs and question programs, so any question can be executed on any image.
  That gives the true answer table (question × image) for every question family, with no model needed.
- **Structure per family:** compute XOR density and additive distance on yes/no families (weighted by how pairs are
  actually sampled), and compare with PID synergy estimated as in Liang et al.
- **Test:** across families, the frozen-feature additive-readout gap (and forgetting and repair from step 1) follows
  structure, not PID synergy.
- **Link to earlier results:** this ties back to P4, where CLEVR had high dataset-level PID synergy (~83%) but was
  solved mostly additively.

### Step 3: an estimator for data without label tables (≈ 1 week)
- **Model-based structure:** sample quadruples (x1a, x1b, x2a, x2b), query a strong joint model on all four cross pairs,
  and count sign-crossover (checkerboard) 2×2 patterns. Separately, train the best additive model (f1(x1) + f2(x2),
  jointly trained) and measure its gap to the joint model.
- **Validation:** check the estimator against the exact values on the tables and on CLEVR before using it elsewhere.
- **Use:** rank the earlier datasets (CIFAR 2-view, AV-MNIST, CREMA-D, Food-101, N24News, VQA, face–voice) by
  non-removable interaction vs PID synergy. This would be the benchmark diagnostic.

### Step 4: theory note (in parallel)
- **Propositions to prove:**
  - (i) the Ferrers characterisation and a density bound, d_add ≥ density·C(K,2)² / (K−1)²;
  - (ii) PID synergy can be high with d_add = 0 (threshold tables) and low with d_add > 0 (a weak XOR block), for
    BROJA / I_ccs / MMI;
  - (iii) the non-uniform sampling version (weighted checkerboards), which is needed for "same?" tasks such as
    face–voice (the identity table has d_add = 1/K under uniform sampling);
  - (iv) multi-class labels via one-vs-rest.
- **Link to the forgetting law:** only non-removable knowledge carries the two-factor damage that a readout can't
  repair.

## What to drop
FIA, the comparative/"kinds" line, κ_c, and PID-weighted curricula. The "needs-both goes first" result on natural data
should be quoted at equal confidence only (−0.06).

## Paper shape (working title)
*"Synergy is not fusion: a structural account of multimodal interaction and what it predicts about forgetting and
compositional generalisation."*
- **§1 Theory:** removable vs non-removable interaction; PID synergy mixes the two.
- **§2 Controlled tables:** readout gap (done), repair, compositional OOD, the composition law.
- **§3 CLEVR:** exact structure on a real benchmark.
- **§4 Estimator and benchmark ranking.**
- **§5 Implications:** for PID-based curricula and synergy-seeking objectives.

## Main risks
- **"XOR isn't linearly separable" reads as textbook.** The paper has to show that the field actually uses PID synergy
  as fusion demand (cite it), and that the consequences are practical (CL repair, OOD), not just the gap.
- **Class-level tables differ from natural data.** Step 3's estimator could be noisy. Step 2 (CLEVR) is the bridge that
  makes the claim credible.
- **PID estimator dependence:** report BROJA, I_ccs and MMI, and Liang et al.'s estimator, throughout.
