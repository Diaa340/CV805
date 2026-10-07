# Findings summary: PID, curriculum learning, and multimodal forgetting

All experiments were run on one GPU (RTX A6000, GPU 2). Key claims from the route-law stage onwards were pre-registered
and hashed before running (`paper/preregistration*.md`).

## 1. The original idea (PID-ordered curriculum on CIFAR-100) does not work

The proposed difficulty score `syn − red` is mathematically identical to the local co-information
`i(x1,x2;y) − i(x1;y) − i(x2;y)` for *any* two-source PID, so the PID adds nothing beyond classical information theory.
Across 15 curricula (3 seeds, 30 epochs), plain training beat every hard-paced curriculum. The pacing itself cost
0.4 points on clean labels and 2.5 points under 40% label noise. Loss reweighting (SuperLoss) was the only large win,
at +9.9 points under noise.

| CIFAR-100, test acc. (%) | clean | 40% label noise |
|---|---|---|
| random (plain training) | **74.10** | 46.16 |
| random order + same pacing | 73.74 | 43.65 |
| PID `syn − red` curriculum (proposed) | 73.51 | 44.20 |
| SuperLoss | 73.50 | **56.06** |

## 2. Multimodal curricula: PID never beat mutual information (MI)

On a synthetic two-modality benchmark with known per-sample types, up-weighting synergistic samples gave +9.3 points,
winning in 10 of 10 seeds. Up-weighting "hard" samples by MI alone gave the same (+9.3), because hard = synergistic
there. Under label noise, PID-based filtering equalled MI filtering (+4.0 vs +3.9). On CREMA-D the real problem was
modality imbalance: the visual branch was unused, even though video alone was the best modality (60.6%). Simple
per-modality auxiliary losses beat the published fix OGM-GE (59.1 vs 55.0), and PID-weighted variants did not beat
MI-weighted ones.

## 3. A real effect: synergistic knowledge is forgotten first

In continual learning (train task 1, then task 2), synergistic knowledge was lost about **twice as fast** as redundant
knowledge **at equal starting confidence**. That is, synergy retained ≈ 0.49–0.51 × redundant, in every robustness
check: equal modality strength, equal type mix, and both combined. Changing only one modality in task 2 destroyed that
modality's unique knowledge and all synergistic knowledge. Redundant knowledge survived on the other route.

## 4. Searching for the law behind it: two refutations, then a mechanism

- **v1: "route law" with modality-level routes** (pre-registered, 447 units on synthetic + 5 real datasets).
  - The zero-parameter prediction ranked forgetting correctly (r = 0.82 for redundant, 0.89 for synergy). It beat
    every baseline for redundant knowledge, but failed calibration: R² 0.39 / 0.32, synergy under-predicted by +0.13.
  - Causal interventions moved retention in the predicted direction. Damaging one route was predicted almost exactly
    (synergy change −0.355 observed vs −0.352 predicted).
- **v2: content lost inside encoders** (pre-registered). **Refuted.** The encoders keep 93–97% of their task-1 content,
  yet synergy retention falls to 0.50 (predicted 0.93).
- **Phase 1 / 1b: where the damage is.**
  - Re-fitting the fusion layer did not restore synergy.
  - Without replay, re-aligning the *encoders* to the old fusion layer restored 79% of the lost synergy; re-fitting the
    fusion layer restored 20%.
  - The fusion layer keeps the interaction; the encoders drift away from it.

## 5. v3: forgetting as composed encoder drift (pre-registered)

Under the task-1 fusion layer, the damage from each modality's encoder drift **multiplies**. Synergy is the only type of
knowledge damaged by *both* modalities, so it multiplies two damage factors, and that is why it is forgotten first.

| v3 criterion | Synthetic (135 held-out units, exact types) | Real data (100 units, estimated types) |
|---|---|---|
| (a) composition: synergy = damage₁ × damage₂ | **PASS**: R² 0.99, MAE 0.018 | FAIL: R² 0.62, MAE 0.156 |
| (b) only synergy has two damage factors | **PASS**: 0.68 / 0.64 (unique: ≈1.0 from the other modality) | FAIL: "unique" samples damaged by the other modality in ~50% of units |
| (c) drifted encoders + old fusion reproduce actual forgetting | **PASS**: R² 0.83 (replay) | FAIL: R² 0.29 |

## 6. P1: the law holds on real features when types are exact (pre-registered)

In a semi-synthetic bridge, modality 1 comes from one real item and modality 2 from a *different* real item: two-view
CIFAR-100, Food-101 image + title, and N24News image + headline. Interaction types are fixed exactly by construction.
All three v3 criteria pass on the 90 pooled units, with synergy learned in 100% of units. Composition for synergy:
R² 0.993 (0.989 on the non-trivial units). Two-factor signature: unique knowledge is undamaged by the other modality in
80% / 98% of units, and synergy is damaged more than redundancy (+0.32). Reproduction for synergy under replay: R² 0.96.
CIFAR and Food-101 pass everything; N24News fails the unique check (67%) and reproduction (R² 0.54). So v3's real-data
failure came from **type estimation, not the law**.

## 7. P2: on natural data the law holds only qualitatively (pre-registered)

On 5 natural datasets (100 units), interaction types were defined by the task-1 model's own behaviour: which modality it
needs to classify each sample, with the other modality replaced by a random real sample. All three strict criteria failed
on the pooled data: synergy composition R² 0.69 (MAE 0.107); unique knowledge undamaged by the other modality in only
56% / 67% of units; reproduction R² 0.64. The qualitative core survived: synergy is damaged by both modalities' drift
(0.66 / 0.67), far more than redundant knowledge (0.89 / 0.88). Two-view CIFAR's synergy composed almost exactly
(R² 0.94). Likely reason: on natural data the "other" modality is never neutral (it carries partial or misleading
evidence), so drift damage leaks across types. Model-defined types agreed with probe-estimated PID types on only 23–39%
of samples.

## 8. P3: on natural data, retraining the readout recovers synergy too (pre-registered)

Without replay, retraining a fresh fusion head on the drifted encoders, using the task-1 data, recovered most of the lost
knowledge of every type. Median recovery: redundant 0.98, unique 0.92 / 0.94, synergy 0.84. Synergy was the least
recoverable type in 93% of units, but the margin was small (+0.07; needed +0.15), and re-aligning the encoders beat
readout retraining for synergy by only +0.12 (needed +0.20). **Claim not supported.** A *linear* readout, which cannot
represent true synergy, recovered almost as much (0.81). So what the network needs "both modalities" for on natural
data is mostly **additive evidence**: two weak cues summed. It is not the XOR-like synergy of the synthetic studies, and
its forgetting is largely an access problem, as in unimodal models.

**Exploratory (not pre-registered): a graded version of the law.** Per-sample survival under one modality's drift falls
steadily with how much the sample relies on that modality (no replay: 0.97 → 0.66). But graded composition only works
with replay (R² 0.82) and fails without it (R² −0.05). The graded law does not predict per-type retention better than
the discrete one.

## 9. P4: the least redundant real tasks (pre-registered)

The tasks were chosen from published PID estimates (CLEVR ~83% synergy, VQA 2.0 ~75%), plus CREMA-D face–voice matching
("same person?"), which is synergistic by construction. A manipulation check came first: can a *linear* readout on the
task-1 encoders reach the model's accuracy on its "needs both" samples? It could for VQA (91%) and CLEVR (73%), so with
frozen CLIP features even these benchmarks are solved mostly additively, and they were not tested. Only face–voice
matching passed (0%). There the law **failed** all three criteria (composition R² 0.31; readout retraining recovered
126%). One encoder drifting cost as much as both (0.79 / 0.82 vs 0.79): the two modalities drift *together* on this
shared skill. Task 2 also mostly *helped*, rather than conflicted with, task 1.

**A robust general finding (exploratory):** across all 140 natural-data units with model-defined types (7 datasets:
two-view CIFAR, AV-MNIST, CREMA-D, Food-101, N24News, CLEVR, VQA), "needs both modalities" knowledge was retained less
than redundant knowledge in **100% of units**, and was the least retained type in **95–100%** of units (replay regime).

## 10. Bottom line and open problem

The mechanism is established wherever interaction types are known exactly: on toy data, and on real learned features
(P1). Forgetting happens in the encoders, not the fusion layer. Synergy is fragile because it depends multiplicatively on
every modality's representation staying aligned with the fusion interface. On natural data only the qualitative
core holds (P2): synergy is damaged by both modalities and forgotten first, but the exact multiplication breaks. And on
these benchmarks, what looks like synergy is mostly additive evidence that readout retraining recovers (P3). **True
synergy is fragile, but rare in standard multimodal classification data.** Even the most synergistic published benchmarks
(CLEVR, VQA) are solved mostly additively by frozen-feature models (P4). On the one genuinely non-additive real task
(face–voice matching), the multiplicative law failed, because the two modalities drift together.

**Final picture.** (1) Robust everywhere: knowledge that needs both modalities is forgotten first (7 natural datasets,
140/140 units). (2) The multiplicative drift law is a *clean-limit* result: it holds when the modalities' damage is
independent (synthetic data, constructed tasks on real features), but not on natural tasks, where modalities co-drift
and "synergy" is mostly additive. (3) Benchmark diagnostic: dataset-level PID synergy does not mean a model uses
non-additive synergy; a linear-readout test exposes the gap.
