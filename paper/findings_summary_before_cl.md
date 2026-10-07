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

## 10. v4: a decomposition of joint survival (pre-registered)

For knowledge that needs both modalities, the share surviving both encoders' drift splits as
**r12 = P1·P2 + C + N**. P1 and P2 are survival under one encoder's drift. C means the same samples tend to survive or
fail under either drift. N means the joint drift does better (or worse) than both single drifts together.

| hypothesis | result | verdict |
|---|---|---|
| H1: N ≈ 0 where the modalities are independent; N > 0 on face–voice | synthetic + bridge median \|N\| 0.028; face–voice median N +0.067 | **PASS** |
| H2: an a-priori coupling score κ_c predicts N | Spearman +0.014 | FAIL |
| H3: a graded evidence law beats the discrete v3 law | R² 0.53 vs 0.69 | FAIL |
| H4: linear-readout recovery falls with interaction share | falls steadily (0.95 → 0.85), but by less than required | FAIL |

**Why H2 failed (post-hoc diagnostics).**
- The metric was faulty. We built drifts with known compensation on face–voice, and true N rose to +0.24, but κ_c stayed
  negative. κ_c measured interaction on mismatched replacement pairs, where compensation cannot exist. A version
  measured on real pairs did turn positive (+0.12).
- N itself is measured reliably (reliability 0.95 pooled, 0.70–0.99 within groups). Within groups, κ_c often tracked N
  (Spearman up to +0.89). The pooled correlation was ≈ 0 because each dataset had a different κ_c offset.

## 11. FIA: anchoring encoders to the old fusion head (pre-registered)

Fusion-Interface Anchoring (FIA) keeps the encoders compatible with the frozen task-1 head. We compared it with EWC,
LwF, L2 and Jacobian distillation, replay, DER++, joint training and oracle re-alignment, with ablations, λ sweeps and
Pareto fronts (48 units, 2,544 runs).

- **Verdicts: all three predictions failed.**
  - P-a (bridge): no regularizer reached matched plasticity, and FIA was below L2 at every λ.
  - P-b (natural data): FIA's advantage on synergy over redundancy was +0.020 (needed 0.03).
  - P-c (face–voice): FIA with only the joint term beat full FIA on both accuracy and retention.
- **Why (post-hoc): forgetting moves into the head.**
  - With FIA, the new encoders plus the *old* head keep ≈ 100% of task-1 synergy.
  - But the task-2 training rewrites the head, which FIA leaves free, so the knowledge is lost there.
  - Stronger anchoring made it worse: on face–voice, S retention was 0.34 with FIA vs 0.63 with plain fine-tuning.
- **Baselines:** LwF, which constrains the whole model's outputs, dominates the Pareto fronts on natural data (S
  retention ≈ 0.90–0.96).
- **Novelty check:** the method is also close to recent work (RCL and PMA, July 2026), and the linear-readout test is
  close to EMAP (Hessel & Lee, 2020).
- **Conclusion:** FIA is not a viable contribution. The head/encoder split is a useful diagnostic.

## 12. Kinds of cross-modal knowledge: comparative vs conjunctive (pre-registered)

**Idea tested.** Comparative knowledge ("is this the same class / person?") might survive coordinated drift, because
only agreement between the codes matters. Conjunctive knowledge (a random class-pair table, XOR-like) should multiply its
damage.

**Setup.** Two-view CIFAR, with view 1 and view 2 from different items. Both tasks pass the linear-readout check (a
linear readout scores 0.50, chance). All numbers use chance-corrected retention k. Excess = k12 − k1·k2. Nesting =
excess / (min(k1, k2) − k1·k2): 0 means the damage multiplies, 1 means the same pairs fail, and above 1 means the drifts
cancel.

**Kill test** (seeds 23–27, coupled task 2 = a cross-modal matching loss):
- Comparative excess +0.086 vs conjunctive ≈ 0: **PASS**, so the idea was kept for follow-up.
- But with both encoders drifted, comparative knowledge was still worse than with either drift alone.
- Raw N rises on any two-answer task just from coin-flip guessing (conjunctive raw N reached +0.17 while its excess was
  ≈ 0).

**Follow-up v2** (fresh seeds 28–32):

| test | result | verdict |
|---|---|---|
| T0 replication | comparative excess +0.063, conjunctive +0.003 | PASS |
| T1 uncoordinated drift (the two encoders come from runs on *different* new classes; their matching accuracy drops from 0.36 to 0.03) | C unchanged (0.067 vs 0.067); nesting 0.34 vs 0.25 | **same weak pairs fail**, no cancelling |
| T2 independent task 2 (each encoder learns its own task), compared at equal damage | comparative nesting 0.44 vs conjunctive 0.09 | **same weak pairs fail**; coupling is not needed |
| T3 re-set the decision threshold (one number) | recovers 8.5% of lost comparative knowledge (0% conjunctive) | FAIL |

**What actually happens (exploratory):**
- **Only "same" pairs break.** "Different" pairs stay at ≈ 0.93 survival while "same" pairs fall to ≈ 0.26: a broken
  code almost never produces a false "same".
- **C comes from mixing labels.** Within each label C ≈ 0–0.03, while pooled over labels it is 0.11. This happens
  because two groups with different survival rates are mixed.
- **Two drifted codes look alike.** On "same" pairs, two drifted encoders say "same" more often than the single drifts
  predict (N +0.10 to +0.16). This is equally true for uncoordinated and independent drift, so it is not cancelling.
- **XOR of attributes behaves like conjunctive** (excess ≈ 0).
- **Face–voice shows the same pattern.** Split by label, C ≈ 0 and N is +0.14 on "same" pairs and ≈ 0 on "different"
  pairs. Its chance-corrected excess is still +0.09 to +0.17. H1's PASS stands as registered, but the reading changes
  from "drifts cancel" to "damage falls on 'same' pairs, and drifted codes look alike".

**Conclusion.** Drop "coupling protects comparative knowledge", the coupling fix, and "comparative knowledge only needs
re-calibration". What survives: conjunctive damage multiplies (nesting ≈ 0); comparative damage is one-sided and grows
with drift, whatever caused it. This is closer to a property of "same?" tasks than to a new kind of knowledge.

## 13. Lessons about the measures

- **Chance floor.** On two-answer tasks, raw survival events inflate N and C. Use chance-corrected k.
- **Label mixing.** On "same?" tasks, damage falls on one label. Split by label before reading C.
- **Off-pair measures.** Anything measured on mismatched replacement pairs (such as κ_c) cannot see effects that exist
  only on real pairs.
- **Damage level.** The raw excess grows with damage. Compare conditions at equal damage (k1·k2), or use nesting.

## 14. Bottom line

1. **Mostly a hard-sample effect on natural data:** knowledge that needs both modalities is forgotten first (140/140
   natural units across 7 datasets, with replay). But about 70% of that gap is explained by the old model's lower
   confidence on those samples (§15). A smaller multimodal-specific gap remains (−0.06, 82% of units).
2. **Clean-limit law:** where interaction types are exact and the modalities' damage is independent (synthetic data,
   P1 bridge), the damage from each encoder's drift multiplies under the old fusion head (R² ≈ 0.99). On natural data
   only the qualitative core holds.
3. **Where forgetting lives:** in the encoders' fit to the fusion head (Phase 1) and, once the encoders are anchored,
   in the head itself (FIA). Methods that protect only one of the two fail. LwF protects both.
4. **Benchmark diagnostic:** high dataset-level PID synergy does not mean a model uses non-additive synergy. CLEVR and
   VQA are solved mostly additively by frozen-feature models; a linear-readout test exposes this.
5. **How damage combines depends on the task:** conjunctive (XOR-like) knowledge multiplies its damage. For "same?"
   knowledge, damage is one-sided and failures are shared, with no evidence that the drifts cancel.
6. **Not viable as contributions:** a PID curriculum, PID-weighted multimodal training, FIA, and the coupling-based fix
   for comparative knowledge.

## 15. Is "needs both goes first" just "hard samples go first"? (exploratory, existing data)

We re-analysed the 140 natural units (P3/P4 dumps). Samples were binned by the old model's confidence in the true class
(deciles of logit p), and needs-both (S) survival was compared with redundant (R) survival *within* each bin.

| regime | raw S − R gap | S < R (raw) | gap at equal confidence | S < R (equal confidence) | share explained by confidence |
|---|---|---|---|---|---|
| replay | −0.221 | 100% of units | −0.058 | 82% | 74% |
| no replay | −0.103 | 70% | −0.035 | 62% | 66% |

- **Most of the effect is the known "hard samples go first" effect:** needs-both samples are the ones the old model is
  less sure about.
- **A smaller gap remains at equal confidence.** It is clearest on Food-101 (−0.08 / −0.15), CLEVR (−0.08, 100% of units,
  replay) and two-view CIFAR (−0.04 / −0.10). It is absent or reversed on AV-MNIST, CREMA-D and VQA without replay.
- **Unique knowledge shrinks the same way:** its gap to redundant knowledge falls to −0.01 to −0.04.
- **Contrast with the synthetic result (§3):** there, the 2× gap held at equal starting confidence. On natural data,
  claims about multimodality must be stated at equal confidence.

## 16. What a linear readout can't do follows XOR structure, not PID synergy (pre-registered)

**Question.** Does the linear-readout gap measure synergy, or something else?

**Theory.** A yes/no table over (class i, class j) can be written as [a_i + b_j > 0] exactly when it has no 2×2
checkerboard ("XOR square"). These tables are the Ferrers relations, or biorders (Ducamp & Falmagne 1969). So AND has
PID synergy (0.50 bits BROJA) but is additive.

**Design.**
- 63 label tables over 10 × 10 two-view CIFAR classes, chosen from a pool of 1,368 so that synergy and XOR density are
  nearly uncorrelated (+0.17).
- Exact PID with `dit` (BROJA primary; I_ccs and MMI reported).
- 3 seeds per table. The gap is the full model's accuracy minus the linear readout's, as a share of the full model's
  accuracy above the majority baseline.

**Results.**
- **P1 (primary): PASS.** The gap follows structure, not synergy:
  - ρ(gap, XOR density | synergy) = **+0.91** [95% interval 0.83, 0.94];
  - ρ(gap, synergy | XOR density) = +0.22 [−0.06, 0.45];
  - the same with I_ccs (+0.85 vs +0.25) and MMI (+0.91 vs +0.22).
- **Tables with no XOR squares:** the gap is ≤ 0.09 for all 16 of them (median 0.06), with synergy up to 0.67 bits.
- **P3:** the additive distance (fewest cells to flip to reach a threshold table) predicts about as well as density
  (Spearman +0.91 vs +0.91), and each adds something to the other. Controlling for additive distance, synergy adds
  nothing (−0.03).
- **P2 (matched pairs): INCONCLUSIVE.** My density cut-off (0.10) was above every selected table (maximum 0.089).
- **Exploratory, with the in-band top tercile as the cut-off:** 15 pairs matched on synergy, median gap 0.06 vs 0.43.
  60% of them meet the strict thresholds.

**Meaning.** PID synergy (as used by multimodal benchmarks) is not a measure of how much a task needs non-additive
fusion. Structural non-additivity (XOR squares, additive distance) is. This also explains P3/P4: benchmarks with high PID
synergy were solved mostly additively.
