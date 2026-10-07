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

## 14. Bottom line (as of 2026-09-30; see §28 for the current one)

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

## 17. Rules and exceptions: exact structure, beyond the readout gap (pre-registered)

**Idea.** For a yes/no answer table over (question, image) cells, the best additive rule f(a,b) = [u_a + v_b > 0] is
computed exactly (MILP); the cells it gets wrong are the **exceptions**. A table has no exceptions exactly when it has no
XOR square, and then it is the only table with its row and column counts (Ryser). Exceptions are what an additive /
late-fusion model cannot get right, whatever its features.

| test | result |
|---|---|
| Step 1, R1 (readout gap follows XOR density on fresh tables) | **STRUCTURE WINS** (+0.86 vs +0.11 for BROJA) |
| Step 1, A (linear repair after drift fails on exceptions) | **STRUCTURE WINS** (+0.86 vs −0.10) |
| Step 1, B (compositional OOD) | SYNERGY WINS (+0.31 vs +0.36) |
| Step 1, D (matched-synergy pairs by construction) | **PASS** (gap +0.50 dense vs zero) |
| Step 2 on CLEVR (exact tables from scene programs): S2-A / S2-B | **PASS** / FAIL |
| Identical PID, different structure | **PASS** (within-family ρ = +0.78) |
| Liang et al. PID estimator on CLEVR | setting-dependent (underestimates in 16/16 settings) |
| Additive estimator recovers the exact ceiling (symbolic) | **PASS** (E1) |

**Meaning.** What an additive model cannot do is fixed by the answer table's exceptions, not by PID synergy.

## 18. VLMs fuse; benchmarks mostly do not need them to (pre-registered)

- **VLM test (Qwen2.5-VL-7B, CLEVR):** accuracy 0.928; verdict MIXED.
- **EMAP:** the VLM uses interactions (M2 PASS: +0.16 over its additive projection); frozen-feature models on 7
  benchmarks are additive (B1 PASS, 7/7); the EMAP gap follows exceptions, not BROJA (T1 PASS, +0.84 vs −0.17); EMAP of a
  strong model does not recover the exact ceiling (M1 FAIL).
- **POPE:** the VLM fuses (gap +0.30); the PID result depends on how a modality is removed (S share 0.56 vs 0).
- **GQA:** the VLM fuses (+0.29); additive fits are only a lower bound on the exact ceiling (G1, G3 FAIL).
- **MMBench / MMStar:** additive recombination recovers 90% of above-chance accuracy on MMBench (PASS); MMStar truly
  needs fusion (FAIL).
- **Do VLMs forget fusion first?** (OCR fine-tuning, CLEVR cells): **KILL**: matched gap +0.012 [−0.003, +0.025].

## 19. Which forgotten samples cannot be repaired? (pre-registered)

- U1 / U2 (the binary exception label predicts unrepairable samples on every benchmark, better than synergy and
  confidence): **FAIL**.
- **F2 (fresh seeds, graded misfit):** **PASS**: AUC 0.79 / 0.94 / 0.99 / 0.78 vs confidence 0.70 / 0.77 / 0.63 / 0.66.
- Linear repair on exceptions is worse than chance (chance-corrected −0.27 to −0.99).

## 20. CL methods on rules vs exceptions: first study (pre-registered; constructed tasks, CLEVR)

| test | result |
|---|---|
| L1 protective methods lose exceptions more (matched confidence) | MIXED (CLEVR ER −0.21, DER++ −0.16, LwF −0.10; unprotected methods at chance) |
| L2a replay coverage explains it | FAIL |
| L2b LwF protects the rule more than exceptions | **PASS** (+0.08 tables, +0.056 CLEVR) |
| L3 rule+exceptions buffer vs best buffer, no overall cost | FAIL (exceptions +0.10 to +0.24, overall −0.03 to −0.11) |
| F1 (fresh seeds, DER++ rule+exceptions, buffer 1000) | **PASS** (tables exceptions +0.064, overall −0.015; CLEVR +0.146, overall +0.001) |

- **Aggregate BWT hides exception loss:** on CLEVR, DER++ has overall BWT +0.001 but −0.096 on exceptions.

## 21. Exception-aware BWT / FWT, 13 methods, four benchmarks (pre-registered)

- **New metrics:** BWT and FWT by sample type; kept share (Acc − 0.5) / (Acc_before − 0.5); FWT against a task-only model;
  type-balanced accuracy.
- **Methods by family:**
  - regularisation: EWC, SI, LwF, feature distillation;
  - memory: ER, A-GEM, GEM, DER++, iCaRL;
  - architectural: DER-expansion (Yan et al.), multi-head.
- **Verdicts:** M1 PASS, M2 PASS (constructed tasks, CLEVR, GQA); M3 FAIL.
- **POPE design flaw (found after the run):** task 2 used cells of the same answer table as task 1, so it taught task 1's
  facts. POPE was dropped.
- **Kept share** is unstable when exceptions start near chance; BWT by type is used instead.

## 22. CLOVE-scene replaces POPE (pre-registered)

- **Benchmark choice:**
  - Of VQACL, CL-CrossVQA, UCo-VQA and CLOVE, only CLOVE has scene graphs and programs, so only CLOVE allows exact
    answer tables.
  - UCo-VQA (CVPR 2026) independently flags the shared-answer flaw we found in POPE.
- **Build:**
  - 43 existence units across 6 scene tasks, all labels proven exact.
  - Relation questions are true on only ~1% of other images, so they were dropped (one addendum tried a fix; no
    change).
- **Small networks, six tasks, 12 runs:**
  - C1 PASS: LwF, ER and DER++ lose exceptions faster, by −0.12 to −0.18.
  - C2 PASS: DER++ hinders new exceptions.
  - C3 FAIL, narrowly: the buffer gains +0.23 on exceptions, but the overall cost is −0.019 (lower bound −0.024).
  - C4 PASS: DER-expansion loses fewer exceptions than ER.
- **Qwen2.5-VL-3B + LoRA, six tasks, 2 orders:**
  - V1 PASS: LwF and ER.
  - V2 FAIL, narrowly: −0.029.
  - V3 FAIL: DER++ already keeps exceptions.

## 23. The answer-class confound, and same-answer metrics (post-hoc check, then pre-registered)

- **The confound.** Exception cells are mostly "yes" on existence tables (GQA 92% vs 6% for rule cells; CLOVE 89% vs
  8%), and models drift toward the majority "no". So the raw exception gap partly measures answer class.
- **Same-answer metrics.** dBWT_s / dFWT_s compare exceptions with rule cells that have the same answer. They are now
  the standard. Answers for saved runs were rebuilt and every row checked against the saved labels.
- **Part A, descriptive** (I saw its output while checking the script, before pre-registering; disclosed):

  | | backward (old exceptions) | forward (new exceptions) |
  |---|---|---|
  | CLEVR | memory methods −0.09 to −0.11, all intervals below 0 | DER++ −0.26, feature distill. −0.13, ER −0.06 |
  | Constructed | about −0.01 (DER++, LwF) | every method −0.05 to −0.08 |
  | GQA | DER++ −0.03; others ≈ 0 | DER++, LwF, feature distill. ≈ −0.07 |
  | CLOVE (small and VLM) | none | small (feature distill. −0.07) |

- **Part B (confirmatory): class-balanced buffer control.**
  - B1 PASS (3/4): exception cells in the buffer help beyond class balance (+0.068 to +0.085). GQA is the exception,
    where balance alone explains the gain.
  - B2 FAIL (2/4): the overall cost is −0.027 / −0.028 on constructed tasks and CLOVE.

## 24. A method guide, checked leave-one-benchmark-out (pre-registered)

- **Design.** Descriptors measured before any CL run: exception share, answer skew, yes share, and how well exceptions
  are learned. The guide predicts each method's final exception accuracy on a benchmark it has not seen.
- **Results.**
  - G1 PASS (3/4): the descriptors beat a fixed ranking.
  - G2 FAIL (1/4): the usual ranking by overall BWT is a strong baseline (constructed tasks: Spearman 0.62 vs 0.18).
- **Target bias (post-hoc).** iCaRL says "yes" more often, so it "wins" on exception accuracy wherever exceptions are
  mostly "yes". An answer-balanced target would fix this; not run.
- **What the fit says.**
  - Answer skew lowers every method's exception accuracy, most for distillation.
  - Well-learned exceptions are kept best by replay and expansion.

## 25. VLM on CLEVR and nonce-word CLEVR

- **Plain CLEVR:** zero-shot accuracy is 0.92–0.98 for rule and exception cells alike (6,000 cells). A CL sequence
  cannot teach anything, so the run was put on hold.
- **Nonce-word CLEVR** (each task renames the 15 attribute concepts with its own made-up words; labels unchanged;
  pre-registered):
  - **The gate was not met:** task-only 0.651 vs zero-shot 0.587, short of the required +0.10. VC1–VC3 are not
    interpreted.
  - **Why (post-hoc).** The model learns the rule (0.64 → 0.86) but exceptions get worse (0.53 → 0.44; exception "no"
    cells 0.75 → 0.28). Fine-tuning installs the additive rule and overrides exceptions, so the bottleneck is
    **acquisition, not retention**.
  - **Post-hoc:** class-balanced ER ends with the highest exception accuracy (0.62 vs 0.47–0.54).
  - Two operational events: an out-of-memory crash, and a resume that retrained task 1 (agreement 0.88–0.89). Both
    disclosed.
- **Pretraining as task 0** (the user's point; post-hoc). Measured from zero-shot with same-answer gaps:
  - **Related fine-tuning (CLOVE, six tasks):** −0.07. Exceptions lose more.
  - **Unrelated fine-tuning (OCR):** +0.04, mixed by answer. Not exception-first, matching the KILL in §18.
  - **Nonce-word CLEVR:** −0.36, but confounded with the zero-shot "no" bias.
  - A clean test (plain CLEVR, measuring untrained tasks) was proposed, not run.

## 26. Main results table

- `paper/table_main.tex`, with a preview at `paper/table_preview/table_main.png`. Built by `make_table.py` and
  `render_table.py`.
- **Columns per benchmark:** ACC, BWT, exception accuracy, and same-answer ΔB / ΔF.
- **Rows:** families as in the reference table. The rule+exceptions buffer is a greyed "diagnostic probe".
- **Headline:** the best method by standard BWT is not the one that keeps exceptions. On CLEVR, DER++ has the top BWT
  (+0.1) but 34.9% exception accuracy; DER-expansion leads ACC and exception accuracy on 3 of 4 benchmarks.

## 27. Disclosures (slips and deviations)

- Importing `gqa_build.py` (no main guard) rebuilt the GQA tables. Verified identical: the saved labels still make all
  48 tables exception-free.
- The Part A same-answer output was seen before the pre-registration, so Part A is descriptive.
- Pilots and smoke runs were seen before several pre-registrations (each file lists what was seen).
- POPE task-1 training labels hit the 30-minute limit (best rule 0.774, bound 0.886).
- `cl_metrics.py` crashed at launch on CLEVR / GQA / POPE (missing cell indices). Fixed, with an addendum.
- In the CLOVE build, relation questions were dropped by the pre-registered yes-share rule; one addendum changed nothing.

## 28. Current bottom line

1. **Structure, not synergy.** Exceptions of the exact additive rule, not PID synergy, decide what additive models,
   readouts and cheap repairs cannot do (§16–17).
2. **Exceptions are learned worse: the most general CL finding.** Within the same answer, sequential training and
   protective methods (DER++, distillation) learn new exceptions worse than the new rule. This holds on constructed
   tasks, CLEVR and GQA, and for a VLM learning new words, where fine-tuning overrides exceptions.
3. **Forgetting old exceptions depends on the data.** It is strong on compositional CLEVR (memory methods −0.09 to
   −0.11); small or absent on object-existence tables, where an "exception" is mostly a rare "yes".
4. **Measure within the same answer.** Raw exception metrics on existence tables mostly measure class imbalance.
5. **Method guidance so far.**
   - DER-expansion keeps exceptions best among standard methods.
   - Class-balanced replay fixes most of the existence-table loss.
   - Distillation is worst for learning new exceptions.
   - Replaying exception cells helps beyond class balance, at an overall cost of 0.01–0.03.
6. **VLMs.**
   - Strong VLMs fuse and already answer in-domain exceptions (CLEVR zero-shot 0.92–0.98).
   - When they must learn new concepts, LoRA learns the rule first and overrides exceptions.
   - Pretrained exceptions are lost faster only under related fine-tuning (post-hoc).
7. **Open.**
   - A clean pretraining-as-task-0 test.
   - Enough VLM budget to learn exceptions before testing their retention.
   - An answer-balanced guide target.
   - Updating the HTML storyline and the methodology to these claims.

## 29. Does task structure predict how much modern VLMs depend on interactions? (pre-registered, three studies)

**Question.** Exception share s (a property of the answer table, model-free) and the EMAP drop Δ = Acc(model) −
Acc(EMAP(model)), the accuracy a model loses when its interactions are removed, are computed on identical, fully labelled
cells. Does s predict Δ, and does PID synergy?

**Theory.** EMAP's hard prediction is additive, so its error is at least s. Hence **Δ ≥ s − e** for any model with error
e. For accurate models part of the s–Δ relation is guaranteed; the tests target what is not.

**Data.** 99 complete 10 × 100 answer tables where every question is well-posed on every image: CLEVR 24, GQA 24,
CLOVE-scene 43, POPE 8.
- Exact rules: 95 proven.
- BROJA: convex solver, matches dit to ≤ 0.0003 bits.

**Models.** Zero-shot, 7–8B: Qwen2.5-VL-7B, InternVL3.5-8B (one tile), LLaVA-OneVision-1.5-8B (publisher code, pinned
and read; run with transformers 4.53).

**Accuracy and EMAP drop.**

| | CLEVR | GQA | CLOVE | POPE |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 0.910 / +0.157 | 0.936 / +0.057 | 0.927 / +0.056 | 0.977 / +0.027 |
| InternVL3.5-8B | 0.973 / +0.198 | 0.900 / +0.059 | 0.860 / +0.056 | 0.935 / −0.002 |
| LLaVA-OV-1.5-8B | 0.988 / +0.165 | 0.944 / +0.062 | 0.925 / +0.054 | 0.974 / +0.024 |

- **The bound Δ ≥ s − e holds for every model on every unit.**
- **Cell level:** graded misfit predicts the cells whose answer EMAP flips (AUC 0.78–0.88) better than low confidence
  (0.64–0.74), for all three models.

**Study A, full grids: all tests FAIL.**
- Structure predicts Δ within CLEVR and CLOVE: InternVL +0.91 / +0.76, LLaVA +0.97 / +0.58, Qwen +0.37 / +0.61.
- But PID predicts almost as well, because within each benchmark s and BROJA are nearly collinear (Spearman +0.88 to
  +0.95; POPE +0.48). The question cannot be answered on these units.

**Study B, decorrelated sub-tables: all tests FAIL, informatively.**
- **Design.** One 50-image sub-table per grid, chosen from 2,673 candidates using answer tables only, so that within
  each benchmark Spearman(s, BROJA) = 0.00. Confirmatory models: Qwen and LLaVA.
- **Structure predicts Δ independently of PID in all three models:** partial ρ(Δ, s | PID, acc) = +0.33 / +0.32 /
  +0.37, all intervals above 0.
- **But PID keeps a signal of its own** (+0.19 / +0.43 / +0.44).
- **Regression:** β_s +0.41 [+0.19, +0.57], β_PID +0.21 [+0.04, +0.38]. s adds R² +0.034 (0.05 needed); PID adds
  +0.006.
- **Post-hoc (exploratory):** after decorrelation, BROJA tracks the yes share (+0.57 to +0.83). With yes share held
  fixed, structure strengthens and PID's signal disappears:

  | | Qwen | LLaVA | InternVL |
  |---|---|---|---|
  | ρ(Δ, s \| PID, acc, yes) | +0.43 [+0.19, +0.58] | +0.37 [+0.15, +0.54] | +0.48 [+0.27, +0.63] |
  | ρ(Δ, PID \| s, acc, yes) | −0.07 [−0.27, +0.17] | +0.19 [−0.01, +0.38] | +0.11 [−0.09, +0.33] |

**Study C, confirmatory (running).**
- **New unseen model family:** Idefics3-8B (Llama-3 + SigLIP).
- **Fresh selection** with s uncorrelated with both BROJA and yes share (|ρ| ≤ 0.02; GQA s–BROJA +0.20).
- **Primary tests:**
  - C1: ρ(Δ, s | PID, acc, yes) ≥ 0.3, lower bound > 0;
  - C2: ρ(Δ, PID | s, acc, yes) interval contains 0 and |ρ| ≤ 0.15.
- **Results:** §30.

## 30. Confirmatory result: structure, not PID, with answer balance held fixed (pre-registered)

**Idefics3-8B** (new family, results unseen), on a fresh selection with structure uncorrelated with both BROJA and
the yes share.

| test | result | criterion | |
|---|---|---|---|
| C1 ρ(Δ, s \| PID, accuracy, yes) | **+0.42** [+0.21, +0.56] | ≥ 0.3, lower > 0 | **PASS** |
| C2 ρ(Δ, PID \| s, accuracy, yes) | **+0.07** [−0.17, +0.29] | contains 0, \|ρ\| ≤ 0.15 | **PASS** |
| C3 linear regression (secondary) | s adds R² +0.014, PID adds +0.021 | ≥ 0.05 / ≤ 0.01 | FAIL |

**Verdict: CONFIRMED** (primary = C1 and C2). The bound Δ ≥ s − e holds on every unit.

**Same units, earlier models (exploratory).**

| | C1 (structure) | C2 (PID ≈ 0) |
|---|---|---|
| InternVL3.5 | +0.42 ✓ | +0.05 ✓ |
| LLaVA-OV-1.5 | +0.43 ✓ | +0.24 ✗ |
| Qwen2.5-VL | +0.26 (just under 0.3) | +0.03 ✓ |

**Caveat.** The secondary linear regression fails for all four models. On raw values (not ranks), PID still adds about
2% of explained variance and structure 1–4%. Rank-based within-benchmark tests and the raw-scale regression weigh the
data differently. The 8 POPE units, where PID and Δ move together strongly, may drive part of this (untested).

**Claim supported.** Within a benchmark and at fixed answer balance, the structural non-additivity of the answer
table predicts how much a VLM's performance depends on cross-modal interactions. PID synergy adds nothing once structure
and answer balance are known. Because BROJA depends only on each modality's answer statistics (§17, Prop. 4), its raw
association with interaction dependence runs through answer balance.

## 31. Current bottom line (supersedes §28 items 1 and 6)

1. **Structure, not synergy, on controlled tables** (§16–17) **and on modern VLMs** (§29–30). Confirmed with an
   unseen model at fixed answer balance, in rank-based within-benchmark tests. A raw-scale regression gives both
   variables little beyond benchmark and answer balance.
2. **A guaranteed floor:** Δ ≥ s − e for every model; held on all 4 × 99 units.
3. **Cell level:** graded misfit identifies the cells where a VLM's answer depends on interactions (AUC 0.78–0.88) better
   than model confidence.
4. **Continual learning (§20–25, unchanged):**
   - exceptions are learned worse (forward), across benchmarks;
   - old exceptions are forgotten faster on compositional CLEVR;
   - exception metrics must be split by answer.
