# Removable vs non-removable interaction, and what PID synergy can see (theory note, draft)

## Setup
- Two modalities, each carrying a latent class: c1 ∈ {1..K1}, c2 ∈ {1..K2}, with cell probabilities p_ij. In our tables
  p is uniform; in CLEVR the "classes" are the images and the questions themselves.
- The label is a deterministic binary table, y = T[c1, c2].
- Each view is drawn from its own class, independently given the classes: x1 ~ P(· | c1), x2 ~ P(· | c2).
- An **additive rule** is any g(x1, x2) = 1[f1(x1) + f2(x2) > 0]. A linear readout on any per-modality codes,
  w1·z1(x1) + w2·z2(x2) + b, is additive, and so is EMAP's projection followed by a threshold.
- **T is threshold-additive** (a Ferrers relation, or biorder) if T_ij = 1[a_i + b_j > 0] for some reals a, b.
- **d_add(T; p)** is the smallest total p-weight of cells that must be flipped to make T threshold-additive.
- **XOR density ρ(T)** is the share of 2×2 submatrices (two rows × two columns) that are checkerboards: [[1,0],[0,1]]
  or [[0,1],[1,0]].

## Proposition 1 (characterisation; classical)
T is threshold-additive **if and only if** it contains no 2×2 checkerboard.

(Ferrers relations / biorders: Riguet 1951; Ducamp & Falmagne 1969; Doignon, Ducamp & Falmagne 1984. In the language
of 2×2 designs, a checkerboard is a crossover interaction; Loftus 1978.)

So AND and OR tables, and every "threshold of a sum" table, are additive, even though they carry PID synergy (AND:
0.50 bits under BROJA). XOR is the smallest non-additive table.

## Proposition 2 (additive ceiling)
Any additive rule, including a linear readout on any learned codes, has error ≥ d_add(T; p).

*Proof.* For each class i draw one x1^(i), and for each class j draw one x2^(j), all independently from their class
distributions. Set a_i = f1(x1^(i)) and b_j = f2(x2^(j)). This defines a class-level additive rule, so its error is
≥ d_add. For every cell (i, j), the pair (x1^(i), x2^(j)) has exactly the distribution of a test pair from that cell,
so the expected error of these random class-level rules equals the error of g. ∎

**Consequence.** The accuracy a linear readout can reach is bounded by 1 − d_add, which is computed from the label
table alone. It is approached when the codes separate the classes. The label-table test and CLEVR step 2 check how
tight this bound is.

## Proposition 3 (density bound)
For uniform p: d_add ≥ ρ / 4.

*Proof.* Let F be the nearest threshold table and m the number of cells where F differs from T. F has no
checkerboards, so every checkerboard of T must contain at least one of those m cells, and each cell lies in
(K1 − 1)(K2 − 1) submatrices. So #checkerboards ≤ m (K1 − 1)(K2 − 1). Dividing by K1·K2 and using
C(K1, 2) C(K2, 2) = K1 K2 (K1 − 1)(K2 − 1) / 4 gives d_add ≥ ρ / 4. ∎

Checked on all 194 tables built so far (it holds in 100%). The bound is loose: d_add / ρ ≈ 1.0–1.8. The two
measures can also move in opposite directions at fixed margins (see Proposition 4's demo), so d_add is the principled
quantity and density is a proxy.

## Proposition 4 (precise version: what BROJA can and cannot see)
Let the label be deterministic, y = T(c1, c2). BROJA fixes p(c1, y) and p(c2, y) and maximises over everything else.
And I(X1X2; Y) = H(Y). So **every BROJA atom is a function of the per-modality label statistics p(c1, y) and
p(c2, y) alone.** BROJA never uses p(c1, c2), i.e. which input pairs actually occur together. (MMI is the same; I_ccs
is not, because its max-entropy reference keeps p(c1, c2).)

**(a) Complete, uniform tables: additivity *is* decided by the margins, so PID is not "blind" there.**
- **Theorem.** A complete 0/1 table can be written as [a_i + b_j > 0] **if and only if** it is the only 0/1 table with
  its row and column counts.
- *Proof.*
  - (⇐) If it is the only one, no interchange [[1,0],[0,1]] ↔ [[0,1],[1,0]] applies, because an interchange would
    produce a second table with the same counts. So T has no checkerboard, and by Prop. 1 it is threshold-additive.
  - (⇒) By Ryser (1957), any two 0/1 matrices with the same margins are connected by a sequence of interchanges.
    A threshold-additive T has no checkerboard, so no interchange can start from it, so it is the unique realisation
    of its margins. ∎
- **Reading.** The XOR square is exactly the move that keeps every per-modality answer rate fixed. A late-fusion (sum
  of scores) model sees each modality's evidence separately, so it cannot tell apart tables that differ only by such
  moves. The additive tables are exactly those with no such twin.
- **Consequences.**
  - Since uniform-cell BROJA is a function of the margins, whether a complete uniform table is additive is, in
    principle, a function of the same data.
  - But the **synergy number** does not decide it: the D pairs have equal BROJA synergy (to 0.003) and opposite
    behaviour (readout gap 0.0 vs 0.4–0.95). They do *not* share margins, and cannot (unique realisation). Their
    other atoms differ.
  - **At identical PID** (all atoms equal), the *degree* of non-additivity, d_add, still varies. Margin-preserving
    interchanges of non-additive tables keep every count and change d_add (`runs/tables/swap_demo.txt`; identical-PID
    experiment, pre-registered, `results_samepid.txt`).

**(b) Incomplete or correlated tables (real benchmarks): PID really cannot see the arrangement.**
- When not every input pair occurs (CLEVR's natural pairs, any dataset with correlated modalities), whether the task
  is additive depends on *which* pairs occur. BROJA does not use that.
- Example: two 3×3 partial tables (6 occupied cells, uniform) with identical per-row and per-column label counts:

      additive            not additive (contains an XOR square)
      0  1  0             0  1  0
      0  .  1             .  0  1
      .  0  .             0  .  .

- BROJA gives *identical* atoms for both: S 0.667, R 0.126, U1 0.000, U2 0.126 bits. I_ccs differs (0.541 vs 0.667),
  because it uses p(c1, c2).

## Relation to existing information theory, and what our quantity is not
- **"Synergy ≠ interaction" is not new.**
  - Jansma (2023, *Entropy* 25:648) computes maximum-entropy higher-order interactions for logic gates; only XOR and
    XNOR are purely synergistic.
  - James, Emenheiser & Crutchfield (2019, *Entropy* 21:12) report an inconsistency between third-order connected
    information, secret key agreement rate and synergy.
  - Our contribution is the **ML consequences**: what an additive or late-fusion model can reach, which forgetting a
    cheap head can repair, and how benchmark PID estimates mislead.
- **d_add is not the classical log-linear three-way interaction either.**
  - With noise, every two-input gate has a nonzero three-point (log-linear) interaction, including AND (Jansma 2023),
    even though AND's answers are decided by adding scores.
  - d_add is about *answers*: the error of the best "sum of per-modality scores, then threshold" rule. That is the
    quantity that bounds additive and late-fusion models (Prop. 2).

## How the pieces fit together
- **Prop. 2:** what an additive readout can do is fixed by d_add.
- **Prop. 4:** the synergy number does not decide additivity; on incomplete tables BROJA cannot see the arrangement at all.
- **Empirics:**
  - the label-table test (P1: gap vs density ρ = +0.91 controlling for synergy; synergy +0.22 controlling for
    density);
  - step 1 (repair, OOD; running);
  - CLEVR step 2 (the ceiling computed from the answer table vs the linear readout of trained models).

## Still to do
- Non-uniform p: the weighted d_add and a weighted density bound, needed for "same?" tasks where positives are
  up-sampled.
- Multi-class labels, via one-vs-rest tables or an argmax of sums: characterise with multi-class biorders.
- Noisy (non-deterministic) labels: Prop. 4 then needs p(y | c1, c2), and PID still depends only on pairwise
  marginals for BROJA.
- A quantitative link to the forgetting law: only the flipped cells (T ≠ F) should show two-factor damage that a
  readout cannot repair (step 1, 1c).
