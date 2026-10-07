# Theory for v4: composition, ordering, co-drift, and the threshold effect

## Setup
Encoders give z_m = f_m(x_m), with features centred on the task-1 mean. The task-1 margin is locally bilinear:

  s(z1, z2) = c + aᵀz1 + bᵀz2 + z1ᵀ W z2.

Take the replacement features at the mean (z̄_m = 0; with random real replacements the same algebra holds in
expectation over the replacement). The evidence profile is then exact:

  bias = c,  α1 = aᵀz1,  α2 = bᵀz2,  β = z1ᵀ W z2,  s = c + α1 + α2 + β.

Drift model after task 2: z'_m = ρ_m z_m + ε_m, where ρ_m ∈ [0, 1] is a per-sample scalar and ε_m is noise. Expanding:

  s' = c + ρ1 α1 + ρ2 α2 + ρ1 ρ2 β + X,
  X = aᵀε1 + bᵀε2 + ρ1 z1ᵀWε2 + ρ2 ε1ᵀW z2 + ε1ᵀW ε2.            (1)

## Proposition 1 (composition with a covariance correction)
If ε1 and ε2 are zero-mean, independent of each other, and independent of (z1, z2, ρ1, ρ2), then E[X | z, ρ] = 0. For a
purely synergistic sample (c = α1 = α2 = 0, β ≠ 0), E[s'/s | z, ρ] = ρ1ρ2. Averaging over synergistic samples:

  E[s'/s]_syn = E[ρ1 ρ2] = E[ρ1] E[ρ2] + Cov(ρ1, ρ2).

*Proof.* Every term of X is linear in ε1 or ε2, or is the bilinear form ε1ᵀWε2. By zero mean and independence,
E[aᵀε1] = E[bᵀε2] = 0, E[z1ᵀWε2 | z] = z1ᵀW E[ε2] = 0, and similarly for ε1ᵀWz2. Finally,
E[ε1ᵀWε2] = E[ε1]ᵀ W E[ε2] = 0. With c = α1 = α2 = 0, (1) gives E[s' | z, ρ] = ρ1ρ2β = ρ1ρ2 s. The last equality is
the definition of covariance. ∎

The v3 law is the special case Cov(ρ1, ρ2) = 0.

## Proposition 2 (ordering of margin retention)
Let ρ1, ρ2 ∈ [0, 1] and α_m ≥ 0. Then:
- the margin retention of a pure synergy sample is ρ1ρ2;
- of a unique-m sample, ρ_m;
- of a redundant (additive) sample, (ρ1α1 + ρ2α2)/(α1 + α2), a convex combination of ρ1 and ρ2.

And ρ1ρ2 ≤ min(ρ1, ρ2) ≤ λρ1 + (1 - λ)ρ2 for every λ ∈ [0, 1].

*Proof.* The retentions follow from E[s'] in (1), under Prop. 1's assumptions, restricted to each profile. Because
ρ_{-m} ≤ 1, ρ1ρ2 ≤ ρ_m for each m. A convex combination of two numbers is at least their minimum. ∎

So synergy retains the least margin and redundancy retains margin through its better-preserved route. This is a
statement about margins. Accuracy-level consequences follow from the threshold analysis below.

## Proposition 3 (co-drift compensation)
Let the drift be linear and coupled: z'_m = Q_m z_m, with Q1ᵀ W Q2 = W (task 2 preserves the cross-modal interaction).
For a pure interaction sample:
- joint-drift retention is r12 = z1ᵀQ1ᵀWQ2z2 / z1ᵀWz2 = 1;
- the single-drift retentions are r1 = z1ᵀQ1ᵀWz2 / z1ᵀWz2 and r2 = z1ᵀWQ2z2 / z1ᵀWz2.

Whenever r1 r2 < 1 (in particular whenever either single drift degrades the sample), r12 > r1 r2, and the composition
law under-predicts joint retention.

*Proof.* Direct substitution. ∎

In the measurable decomposition r12 = P(S1)P(S2) + C + N, this is the case N > 0: the two drifts are not separable.
Face–voice matching, where task 2 trains the same cross-modal skill, is the predicted instance.

## Threshold effect for additive "needs-both" evidence
Accuracy survives if and only if the drifted margin is positive. Take additive evidence (β = 0) with bias c = -τ, τ > 0.
- A sample is *needs-both* if α1 < τ, α2 < τ and α1 + α2 > τ (neither route suffices alone, together they do).
- It is *redundant* if α1 > τ and α2 > τ.

**Lemma (AND-gate geometry).** Under drift ρ1, ρ2 ∈ [0, 1] (noise-free):
- (a) a needs-both sample can only survive if ρ1 + ρ2 > 1;
- (b) a redundant sample survives whenever ρ1 + ρ2 ≥ 1.

*Proof.*
- (a) Survival requires ρ1α1 + ρ2α2 > τ. But α_m < τ, so ρ1α1 + ρ2α2 < τ(ρ1 + ρ2).
- (b) α_m > τ, so ρ1α1 + ρ2α2 > τ(ρ1 + ρ2) ≥ τ. ∎

So when the two routes together retain less than one route's worth (ρ1 + ρ2 ≤ 1), all needs-both samples fail. For
ρ1 + ρ2 ≥ 1, every redundant sample survives while only part of the needs-both samples do. This holds for any
AND-structured evidence, with no multiplicative interaction at all. It is why "needs-both is forgotten first" holds on
natural data (140/140 units) even though that evidence is mostly additive (P3).

**Gaussian survival gap (symmetric drift ρ1 = ρ2 = ρ).** Let α1, α2 be i.i.d. N(μ, σ²). Write T = α1 + α2 ~ N(2μ, 2σ²)
and D = α1 - α2 ~ N(0, 2σ²), which are independent. The needs-both region is τ < T and |D| < 2τ - T. Survival requires
ρT > τ, so

  P(survive | needs-both) = ∫_{τ/ρ}^{2τ} φ_T(t) g(t) dt / ∫_{τ}^{2τ} φ_T(t) g(t) dt,
  g(t) = 2Φ((2τ - t)/(σ√2)) - 1,

which is 0 for ρ ≤ 1/2. Redundant samples have T > 2τ, so they survive with probability 1 for all ρ ≥ 1/2. The survival
gap (redundant minus needs-both) is therefore exactly the share of needs-both mass with T < τ/ρ:
- it equals 1 at ρ = 1/2;
- it decreases monotonically to 0 as ρ -> 1;
- it grows as the needs-both mass concentrates near the threshold (small σ relative to τ).

For true synergy (β-dominated samples), the multiplicative factor ρ1ρ2 from Prop. 1 adds to this threshold effect.
