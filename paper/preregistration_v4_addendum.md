# Addendum to preregistration_v4_fia.md (written before any v4 run)

## Problem found while implementing
The fusion heads are ReLU MLPs, so they are piecewise linear. The mixed second derivative ∂²s/∂z1∂z2 is therefore zero
almost everywhere, and W as defined (the mean mixed Jacobian) is ≈ 0. That makes κ and κ_c undefined.

## Replacement definition (finite-difference / Harsanyi form; same spirit as Prop 3)
With β(z1, z2; z̄1, z̄2) the per-sample mixed difference through the task-1 head g⁰ (as in the definitions), and Q̂_m
the fitted linear old -> new maps applied to the centred features, including the replacement features:
- joint distortion: κ = E|β(Q̂1 z1, Q̂2 z2; Q̂1 z̄1, Q̂2 z̄2) - β(z1, z2; z̄1, z̄2)| / E|β(z1, z2; z̄1, z̄2)|;
- single distortion for modality 1: κ_1 = E|β(Q̂1 z1, z2; Q̂1 z̄1, z̄2) - β(z1, z2; z̄1, z̄2)| / E|β|; symmetrically κ_2;
- compensation index: κ_c = ½(κ_1 + κ_2) - κ.
Expectations are over task-1 test samples (3 replacement draws). H2 is unchanged:
Spearman(κ_c, N) >= 0.6 across units and regimes.

Nothing else changes. The Jacobian-weighted feature-distillation baseline uses first derivatives, which exist almost
everywhere for ReLU heads, so it is unaffected.
