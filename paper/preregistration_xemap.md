# Pre-registration: exact additive projection of VLMs (replacing EMAP's least-squares projection)

Written 2026-10-19 before computing any model result with it. SHA-256 in `preregistration_xemap.sha256` (with
`xemap.py`, `analyze_xemap.py`).

## Why
A post-hoc decomposition (seen) found that 54-67% of the cells EMAP flips are stable rule cells. Of those, 34-42% correct
a model error, and 58-66% break a correct answer, many of which a perfect model would also lose under EMAP. EMAP's
least-squares projection is not the best additive rule, so part of its "interaction dependence" is approximation
error, not the model.

## Measure
Exact projection: the additive rule that agrees with the model's own yes/no answers on the most cells (MILP), and a
confidence-weighted variant (weights |log-odds| + 0.01). Flips are the cells where the projection disagrees with the
model; the exact drop is accuracy minus the projection's accuracy. Check on four perfect models (seen): drop = s and all
flips are exceptions under some optimal rule.

## Data
The 99 Study C sub-tables, four VLMs; the accuracy-matched null scorer of `rev_null.py` (20 draws, unweighted).

## Tests (`analyze_xemap.py`)
- X1: for every model, at least 60% of the exact projection's flips are stable or ambiguous exceptions (EMAP: 33-46%).
- X2: for Idefics3, rho(exact drop - null exact drop, s | synergy, accuracy, yes-share) has a 95% lower bound > 0.
  The same is reported for the other models, with rho(exact drop, s) and rho(exact drop, synergy).
- Reported: flip composition for EMAP, exact and weighted projections; mean drops; proven-optimal shares; floor check.

## Addendum 1 (no X2 result seen)
The first analysis run lost the null-scorer rows: pandas reads the label "null" as a missing value. Section 1 (flip
composition, which decides X1) had printed before the crash and was seen; sections 2-3 and X2 were not computed. Fixed by
restoring the label after loading; nothing else changed.
