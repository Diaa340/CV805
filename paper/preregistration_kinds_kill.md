# Pre-registration: comparative vs conjunctive knowledge under coupled drift (kill test)

Written 2026-09-30, before any run. SHA-256 in `preregistration_kinds_kill.sha256`. Script: `kinds_kill.py`.

## Question
Does comparative cross-modal knowledge ("do the two inputs match?") survive a coupled drift of both encoders better than
the product of its single-encoder survivals, while conjunctive knowledge does not? If not, drop the "kinds of knowledge"
idea.

## Data and tasks (two-view CIFAR-100 features, cifar2v.npz; P1-bridge style)
- Each pair = view 1 of item i and view 2 of a different item j. Task-1 class pool: 10 classes, task-2 pool: 10 other
  classes (class permutation seed 900, as in P1). Train pairs from the train split, test pairs from the test split.
- Comparative task 1: y = [class(i) == class(j)], 50/50.
- Conjunctive task 1 (primary): y = T[class(i), class(j)] with T a random 10x10 binary table whose rows and columns each
  contain five ones (one side alone carries no label information). Pairs uniform over class pairs.
- XOR task 1 (secondary, reported): y = a(class(i)) XOR a(class(j)), a = random balanced binary attribute. Note: this
  equals "different group?", so the theory predicts it behaves like comparative; reported only.
- Task-1 model: Net (depth 2, width 256, emb 128), 30 epochs, Adam 1e-3, as in all earlier runs.

## Coupled task 2 (same for all kinds)
- Task-2 classes only. Encoders trained with a symmetric cross-modal InfoNCE matching loss (L2-normalised codes,
  temperature 0.1; positive = the two views of the same item; batch 256, Adam 1e-3). The task-1 head is frozen and unused,
  so all change is in the encoders.
- Drift levels: 1, 3, 10, 30 epochs. Seeds 23-27. 5 seeds x 4 levels x 3 kinds.

## Measures (task-1 test pairs, through the frozen task-1 head)
- acc0 (old encoders), acc1 (new e1 + old e2), acc2 (old e1 + new e2), acc12 (both new).
- Chance-corrected retention k = (acc - 0.5) / (acc0 - 0.5). Excess E = k12 - k1*k2.
- Also reported: raw survival events on task-1-correct pairs, P(S1), P(S2), r12, C, N.

## Manipulation checks (must hold, else the unit is excluded and reported)
- Task-1 accuracy >= 0.75 for each kind.
- Linear readout on frozen task-1 codes [z1, z2] (logistic, 500 steps): accuracy <= 0.60 (true interaction, not additive).

## Primary test
- Informative levels: those where, averaged over seeds, both kinds have min(k1, k2) in [0.1, 0.9].
- Statistic: E averaged over informative levels, per seed.
- PASS (keep the idea) iff (a) comparative mean E >= 0.05 with the one-sided 95% t-interval over 5 seeds above 0, and
  (b) comparative E - conjunctive E >= 0.05 (paired over seeds, one-sided 95% t-interval above 0).
- FAIL -> drop the idea and fall back to the science paper. No informative level -> INCONCLUSIVE (reported, not rescued).
