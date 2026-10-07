# Addendum (before the real run; only a 1-seed smoke test, seed 99, was run and is discarded)
The smoke test showed independent task 2 damages much less than coupled task 2 at equal epochs (k about 0.8 vs 0.3),
and the comparative excess grows with damage. To avoid a damage confound:
- The independent run continues to 30 and 100 epochs (snapshots 1, 3, 10, 30, 100). T0-T2 still use levels 1, 3, 10 only.
- T2m (damage-matched, decisive for the coupling question): reference = seed-mean comparative k12 of the matched
  (coupled) condition at level 3 (the kill test's peak). Pick the independent level whose seed-mean comparative k12 is
  closest. G = E_comp - E_conj at those levels.
  - "comparative advantage without coupling" iff G(independent, matched level) >= 0.05 (lower bound > 0);
  - "coupling needed" iff G(coupled, L3) - G(independent, matched level) >= 0.05 (lower bound > 0) and not the above;
  - otherwise mixed. If T2 and T2m disagree, T2m decides the coupling question.
- Analysis code fixed as `analyze_kinds_2x2.py` (hash below).
