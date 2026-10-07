# Pre-registration: CLOVE-scene answer tables (build only; outcome-free)

Written 2026-10-02 before building. SHA-256 in `preregistration_clove.sha256` (with `clove_exec.py`, `clove_build.py`).
CLOVE (Lei et al., AAAI 2023) replaces POPE, whose task 2 shared task 1's answer table. Seen before writing: CLOVE
question counts per family and the executor's own-image agreement per family (all six tasks, train + val; after
removing category words and rare names: 0.98-1.00; existRelS 0.984).

- Executor (`clove_exec.py`): set-lifted GQA programs (select, filter of any attribute type except position, relate,
  exist, and, or); object ids ignored. Only existence questions are used, because they are well-posed on any image.
- Build (`clove_build.py`): rules in its docstring. They copy the pre-registered GQA build rules (family agreement
  >= 99%, own-image agreement per question, no category words, 10 questions per unit, yes-share in [0.10, 0.90]),
  with CLOVE's own train / val image splits per task (150 / 50 images per unit) and up to 8 units per task.
- Labels: exact additive rule (MILP, 300 s per table) on each unit's training table (10 x 150) and test table
  (10 x 50), as for GQA.
- Next (separate pre-registration): the six-task continual-learning study with the 13 methods, CLOVE's task orders.

## Addendum 1 (2026-10-02; written after seeing the build's yes-shares, before running this change)
Result of the build as registered: 43 units, all from the two plain existence families (exist, existC). Every
relation family was dropped because relation questions are almost never true on other images (mean yes-share
0.002-0.024 against the 0.10 floor). This leaves only "Is there an X?" tables.
Change (`python clove_build.py --pool-all`, written to units_v2): for units whose questions use `relate`, images are
sampled among those containing *all* objects named by at least one of the unit's questions (instead of any one of
them). Everything else is unchanged, including the yes-share floor. The study uses units_v2; units (v1) is kept and
reported. This is a design change made after seeing yes-shares (no model results exist yet).
Outcome of addendum 1: relation units still have yes-share ~0.012-0.014 (scene graphs annotate few relations), so all
are dropped again; units_v2 is identical to units (43 units, exist / existC). The study uses these 43 units.
