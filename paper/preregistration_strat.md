# Pre-registration: same-answer exception metrics and the class-balance buffer control

Written 2026-10-03. SHA-256 in `preregistration_strat.sha256` (with `recon_y.py`, `cl_bal.py`, `cl_seq.py`,
`analyze_strat.py`). Motivation: the post-hoc check (results_answer_confound.txt) found that exception cells are
mostly "yes" on existence tables (GQA, CLOVE), so the aggregate exception-minus-rule gap partly measures answer class.

## Part A: same-answer metrics on saved results (DESCRIPTIVE)
- dBWT_s = sum over answers y of pi_y [BWT(exceptions, y) - BWT(rule, y)], pi_y = share of exception cells with answer
  y (answers with no cells of one type are dropped and pi renormalised); dFWT_s the same for forward transfer.
  These become the standard exception metrics from now on; raw (unstratified) values are reported alongside.
- Data: the two-task study (tables, CLEVR, GQA; answers rebuilt by `recon_y.py`, every row verified against the
  saved exception label), the CLOVE six-task small-model study, and the CLOVE VLM study.
- Planned tests: SA1 (CLEVR: ER and DER++ dBWT_s <= -0.03, upper bound < 0) and SA2 (DER++ dFWT_s <= -0.03, upper
  bound < 0, on >= 2 of tables / CLEVR / GQA / CLOVE).
- PROTOCOL SLIP, disclosed: while checking that `analyze_strat.py` runs, its Part A output (including SA1 and SA2
  verdicts, and which methods have dBWT_s upper bound < 0) was printed and seen before this file was written. Part A
  is therefore descriptive, not confirmatory; within-answer BWT for some methods had also been seen in the post-hoc
  check.

## Part B: class-balance control (CONFIRMATORY; no data exist yet)
- Arms (DER++, memory 1,000): random; rule+exceptions (as before); class-balanced random (50% yes); class-balanced
  rule+exceptions (same exception half, the rest chosen so the buffer is 50% yes).
- Two-task benchmarks (`cl_bal.py`): tables, CLEVR, GQA, same units / seeds / task-1 training as the two-task study.
  CLOVE (`cl_seq.py --bal`): 12 runs (6 orders x 2 seeds), memories interleaved so shrinking keeps the mix.
- B1 (exceptions matter beyond class): class-balanced rule+exceptions vs class-balanced random, exception BWT gain
  >= 0.03 with lower bound > 0. PASS iff it holds on >= 3 of the 4 benchmarks.
- B2 (cost): class-balanced rule+exceptions vs random, overall BWT difference lower bound > -0.02, on >= 3 of 4.
- Reported: every arm's exception BWT, overall BWT, BWT by answer for both types, dBWT_s.
