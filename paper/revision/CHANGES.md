# Revision: what changed and why

Files in `paper/revision/`: `0_abstract.tex`, `1_intro.tex`, `2_related.tex`, `3_method.tex`, `4_experiments.tex`,
`5_conclusion.tex` (limitations + conclusion), `supplementary.tex`, `refs_revision.bib`, `rebuttal.md`.
New analyses: pre-registered in `paper/preregistration_revision.md` (hash 73d425b2); results in
`paper/results_rev_null.txt`, `results_rev_unique.txt`, `results_rev_difficulty.txt`, `results_rev_stability.txt`.

## New analyses and what they found

| Review point | Analysis | Result |
|---|---|---|
| W2 / Q1: is the VLM correlation arithmetic? | accuracy-matched "oracle + noise" null, same pipeline | **Mostly yes.** The null reproduces most of the correlation. Pre-registered test (Idefics3) **fails**: excess +0.17 [−0.07, 0.38]. Only InternVL3.5 goes beyond the requirement (+0.26 [0.05, 0.43]; slack +0.28). |
| Q5: yes-share's own coefficient | partial ρ(Δ, yes \| s, PID, acc) | +0.23 to +0.34, all intervals above 0. |
| W4 / Q2: exceptions not unique | random tie-breaking re-solves | 3–6% of cells ambiguous (lower bound); 64–75% of saved exception labels are stable. CLEVR forgetting survives (**pass**, 4/4). Learning gap with stable labels **fails** (1/3): constructed task-2 table has **no** stable labels; GQA interval reaches 0. |
| W5 / Q3: just difficulty? | re-runs saving margins; answer × margin-decile matching | CLEVR: forgetting gap **survives** (ER −12.5, DER++ −7.8; pass); learning gap survives for every method (−10.6 to −27.0). Finetune/LwF positive gaps were difficulty. GQA: only 18–19% of exceptions can be matched; DER++ learning gap −9.5 [−18.6, −0.0], unmatched gap not significant in the re-run. Pre-registered D1 passes (3/3) but is robust only on CLEVR. |
| Q4: does s(T) depend on table size? | subsampling 25–1000 images, 5/10 questions | s grows slowly with n (CLEVR .079 → .099), lower with fewer questions; rankings stable (ρ ≥ 0.97; ICC 0.84–0.97). |

## Text changes, by review point

- **W1 (novelty overstated):** Theorem 1 credited as classical (Ferrers / Ryser) in the statement itself; Props. 2–3
  described as short consequences; contributions reworded ("the underlying characterisation is classical; the
  contribution is its use for multimodal evaluation").
- **W2 / W3 (overclaiming):** abstract, intro, Sec. 4.2 and conclusion now state the narrow claim: exception share sets
  the interaction dependence an accurate model needs; VLMs mostly use about that much; structure beats synergy only
  within benchmark at fixed answer balance; LLaVA keeps a synergy signal; yes-share coefficient reported; null table
  added (Tab. `tab:null`). "PID cannot answer this question" removed. Floor check now called a sanity check, with the
  vacuous case (Idefics3 on CLEVR) named.
- **W4 (exceptions not unique; wrong XOR statement):** "every exception corresponds to / exceptions are exactly the
  cells of XOR squares" removed. New paragraph: minimum hitting set of XOR squares, non-unique; stable labels defined;
  cell-level results also with stable labels. NP-hardness / scalability note with the ten-row limitation.
- **W5 (difficulty, E-ACC below 50):** new "three confounds" paragraph (answer, difficulty, labels); difficulty-matched
  table (`tab:difficulty`); E-ACC < 50 explained as expected for a rule-learning model; Feldman / simplicity-bias cited.
- **W6 (two halves loosely coupled; toy CL setting):** CL setting described plainly as far from VLMs; CL claims narrowed
  to what survives all controls (CLEVR). Condensing CL further is a judgement call left to you.
- **Moderate:** scope of existence questions moved into the main text; measure stability (new supplement section);
  PID-as-used caveat in Limitations; graded misfit defined in Methods and binary-label AUC reported next to it;
  pre-registration link placeholder.
- **Related work:** nestedness (NODF, temperature), Tukey's test, sign-rank / margin complexity, Shapley interaction
  indices and neural interaction detection, simplicity bias and memorisation (Feldman, Shah et al.).
- **Presentation:** broken "figure ??" fixed (→ Fig. `fig:teaser`); "as demonstrated in 3" → Fig. ref; stray comma in
  Eq. 1; `\{0,1\}` braces; "Codea"; stray markdown fence in the supplement; theorem and propositions stated formally in
  the main text; units switched to points throughout the text; abstract says which claims rest on which models;
  definitions of rule cells, graded misfit and sub-table weights added.

## Still needs you

1. **`main.tex`** (not uploaded): replace the abstract with `0_abstract.tex`, replace Secs. 6–7 with `5_conclusion.tex`,
   add `refs_revision.bib`, and update the Fig. 1 caption. Suggested Fig. 1 VLM caption sentence: *"VLMs: within each
   benchmark and at fixed answer balance, the exception share ranks tables by EMAP drop and synergy adds little; an
   accuracy-matched null reproduces most of this, so models rely on interactions roughly as much as their accuracy
   requires (Sec. 4.2)."*
2. **Figures (done, in `paper/revision/fig/`, made by `make_rev_figs.py`):** `fig_gaps.pdf` replaces the old one (constructed-task forward panel removed, with a note); `fig_vlm_panel.pdf` is a drop-in for Fig. 1's VLM panel, adding the accuracy-matched null scorer as a hatched blue bar. Fig. 1's other panels are unchanged.
3. **Pre-registration link:** create an anonymised OSF registration with the `preregistration_*.md` files and hashes,
   and replace the placeholder in the supplement.
4. **Check the new BibTeX entries** against the published versions (several are older papers cited from memory).
