# Rebuttal draft

We thank the reviewer for a careful review. We ran every requested analysis, pre-registered before computing it, and
report the results whether or not they favour us. Two change the paper's claims, and we have revised accordingly.

**Q1 / W2: null baseline.** We built the requested accuracy-matched "oracle + noise" scorer (logits
(2T−1)μ_y + N(0,1), μ_y matched to each model's accuracy on yes and no cells of each table) and ran it through the
identical Study C pipeline. It reproduces most of the correlation. Partial ρ with s:

| | observed | null | excess | survives |
|---|---|---|---|---|
| Idefics3 | +0.42 | +0.37 | +0.17 [−0.07, 0.38] | 42% |
| InternVL3.5 | +0.42 | +0.50 | +0.26 [0.05, 0.43] | 63% |
| LLaVA-OV-1.5 | +0.43 | +0.40 | +0.12 [−0.11, 0.32] | 28% |
| Qwen2.5-VL | +0.26 | +0.30 | +0.05 [−0.17, 0.25] | 19% |

Our pre-registered test of the excess on Idefics3 failed. The slack Δ−(s−e) tracks structure only for InternVL3.5
(+0.28 [0.06, 0.45]), whose mean EMAP drop is also twice the null's. We agree with the reviewer's reading and now state
the narrower claim throughout: the exception share sets the interaction dependence an accurate model requires, and
current VLMs mostly depend on interactions about as much as that requirement and their accuracy imply.

**Q5.** The yes-share's own partial correlation with the EMAP drop is +0.23 to +0.34 (all intervals above 0). We now
report it and note that partialling it out removes variance synergy shares with answer balance.

**W3 (overclaiming).** Abstract, introduction, Fig. 1 caption and conclusion now match the limitations section: a
within-benchmark ranking effect at fixed answer balance; LLaVA keeps a synergy signal; GQA shows no signal on the full
tables. "PID cannot answer this question" is removed.

**Q2 / W4 (non-uniqueness).** The reviewer is right that minimum exception sets are hitting sets of XOR squares, not
their union, and not unique; we corrected both statements. Re-solving every table with random tie-breaking, 3–6% of
cells are ambiguous (a lower bound). Using only stable labels, the CLEVR forgetting result holds for all four replay
methods (−9.3 to −12.0 points). The learning result holds on CLEVR (DER++ −27.8) but not as pre-registered across
benchmarks: on GQA the interval reaches zero, and the constructed tasks' second table has no stable exception at all,
so we removed those forward results. Cell-level: stable binary labels do not beat model confidence for predicting EMAP
flips; graded misfit (now defined in the main text) does. We added a note on NP-hardness and the ten-row constraint.

**Q3 / W5 (difficulty).** We re-ran six methods saving each cell's margin and compared exceptions with rule cells of
the same answer and the same margin decile (99–100% of exceptions matched). On CLEVR, forgetting gaps survive (ER
−12.5 [−14.5, −10.4], DER++ −7.8 [−9.6, −5.6]; pre-registered pass), and every method learns exceptions worse than
equally difficult rule cells (−10.6 to −27.0). Exceptions also do worse than the hardest quarter of same-answer rule
cells. The large positive gaps of finetuning and LwF were difficulty effects and vanish. On GQA only 18–19% of exceptions have difficulty-matched rule cells, and the DER++ learning gap there is only marginal (−9.5 [−18.6, −0.0]); the pre-registered test passes (3/3) but, given this and the constructed-task label ambiguity, we claim the learning result for CLEVR only. We added
the simplicity-bias and memorisation literature and explain why E-ACC below 50 is expected.

**Q4 (size).** s grows slowly with the number of images (CLEVR 0.079 at 25 images, 0.099 at 1,000) and is lower with
fewer questions; table rankings are stable across sizes (Spearman ≥ 0.97; 84–97% of variance between tables). We now
say s should be compared at a fixed table shape.

**W1 (novelty).** Agreed. Theorem 1 is now credited as classical in its statement, Props. 2–3 are presented as short
consequences, and the contribution is framed as bringing this structure into multimodal evaluation and testing it.

**W6 / scope.** The CL setting is described plainly as far from VLMs, and CL claims are limited to what survives all
three controls (answer, difficulty, label ambiguity), which is CLEVR.

**Other.** Related work added (nestedness, Tukey's test, sign-rank, interaction attribution, simplicity bias,
memorisation). All presentation issues fixed. An anonymised pre-registration link will be provided.
