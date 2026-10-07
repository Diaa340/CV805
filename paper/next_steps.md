# Next steps, anchored on novelty

## What is already known (so we must not claim it)
- **Forgetting is mostly an access problem, not erasure** (unimodal). Representations stay linearly decodable, and
  retraining only the classifier recovers much of the lost accuracy: Davari et al., CVPR 2022; "Catastrophic forgetting
  as accessibility collapse", arXiv 2606.06032 (retraining the readout restores ~76%). Our v2 finding ("encoders keep
  93–97% of their content") is consistent with this and is **not new on its own**.
- **Modality reliance drifts in multimodal continual learning.** "Modality Contribution Drift" (arXiv 2607.27260) measures
  it with modality-subset interventions and regularises it (CMCDR). "Hidden forgetting" (arXiv 2607.02020) shows accuracy
  can survive while evidence grounding shifts. Neither decomposes knowledge by PID atom, studies synergy, or proposes a
  composition law.
- **PID of multimodal data** (Liang et al., NeurIPS 2023). Redundancy is robust and synergy is fragile (neuroscience).

## What is plausibly new (our contribution candidates)
1. **Per-atom forgetting:** synergy is forgotten first, about 2× faster than redundancy at equal confidence.
2. **A multiplicative drift-composition law:** atom retention = product of per-modality encoder-drift damage; synergy is
   the only atom with two factors. Pre-registered PASS on synthetic data; not yet shown on real data.
3. **A limit of the "accessibility" view in multimodal models:** retraining the readout recovers unique and redundant
   knowledge but **not synergy**; without replay, only re-aligning the encoders does. Seen in Phase 1/1b on synthetic
   data. This is the sharpest contrast with the 2026 unimodal work.
4. **One law for forgetting and robustness (OOD link):** the same per-modality factors predicted retention under modality
   corruption (exp 5: R² 0.53–0.77).

## Plan (in order; each step has a kill criterion)
**P1: Semi-synthetic bridge (real inputs, exact types). ~2 days.**
Pairs of real samples with interaction types set by construction:
- two-view CIFAR-100: image A + image B;
- AV-MNIST-style: real image + real speech;
- CREMA-D: face of clip A + voice of clip B.

Labels by rule:
- R: both carry the class;
- U: one carries it;
- S: label = combination of the two classes (mod K, with K = 3–5 so that it is learnable).

Re-run v3 unchanged. **Kill if** (a)–(c) fail with exact types on real features: then the law is a property of toy data.

**P2: Model-functional types on natural data. ~2 days.**
Replace the unreliable PID probes with the task-1 model's own counterfactual behaviour:
- a sample needs modality m if ablating m flips it;
- it is "synergistic for the model" if it needs both;
- it is "redundant for the model" if it needs neither alone.

This is not circular: types are fixed before task 2, and the drift test comes after. Test composition and the readout
limit on all 5 real datasets and on end-to-end CREMA-D. **Kill if** composition fails for model-synergistic samples.

**P3: The readout-limit claim (contribution 3), head-to-head with the 2026 work. ~3 days.**
On real data:
- compare readout retraining (the accessibility remedy) with encoder re-alignment, per atom;
- compare against CMCDR-style contribution-drift regularisation.

The claim to win: readout retraining fixes R/U but not S, and contribution-drift regularisation does not protect S
specifically.

**P4: Scale the law.**
- Three modalities: 3-way synergy is learnable with 3 classes; the prediction is s1·s2·s3.
- Corruption/OOD: modality noise at test time, using the same factors.

**P5: Method, only if P1–P3 hold.**
Protect the interaction, not the whole representation. For example: anchor each encoder's output on the samples the model
needs both modalities for, and compare with CMCDR, LwF and feature distillation. The novelty must come from targeting the
interaction term, since generic feature distillation already exists.

## Framing if P1–P3 hold
*"Synergy is forgotten first: a multiplicative drift law for multimodal continual learning."* Contributions: per-atom
forgetting measurement; the composition law; the synergy exception to readout recovery; one law for forgetting and
modality robustness; and (optionally) an interaction-targeted fix.
