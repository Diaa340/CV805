# Pre-registration: POPE (exact table on real photos) and a head-to-head with published PID claims

Written 2026-10-01 before any model output on POPE. SHA-256 in `preregistration_pope.sha256` (with `pope_build.py`,
`pope_vlm.py`, `pope_train.py`, `pope_liang.py`, `xiu/qwen25_small_devi.py`, `xiu/qwen25_small_embed.py`,
`xiu/real_replace.py`, `xiu/batch_vlm_final_pope_drop.py`, `analyze_pope.py`).

## Exact table (outcome-free, seen before writing; `pope_build.py`)
Truth = POPE's own ground-truth object lists (segmentations + captions) for the 500 POPE images x 80 COCO categories;
it reproduces all 9,000 POPE labels. Full table: yes 5.4%, additive-fit error 0.049, XOR density 0.004, MMI synergy 0.25
bits. POPE's own cells (5,127 unique): yes 29%, additive-fit error 0.25, XOR density 0.063, MMI synergy 0.75 bits.
"Is there an X?" is well-posed on every image x object pair.

## Published claims compared
Xiu et al. (ICLR 2026): POPE (COCO14 adversarial) is "synergy-driven" (high S share; S correlates with accuracy
rho = 0.74 across 26 models). Fang et al. (ICML 2026): POPE is synergy-dominant. No directional prediction: existence
questions are matching questions and may genuinely need fusion; either outcome is informative.

## Parts
- B (primary): Qwen2.5-VL-7B log-odds on all 9,000 POPE questions, and on a complete 80-object x 100-image sub-table
  (seed 61). EMAP on the sub-table; balanced accuracy (the table is 5% yes).
  "VLM fuses on POPE" iff VLM - EMAP(VLM) >= 0.05 with 95% bootstrap (over objects) lower bound > 0; "no fusion" iff the
  upper bound < 0.02; else mixed.
- C: trained additive vs full vs linear readout on CLIP features (images split 3:1, seed 71; seeds 23-25), compared with
  the exact ceiling on the test cells. PASS iff |additive - ceiling| <= 0.03.
- D (pitfall 3): Xiu et al.'s pipeline (their code; only change: sdpa attention instead of flash-attention) for
  Qwen2.5-VL-7B on POPE adversarial in their two-option format (A = Yes, B = No), random 3:1 split (seed 0, 2,250 /
  750), Stage 1 deviation statistics, Stage 2 three-way inference with Gaussian-noise removal, Stage 3 BATCH. The same
  Stage 3 on a variant where a modality is removed by a random real sample instead (`real_replace.py`).
  "Removal method matters" iff |S_noise - S_real| > 0.1 bits or the S share falls on opposite sides of 0.5.
- Reported: error odds ratio on non-additive cells (object and image fixed effects); task-level clustered PID
  (Liang et al., CLIP, k = 30); exact MMI synergy.
- Bug fix (before any POPE result was read): pope_train.py computed the readout codes with gradients enabled (crash). New hash: aeb7af292b2632b3ab67e9ac7696bebbc227d72a68db8e809a88de3650fc5b6f  pope_train.py
