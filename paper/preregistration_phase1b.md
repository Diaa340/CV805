# Pre-registration, Phase 1b: the fusion layer keeps the interaction; encoders drift away from it

Written after Phase 1 (preregistration_phase1.md), whose criterion 1 was unreachable (ceiling arm: synergy 0.65) and
whose data suggested the reverse mechanism (no replay: re-fitting the encoders to the frozen fusion layer lifted
synergy 0.19 -> 0.50; re-fitting the fusion layer only to 0.26). Written before any Phase-1b run. SHA-256 in
`preregistration_phase1b.sha256`.

## Hypotheses
- H1 (interface): after task 2 the task-1 fusion layer still holds the cross-modal interaction; the encoders drifted
  away from it. Re-aligning the encoders to the task-1 fusion layer ("anchoring") restores synergy better than
  re-fitting a fusion layer.
- H2 (weak modality): re-aligning the weak modality's encoder restores more synergy than re-aligning the strong one.

## Runs (synthetic, same generator / training / buffer as Phase 1)
- SNR {4.5/3.0, 4.5/4.5, 3.5/3.5} x shift {both, m1, m2}, seeds 3–7; plus seeds 8–12 for (4.5/3.0, both).
- Regimes: replay (control) and no replay (none).
- Trained repair arms (6000 Adam steps, lr 1e-3, batches of 64 buffer + 64 task-2 samples; 1000-sample buffer):
  - `ceiling`: task-1 encoders frozen + new fusion layer
  - `fusion_refit`: drifted encoders frozen, fusion layer trained further
  - `encoder_refit`: post-task-2 fusion layer frozen, both encoders trained
  - `anchor_1`, `anchor_2`, `anchor_both`: TASK-1 fusion layer frozen; encoder 1 / encoder 2 / both trained, other
    encoder(s) left drifted
- Training-free swaps (diagnostic): task-1 fusion + both drifted encoders (`swap_both`); task-1 fusion + drifted
  encoder 1 + task-1 encoder 2 (`swap_1`); task-1 fusion + task-1 encoder 1 + drifted encoder 2 (`swap_2`).
- Measured: chance-corrected task-1 retention per type; task-2 accuracy; embedding CKA per modality (drift).
- Recovery of type t by an arm: (ret_arm - ret_no_repair) / (ret_ceiling - ret_no_repair), only where the
  denominator is >= 0.05.

## Success criteria
1. anchor_both recovers >= 50% of the synergy gap to the ceiling (median recovery over units), in BOTH regimes.
2. S retention of anchor_both exceeds fusion_refit by >= 0.10 on average, in BOTH regimes.
3. Task-2 accuracy of anchor_both >= no_repair - 0.05, in both regimes.
4. H2: on (4.5/3.0, both) units (10 seeds x 2 regimes), S retention of anchor_2 (weak modality) > anchor_1 in >= 70%
   of the 20 paired comparisons.
5. H2 (secondary, all units): anchoring the modality with the lower embedding CKA (more drifted) beats anchoring the
   other in >= 70% of paired comparisons.
H1 supported if 1–3 pass; H2 supported if 4 passes (5 reported alongside).
