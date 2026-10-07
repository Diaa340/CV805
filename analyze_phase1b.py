"""Pre-registered analysis of Phase 1b (paper/preregistration_phase1b.md). Writes paper/results_phase1b.txt."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["R", "U1", "U2", "S"]
ARMS = ["no_repair", "swap_both", "swap_1", "swap_2", "ceiling", "fusion_refit", "encoder_refit", "anchor_1", "anchor_2", "anchor_both"]
lines = []


def say(s=""):
    print(s); lines.append(s)


D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs", "p1b_*_phase1b.csv"))], ignore_index=True)
D = D[D.before_S >= D.c + 0.3]
key = ["snr", "shift", "seed", "regime"]
say(f"Phase 1b: {D.groupby(['snr', 'shift', 'seed']).ngroups} units with synergy learned in task 1")
W = D.pivot_table(index=key, columns="arm", values=[f"ret_{t}" for t in TYPES] + ["task2_acc"])
ok = {}
for regime in ("control", "none"):
    w = W.xs(regime, level="regime")
    say(f"\n=== {'REPLAY' if regime == 'control' else 'NO-REPLAY'} regime: mean retention per arm (task-2 accuracy) ===")
    for arm in ARMS:
        say(f"  {arm:13s} " + " ".join(f"{t}={w[(f'ret_{t}', arm)].mean():.2f}" for t in TYPES) + f" | task2 {w[('task2_acc', arm)].mean():.2f}")
    gap = w[("ret_S", "ceiling")] - w[("ret_S", "no_repair")]
    rec = ((w[("ret_S", "anchor_both")] - w[("ret_S", "no_repair")]) / gap)[gap >= 0.05]
    d_fr = (w[("ret_S", "anchor_both")] - w[("ret_S", "fusion_refit")]).mean()
    t2 = w[("task2_acc", "anchor_both")].mean() >= w[("task2_acc", "no_repair")].mean() - 0.05
    ok[regime] = (rec.median() >= 0.5, d_fr >= 0.10, t2, rec.median(), len(rec), d_fr)
    say("  synergy recovery (share of gap to ceiling), median: " + " ".join(
        f"{a}={(((w[('ret_S', a)] - w[('ret_S', 'no_repair')]) / gap)[gap >= 0.05]).median():.2f}"
        for a in ("fusion_refit", "encoder_refit", "anchor_1", "anchor_2", "anchor_both", "swap_both")))
say("\n=== PRE-REGISTERED VERDICT ===")
for i, (name, idx) in enumerate((("anchor_both recovers >= 50% of synergy gap (median)", 0), ("S(anchor_both) - S(fusion_refit) >= 0.10", 1),
                                 ("task-2 accuracy kept", 2)), start=1):
    detail = {0: lambda r: f"{ok[r][3]:.2f} (n={ok[r][4]})", 1: lambda r: f"{ok[r][5]:+.3f}", 2: lambda r: ""}[idx]
    say(f"  {i}. {name}: replay {detail('control')} -> {'PASS' if ok['control'][idx] else 'FAIL'} | no replay {detail('none')} -> {'PASS' if ok['none'][idx] else 'FAIL'}")
h1 = all(ok[r][0] and ok[r][1] and ok[r][2] for r in ok)
a = W.loc[(W.index.get_level_values("snr") == "4.5/3.0") & (W.index.get_level_values("shift").astype(str) == "12")]
h2 = (a[("ret_S", "anchor_2")] > a[("ret_S", "anchor_1")]).mean()
say(f"  4. H2 (4.5/3.0, both): anchor_2 (weak) > anchor_1 in {100 * h2:.0f}% of {len(a)} pairs (need >= 70%) -> {'PASS' if h2 >= 0.7 else 'FAIL'}")
C = D[D.arm == "no_repair"].set_index(key)[["cka_emb1", "cka_emb2"]]
more = np.where(C.cka_emb1 < C.cka_emb2, 1, 2)
Wc = W.loc[C.index]
win = np.where(more == 1, Wc[("ret_S", "anchor_1")] > Wc[("ret_S", "anchor_2")], Wc[("ret_S", "anchor_2")] > Wc[("ret_S", "anchor_1")])
say(f"  5. H2 secondary (all units): anchoring the more-drifted modality wins in {100 * win.mean():.0f}% of {len(win)} pairs (need >= 70%) -> {'PASS' if win.mean() >= 0.7 else 'FAIL'}")
say(f"  -> H1 (fusion keeps the interaction; encoders drift): {'SUPPORTED' if h1 else 'NOT SUPPORTED'}; H2 (weak modality): {'SUPPORTED' if h2 >= 0.7 else 'NOT SUPPORTED'}")
open(os.path.join(ROOT, "paper", "results_phase1b.txt"), "w").write("\n".join(lines) + "\n")
