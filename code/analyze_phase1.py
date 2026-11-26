"""Pre-registered analysis of Phase 1 (paper/preregistration_phase1.md). Writes paper/results_phase1.txt."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["R", "U1", "U2", "S"]
ARMS = ["no_repair", "fusion_refit", "encoder_refit", "additive_refit", "fresh_encoders", "task1_encoders", "fusion_fresh"]
lines = []


def say(s=""):
    print(s); lines.append(s)


D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs", "p1_*_phase1.csv"))], ignore_index=True)
D = D[D.before_S >= D.c + 0.3]  # synergy must have been learned in task 1
say(f"Phase 1: {D.groupby(['snr', 'shift', 'seed']).ngroups} units with synergy learned in task 1")
for regime in ("control", "none"):
    g = D[D.regime == regime]
    say(f"\n=== {'REPLAY regime [primary]' if regime == 'control' else 'NO-REPLAY regime'}: mean chance-corrected task-1 retention (+- std) and task-2 accuracy ===")
    S = g.groupby("arm")[[f"ret_{t}" for t in TYPES] + ["task2_acc"]].agg(["mean", "std"]).reindex(ARMS)
    for arm in ARMS:
        say(f"  {arm:15s} " + " ".join(f"{t}={S.loc[arm, (f'ret_{t}', 'mean')]:.2f}+-{S.loc[arm, (f'ret_{t}', 'std')]:.2f}" for t in TYPES)
            + f" | task2 {S.loc[arm, ('task2_acc', 'mean')]:.2f}")
    m = lambda arm, col: g[g.arm == arm][col].mean()
    if regime == "control":
        c1 = m("fusion_refit", "ret_S") >= 0.8
        c2 = m("fusion_refit", "ret_S") - m("fresh_encoders", "ret_S") >= 0.3
        c3 = (m("fusion_refit", "ret_S") - m("additive_refit", "ret_S") >= 0.3) and m("additive_refit", "ret_R") >= 0.8
        c4 = m("fusion_refit", "task2_acc") >= m("no_repair", "task2_acc") - 0.05
        say("\n=== PRE-REGISTERED VERDICT (replay regime) ===")
        say(f"  1. fusion_refit S retention {m('fusion_refit', 'ret_S'):.2f} >= 0.8 -> {'PASS' if c1 else 'FAIL'}")
        say(f"  2. S(fusion_refit) - S(fresh_encoders) = {m('fusion_refit', 'ret_S') - m('fresh_encoders', 'ret_S'):.2f} >= 0.3 -> {'PASS' if c2 else 'FAIL'}")
        say(f"  3. S(fusion_refit) - S(additive_refit) = {m('fusion_refit', 'ret_S') - m('additive_refit', 'ret_S'):.2f} >= 0.3 and "
            f"R(additive_refit) = {m('additive_refit', 'ret_R'):.2f} >= 0.8 -> {'PASS' if c3 else 'FAIL'}")
        say(f"  4. task-2 acc fusion_refit {m('fusion_refit', 'task2_acc'):.2f} >= no_repair {m('no_repair', 'task2_acc'):.2f} - 0.05 -> {'PASS' if c4 else 'FAIL'}")
        say(f"  -> hypothesis 'synergy is disconnected, not erased': {'SUPPORTED' if c1 and c2 and c3 and c4 else 'NOT SUPPORTED'}")
    # exploratory: recovery by the fusion layer relative to the damage, per type
    nr, fr = g[g.arm == "no_repair"].set_index(["snr", "shift", "seed"]), g[g.arm == "fusion_refit"].set_index(["snr", "shift", "seed"])
    say("  [exploratory] share of the lost retention that fusion_refit recovers: "
        + " ".join(f"{t}={((fr[f'ret_{t}'] - nr[f'ret_{t}']) / (1 - nr[f'ret_{t}']).clip(lower=0.05)).median():.2f}" for t in TYPES))
say("\n=== [exploratory, Phase 2] drift measures after task 2 (before repair), mean over units ===")
cols = ["cka_fusion_hidden", "cka_emb1", "cka_emb2", "wdrift_fusion_m1", "wdrift_fusion_m2", "crossmodal_cka_before", "crossmodal_cka_after"]
say(D[D.arm == "no_repair"].groupby("regime")[cols].mean().round(3).to_string())
nr = D[(D.arm == "no_repair") & (D.regime == "control")]
say("\n  [exploratory] correlation with synergy retention (replay regime, no repair): "
    + " ".join(f"{c}={np.corrcoef(nr[c], nr.ret_S)[0, 1]:+.2f}" for c in cols[:5]))
open(os.path.join(ROOT, "paper", "results_phase1.txt"), "w").write("\n".join(lines) + "\n")
