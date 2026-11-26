"""Pre-registered analysis of P3 (paper/preregistration_p3_readout.md). Writes paper/results_p3_readout.txt."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["R", "U1", "U2", "S"]
ARMS = ["no_repair", "readout_linear", "readout_mlp", "encoder_realign", "ceiling"]
lines = []


def say(s=""):
    print(s); lines.append(s)


D = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "runs", "p3_*_p3.csv"))], ignore_index=True)
key = ["data", "class_split", "order", "seed", "regime"]
W = D.pivot_table(index=key, columns="arm", values=[f"ret_{t}" for t in TYPES])
N = D[D.arm == "no_repair"].set_index(key)[[f"n_{t}" for t in TYPES]]


def recovery(w, arm, t, n):
    gap = w[(f"ret_{t}", "ceiling")] - w[(f"ret_{t}", "no_repair")]
    r = (w[(f"ret_{t}", arm)] - w[(f"ret_{t}", "no_repair")]) / gap
    return r.where((gap >= 0.05) & (n[f"n_{t}"] >= 30))


verdict = {}
for regime in ("none", "control"):
    w = W.xs(regime, level="regime"); n = N.xs(regime, level="regime")
    say(f"\n=== {'NO REPLAY [primary]' if regime == 'none' else 'REPLAY [reported]'}: {len(w)} units ===")
    say("  mean retention per arm: ")
    for arm in ARMS:
        say(f"    {arm:15s} " + " ".join(f"{t}={w[(f'ret_{t}', arm)].mean():.2f}" for t in TYPES))
    rec = {arm: pd.DataFrame({t: recovery(w, arm, t, n) for t in TYPES}) for arm in ("readout_linear", "readout_mlp", "encoder_realign")}
    for arm, r in rec.items():
        say(f"  median recovery ({arm:15s}): " + " ".join(f"{t}={r[t].median():.2f}" for t in TYPES) + f"  (units with S defined: {r.S.notna().sum()})")
    r = rec["readout_mlp"].dropna()
    gapS = (r[["R", "U1", "U2"]].min(axis=1) - r.S).median()
    lowest = (r.S <= r[["R", "U1", "U2"]].min(axis=1)).mean()
    dre = (rec["encoder_realign"].S - rec["readout_mlp"].S).dropna().median()
    verdict[regime] = (gapS, lowest, dre, len(r))
    say(f"  C1: median[min(rec R,U1,U2) - rec S] = {gapS:+.3f} (need >= 0.15); S lowest in {100 * lowest:.0f}% of {len(r)} units (need >= 70%)")
    say(f"  C2: median[rec_S(encoder_realign) - rec_S(readout_mlp)] = {dre:+.3f} (need >= 0.20)")
    say("  per dataset, readout_mlp median recovery R/U1/U2/S: " + "; ".join(
        f"{d}: " + "/".join(f"{rec['readout_mlp'][t][rec['readout_mlp'].index.get_level_values('data') == d].median():.2f}" for t in TYPES)
        for d in sorted(set(rec["readout_mlp"].index.get_level_values("data")))))
g, l, d, nn_ = verdict["none"]
c1, c2 = g >= 0.15 and l >= 0.7, d >= 0.20
say("\n=== PRE-REGISTERED VERDICT (no replay) ===")
say(f"  C1 readout limit: {'PASS' if c1 else 'FAIL'} | C2 re-alignment beats readout for synergy: {'PASS' if c2 else 'FAIL'}")
say(f"  -> claim 'readout retraining recovers R/U but not synergy': {'SUPPORTED' if c1 and c2 else 'NOT SUPPORTED'}")
open(os.path.join(ROOT, "paper", "results_p3_readout.txt"), "w").write("\n".join(lines) + "\n")
