"""Pre-registered analysis of v2 (paper/preregistration_v2.md). Writes paper/results_route_law_v2.txt."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["R", "U1", "U2", "S"]
lines = []


def say(s=""):
    print(s); lines.append(s)


def metrics(o, p):
    m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
    return dict(n=len(o), r=pearsonr(o, p)[0], R2=1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum(), MAE=np.abs(o - p).mean())


def preds(df, q):
    return {"U1": df[f"{q}_1U1"], "U2": df[f"{q}_2U2"], "R": 1 - (1 - df[f"{q}_1R"]) * (1 - df[f"{q}_2R"]), "S": df[f"{q}_1S"] * df[f"{q}_2S"]}


D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs", "*_v2.csv")) if "smoke" not in os.path.basename(f)], ignore_index=True)
say(f"v2: {len(D)} rows, {D.groupby(['snr', 'shift', 'depth', 'width', 'seed', 'batch']).ngroups} units")
valid = lambda df, t: df[f"before_{t}"] >= df.c + 0.3
H = D[(D.batch == "heldout") & (D.cond != "none")]
passed = {}
for regime, sub in (("REPLAY REGIME (control + interventions) [pre-registered test]", H), ("no replay [reported separately]", D[(D.batch == "heldout") & (D.cond == "none")])):
    say(f"\n=== {regime} ===")
    for q, name in (("qp", "primary: code decodability, re-fit"), ("qf", "secondary: code decodability, fixed readout"), ("qr", "tertiary: representation recoverability")):
        P = preds(sub, q)
        say(f"  {name}")
        for t in TYPES:
            s = valid(sub, t); m = metrics(sub[f"ret_{t}"][s].values, P[t][s].values)
            say(f"    {t:2s} n={m['n']:4d} r={m['r']:.3f} R2={m['R2']:.3f} MAE={m['MAE']:.3f}  (mean obs {sub[f'ret_{t}'][s].mean():.3f}, pred {P[t][s].mean():.3f})")
            if regime.startswith("REPLAY"):
                passed[(q, t)] = m
# criterion 1
c1 = all(passed[("qp", t)]["R2"] >= 0.6 and passed[("qp", t)]["MAE"] <= 0.08 for t in TYPES)
# criterion 2: vs v1 synergy MAE (0.140) and additive baseline fitted on discovery
disc = D[(D.batch == "discovery") & (D.cond != "none")]; ds, hs = disc[valid(disc, "S")], H[valid(H, "S")]
reg = LinearRegression().fit(ds[["qp_1S", "qp_2S"]].values, ds.ret_S.values)
add_mae = np.abs(hs.ret_S.values - reg.predict(hs[["qp_1S", "qp_2S"]].values)).mean()
c2 = passed[("qp", "S")]["MAE"] < 0.140 and passed[("qp", "S")]["MAE"] < add_mae
# criterion 3: AND bounds
lo = np.maximum(0, hs.qp_1S + hs.qp_2S - 1) - 0.05; hi = np.minimum(hs.qp_1S, hs.qp_2S) + 0.05
inb = ((hs.ret_S >= lo) & (hs.ret_S <= hi)).mean(); above = (hs.ret_S > hi).mean(); below = (hs.ret_S < lo).mean()
c3 = inb >= 0.8
gate = c1 and c2 and c3 and passed[("qr", "R")]["R2"] >= 0.5 and passed[("qr", "S")]["R2"] >= 0.5
say("\n=== PRE-REGISTERED VERDICT ===")
say(f"  1. primary q: R2>=0.6 and MAE<=0.08 for R, S, U1, U2 -> {'PASS' if c1 else 'FAIL'}")
say(f"  2. synergy MAE {passed[('qp', 'S')]['MAE']:.3f} < v1 0.140 and < additive baseline {add_mae:.3f} -> {'PASS' if c2 else 'FAIL'}")
say(f"  3. synergy inside AND bounds in {100 * inb:.0f}% of runs (need >= 80%; above {100 * above:.0f}%, below {100 * below:.0f}%) -> {'PASS' if c3 else 'FAIL'}")
say(f"  gate to real data (1-3 and tertiary R2>=0.5 for R and S): {'OPEN' if gate else 'CLOSED'}")
open(os.path.join(ROOT, "paper", "results_route_law_v2.txt"), "w").write("\n".join(lines) + "\n")
