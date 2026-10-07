"""Pre-registered analysis of v4 H1–H4 (paper/preregistration_v4_fia.md + addendum). Writes paper/results_v4.txt."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.abspath(__file__))
lines = []


def say(s=""):
    print(s); lines.append(s)


NAT = ("cifar2v", "avmnist", "cremad", "food101", "n24news")
rows, samp = [], []
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "v4", "v4_*.npz"))):
    z = np.load(f, allow_pickle=True); name = os.path.basename(f)[:-4]
    data = str(z["data"]); regime = name.rsplit("_", 1)[1]
    group = "synth" if data == "synth" else ("bridge" if data.startswith("bridge_") else data)
    ok = z["before"].astype(bool); types = z["types"]
    for t in ("R", "U1", "U2", "S"):
        k = ok & (types == t)
        if k.sum() < 30:
            continue
        p1, p2, p12, p1n2 = z["S1"][k].mean(), z["S2"][k].mean(), z["S12"][k].mean(), (z["S1"][k] & z["S2"][k]).mean()
        grad = ((z["rho1"] * z["a1"] + z["rho2"] * z["a2"] + z["rho1"] * z["rho2"] * z["beta"] + z["b"]) > 0)[k].mean()
        rows.append(dict(unit=name.rsplit("_", 1)[0], data=data, group=group, regime=regime, type=t, n=int(k.sum()), P1=p1, P2=p2,
                         r12=p12, C=p1n2 - p1 * p2, N=p12 - p1n2, discrete=p1 * p2, graded=grad,
                         kappa=float(z["kappa"]), kappa_c=float(z["kappa_c"])))
    if data in NAT and regime == "none":
        lost = ok & ~z["actual"].astype(bool)
        share = np.abs(z["beta"]) / (np.abs(z["a1"]) + np.abs(z["a2"]) + np.abs(z["beta"]) + 1e-8)
        samp.append(pd.DataFrame(dict(share=share[lost], recovered=z["lin"][lost].astype(float))))
D = pd.DataFrame(rows)
S = D[D.type == "S"]
say(f"v4: {D.unit.nunique()} units x 2 regimes")
say("\n=== H1: decomposition separates the regimes (needs-both group; both regimes) ===")
for g, x in S.groupby("group"):
    say(f"  {g:10s} n={len(x):3d} median C {x.C.median():+.3f} | median N {x.N.median():+.3f} | median |N| {x.N.abs().median():.3f} | "
        f"r12 {x.r12.mean():.2f} vs P1*P2 {x.discrete.mean():.2f}")
sb = S[S.group.isin(["synth", "bridge"])]; fv = S[S.group == "fv_cremad"]
h1 = sb.N.abs().median() <= 0.05 and fv.N.median() >= 0.05
say(f"  -> synthetic + bridge median |N| {sb.N.abs().median():.3f} (need <= 0.05); face-voice median N {fv.N.median():+.3f} (need >= 0.05): {'PASS' if h1 else 'FAIL'}")
say("\n=== H2: a-priori coupling statistic predicts N (needs-both group; all units, both regimes) ===")
r_c = spearmanr(S.kappa_c, S.N)[0]; r_k = spearmanr(S.kappa, S.N)[0]
say(f"  Spearman(kappa_c, N) = {r_c:+.3f} (need >= 0.6) -> {'PASS' if r_c >= 0.6 else 'FAIL'} | [reported] Spearman(kappa, N) = {r_k:+.3f}")
say("  per group: " + "; ".join(f"{g}: kappa_c {x.kappa_c.median():+.2f}, N {x.N.median():+.3f}" for g, x in S.groupby("group")))
say("\n=== H3: graded evidence law vs discrete v3 law (natural datasets, all types, both regimes; target = share surviving S12) ===")
Nd = D[D.data.isin(NAT)]
r2 = lambda p: 1 - ((Nd.r12 - p) ** 2).sum() / ((Nd.r12 - Nd.r12.mean()) ** 2).sum()
rg, rd = r2(Nd.graded), r2(Nd.discrete)
say(f"  R2 graded {rg:.3f} | R2 discrete {rd:.3f} | difference {rg - rd:+.3f} (need >= 0.15) -> {'PASS' if rg - rd >= 0.15 else 'FAIL'}")
for t, x in Nd.groupby("type"):
    say(f"    {t:2s} n={len(x):3d} mean obs {x.r12.mean():.2f} graded {x.graded.mean():.2f} discrete {x.discrete.mean():.2f}")
say("\n=== H4: linear-readout recovery falls with interaction share (natural datasets, no replay, lost samples) ===")
Sm = pd.concat(samp, ignore_index=True); Sm["q"] = pd.qcut(Sm.share.rank(method="first"), 5, labels=False)
rec = Sm.groupby("q").recovered.mean()
rho4 = spearmanr(rec.index, rec.values)[0]; d4 = rec.iloc[-1] - rec.iloc[0]
say("  recovered share by interaction-share quintile: " + " ".join(f"Q{i + 1}={v:.2f}" for i, v in rec.items())
    + f" | quintile share medians " + " ".join(f"{v:.2f}" for v in Sm.groupby('q').share.median()))
say(f"  Spearman {rho4:+.2f} (need <= -0.9), Q5-Q1 {d4:+.3f} (need <= -0.15) -> {'PASS' if rho4 <= -0.9 and d4 <= -0.15 else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_v4.txt"), "w").write("\n".join(lines) + "\n")
