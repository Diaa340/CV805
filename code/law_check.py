"""Check the route law on all existing runs: v1 (independent routes, no shared bottleneck) and the parameter-free
inclusion-exclusion identity R' + S' = U1' + U2' (chance-floor corrected), which holds for any shared-bottleneck h."""
import glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def stats(o, p):
    x = pd.DataFrame(dict(o=o, p=p)).replace([np.inf, -np.inf], np.nan).dropna()
    return pearsonr(x.o, x.p)[0], (x.o - x.p).abs().mean(), len(x)


rows = []
for f in sorted(glob.glob("results/cl_step1/*_results.csv")):  # synthetic: per-type accuracy before/after; chance 0.1
    R = pd.read_csv(f); R = R[R.before_S >= 0.5]; c = 0.1
    for _, r in R.iterrows():
        fc = lambda t: (r[f"after_{t}"] - c) / (r[f"before_{t}"] - c)
        rows.append(dict(src="synthetic", cond=r.cond, U1=fc("U1"), U2=fc("U2"), R=fc("R"), S=fc("S")))
F = pd.read_csv("results/feat_cl_3ds_results.csv"); F = F[~((F.dataset == "cremad") & (F.split == 1))]
C1 = {"cifar2v": 50, "avmnist": 5, "cremad": 3}
for _, r in F.iterrows():  # real features: kept_* = share of correct-before still correct; chance 1/C1
    c = 1 / C1[r.dataset]; fc = lambda t: (r[f"kept_{t}"] - c) / (1 - c)
    rows.append(dict(src="real:" + r.dataset, cond=r.cond, U1=fc("u1"), U2=fc("u2"), R=fc("red"), S=fc("syn")))
D = pd.DataFrame(rows)
num = ["U1", "U2", "R", "S"]; D[num] = D[num].clip(-0.2, 1.2)
D["lhs"] = D.R + D.S; D["rhs"] = D.U1 + D.U2
q1, q2 = D.U1.clip(0, 1), D.U2.clip(0, 1)
D["R_v1"] = 1 - (1 - q1) * (1 - q2); D["S_v1"] = q1 * q2; D["h"] = D.U1 * D.U2 / D.S
print("identity R'+S' = U1'+U2' per source/condition:")
for (s, cond), g in D.groupby(["src", "cond"]):
    r, mae, n = stats(g.lhs, g.rhs)
    print(f"  {s:16s} {cond:15s} n={n:2d} mean lhs {g.lhs.mean():.2f} rhs {g.rhs.mean():.2f} |err| {mae:.3f}  median h {g.h.median():.2f}")
for name, g in (("all", D), ("synthetic", D[D.src == "synthetic"]), ("real", D[D.src != "synthetic"])):
    r, mae, n = stats(g.lhs, g.rhs); r1, m1, _ = stats(g.R, g.R_v1); r2, m2, _ = stats(g.S, g.S_v1)
    print(f"{name:9s}: identity r={r:.2f} |err|={mae:.3f} (n={n}) | v1: R r={r1:.2f} |err|={m1:.3f}; S r={r2:.2f} |err|={m2:.3f}")
D.to_csv("results/law_check.csv", index=False)
