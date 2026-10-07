"""Pre-registered analysis of step 1 (paper/preregistration_step1.md). Writes paper/results_step1.txt"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = os.path.dirname(os.path.abspath(__file__)); TD = os.path.join(ROOT, "runs", "tables"); lines = []
def say(s=""): print(s); lines.append(s)


def meta(setname, pidfile, Fkey):
    S = np.load(os.path.join(TD, f"{setname}.npz"))
    M = pd.DataFrame(dict(table=range(len(S["T"])), fam=S["fam"], dens=S["dens"], d_add=S["d_add"]))
    if "pair" in S.files:
        M["pair"], M["role"] = S["pair"], S["role"]
    return M.merge(pd.read_csv(os.path.join(TD, pidfile))[["table", "syn_broja", "syn_ccs"]], on="table").assign(set=setname)


M = pd.concat([meta("selected", "pid.csv", None), meta("fresh", "pid_fresh.csv", None), meta("pairs", "pid_pairs.csv", None)])
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(TD, "step1_*.csv"))])
R = R.groupby(["set", "table"], as_index=False).mean(numeric_only=True)
D = M.merge(R, on=["set", "table"]); D["learn"] = D.acc_before - D.acc_maj
D["gap"] = (D.acc_before - D.acc_lin) / D.learn
D["shortfall"] = (D.acc_rep_mlp - D.acc_rep_lin) / D.learn
D["rec_lin"] = (D.acc_rep_lin - D.acc_drift) / (D.acc_before - D.acc_drift); D["rec_mlp"] = (D.acc_rep_mlp - D.acc_drift) / (D.acc_before - D.acc_drift)
D["ood_drop"] = D.seen_bal - D.ood_bal
bad = D[D.learn < 0.10]; D = D[D.learn >= 0.10]
say(f"tables: selected {sum(D.set == 'selected')}, fresh {sum(D.set == 'fresh')}, pairs {sum(D.set == 'pairs')}; excluded (not learned): "
    f"{bad[['set', 'table']].values.tolist()}")


def pspear(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z); Z = np.c_[np.ones(len(rz)), rz]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return np.corrcoef(res(ry), res(rx))[0, 1]


def boot(X, f, n=2000):
    rng = np.random.default_rng(0); v = [f(X.iloc[rng.integers(0, len(X), len(X))]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def ptest(X, y, s, sign, label):
    a = pspear(X[y], X[s], X.syn_broja); b = pspear(X[y], X.syn_broja, X[s])
    ca = boot(X, lambda x: pspear(x[y], x[s], x.syn_broja)); cb = boot(X, lambda x: pspear(x[y], x.syn_broja, x[s]))
    say(f"  rho({y}, {s} | BROJA) = {a:+.3f} [{ca[0]:+.2f}, {ca[1]:+.2f}] | rho({y}, BROJA | {s}) = {b:+.3f} [{cb[0]:+.2f}, {cb[1]:+.2f}]")
    a, b = sign * a, sign * b
    v = "STRUCTURE WINS" if a >= 0.5 and a - abs(b) >= 0.3 else ("SYNERGY WINS" if b >= a else "MIXED")
    say(f"  -> {label}: {v}"); return v


V = {}
say("\n=== R1: P1 replication on the fresh tables (readout gap ~ XOR density vs BROJA) ===")
V["R1"] = ptest(D[D.set == "fresh"], "gap", "dens", +1, "R1")
O = D[D.set.isin(["selected", "fresh"])]
say(f"\n=== A (1a): repair after drift, additive shortfall = (MLP repair - linear repair) / (before - majority); {len(O)} tables ===")
V["A"] = ptest(O, "shortfall", "d_add", +1, "A")
say(f"\n=== B (1b): compositional OOD, drop = seen - held-out balanced accuracy; {len(O)} tables ===")
V["B"] = ptest(O, "ood_drop", "d_add", +1, "B")
say("\n=== D (1d): matched pairs by construction ===")
P = D[D.set == "pairs"].pivot(index="pair", columns="role", values=["gap", "syn_broja", "dens", "d_add"]).dropna()
dg = P[("gap", "dense")] - P[("gap", "zero")]
say(P.round(3).to_string())
if len(P) < 8:
    V["D"] = "INCONCLUSIVE"; say(f"  only {len(P)} pairs -> INCONCLUSIVE")
else:
    ok = (dg >= 0.20).mean(); V["D"] = "PASS" if ok >= 0.8 and dg.mean() >= 0.25 else "FAIL"
    say(f"  gap(dense) - gap(zero): mean {dg.mean():+.3f}; >= 0.20 in {100 * ok:.0f}% of {len(P)} pairs -> {V['D']}")
say("\n[reported] 1c survival in removable vs non-removable cells (tables with non-removable cells; seed means, then table means):")
C = O[O.n_non > 0].copy()
for g in ("rem", "non"):
    C[f"E_{g}"] = C[f"r12_{g}"] - C[f"P1_{g}"] * C[f"P2_{g}"]; C[f"N_{g}"] = C[f"r12_{g}"] - C[f"P1n2_{g}"]
say(C[["P1_rem", "P2_rem", "r12_rem", "E_rem", "N_rem", "P1_non", "P2_non", "r12_non", "E_non", "N_non"]].mean().round(3).to_string())
say("\n[reported] with XOR density instead of d_add, and with I_ccs:")
for y in ("shortfall", "ood_drop"):
    say(f"  {y}: rho(., density | BROJA) {pspear(O[y], O.dens, O.syn_broja):+.3f}; rho(., BROJA | density) {pspear(O[y], O.syn_broja, O.dens):+.3f}; "
        f"rho(., d_add | CCS) {pspear(O[y], O.d_add, O.syn_ccs):+.3f}; rho(., CCS | d_add) {pspear(O[y], O.syn_ccs, O.d_add):+.3f}")
say(f"  R1 with d_add: rho(gap, d_add | BROJA) {pspear(D[D.set == 'fresh'].gap, D[D.set == 'fresh'].d_add, D[D.set == 'fresh'].syn_broja):+.3f}")
say("\n[reported] recovery fractions and OOD by structure (tables split at d_add = 0):")
say(O.assign(additive=O.d_add == 0).groupby("additive")[["rec_lin", "rec_mlp", "shortfall", "seen_bal", "ood_bal", "ood_drop", "ood_lin_bal", "ood_onehot_cells"]].mean().round(3).to_string())
say(f"  Spearman(rec_mlp, d_add) {spearmanr(O.rec_mlp, O.d_add)[0]:+.3f}; Spearman(rec_mlp, BROJA) {spearmanr(O.rec_mlp, O.syn_broja)[0]:+.3f}")
say("\nVERDICTS: " + " | ".join(f"{k} {v}" for k, v in V.items()))
open(os.path.join(ROOT, "paper", "results_step1.txt"), "w").write("\n".join(lines) + "\n")
