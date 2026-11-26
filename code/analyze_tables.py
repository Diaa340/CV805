"""Pre-registered analysis of the label-table kill test (paper/preregistration_tables.md). Writes paper/results_tables.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []
def say(s=""): print(s); lines.append(s)
S = np.load(os.path.join(ROOT, "runs", "tables", "selected.npz"))
M = pd.DataFrame(dict(table=range(len(S["T"])), fam=S["fam"], dens=S["dens"], d_add=S["d_add"], syn_mmi=S["syn_mmi"]))
M = M.merge(pd.read_csv(os.path.join(ROOT, "runs", "tables", "pid.csv")), on="table")
R = pd.read_csv(os.path.join(ROOT, "runs", "tables", "train.csv")).groupby("table", as_index=False).mean(numeric_only=True)
D = M.merge(R, on="table"); D["learn"] = D.acc_full - D.acc_maj
D["gap"] = (D.acc_full - D.acc_lin) / D.learn; D["gap_onehot"] = (1 - D.acc_onehot)
bad = D[D.learn < 0.10]; D = D[D.learn >= 0.10]
say(f"tables {len(D) + len(bad)}; excluded (full - majority < 0.10): {bad.table.tolist()}")
say(f"Spearman among analysed tables: syn_broja vs dens {spearmanr(D.syn_broja, D.dens)[0]:+.2f}; syn_broja vs d_add {spearmanr(D.syn_broja, D.d_add)[0]:+.2f}")


def pspear(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z); Z = np.c_[np.ones(len(rz)), rz]
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return np.corrcoef(res(ry), res(rx))[0, 1]


def boot(f, n=2000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(D), len(D)); v.append(f(D.iloc[i]))
    return np.percentile(v, [2.5, 97.5])


say("\n=== P1 (primary): partial Spearman of the readout gap ===")
a = pspear(D.gap, D.dens, D.syn_broja); b = pspear(D.gap, D.syn_broja, D.dens)
ca = boot(lambda x: pspear(x.gap, x.dens, x.syn_broja)); cb = boot(lambda x: pspear(x.gap, x.syn_broja, x.dens))
say(f"  rho(gap, XOR density | BROJA synergy) = {a:+.3f} [95% boot {ca[0]:+.2f}, {ca[1]:+.2f}]")
say(f"  rho(gap, BROJA synergy | XOR density) = {b:+.3f} [95% boot {cb[0]:+.2f}, {cb[1]:+.2f}]")
v1 = "STRUCTURE WINS (keep)" if a >= 0.5 and a - abs(b) >= 0.3 else ("SYNERGY WINS (drop the idea)" if b >= a else "MIXED")
say(f"  -> {v1}")
say("\n=== P2 (secondary): matched-synergy pairs, zero XOR squares vs XOR-dense (density >= 0.10), |dBROJA| <= 0.05 ===")
F_, X_ = D[D.dens == 0].sort_values("syn_broja"), D[D.dens >= 0.10].copy(); used, P = set(), []
for _, f in F_.iterrows():
    c = X_[~X_.table.isin(used)]; c = c.assign(d=(c.syn_broja - f.syn_broja).abs()); c = c[c.d <= 0.05]
    if len(c):
        x = c.sort_values("d").iloc[0]; used.add(x.table); P.append((f.table, x.table, f.syn_broja, x.syn_broja, f.gap, x.gap))
P = pd.DataFrame(P, columns=["ferrers", "xor", "syn_f", "syn_x", "gap_f", "gap_x"])
say(P.round(3).to_string(index=False) if len(P) else "  no pairs")
if len(P) < 5:
    say("  -> INCONCLUSIVE (fewer than 5 pairs)")
else:
    ok = ((P.gap_f <= 0.10) & (P.gap_x >= 0.40)).mean()
    say(f"  pairs meeting gap_f <= 0.10 and gap_x >= 0.40: {100 * ok:.0f}% (need >= 80%) -> {'PASS' if ok >= 0.8 else 'FAIL'}")
say("\n=== P3 (secondary): additive distance vs XOR density ===")
say(f"  Spearman(gap, d_add) {spearmanr(D.gap, D.d_add)[0]:+.3f} | Spearman(gap, density) {spearmanr(D.gap, D.dens)[0]:+.3f} | "
    f"Spearman(gap, BROJA synergy) {spearmanr(D.gap, D.syn_broja)[0]:+.3f}")
say(f"  rho(gap, d_add | BROJA) {pspear(D.gap, D.d_add, D.syn_broja):+.3f}; rho(gap, BROJA | d_add) {pspear(D.gap, D.syn_broja, D.d_add):+.3f}; "
    f"rho(gap, d_add | density) {pspear(D.gap, D.d_add, D.dens):+.3f}; rho(gap, density | d_add) {pspear(D.gap, D.dens, D.d_add):+.3f}")
say("\n[reported] other synergy measures in P1:")
for s in ("syn_ccs", "syn_mmi"):
    say(f"  {s}: rho(gap, density | syn) {pspear(D.gap, D.dens, D[s]):+.3f}; rho(gap, syn | density) {pspear(D.gap, D[s], D.dens):+.3f}")
say(f"\n[reported] ideal one-hot additive readout: Spearman(1 - acc_onehot, d_add) {spearmanr(D.gap_onehot, D.d_add)[0]:+.3f}; "
    f"max |(1 - acc_onehot) - d_add| {np.abs(D.gap_onehot - D.d_add).max():.3f}")
say("\n[reported] by family (means):")
say(D.groupby("fam")[["dens", "d_add", "syn_broja", "acc_full", "acc_lin", "gap"]].mean().round(3).to_string())
open(os.path.join(ROOT, "paper", "results_tables.txt"), "w").write("\n".join(lines) + "\n")
