"""Pre-registered analysis of the estimator robustness sweep (paper/preregistration_estimator.md). Writes paper/results_estimator.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2"); lines = []
def say(s=""): print(s); lines.append(s)
U = pd.read_csv(os.path.join(OUT, "families.csv")).query("keep").set_index("unit")
S = pd.read_csv(os.path.join(OUT, "liang_sweep.csv")); S = S.join(U[["syn_mmi_exact", "d_add_hat", "dens"]], on="unit")
say(f"exact reference (MMI = exact BROJA, checked on 10x100 sub-tables): median synergy {U.syn_mmi_exact.median():.3f} bits")
rows = []
for (f, p, k), x in S.groupby(["feat", "pca", "k"]):
    rows.append(dict(feat=f, pca=p, k=k, median_S=x.S.median(), median_ratio=(x.S / x.syn_mmi_exact).median(),
                     rho_dadd=spearmanr(x.S, x.d_add_hat)[0], rho_exact=spearmanr(x.S, x.syn_mmi_exact)[0], n=len(x)))
R = pd.DataFrame(rows); say(R.round(3).to_string(index=False))
d = R[(R.feat == "clip") & R.pca & (R.k == 30)]
say("\ntheir default (CLIP-like features, PCA 5, k = 30): " + (d.round(3).to_string(index=False, header=False) if len(d) else "n/a"))
ok_all = (R.median_ratio < 0.5) & (R.rho_dadd < 0.5)
say(f"\nclaim 1 (underestimates > 2x) holds in {int((R.median_ratio < 0.5).sum())}/{len(R)} settings; "
    f"claim 2 (ranks units by structure weakly, rho < 0.5) in {int((R.rho_dadd < 0.5).sum())}/{len(R)}")
v = "ROBUST" if ok_all.all() else ("SETTING-DEPENDENT" if ok_all.any() else "NOT SUPPORTED")
best = R.loc[R.rho_dadd.idxmax()]
say(f"best setting for structure: {best.feat}, PCA {best.pca}, k {best.k}: rho {best.rho_dadd:.2f}, median ratio {best.median_ratio:.2f}")
say(f"VERDICT: {v}")
open(os.path.join(ROOT, "paper", "results_estimator.txt"), "w").write("\n".join(lines) + "\n")
