"""Pre-registered analysis of the identical-PID experiment (paper/preregistration_samepid.md). Writes paper/results_samepid.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TD = os.path.join(ROOT, "runs", "tables"); lines = []
def say(s=""): print(s); lines.append(s)
S = np.load(os.path.join(TD, "samepid.npz")); P = pd.read_csv(os.path.join(TD, "pid_samepid.csv"))
M = pd.DataFrame(dict(table=range(len(S["T"])), family=S["family"], kind=S["kind"], d_add=S["d_add"], dens=S["dens"])).merge(P, on="table")
R = pd.read_csv(os.path.join(TD, "train_samepid.csv")).groupby("table", as_index=False).mean(numeric_only=True)
D = M.merge(R, on="table"); D["gap"] = (D.acc_full - D.acc_lin) / (D.acc_full - D.acc_maj)
spread = D.groupby("family")[["syn_broja", "red_broja", "syn_ccs"]].agg(lambda x: x.max() - x.min()).max()
say(f"{D.family.nunique()} families x 3 arrangements; max within-family spread: BROJA synergy {spread.syn_broja:.4f}, BROJA redundancy {spread.red_broja:.4f}, I_ccs synergy {spread.syn_ccs:.4f}")
say(D.pivot(index="family", columns="kind", values=["d_add", "gap"]).round(3).to_string())
C = D.assign(dc=D.d_add - D.groupby("family").d_add.transform("mean"), gc=D.gap - D.groupby("family").gap.transform("mean"))
r = spearmanr(C.dc, C.gc)[0]
W = D.pivot(index="family", columns="kind", values="gap"); dd = W["max"] - W["min"]
lo = dd.mean() - tdist.ppf(0.95, len(dd) - 1) * dd.std(ddof=1) / np.sqrt(len(dd))
ok = (dd >= 0.05).mean()
say(f"\nwithin-family Spearman(d_add, gap) {r:+.3f} (need >= 0.4)")
say(f"gap(max) - gap(min): mean {dd.mean():+.3f}, one-sided 95% lower {lo:+.3f}; >= 0.05 in {100 * ok:.0f}% of families (need >= 75%)")
v = "PASS" if r >= 0.4 and lo > 0 and ok >= 0.75 else "FAIL"
say(f"VERDICT: {v} (identical PID, the readout gap follows the degree of non-additivity)" if v == "PASS" else f"VERDICT: {v}")
open(os.path.join(ROOT, "paper", "results_samepid.txt"), "w").write("\n".join(lines) + "\n")
