"""Pre-registered analysis of the kinds kill test. Writes paper/results_kinds_kill.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import t as tdist
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []
def say(s=""): print(s); lines.append(s)
D = pd.read_csv(os.path.join(ROOT, "runs", "kinds_kill.csv"))
ok = (D.acc0 >= 0.75) & (D.lin_acc <= 0.60)
say(f"units: {D.seed.nunique()} seeds; manipulation checks failed on {(~ok).sum()} rows: "
    + str(D[~ok][['seed', 'kind', 'acc0', 'lin_acc']].drop_duplicates().values.tolist()))
D = D[ok]
say("\nmeans over seeds by kind and drift level:")
say(D.groupby(["kind", "level"])[["acc0", "lin_acc", "k1", "k2", "k12", "E", "P1", "P2", "r12", "C", "N", "drift1", "drift2"]].mean().round(3).to_string())
m = D.groupby(["kind", "level"])[["k1", "k2"]].mean().min(1).unstack(0)
inf = [l for l in m.index if all(0.1 <= m.loc[l, k] <= 0.9 for k in ("comparative", "conjunctive"))]
say(f"\ninformative levels (both kinds, min(k1,k2) in [0.1, 0.9]): {inf}")
if not inf:
    say("VERDICT: INCONCLUSIVE")
else:
    E = D[D.level.isin(inf)].groupby(["seed", "kind"]).E.mean().unstack()
    lo = lambda x: x.mean() - tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    ec, d = E["comparative"], E["comparative"] - E["conjunctive"]
    a = ec.mean() >= 0.05 and lo(ec) > 0; b = d.mean() >= 0.05 and lo(d) > 0
    say(f"per-seed E comparative {np.round(ec.values, 3)} conjunctive {np.round(E['conjunctive'].values, 3)} xor {np.round(E['xor'].values, 3)}")
    say(f"(a) comparative E {ec.mean():+.3f} (one-sided 95% lower {lo(ec):+.3f}) -> {'PASS' if a else 'FAIL'}")
    say(f"(b) comparative - conjunctive {d.mean():+.3f} (lower {lo(d):+.3f}) -> {'PASS' if b else 'FAIL'}")
    say(f"[reported] xor E {E['xor'].mean():+.3f}")
    say(f"VERDICT: {'PASS - keep the idea' if a and b else 'FAIL - drop the idea'}")
open(os.path.join(ROOT, "paper", "results_kinds_kill.txt"), "w").write("\n".join(lines) + "\n")
