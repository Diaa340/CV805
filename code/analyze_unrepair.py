"""Pre-registered analysis of the per-sample unrepairability test (paper/preregistration_unrepair.md). Writes paper/results_unrepair.txt"""
import glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "unrepair"); lines = []
def say(s=""): print(s); lines.append(s)
PRED = {"non-additive cell": "nonadd", "local syn - red": "synred", "low confidence": "negconf"}
V = {}
for b in ("tables", "clevr", "gqa", "pope"):
    fs = glob.glob(os.path.join(OUT, f"samples_{b}_*.csv.gz"))
    if not fs:
        continue
    D = pd.concat([pd.read_csv(f) for f in fs]); D["negconf"] = -D.conf; L = D[(D.before == 1) & (D.after == 0)].copy(); L["unrec"] = 1 - L.rec_lin
    say(f"=== {b}: {D.unit.nunique()} runs, {len(D)} test samples; lost after drift {len(L)} (non-additive {int(L.nonadd.sum())}) ===")
    say(f"  linear repair recovers: additive cells {L[L.nonadd == 0].rec_lin.mean():.3f} | non-additive cells {L[L.nonadd == 1].rec_lin.mean():.3f}  "
        f"(MLP repair: {L[L.nonadd == 0].rec_mlp.mean():.3f} | {L[L.nonadd == 1].rec_mlp.mean():.3f})")
    if L.nonadd.sum() < 30 or L.unrec.nunique() < 2:
        V[b] = "INCONCLUSIVE (too few lost non-additive samples)"; say(f"  -> {V[b]}"); continue
    auc = {k: roc_auc_score(L.unrec, L[c]) for k, c in PRED.items()}
    rng = np.random.default_rng(0); units = L.unit.unique(); g = {u: x for u, x in L.groupby("unit")}; bs = []
    for _ in range(500):
        X = pd.concat([g[u] for u in rng.choice(units, len(units))])
        if X.unrec.nunique() == 2:
            a = {k: roc_auc_score(X.unrec, X[c]) for k, c in PRED.items()}; bs.append((a["non-additive cell"], a["non-additive cell"] - max(a["local syn - red"], a["low confidence"])))
    bs = np.array(bs); ci = np.percentile(bs[:, 0], [2.5, 97.5]); cd = np.percentile(bs[:, 1], [2.5, 97.5])
    say("  AUC for predicting 'not recovered by the linear repair' among lost samples: " + " | ".join(f"{k} {v:.3f}" for k, v in auc.items()))
    say(f"  non-additive AUC 95% unit-bootstrap [{ci[0]:.3f}, {ci[1]:.3f}]; margin over the best competitor {auc['non-additive cell'] - max(auc['local syn - red'], auc['low confidence']):+.3f} [{cd[0]:+.3f}, {cd[1]:+.3f}]")
    V[b] = (auc["non-additive cell"] >= 0.70, auc["non-additive cell"] - max(auc["local syn - red"], auc["low confidence"]) >= 0.05)
    L["cbin"] = pd.qcut(L.conf.rank(method="first"), 5, labels=False)
    within = [roc_auc_score(x.unrec, x.nonadd) for _, x in L.groupby("cbin") if x.unrec.nunique() == 2 and x.nonadd.nunique() == 2]
    say(f"  [reported] non-additive AUC within confidence quintiles: {np.round(within, 3).tolist()}")
res = {b: v for b, v in V.items() if isinstance(v, tuple)}
U1 = all(v[0] for v in res.values()) and len(res) == 4; U2 = sum(v[1] for v in res.values()) >= 3
say(f"\nU1 (AUC >= 0.70 on every benchmark): {'PASS' if U1 else 'FAIL'} ({ {b: v[0] for b, v in res.items()} })")
say(f"U2 (beats synergy and confidence by >= 0.05 on >= 3 of 4): {'PASS' if U2 else 'FAIL'} ({ {b: v[1] for b, v in res.items()} })")
say("VERDICTS: U1 " + ("PASS" if U1 else "FAIL") + " | U2 " + ("PASS" if U2 else "FAIL") + "".join(f" | {b} {v}" for b, v in V.items() if not isinstance(v, tuple)))
open(os.path.join(ROOT, "paper", "results_unrepair.txt"), "w").write("\n".join(lines) + "\n")
