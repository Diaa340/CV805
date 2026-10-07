"""Pre-registered analysis of the confirmatory follow-ups (paper/preregistration_f1f2.md). Writes paper/results_f1f2.txt"""
import glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []
def say(s=""): print(s); lines.append(s)
rng0 = np.random.default_rng(0)
def boot(X, f, n=1000):
    us = X.unit.unique(); g = {u: x for u, x in X.groupby("unit")}; v = [f(pd.concat([g[u] for u in rng0.choice(us, len(us))])) for _ in range(n)]
    return np.nanpercentile(v, [2.5, 97.5])
V = {}
say("=== F1: DER++ with a rule + exceptions buffer vs DER++ with a random buffer (fresh seed; fresh CLEVR units) ===")
for b in ("tables", "clevr"):
    fs = glob.glob(os.path.join(ROOT, "runs", "clx2", f"samples_{b}_*.csv.gz"))
    if not fs:
        continue
    D = pd.concat([pd.read_csv(f) for f in fs]); B = D[D.before == 1]
    S = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs", "clx2", f"summary_{b}_*.csv"))]).pivot(index="unit", columns="method", values="task2_acc")
    say(f"  -- {b}: {B.unit.nunique()} units, {len(B)} task-1-correct samples ({int(B.nonadd.sum())} exceptions)")
    for m in (1000, 250):
        a, r = f"after__derpp_ruleexc_{m}", f"after__derpp_random_{m}"
        dex = lambda X: X[X.nonadd == 1][a].mean() - X[X.nonadd == 1][r].mean(); dov = lambda X: X[a].mean() - X[r].mean()
        e, (elo, ehi) = dex(B), boot(B, dex); o, (olo, ohi) = dov(B), boot(B, dov); dt = (S[f"derpp_ruleexc_{m}"] - S[f"derpp_random_{m}"]).mean()
        ok = e >= 0.05 and elo > 0 and olo > -0.02 and abs(dt) <= 0.02
        if m == 1000:
            V[f"F1 {b}"] = ok
        say(f"     buffer {m}: exceptions {e:+.3f} [{elo:+.3f}, {ehi:+.3f}] | overall {o:+.3f} [{olo:+.3f}, {ohi:+.3f}] | rule {B[B.nonadd == 0][a].mean() - B[B.nonadd == 0][r].mean():+.3f} | "
            f"task 2 {dt:+.3f} -> {'PASS' if ok else 'FAIL'}{' (primary)' if m == 1000 else ' (reported)'}")
        comp = {k: B[f"after__derpp_{k}_{m}"].mean() for k in ("random", "herding", "forget")}; best = max(comp, key=comp.get)
        say(f"       [reported] vs best DER++ competitor ({best}): exceptions {B[B.nonadd == 1][a].mean() - B[B.nonadd == 1][f'after__derpp_{best}_{m}'].mean():+.3f}, "
            f"overall {B[a].mean() - B[f'after__derpp_{best}_{m}'].mean():+.3f}")
say("\n=== F2: graded additive misfit predicts which forgotten samples cheap repair cannot recover (fresh seeds) ===")
f2a = f2b = 0; nb = 0
for b in ("tables", "clevr", "gqa", "pope"):
    fs = glob.glob(os.path.join(ROOT, "runs", "unrepair2", f"samples_{b}_*.csv.gz"))
    if not fs:
        continue
    D = pd.concat([pd.read_csv(f) for f in fs]); L = D[(D.before == 1) & (D.after == 0)].copy(); L["unrec"] = 1 - L.rec_lin; L["negconf"] = -L.conf; nb += 1
    am, ac = roc_auc_score(L.unrec, L.synred), roc_auc_score(L.unrec, L.negconf)
    d = lambda X: roc_auc_score(X.unrec, X.synred) - roc_auc_score(X.unrec, X.negconf) if X.unrec.nunique() == 2 else np.nan
    lo, hi = boot(L, d, 500); f2a += am >= 0.75; f2b += (am - ac >= 0.05 and lo > 0)
    L["cbin"] = pd.qcut(L.conf.rank(method="first"), 5, labels=False)
    within = [roc_auc_score(x.unrec, x.synred) for _, x in L.groupby("cbin") if x.unrec.nunique() == 2]
    say(f"  {b:6s} lost {len(L)}: AUC misfit {am:.3f} | confidence {ac:.3f} | difference {am - ac:+.3f} [{lo:+.3f}, {hi:+.3f}] | within confidence quintiles median {np.median(within):.3f}")
V["F2a (AUC >= 0.75 on >= 3 of 4)"] = f2a >= 3; V["F2b (beats confidence by >= 0.05 on >= 3 of 4)"] = f2b >= 3
say(f"  counts: F2a {f2a}/{nb}, F2b {f2b}/{nb}")
say("\nVERDICTS: " + " | ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in V.items()))
open(os.path.join(ROOT, "paper", "results_f1f2.txt"), "w").write("\n".join(lines) + "\n")
