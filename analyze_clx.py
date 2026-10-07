"""Pre-registered analysis of the CL-exceptions study (paper/preregistration_clx.md). Writes paper/results_clx.txt"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clx"); lines = []
def say(s=""): print(s); lines.append(s)
rng0 = np.random.default_rng(0)


def load(b):
    D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, f"samples_{b}_*.csv.gz")))])
    S = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, f"summary_{b}_*.csv")))]).pivot(index="unit", columns="method", values="task2_acc")
    B = D[D.before == 1].copy()
    for fam, lams in (("ewc", ("1", "100", "10000")), ("lwf", ("0.3", "3")), ("l2", ("0.3", "3"))):   # matched plasticity per unit
        pick = {}
        for u, x in B.groupby("unit"):
            ok = [l for l in lams if S.loc[u, f"{fam}_{l}"] >= S.loc[u, "finetune"] - 0.02]
            pick[u] = max(ok, key=lambda l: x[f"after__{fam}_{l}"].mean()) if ok else lams[0]
        B[f"after__{fam}"] = [r[f"after__{fam}_{pick[u]}"] for u, r in zip(B.unit, B.to_dict("records"))]
        S[fam] = [S.loc[u, f"{fam}_{pick[u]}"] for u in S.index]
    B["cbin"] = pd.qcut(B.conf.rank(method="first"), 10, labels=False)
    return B, S


def matched_gap(X, col):
    num = den = 0.0
    for _, s in X.groupby(["unit", "cbin"]):
        a, b = s[s.nonadd == 1][col], s[s.nonadd == 0][col]
        if len(a) >= 3 and len(b) >= 3:
            num += len(a) * (a.mean() - b.mean()); den += len(a)
    return num / den if den else np.nan


def ci(X, f, n=1000):
    us = X.unit.unique(); g = {u: x for u, x in X.groupby("unit")}; v = [f(pd.concat([g[u] for u in rng0.choice(us, len(us))])) for _ in range(n)]
    return np.nanpercentile(v, [2.5, 97.5])


V = {}
for b in ("tables", "clevr"):
    if not glob.glob(os.path.join(OUT, f"samples_{b}_*.csv.gz")):
        continue
    B, S = load(b); say(f"\n=== {b}: {B.unit.nunique()} units, {len(B)} task-1-correct test samples (non-additive {int(B.nonadd.sum())}) ===")
    say("  retention (additive / non-additive), matched gap [95% unit bootstrap], task-2 accuracy:")
    l1 = 0
    for m in ("finetune", "ewc", "lwf", "l2", "er_random_1000", "derpp_random_1000"):
        g = matched_gap(B, f"after__{m}"); lo, hi = ci(B, lambda X: matched_gap(X, f"after__{m}")); ok = g <= -0.03 and hi < 0; l1 += ok
        say(f"    {m:18s} {B[B.nonadd == 0][f'after__{m}'].mean():.3f} / {B[B.nonadd == 1][f'after__{m}'].mean():.3f} | gap {g:+.3f} [{lo:+.3f}, {hi:+.3f}] {'*' if ok else ' '} | task2 {S[m].mean():.3f}")
    V[f"L1 {b}"] = l1
    ft, lw = "after__finetune", "after__lwf"
    sel = lambda X: (X[X.nonadd == 0][lw].mean() - X[X.nonadd == 0][ft].mean()) - (X[X.nonadd == 1][lw].mean() - X[X.nonadd == 1][ft].mean())
    d = sel(B); lo, hi = ci(B, sel); V[f"L2b {b}"] = d >= 0.05 and lo > 0
    say(f"  L2b LwF selective protection (gain on additive - gain on non-additive): {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if V[f'L2b {b}'] else 'FAIL'}")
    if b == "tables" and "cover__random_50" in B:
        def inter(X):
            r = lambda na, cv: X[(X.nonadd == na) & (X.cover__random_50 == cv)].after__er_random_50.mean()
            return (r(1, 1) - r(1, 0)) - (r(0, 1) - r(0, 0))
        d = inter(B); lo, hi = ci(B, inter); V["L2a tables"] = d >= 0.07 and lo > 0
        say(f"  L2a replay coverage (ER random 50; cells covered {B.cover__random_50.mean():.2f}): non-additive covered-uncovered minus additive: {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASS' if V['L2a tables'] else 'FAIL'}")
    for m in ((50, 250, 1000) if b == "tables" else (250, 1000)):
        comp = {s: B[f"after__er_{s}_{m}"].mean() for s in ("random", "herding", "lowconf", "forget")}; best = max(comp, key=comp.get)
        dov = lambda X: X[f"after__er_ruleexc_{m}"].mean() - X[f"after__er_{best}_{m}"].mean()
        dno = lambda X: X[X.nonadd == 1][f"after__er_ruleexc_{m}"].mean() - X[X.nonadd == 1][f"after__er_{best}_{m}"].mean()
        a, (alo, _) = dov(B), ci(B, dov); c, (clo, _) = dno(B), ci(B, dno); dt = S[f"er_ruleexc_{m}"].mean() - S[f"er_{best}_{m}"].mean()
        ok = a >= 0.02 and alo > 0 and c >= 0.05 and clo > 0 and abs(dt) <= 0.02
        if m == 250:
            V[f"L3 {b}"] = ok
        say(f"  L3 buffer {m}: rule+exceptions vs best competitor ({best}): overall {a:+.3f} (lower {alo:+.3f}), non-additive {c:+.3f} (lower {clo:+.3f}), task2 {dt:+.3f} -> {'PASS' if ok else 'FAIL'}"
            f"   [reported] pure exception buffer overall {B[f'after__er_exception_{m}'].mean() - B[f'after__er_{best}_{m}'].mean():+.3f}")
    for m in (250, 1000):
        say(f"  [reported] DER++ {m}: rule+exceptions vs random overall {B[f'after__derpp_ruleexc_{m}'].mean() - B[f'after__derpp_random_{m}'].mean():+.3f}, "
            f"non-additive {B[B.nonadd == 1][f'after__derpp_ruleexc_{m}'].mean() - B[B.nonadd == 1][f'after__derpp_random_{m}'].mean():+.3f}")
l1 = {k: v for k, v in V.items() if k.startswith("L1")}
say("\nVERDICTS: L1 " + ("PASS" if all(v >= 4 for v in l1.values()) and len(l1) == 2 else ("KILL" if all(v == 0 for v in l1.values()) else "MIXED")) + f" ({l1})"
    + " | " + " | ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in V.items() if not k.startswith("L1")))
open(os.path.join(ROOT, "paper", "results_clx.txt"), "w").write("\n".join(lines) + "\n")
