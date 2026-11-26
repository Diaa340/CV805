"""Revision R4 analysis (paper/preregistration_revision.md): are exceptions learned worse / forgotten faster than rule cells
of the same answer AND the same difficulty? Data: runs/rev/difficulty (cl_difficulty.py).
Difficulty: backward = the task-1 model's signed margin for the true answer before task 2; forward = the task-2-only
model's signed margin. Strata = unit x answer x difficulty decile (deciles within unit and answer, over all its cells).
Per stratum with >= 3 exception and >= 3 rule cells: (mean change on exceptions) - (mean change on rule cells), where
change = after - before (backward) or method - task-2-only (forward). Gaps are weighted by the number of exceptions.
Also reported: the same-answer gap without difficulty matching (strata = unit x answer), the comparison with the hardest
quarter of rule cells, and the share of exceptions inside usable strata. 95% bootstrap over units (GQA: base units).
Writes paper/results_rev_difficulty.txt"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); DIR = os.path.join(ROOT, "runs", "rev", "difficulty"); lines = []; NB = 1000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
METH = {"finetune": "Finetune", "lwf": "LwF", "l2": "Feature distill.", "er_random_1000": "ER", "derpp_random_1000": "DER++", "der_expand_1000": "DER-expansion"}


def load(b):
    D1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DIR, f"task1_{b}_*.csv.gz")))]).reset_index(drop=True)
    D2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DIR, f"task2_{b}_*.csv.gz")))]).reset_index(drop=True)
    for D in (D1, D2):
        for c in D.columns:
            if D[c].dtype == bool:
                D[c] = D[c].astype(float)
    a1, a2 = D1.groupby("unit").mean(numeric_only=True), D2.groupby("unit").mean(numeric_only=True)
    for fam, lams in (("lwf", ("0.3", "3")), ("l2", ("0.3", "3"))):   # matched plasticity, as in the original analyses
        pick = {u: (max(ok, key=lambda l: a1.loc[u, f"{fam}_{l}"]) if (ok := [l for l in lams if a2.loc[u, f"{fam}_{l}"] >= a2.loc[u, "finetune"] - 0.02]) else lams[0]) for u in a2.index}
        for D in (D1, D2):
            D[fam] = 0.0
            for l in lams:
                m = D.unit.map(pick) == l; D.loc[m, fam] = D.loc[m, f"{fam}_{l}"].astype(float)
    for D in (D1, D2):
        D["cluster"] = D.unit.str.rsplit("_s", n=1).str[0] if b == "gqa" else D.unit
    return D1, D2


def decile(D, col):
    r = D.groupby(["unit", "y"])[col].rank(pct=True, method="first"); return np.minimum((r * 10).astype(int), 9)


def strata(D, change, strat, hard=None):
    """Per usable stratum: its unit's cluster, number of exceptions, and (exceptions - rule cells) mean change."""
    rows = []; tot = D.groupby("cluster").apply(lambda x: (x.nonadd == 1).sum())
    for key, x in D.groupby(strat):
        e, r = x[x.nonadd == 1], x[x.nonadd == 0]
        if hard is not None:
            r = r[r[hard[0]] <= r[hard[0]].quantile(0.25)]
        if len(e) >= 3 and len(r) >= 3:
            rows.append((x.cluster.iloc[0], len(e), change(e).mean() - change(r).mean()))
    return pd.DataFrame(rows, columns=["cluster", "n", "d"]), tot


def summarise(S, tot):
    """Point estimate, 95% bootstrap over clusters (each stratum belongs to one cluster), coverage of exceptions."""
    cl = tot.index.values; k = {c: i for i, c in enumerate(cl)}; ci = S.cluster.map(k).values; n, d = S.n.values, S.d.values
    est = lambda w: (w[ci] * n * d).sum() / (w[ci] * n).sum() if (w[ci] * n).sum() > 0 else np.nan
    B = [est(np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl)).astype(float)) for _ in range(NB)]
    return est(np.ones(len(cl))), np.nanpercentile(B, [2.5, 97.5]), n.sum() / tot.sum()


R = {}
for b in ("tables", "clevr", "gqa"):
    if not glob.glob(os.path.join(DIR, f"task1_{b}_*.csv.gz")):
        continue
    D1, D2 = load(b); D1["dec"] = decile(D1, "margin_before"); D2["dec"] = decile(D2, "margin_scratch")
    say(f"\n=== {b}: {D1.unit.nunique()} runs ===   gap = exceptions minus rule cells (points of accuracy change); [95% CI]; coverage = share of exceptions matched")
    say("  method             | BACKWARD same answer | + same difficulty (coverage) | vs hardest rule quarter | FORWARD same answer | + same difficulty (coverage) | vs hardest rule quarter")
    for m, nm in METH.items():
        cb = lambda X, m=m: X[m] - X.before; cf = lambda X, m=m: X[m] - X.scratch; out = []
        for D, ch, mcol in ((D1, cb, "margin_before"), (D2, cf, "margin_scratch")):
            for strat, hard in ((["unit", "y"], None), (["unit", "y", "dec"], None), (["unit", "y"], (mcol,))):
                out.append(summarise(*strata(D, ch, strat, hard)))
        R[(b, m)] = out
        f = lambda t, cov=False: f"{100 * t[0]:+5.1f} [{100 * t[1][0]:+5.1f},{100 * t[1][1]:+5.1f}]" + (f" ({t[2]:.2f})" if cov else "")
        say(f"  {nm:18s} | {f(out[0])} | {f(out[1], True)} | {f(out[2])} | {f(out[3])} | {f(out[4], True)} | {f(out[5])}")
ok = lambda t: t[0] <= -0.03 and t[1][1] < 0
d1 = sum(ok(R[(b, "derpp_random_1000")][4]) for b in ("tables", "clevr", "gqa") if (b, "derpp_random_1000") in R)
d2 = all(ok(R[("clevr", m)][1]) for m in ("er_random_1000", "derpp_random_1000")) if ("clevr", "er_random_1000") in R else False
say(f"\nD1 (forward, difficulty-matched): DER++ gap <= -3 points with upper bound < 0 on {d1}/3 benchmarks -> {'PASS' if d1 >= 2 else 'FAIL'}")
say(f"D2 (backward, difficulty-matched, CLEVR): ER and DER++ both <= -3 points with upper bound < 0 -> {'PASS' if d2 else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_rev_difficulty.txt"), "w").write("\n".join(lines) + "\n")
