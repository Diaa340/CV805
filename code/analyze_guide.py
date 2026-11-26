"""Leave-one-benchmark-out check of a data-driven method guide (paper/preregistration_guide.md). Saved results only.
Observations: units of the two-task study (tables, CLEVR, GQA; GQA runs per seed) and (run, unit) pairs of the CLOVE
six-task study (tasks 1-5). Twelve standard methods (matched plasticity as in the original analyses).
Target (primary): final accuracy on the unit's exception cells; secondary: type-balanced final accuracy.
Descriptors (data only, computed before any CL): d1 exception share, d2 answer skew (yes share of exceptions minus
yes share of rule cells), d3 yes share, d4 exception accuracy right after learning the task.
Guide: per method, ridge regression (alpha 1, standardised descriptors) of the target, fit on three benchmarks,
predicting the fourth. Baselines: M = each method's mean target on the three benchmarks (no descriptors); B = each
method's mean overall BWT on the three benchmarks (the usual CL ranking). Scores per held-out observation: Spearman
between predicted and actual method ordering; regret = best actual target minus actual target of the predicted best.
95% intervals: bootstrap over held-out units. Writes paper/results_guide.txt, figures/guide_obs.csv"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []; NB = 1000; rng0 = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
METH = ["finetune", "ewc", "si", "lwf", "l2", "er", "agem", "gem", "derpp", "icarl", "der_expand", "multihead"]
TWO = {"er": "er_random_1000", "agem": "agem_random_1000", "gem": "gem_random_1000", "derpp": "derpp_random_1000", "der_expand": "der_expand_1000"}
G = [("ewc", ("1", "100", "10000")), ("si", ("0.1", "1", "10", "100")), ("lwf", ("0.3", "3")), ("l2", ("0.3", "3")), ("icarl", ("0.1", "1"))]
obs = []
for b in ("tables", "clevr", "gqa"):
    Y = np.load(os.path.join(ROOT, "runs", "clm", f"y_{b}.npz"))
    D1 = pd.concat([pd.read_csv(f) for f in Y["task1_files"]]).reset_index(drop=True); D2 = pd.concat([pd.read_csv(f) for f in Y["task2_files"]]).reset_index(drop=True)
    D1["y"] = Y["task1"]; a2 = D2.groupby("unit").mean(numeric_only=True); a1 = D1.groupby("unit").mean(numeric_only=True)
    for fam, lams in G:
        pick = {u: (max(ok, key=lambda l: a1.loc[u, f"{fam}_{l}"]) if (ok := [l for l in lams if a2.loc[u, f"{fam}_{l}"] >= a2.loc[u, "finetune"] - 0.02]) else lams[0]) for u in a2.index}
        D1[fam] = 0.0
        for l in lams:
            m = D1.unit.map(pick) == l; D1.loc[m, fam] = D1.loc[m, f"{fam}_{l}"].astype(float)
    for u, x in D1.groupby("unit"):
        e, r = x[x.nonadd == 1], x[x.nonadd == 0]
        if len(e) < 5:
            continue
        o = dict(bench=b, cluster=u.rsplit("_s", 1)[0] if b == "gqa" else u, d1=len(e) / len(x), d2=e.y.mean() - r.y.mean(), d3=x.y.mean(), d4=e.before.mean())
        for m in METH:
            c = TWO.get(m, m); o[f"E_{m}"] = e[c].mean(); o[f"R_{m}"] = r[c].mean(); o[f"BWT_{m}"] = x[c].mean() - x.before.mean()
        obs.append(o)
C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "clove_cl", "run_[0-9]*.csv.gz")))])
for fam, lams in G:
    late = C[C.pos >= 2].groupby("run").mean(numeric_only=True); allr = C.groupby("run").mean(numeric_only=True); pick = {}
    for r in late.index:
        ok = [l for l in lams if late.loc[r, f"{fam}_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]; pick[r] = max(ok, key=lambda l: allr.loc[r, f"{fam}_{l}__final"]) if ok else lams[0]
    for s in ("own", "final"):
        C[f"{fam}__{s}"] = 0.0
        for l in lams:
            m = C.run.map(pick) == l; C.loc[m, f"{fam}__{s}"] = C.loc[m, f"{fam}_{l}__{s}"]
for (run, u), x in C[C.pos <= 5].groupby(["run", "unit"]):
    e, r = x[x.nonadd == 1], x[x.nonadd == 0]
    if len(e) < 5:
        continue
    o = dict(bench="clove", cluster=u, d1=len(e) / len(x), d2=e.y.mean() - r.y.mean(), d3=x.y.mean(), d4=e["finetune__own"].mean())
    for m in METH:
        o[f"E_{m}"] = e[f"{m}__final"].mean(); o[f"R_{m}"] = r[f"{m}__final"].mean(); o[f"BWT_{m}"] = x[f"{m}__final"].mean() - x[f"{m}__own"].mean()
    obs.append(o)
O = pd.DataFrame(obs); O.round(4).to_csv(os.path.join(ROOT, "figures", "guide_obs.csv"), index=False)
for m in METH:
    O[f"BAL_{m}"] = 0.5 * (O[f"E_{m}"] + O[f"R_{m}"])
DESC = ["d1", "d2", "d3", "d4"]; BENCH = ["tables", "clevr", "gqa", "clove"]
say(f"Observations: {O.bench.value_counts().to_dict()}")
say("Descriptor means by benchmark:\n" + O.groupby("bench")[DESC].mean().round(3).loc[BENCH].to_string())


def ridge(X, y, a=1.0):
    X1 = np.c_[np.ones(len(X)), X]; I = np.eye(X1.shape[1]); I[0, 0] = 0; return np.linalg.solve(X1.T @ X1 + a * I, X1.T @ y)


def lobo(target):
    out = {}
    for hb in BENCH:
        tr, te = O[O.bench != hb], O[O.bench == hb]; mu, sd = tr[DESC].mean(), tr[DESC].std() + 1e-9
        Xtr, Xte = ((tr[DESC] - mu) / sd).values, ((te[DESC] - mu) / sd).values
        Pg = np.stack([np.c_[np.ones(len(Xte)), Xte] @ ridge(Xtr, tr[f"{target}_{m}"].values) for m in METH], 1)
        Pm = np.tile([tr[f"{target}_{m}"].mean() for m in METH], (len(te), 1)); Pb = np.tile([tr[f"BWT_{m}"].mean() for m in METH], (len(te), 1))
        A = te[[f"{target}_{m}" for m in METH]].values
        sp = lambda P: np.array([spearmanr(p, a).correlation if np.std(a) > 0 else np.nan for p, a in zip(P, A)])
        rg = lambda P: A.max(1) - A[np.arange(len(A)), P.argmax(1)]
        out[hb] = dict(cl=te.cluster.values, sg=sp(Pg), sm=sp(Pm), sb=sp(Pb), rgg=rg(Pg), rgm=rg(Pm), rgb=rg(Pb),
                       best_g=pd.Series([METH[k] for k in Pg.argmax(1)]).value_counts().head(3).to_dict(), best_m=METH[int(Pm[0].argmax())], best_b=METH[int(Pb[0].argmax())],
                       best_actual=pd.Series([METH[k] for k in A.argmax(1)]).value_counts().head(3).to_dict())
    return out


def boot(cl, v):
    u = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in u}; s = []
    for _ in range(NB):
        k = np.concatenate([idx[c] for c in rng0.choice(u, len(u))]); s.append(np.nanmean(v[k]))
    return np.nanpercentile(s, [2.5, 97.5])


V = {}
for target, label in (("E", "PRIMARY: final accuracy on exception cells"), ("BAL", "SECONDARY: type-balanced final accuracy")):
    R = lobo(target); say(f"\n=== {label} ===")
    say("  held-out   Spearman guide / mean-ranking / BWT-ranking | guide - mean [95%] | mean - BWT [95%] | regret guide / mean / BWT")
    g1 = g2 = 0
    for hb, r in R.items():
        d1 = r["sg"] - r["sm"]; d2 = r["sm"] - r["sb"]; c1 = boot(r["cl"], d1); c2 = boot(r["cl"], d2)
        g1 += bool(c1[0] > 0); g2 += bool(c2[0] > 0)
        say(f"  {hb:8s} {np.nanmean(r['sg']):+.3f} / {np.nanmean(r['sm']):+.3f} / {np.nanmean(r['sb']):+.3f} | {np.nanmean(d1):+.3f} [{c1[0]:+.3f},{c1[1]:+.3f}] | "
            f"{np.nanmean(d2):+.3f} [{c2[0]:+.3f},{c2[1]:+.3f}] | {r['rgg'].mean():.3f} / {r['rgm'].mean():.3f} / {r['rgb'].mean():.3f}")
        say(f"           picks: guide {r['best_g']}, mean-ranking {r['best_m']}, BWT-ranking {r['best_b']}; actually best {r['best_actual']}")
    V[target] = (g1, g2)
say("\nGuide fitted on all four benchmarks (primary target): standardised ridge coefficients per method")
mu, sd = O[DESC].mean(), O[DESC].std(); X = ((O[DESC] - mu) / sd).values
say(pd.DataFrame({m: ridge(X, O[f"E_{m}"].values) for m in METH}, index=["intercept"] + DESC).T.round(3).to_string())
say(f"\nVERDICTS (primary): G1 guide beats mean-ranking on {V['E'][0]}/4 -> {'PASS' if V['E'][0] >= 3 else 'FAIL'} | "
    f"G2 exception-aware mean-ranking beats BWT-ranking on {V['E'][1]}/4 -> {'PASS' if V['E'][1] >= 3 else 'FAIL'}")
say(f"(secondary target: G1 {V['BAL'][0]}/4, G2 {V['BAL'][1]}/4)")
open(os.path.join(ROOT, "paper", "results_guide.txt"), "w").write("\n".join(lines) + "\n")
