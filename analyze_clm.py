"""Pre-registered analysis of CL metrics by sample type (paper/preregistration_clm.md). Writes paper/results_clm.txt and
paper/figs/clm_metrics.csv"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clm"); lines = []
def say(s=""): print(s); lines.append(s)
rng0 = np.random.default_rng(0); PROT = ("lwf", "er_random_1000", "derpp_random_1000"); NB = 1000
FAM = [("none", ["finetune"]), ("regularisation: parameter", ["ewc", "si"]), ("regularisation: distillation", ["lwf", "l2"]),
       ("memory", ["er_random_1000", "agem_random_1000", "gem_random_1000", "derpp_random_1000", "icarl"]),
       ("architectural", ["der_expand_1000", "multihead"]), ("our fix", ["derpp_ruleexc_1000"])]
METH = tuple(m for _, ms in FAM for m in ms); fam_of = {m: f for f, ms in FAM for m in ms}
GRIDS = (("ewc", ("1", "100", "10000")), ("si", ("0.1", "1", "10", "100")), ("lwf", ("0.3", "3")), ("l2", ("0.3", "3")), ("icarl", ("0.1", "1")))


def load(b):
    D1 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, f"task1_{b}_*.csv.gz")))])
    D2 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, f"task2_{b}_*.csv.gz")))])
    a2 = D2.groupby("unit").mean(numeric_only=True); a1 = D1.groupby("unit").mean(numeric_only=True)
    for fam, lams in GRIDS:   # matched plasticity: best task-1 accuracy among settings within 0.02 of finetune on task 2
        pick = {u: (max(ok, key=lambda l: a1.loc[u, f"{fam}_{l}"]) if (ok := [l for l in lams if a2.loc[u, f"{fam}_{l}"] >= a2.loc[u, "finetune"] - 0.02]) else lams[0]) for u in a2.index}
        for D in (D1, D2):
            col = D.unit.map(pick); D[fam] = 0.0
            for l in lams:
                D.loc[col == l, fam] = D.loc[col == l, f"{fam}_{l}"].astype(float)
    return D1, D2


def agg(D, cols):   # per unit and type: number of samples and number correct for each column -> arrays [unit, type, col]
    us = sorted(D.unit.unique()); G = D.groupby(["unit", "nonadd"])[cols].agg(["sum", "count"])
    S = np.zeros((len(us), 2, len(cols))); N = np.zeros((len(us), 2))
    for (u, g), r in G.iterrows():
        S[us.index(u), int(g)] = [r[(c, "sum")] for c in cols]; N[us.index(u), int(g)] = r[(cols[0], "count")]
    return S, N


def metrics(S1, N1, S2, N2, w, c1, c2):
    """w: unit weights (1 = point estimate, bootstrap counts otherwise). c1/c2: column index maps."""
    acc1 = np.einsum("u,ugc->gc", w, S1) / np.einsum("u,ug->g", w, N1)[:, None]; acc2 = np.einsum("u,ugc->gc", w, S2) / np.einsum("u,ug->g", w, N2)[:, None]
    all1 = np.einsum("u,ugc->c", w, S1) / np.einsum("u,ug->", w, N1); all2 = np.einsum("u,ugc->c", w, S2) / np.einsum("u,ug->", w, N2)
    out = {}
    for m in METH:
        r = {}; b0 = acc1[:, c1["before"]]; a = acc1[:, c1[m]]
        for g, nm in ((0, "R"), (1, "E")):
            r[f"acc1_before_{nm}"], r[f"acc1_after_{nm}"], r[f"BWT_{nm}"] = b0[g], a[g], a[g] - b0[g]
            r[f"KS_{nm}"] = (a[g] - 0.5) / (b0[g] - 0.5) if b0[g] >= 0.55 else np.nan
            r[f"acc2_{nm}"] = acc2[g, c2[m]]; r[f"FWT_{nm}"] = acc2[g, c2[m]] - acc2[g, c2["scratch"]]
        r.update(KS_gap=r["KS_E"] - r["KS_R"], FWT_gap=r["FWT_E"] - r["FWT_R"], BWT_all=all1[c1[m]] - all1[c1["before"]], acc2_all=all2[c2[m]],
                 ACC_bal=np.mean([0.5 * (a[0] + a[1]), 0.5 * (acc2[0, c2[m]] + acc2[1, c2[m]])]))
        out[m] = r
    return out


rows, V = [], {}
for b in ("tables", "clevr", "gqa", "pope"):
    if not glob.glob(os.path.join(OUT, f"task1_{b}_*.csv.gz")):
        continue
    D1, D2 = load(b); C1, C2 = ["before", *METH], [*METH, "scratch"]; S1, N1 = agg(D1, C1); S2, N2 = agg(D2, C2)
    c1, c2 = {c: i for i, c in enumerate(C1)}, {c: i for i, c in enumerate(C2)}; nu = len(N1)
    P = metrics(S1, N1, S2, N2, np.ones(nu), c1, c2); W = [np.bincount(rng0.integers(0, nu, nu), minlength=nu).astype(float) for _ in range(NB)]
    B = [metrics(S1, N1, S2, N2, w, c1, c2) for w in W]
    ci = lambda f: np.nanpercentile([f(x) for x in B], [2.5, 97.5])
    say(f"\n=== {b}: {nu} runs; task-1 test {int(N1.sum())} ({int(N1[:, 1].sum())} exceptions), task-2 test {int(N2.sum())} ({int(N2[:, 1].sum())} exceptions) ===")
    say(f"  task-1 accuracy before task 2: rule {P['finetune']['acc1_before_R']:.3f}, exceptions {P['finetune']['acc1_before_E']:.3f}; "
        f"task-2-only model: rule {S2[:, 0, c2['scratch']].sum() / N2[:, 0].sum():.3f}, exceptions {S2[:, 1, c2['scratch']].sum() / N2[:, 1].sum():.3f}")
    say("  family                         method              BWT_R   BWT_E | KS_R  KS_E  KS gap [95%]           | FWT_R  FWT_E  FWT gap [95%]          | ACC_bal")
    m1 = m2 = 0
    for m in METH:
        r = dict(P[m], bench=b, method=m, family=fam_of[m]); kc = ci(lambda x: x[m]["KS_gap"]); fc = ci(lambda x: x[m]["FWT_gap"])
        r.update(KS_gap_lo=kc[0], KS_gap_hi=kc[1], FWT_gap_lo=fc[0], FWT_gap_hi=fc[1], holds=bool(r["KS_R"] >= 0.5 and kc[0] > -0.05)); rows.append(r)
        if m in PROT:
            m1 += bool(r["KS_gap"] <= -0.05 and kc[1] < 0); m2 += bool(r["FWT_gap"] <= -0.03 and fc[1] < 0)
        say(f"  {fam_of[m]:30s} {m:18s} {r['BWT_R']:+.3f} {r['BWT_E']:+.3f} | {r['KS_R']:.2f}  {r['KS_E']:.2f}  {r['KS_gap']:+.3f} [{kc[0]:+.3f},{kc[1]:+.3f}] | "
            f"{r['FWT_R']:+.3f} {r['FWT_E']:+.3f} {r['FWT_gap']:+.3f} [{fc[0]:+.3f},{fc[1]:+.3f}] | {r['ACC_bal']:.3f}")
    V[b] = dict(M1=m1 >= 2, M2=m2 >= 2)
    x, y = P["derpp_ruleexc_1000"], P["derpp_random_1000"]
    dk = ci(lambda z: z["derpp_ruleexc_1000"]["KS_E"] - z["derpp_random_1000"]["KS_E"]); db = ci(lambda z: z["derpp_ruleexc_1000"]["BWT_all"] - z["derpp_random_1000"]["BWT_all"])
    V[b]["M3"] = bool(x["KS_E"] - y["KS_E"] >= 0.05 and dk[0] > 0 and db[0] > -0.02)
    say(f"  M3 DER++ rule+exceptions vs random: KS_E {x['KS_E'] - y['KS_E']:+.3f} [{dk[0]:+.3f},{dk[1]:+.3f}], overall BWT {x['BWT_all'] - y['BWT_all']:+.3f} "
        f"[{db[0]:+.3f},{db[1]:+.3f}], task-2 acc {x['acc2_all'] - y['acc2_all']:+.3f}")
    say(f"  -> LwF / ER / DER++ with KS gap: {m1}/3, with FWT gap: {m2}/3")
R = pd.DataFrame(rows)
say("\nM4 (exploratory, criterion fixed in advance): keeps exceptions as well as the rule = KS_R >= 0.5 and KS gap lower bound > -0.05")
for m in METH:
    x = R[R.method == m]; say(f"  {fam_of[m]:30s} {m:18s} " + " ".join(f"{b}:{'Y' if h else '.'}" for b, h in zip(x.bench, x.holds)) + f"  -> {int(x.holds.sum())}/{len(x)}")
say("\nBy family (mean over the family's methods): KS_R, KS_E, KS gap, FWT gap")
for b in R.bench.unique():
    for f, _ in FAM:
        x = R[(R.bench == b) & (R.family == f)]; say(f"  {b:7s} {f:30s} {x.KS_R.mean():.2f} {x.KS_E.mean():.2f} {x.KS_gap.mean():+.3f} {x.FWT_gap.mean():+.3f}")
R.round(4).to_csv(os.path.join(ROOT, "paper", "figs", "clm_metrics.csv"), index=False)
nb = lambda k: sum(bool(v[k]) for v in V.values())
say(f"\nVERDICTS: M1 {'PASS' if nb('M1') >= 3 else 'FAIL'} ({ {b: bool(v['M1']) for b, v in V.items()} }) | "
    f"M2 {'PASS' if nb('M2') >= 3 else 'FAIL'} ({ {b: bool(v['M2']) for b, v in V.items()} }) | "
    f"M3 (GQA, POPE) {'PASS' if V.get('gqa', {}).get('M3') and V.get('pope', {}).get('M3') else 'FAIL'} ({ {b: bool(v['M3']) for b, v in V.items()} })")
open(os.path.join(ROOT, "paper", "results_clm.txt"), "w").write("\n".join(lines) + "\n")
