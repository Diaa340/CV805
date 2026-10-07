"""Same-answer (answer-stratified) exception metrics and the class-balance buffer control (paper/preregistration_strat.md).
Part A re-analyses saved results (two-task study: tables, CLEVR, GQA; CLOVE six-task small models; CLOVE VLM) with
  dBWT_s = sum_y pi_y [BWT(exceptions, answer y) - BWT(rule, answer y)],   pi_y = share of exception cells with answer y
and dFWT_s the same with forward transfer. Part B compares DER++ buffers: random, rule+exceptions, class-balanced
random, class-balanced rule+exceptions. 95% intervals: bootstrap over units. Writes paper/results_strat.txt,
paper/figs/strat_metrics.csv, paper/figs/strat_buffers.csv"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []; NB = 1000
def say(s=""): print(s, flush=True); lines.append(s)


def pick_matched(D1, D2, grids, own1="before", unitcol="unit", late=None):
    """Matched plasticity (as in the original analyses): per unit / run, best task-1 (or final) accuracy among settings
    within 0.02 of finetune on task-2 (or own-task) accuracy; else the first setting."""
    a2 = late.groupby(unitcol).mean(numeric_only=True) if late is not None else D2.groupby(unitcol).mean(numeric_only=True)
    a1 = D1.groupby(unitcol).mean(numeric_only=True)
    for fam, lams, c2, c1 in grids:
        pick = {u: (max(ok, key=lambda l: a1.loc[u, c1(fam, l)]) if (ok := [l for l in lams if a2.loc[u, c2(fam, l)] >= a2.loc[u, c2("finetune", None)] - 0.02]) else lams[0]) for u in a2.index}
        yield fam, pick


class Strat:
    """Per-unit sums by (type, answer) for pairs of columns; bootstrap over units."""
    def __init__(self, X, pairs, seed=0):
        self.units = sorted(X.unit.unique()); k = {u: i for i, u in enumerate(self.units)}; self.pairs = pairs; cols = sorted({c for p in pairs.values() for c in p})
        self.ci = {c: i for i, c in enumerate(cols)}; nu = len(self.units); self.S = np.zeros((nu, 2, 2, len(cols))); self.N = np.zeros((nu, 2, 2))
        for (u, g, y), x in X.groupby(["unit", "nonadd", "y"]):
            self.S[k[u], g, y] = x[cols].astype(float).sum().values; self.N[k[u], g, y] = len(x)
        self.rng = np.random.default_rng(seed)

    def stats(self, w):
        S = np.einsum("u,ugyc->gyc", w, self.S); N = np.einsum("u,ugy->gy", w, self.N); out = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            A = S / N[..., None]; pi = N[1] / N[1].sum()
            Ag = S.sum(1) / N.sum(1)[:, None]; At = S.sum((0, 1)) / N.sum()
        for name, (after, ref) in self.pairs.items():
            d = A[..., self.ci[after]] - A[..., self.ci[ref]]   # [type, answer]
            ok = ~np.isnan(d[0]) & ~np.isnan(d[1]) & (N[1] > 0); p = pi * ok; p = p / p.sum()
            out[name] = dict(R=Ag[0, self.ci[after]] - Ag[0, self.ci[ref]], E=Ag[1, self.ci[after]] - Ag[1, self.ci[ref]],
                             d_raw=(Ag[1, self.ci[after]] - Ag[1, self.ci[ref]]) - (Ag[0, self.ci[after]] - Ag[0, self.ci[ref]]),
                             d_strat=float(np.nansum(p * (d[1] - d[0]))), all=At[self.ci[after]] - At[self.ci[ref]],
                             R_yes=d[0, 1], R_no=d[0, 0], E_yes=d[1, 1], E_no=d[1, 0])
        return out

    def run(self):
        nu = len(self.units); self.P = self.stats(np.ones(nu))
        self.B = [self.stats(np.bincount(self.rng.integers(0, nu, nu), minlength=nu).astype(float)) for _ in range(NB)]; return self

    def ci_(self, f):
        return np.nanpercentile([f(b) for b in self.B], [2.5, 97.5])


# ------------------------------------------------------------------------------------------------ Part A: saved results
TWO = ["finetune", "ewc", "si", "lwf", "l2", "er_random_1000", "agem_random_1000", "gem_random_1000", "derpp_random_1000", "icarl", "der_expand_1000",
       "multihead", "derpp_ruleexc_1000"]
G2 = [("ewc", ("1", "100", "10000")), ("si", ("0.1", "1", "10", "100")), ("lwf", ("0.3", "3")), ("l2", ("0.3", "3")), ("icarl", ("0.1", "1"))]
rowsA, res = [], {}
say("PART A: same-answer gaps. dBWT_s / dFWT_s = exception minus rule, compared within the same answer; raw = not stratified.")
for b in ("tables", "clevr", "gqa"):
    Y = np.load(os.path.join(ROOT, "runs", "clm", f"y_{b}.npz"))
    D1 = pd.concat([pd.read_csv(f) for f in Y["task1_files"]]).reset_index(drop=True); D2 = pd.concat([pd.read_csv(f) for f in Y["task2_files"]]).reset_index(drop=True)
    D1["y"], D2["y"] = Y["task1"], Y["task2"]
    for fam, pick in pick_matched(D1, D2, [(f, l, lambda f, l: f if l is None else f"{f}_{l}", lambda f, l: f"{f}_{l}") for f, l in G2]):
        for D in (D1, D2):
            D[fam] = 0.0
            for l in dict(G2)[fam]:
                m = D.unit.map(pick) == l; D.loc[m, fam] = D.loc[m, f"{fam}_{l}"].astype(float)
    res[b] = (Strat(D1, {m: (m, "before") for m in TWO}).run(), Strat(D2, {m: (m, "scratch") for m in TWO}).run(), TWO)
C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "clove_cl", "run_[0-9]*.csv.gz")))]); C["unit_run"] = C.unit
SEQ = ["finetune", "ewc", "si", "lwf", "l2", "er", "agem", "gem", "derpp", "icarl", "der_expand", "multihead", "derpp_ruleexc"]
for fam, lams in [("ewc", ("1", "100", "10000")), ("si", ("0.1", "1", "10", "100")), ("lwf", ("0.3", "3")), ("l2", ("0.3", "3")), ("icarl", ("0.1", "1"))]:
    late = C[C.pos >= 2].groupby("run").mean(numeric_only=True); allr = C.groupby("run").mean(numeric_only=True); pick = {}
    for r in late.index:
        ok = [l for l in lams if late.loc[r, f"{fam}_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]; pick[r] = max(ok, key=lambda l: allr.loc[r, f"{fam}_{l}__final"]) if ok else lams[0]
    for s in ("own", "final"):
        C[f"{fam}__{s}"] = 0.0
        for l in lams:
            m = C.run.map(pick) == l; C.loc[m, f"{fam}__{s}"] = C.loc[m, f"{fam}_{l}__{s}"]
res["clove"] = (Strat(C[C.pos <= 5], {m: (f"{m}__final", f"{m}__own") for m in SEQ}).run(), Strat(C[C.pos >= 2], {m: (f"{m}__own", "scratch") for m in SEQ}).run(), SEQ)
V = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "clove_vlm", "order_[0-9].csv.gz")))])
late = V[V.pos >= 2].groupby("order").mean(numeric_only=True); allr = V.groupby("order").mean(numeric_only=True); pick = {}
for r in late.index:
    ok = [l for l in ("100", "10000") if late.loc[r, f"ewc_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]; pick[r] = max(ok, key=lambda l: allr.loc[r, f"ewc_{l}__final"]) if ok else "100"
for s in ("own", "final"):
    V[f"ewc__{s}"] = 0.0
    for l in ("100", "10000"):
        m = V.order.map(pick) == l; V.loc[m, f"ewc__{s}"] = V.loc[m, f"ewc_{l}__{s}"]
VM = ["finetune", "ewc", "lwf_1", "er", "derpp", "derpp_ruleexc"]
res["clove_vlm"] = (Strat(V[V.pos <= 5], {m: (f"{m}__final", f"{m}__own") for m in VM}).run(), Strat(V[V.pos >= 2], {m: (f"{m}__own", "scratch") for m in VM}).run(), VM)
for b, (SB, SF, ms) in res.items():
    say(f"\n== {b} ==   method: dBWT raw | dBWT_s [95%] | BWT exceptions yes / no, rule yes / no | dFWT raw | dFWT_s [95%]")
    for m in ms:
        p, f = SB.P[m], SF.P[m]; a = SB.ci_(lambda x: x[m]["d_strat"]); c = SF.ci_(lambda x: x[m]["d_strat"])
        rowsA.append(dict(bench=b, method=m, dBWT_raw=p["d_raw"], dBWT_s=p["d_strat"], dBWT_s_lo=a[0], dBWT_s_hi=a[1], BWT_E_yes=p["E_yes"], BWT_E_no=p["E_no"],
                          BWT_R_yes=p["R_yes"], BWT_R_no=p["R_no"], dFWT_raw=f["d_raw"], dFWT_s=f["d_strat"], dFWT_s_lo=c[0], dFWT_s_hi=c[1]))
        say(f"  {m:20s} {p['d_raw']:+.3f} | {p['d_strat']:+.3f} [{a[0]:+.3f},{a[1]:+.3f}] | E {p['E_yes']:+.3f}/{p['E_no']:+.3f}  R {p['R_yes']:+.3f}/{p['R_no']:+.3f} | "
            f"{f['d_raw']:+.3f} | {f['d_strat']:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]")
A = pd.DataFrame(rowsA); A.round(4).to_csv(os.path.join(ROOT, "paper", "figs", "strat_metrics.csv"), index=False)
key = lambda b, m: A[(A.bench == b) & (A.method == m)].iloc[0]
sa1 = all(key("clevr", m).dBWT_s <= -0.03 and key("clevr", m).dBWT_s_hi < 0 for m in ("er_random_1000", "derpp_random_1000"))
dp = {"tables": "derpp_random_1000", "clevr": "derpp_random_1000", "gqa": "derpp_random_1000", "clove": "derpp"}
sa2 = sum(bool(key(b, m).dFWT_s <= -0.03 and key(b, m).dFWT_s_hi < 0) for b, m in dp.items())
say(f"\nSA1 (CLEVR, ER and DER++: dBWT_s <= -0.03, upper < 0): {'PASS' if sa1 else 'FAIL'}")
say(f"SA2 (DER++ dFWT_s <= -0.03, upper < 0, on >= 2 of tables / CLEVR / GQA / CLOVE): {sa2}/4 -> {'PASS' if sa2 >= 2 else 'FAIL'}")
say("Methods with dBWT_s upper bound < 0 (exceptions lost more within the same answer): " + "; ".join(
    f"{b}: " + ", ".join(A[(A.bench == b) & (A.dBWT_s_hi < 0)].method) for b in res))

# ------------------------------------------------------------------------------------------------ Part B: buffer control
say("\nPART B: DER++ buffers (memory 1000). Exception BWT, overall BWT, and BWT by answer for exception / rule cells.")
ARMS = ["random", "ruleexc", "classbal", "ruleexc_bal"]; rowsB = []; vb = {}
for b in ("tables", "clevr", "gqa", "clove"):
    if b == "clove":
        fs = sorted(glob.glob(os.path.join(ROOT, "runs", "clove_cl", "run_bal_*.csv.gz")))
        if not fs:
            continue
        X = pd.concat([pd.read_csv(f) for f in fs]); X = X[X.pos <= 5]; nm = {"random": "derpp", "ruleexc": "derpp_ruleexc", "classbal": "derpp_classbal", "ruleexc_bal": "derpp_ruleexc_bal"}
        S = Strat(X, {a: (f"{nm[a]}__final", f"{nm[a]}__own") for a in ARMS} | {f"{a}-vs-classbal": (f"{nm[a]}__final", "derpp_classbal__final") for a in ARMS}
                  | {"ruleexc_bal-vs-random": (f"{nm['ruleexc_bal']}__final", "derpp__final")}).run()
        # differences of BWT between arms = differences of final accuracy minus differences of own accuracy; own differs by arm only through earlier tasks' training
        S2 = Strat(X, {"own_rb_minus_cb": ("derpp_ruleexc_bal__own", "derpp_classbal__own"), "own_rb_minus_r": ("derpp_ruleexc_bal__own", "derpp__own")}).run()
        diffE = (S.P["ruleexc_bal-vs-classbal"]["E"] - S2.P["own_rb_minus_cb"]["E"], np.nanpercentile([S.B[i]["ruleexc_bal-vs-classbal"]["E"] - S2.B[i]["own_rb_minus_cb"]["E"] for i in range(NB)], [2.5, 97.5]))
        diffA = (S.P["ruleexc_bal-vs-random"]["all"] - S2.P["own_rb_minus_r"]["all"], np.nanpercentile([S.B[i]["ruleexc_bal-vs-random"]["all"] - S2.B[i]["own_rb_minus_r"]["all"] for i in range(NB)], [2.5, 97.5]))
    else:
        fs = sorted(glob.glob(os.path.join(ROOT, "runs", "clbal", f"task1_{b}_*.csv.gz")))
        if not fs:
            continue
        X = pd.concat([pd.read_csv(f) for f in fs])
        S = Strat(X, {a: (a, "before") for a in ARMS} | {"ruleexc_bal-vs-classbal": ("ruleexc_bal", "classbal"), "ruleexc_bal-vs-random": ("ruleexc_bal", "random")}).run()
        diffE = (S.P["ruleexc_bal-vs-classbal"]["E"], S.ci_(lambda x: x["ruleexc_bal-vs-classbal"]["E"]))
        diffA = (S.P["ruleexc_bal-vs-random"]["all"], S.ci_(lambda x: x["ruleexc_bal-vs-random"]["all"]))
    say(f"\n== {b} ==   arm: exception BWT | overall BWT | exceptions yes / no | rule yes / no | dBWT_s [95%]")
    for a in ARMS:
        p = S.P[a]; c = S.ci_(lambda x: x[a]["d_strat"]); rowsB.append(dict(bench=b, arm=a, BWT_E=p["E"], BWT_all=p["all"], dBWT_s=p["d_strat"], lo=c[0], hi=c[1]))
        say(f"  {a:12s} {p['E']:+.3f} | {p['all']:+.3f} | {p['E_yes']:+.3f}/{p['E_no']:+.3f} | {p['R_yes']:+.3f}/{p['R_no']:+.3f} | {p['d_strat']:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]")
    vb[b] = dict(B1=bool(diffE[0] >= 0.03 and diffE[1][0] > 0), B2=bool(diffA[1][0] > -0.02))
    say(f"  B1 balanced rule+exceptions vs balanced random, exception BWT: {diffE[0]:+.3f} [{diffE[1][0]:+.3f},{diffE[1][1]:+.3f}] -> {vb[b]['B1']}")
    say(f"  B2 balanced rule+exceptions vs random, overall BWT: {diffA[0]:+.3f} [{diffA[1][0]:+.3f},{diffA[1][1]:+.3f}] -> {vb[b]['B2']}")
pd.DataFrame(rowsB).round(4).to_csv(os.path.join(ROOT, "paper", "figs", "strat_buffers.csv"), index=False)
n1, n2 = sum(v["B1"] for v in vb.values()), sum(v["B2"] for v in vb.values())
say(f"\nVERDICTS: SA1 {'PASS' if sa1 else 'FAIL'} | SA2 {'PASS' if sa2 >= 2 else 'FAIL'} ({sa2}/4) | B1 {'PASS' if n1 >= 3 else 'FAIL'} ({n1}/{len(vb)}) | "
    f"B2 {'PASS' if n2 >= 3 else 'FAIL'} ({n2}/{len(vb)})")
open(os.path.join(ROOT, "paper", "results_strat.txt"), "w").write("\n".join(lines) + "\n")
