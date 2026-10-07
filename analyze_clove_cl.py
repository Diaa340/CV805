"""Pre-registered analysis of the six-task CLOVE-scene study (paper/preregistration_clove_cl.md).
Writes paper/results_clove_cl.txt and paper/figs/clove_cl_metrics.csv"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []; rng0 = np.random.default_rng(0); NB = 1000
def say(s=""): print(s); lines.append(s)
FAM = [("none", ["finetune"]), ("regularisation: parameter", ["ewc", "si"]), ("regularisation: distillation", ["lwf", "l2"]),
       ("memory", ["er", "agem", "gem", "derpp", "icarl"]), ("architectural", ["der_expand", "multihead"]), ("our fix", ["derpp_ruleexc"])]
METH = [m for _, ms in FAM for m in ms]; fam_of = {m: f for f, ms in FAM for m in ms}
GRIDS = {"ewc": ("1", "100", "10000"), "si": ("0.1", "1", "10", "100"), "lwf": ("0.3", "3"), "l2": ("0.3", "3"), "icarl": ("0.1", "1")}
D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "clove_cl", "run_*.csv.gz")))])
for fam, lams in GRIDS.items():   # matched plasticity per run: best final accuracy among settings whose own-task accuracy on tasks 2-6 is within 0.02 of finetune
    late = D[D.pos >= 2].groupby("run").mean(numeric_only=True); allr = D.groupby("run").mean(numeric_only=True); pick = {}
    for r in late.index:
        ok = [l for l in lams if late.loc[r, f"{fam}_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]
        pick[r] = max(ok, key=lambda l: allr.loc[r, f"{fam}_{l}__final"]) if ok else lams[0]
    for s in ("own", "final"):
        D[f"{fam}__{s}"] = 0.0
        for l in lams:
            m = D.run.map(pick) == l; D.loc[m, f"{fam}__{s}"] = D.loc[m, f"{fam}_{l}__{s}"]
units = sorted(D.unit.unique()); D["u"] = D.unit.map({u: k for k, u in enumerate(units)}); nu = len(units)
cols = [f"{m}__{s}" for m in METH for s in ("own", "final")] + ["scratch"]


def sums(mask):   # per unit and type: sums of each column and counts -> [unit, type, col], [unit, type]
    X = D[mask]; S = np.zeros((nu, 2, len(cols))); N = np.zeros((nu, 2))
    for (u, g), x in X.groupby(["u", "nonadd"]):
        S[u, g] = x[cols].sum().values; N[u, g] = len(x)
    return S, N


B5, N5 = sums(D.pos <= 5); B6, N6 = sums(D.pos >= 2); BA, NA = sums(D.pos >= 1); ci_ = {c: k for k, c in enumerate(cols)}


def metrics(w):
    agg = lambda S, N: np.einsum("u,ugc->gc", w, S) / np.einsum("u,ug->g", w, N)[:, None]
    a5, a6, aa = agg(B5, N5), agg(B6, N6), agg(BA, NA); out = {}
    for m in METH:
        r = {}
        for g, nm in ((0, "R"), (1, "E")):
            r[f"BWT_{nm}"] = a5[g, ci_[f"{m}__final"]] - a5[g, ci_[f"{m}__own"]]; r[f"FWT_{nm}"] = a6[g, ci_[f"{m}__own"]] - a6[g, ci_["scratch"]]
            r[f"ACC_{nm}"] = aa[g, ci_[f"{m}__final"]]
        r["dBWT"] = r["BWT_E"] - r["BWT_R"]; r["dFWT"] = r["FWT_E"] - r["FWT_R"]
        tot5 = np.einsum("u,ugc->c", w, B5) / np.einsum("u,ug->", w, N5); r["BWT_all"] = tot5[ci_[f"{m}__final"]] - tot5[ci_[f"{m}__own"]]
        r["ACC_bal"] = 0.5 * (r["ACC_R"] + r["ACC_E"]); out[m] = r
    return out


P = metrics(np.ones(nu)); W = [np.bincount(rng0.integers(0, nu, nu), minlength=nu).astype(float) for _ in range(NB)]; BS = [metrics(w) for w in W]
ci = lambda f: np.nanpercentile([f(x) for x in BS], [2.5, 97.5])
say(f"=== CLOVE-scene, six tasks: {D.run.nunique()} runs (orders x seeds), {nu} units; test samples per run {len(D) // D.run.nunique()} "
    f"({D.nonadd.mean():.3f} exceptions, {D.y.mean():.3f} yes) ===")
say("BWT over tasks 1-5 (final minus right after the task); FWT over tasks 2-6 (right after the task minus a task-only model)")
say("  family                         method          BWT_R   BWT_E   dBWT [95%]              | FWT_R   FWT_E   dFWT [95%]              | ACC_R ACC_E")
rows = []
for m in METH:
    r = dict(P[m], method=m, family=fam_of[m]); a = ci(lambda x: x[m]["dBWT"]); b = ci(lambda x: x[m]["dFWT"])
    r.update(dBWT_lo=a[0], dBWT_hi=a[1], dFWT_lo=b[0], dFWT_hi=b[1]); rows.append(r)
    say(f"  {fam_of[m]:30s} {m:14s} {r['BWT_R']:+.3f} {r['BWT_E']:+.3f}  {r['dBWT']:+.3f} [{a[0]:+.3f},{a[1]:+.3f}] | {r['FWT_R']:+.3f} {r['FWT_E']:+.3f}  "
        f"{r['dFWT']:+.3f} [{b[0]:+.3f},{b[1]:+.3f}] | {r['ACC_R']:.3f} {r['ACC_E']:.3f}")
R = pd.DataFrame(rows); R.round(4).to_csv(os.path.join(ROOT, "paper", "figs", "clove_cl_metrics.csv"), index=False); Rm = R.set_index("method")
c1 = sum(bool(Rm.loc[m, "dBWT"] <= -0.03 and Rm.loc[m, "dBWT_hi"] < 0) for m in ("lwf", "er", "derpp"))
c2 = bool(Rm.loc["derpp", "dFWT"] <= -0.03 and Rm.loc["derpp", "dFWT_hi"] < 0)
g = ci(lambda x: x["derpp_ruleexc"]["BWT_E"] - x["derpp"]["BWT_E"]); h = ci(lambda x: x["derpp_ruleexc"]["BWT_all"] - x["derpp"]["BWT_all"])
gd = P["derpp_ruleexc"]["BWT_E"] - P["derpp"]["BWT_E"]; c3 = bool(gd >= 0.03 and g[0] > 0 and h[0] > -0.02)
k = ci(lambda x: x["der_expand"]["dBWT"] - x["er"]["dBWT"]); kd = P["der_expand"]["dBWT"] - P["er"]["dBWT"]; c4 = bool(k[0] > 0)
say(f"\nC1 exceptions forgotten more (LwF, ER, DER++ with dBWT <= -0.03, upper < 0): {c1}/3 -> {'PASS' if c1 >= 2 else 'FAIL'}")
say(f"C2 DER++ hinders new exceptions more (dFWT {Rm.loc['derpp', 'dFWT']:+.3f} [{Rm.loc['derpp', 'dFWT_lo']:+.3f},{Rm.loc['derpp', 'dFWT_hi']:+.3f}]) -> {'PASS' if c2 else 'FAIL'}")
say(f"C3 rule+exceptions buffer: exception BWT {gd:+.3f} [{g[0]:+.3f},{g[1]:+.3f}], overall BWT {P['derpp_ruleexc']['BWT_all'] - P['derpp']['BWT_all']:+.3f} "
    f"[{h[0]:+.3f},{h[1]:+.3f}] -> {'PASS' if c3 else 'FAIL'}")
say(f"C4 DER-expansion loses exceptions less than ER (dBWT difference {kd:+.3f} [{k[0]:+.3f},{k[1]:+.3f}]) -> {'PASS' if c4 else 'FAIL'}")
say("\nBy family (mean over methods): BWT_R, BWT_E, dBWT, dFWT")
for f, ms in FAM:
    x = R[R.family == f]; say(f"  {f:30s} {x.BWT_R.mean():+.3f} {x.BWT_E.mean():+.3f} {x.dBWT.mean():+.3f} {x.dFWT.mean():+.3f}")
open(os.path.join(ROOT, "paper", "results_clove_cl.txt"), "w").write("\n".join(lines) + "\n")
