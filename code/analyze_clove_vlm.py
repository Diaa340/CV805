"""Pre-registered analysis of the VLM six-task CLOVE-scene study (paper/preregistration_clove_vlm.md).
Writes paper/results_clove_vlm.txt and figures/clove_vlm_metrics.csv"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []; rng0 = np.random.default_rng(0); NB = 1000
def say(s=""): print(s); lines.append(s)
METH = ["finetune", "ewc", "lwf_1", "er", "derpp", "derpp_ruleexc"]
D = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "clove_vlm", "order_[0-9].csv.gz")))])
late = D[D.pos >= 2].groupby("order").mean(numeric_only=True); allr = D.groupby("order").mean(numeric_only=True); pick = {}
for r in late.index:   # matched plasticity for EWC, as in the small-model study
    ok = [l for l in ("100", "10000") if late.loc[r, f"ewc_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]
    pick[r] = max(ok, key=lambda l: allr.loc[r, f"ewc_{l}__final"]) if ok else "100"
for s in ("own", "final"):
    D[f"ewc__{s}"] = 0.0
    for l in ("100", "10000"):
        m = D.order.map(pick) == l; D.loc[m, f"ewc__{s}"] = D.loc[m, f"ewc_{l}__{s}"]
units = sorted(D.unit.unique()); D["u"] = D.unit.map({u: k for k, u in enumerate(units)}); nu = len(units)
cols = [f"{m}__{s}" for m in METH for s in ("own", "final")] + ["scratch", "zeroshot"]; ci_ = {c: k for k, c in enumerate(cols)}


def sums(mask):
    X = D[mask]; S = np.zeros((nu, 2, len(cols))); N = np.zeros((nu, 2))
    for (u, g), x in X.groupby(["u", "nonadd"]):
        S[u, g] = x[cols].astype(float).sum().values; N[u, g] = len(x)
    return S, N


B5, N5 = sums(D.pos <= 5); B6, N6 = sums(D.pos >= 2); BA, NA = sums(D.pos >= 1)


def metrics(w):
    agg = lambda S, N: np.einsum("u,ugc->gc", w, S) / np.einsum("u,ug->g", w, N)[:, None]; a5, a6, aa = agg(B5, N5), agg(B6, N6), agg(BA, NA); out = {}
    for m in METH:
        r = {}
        for g, nm in ((0, "R"), (1, "E")):
            r[f"BWT_{nm}"] = a5[g, ci_[f"{m}__final"]] - a5[g, ci_[f"{m}__own"]]; r[f"FWT_{nm}"] = a6[g, ci_[f"{m}__own"]] - a6[g, ci_["scratch"]]
            r[f"ACC_{nm}"] = aa[g, ci_[f"{m}__final"]]; r[f"ZS_{nm}"] = aa[g, ci_["zeroshot"]]
        r["dBWT"] = r["BWT_E"] - r["BWT_R"]; r["dFWT"] = r["FWT_E"] - r["FWT_R"]
        tot = np.einsum("u,ugc->c", w, B5) / np.einsum("u,ug->", w, N5); r["BWT_all"] = tot[ci_[f"{m}__final"]] - tot[ci_[f"{m}__own"]]; out[m] = r
    return out


P = metrics(np.ones(nu)); BS = [metrics(np.bincount(rng0.integers(0, nu, nu), minlength=nu).astype(float)) for _ in range(NB)]
ci = lambda f: np.nanpercentile([f(x) for x in BS], [2.5, 97.5])
say(f"=== CLOVE-scene with Qwen2.5-VL-3B + LoRA: {D.order.nunique()} task orders, {nu} units; test cells per order {len(D) // D.order.nunique()} "
    f"({D.nonadd.mean():.3f} exceptions) ===")
say(f"zero-shot accuracy: rule {P['finetune']['ZS_R']:.3f}, exceptions {P['finetune']['ZS_E']:.3f}; EWC strength picked: {pick}")
say("  method          BWT_R   BWT_E   dBWT [95%]              | FWT_R   FWT_E   dFWT [95%]              | ACC_R ACC_E")
rows = []
for m in METH:
    r = dict(P[m], method=m); a = ci(lambda x: x[m]["dBWT"]); b = ci(lambda x: x[m]["dFWT"]); r.update(dBWT_lo=a[0], dBWT_hi=a[1], dFWT_lo=b[0], dFWT_hi=b[1]); rows.append(r)
    say(f"  {m:14s} {r['BWT_R']:+.3f} {r['BWT_E']:+.3f}  {r['dBWT']:+.3f} [{a[0]:+.3f},{a[1]:+.3f}] | {r['FWT_R']:+.3f} {r['FWT_E']:+.3f}  "
        f"{r['dFWT']:+.3f} [{b[0]:+.3f},{b[1]:+.3f}] | {r['ACC_R']:.3f} {r['ACC_E']:.3f}")
R = pd.DataFrame(rows).set_index("method"); R.round(4).to_csv(os.path.join(ROOT, "figures", "clove_vlm_metrics.csv"))
v1 = sum(bool(R.loc[m, "dBWT"] <= -0.03 and R.loc[m, "dBWT_hi"] < 0) for m in ("lwf_1", "er", "derpp")); v2 = bool(R.loc["derpp", "dFWT"] <= -0.03 and R.loc["derpp", "dFWT_hi"] < 0)
g = ci(lambda x: x["derpp_ruleexc"]["BWT_E"] - x["derpp"]["BWT_E"]); h = ci(lambda x: x["derpp_ruleexc"]["BWT_all"] - x["derpp"]["BWT_all"])
gd = P["derpp_ruleexc"]["BWT_E"] - P["derpp"]["BWT_E"]; v3 = bool(gd >= 0.03 and g[0] > 0 and h[0] > -0.02)
say(f"\nV1 exceptions forgotten more (LwF, ER, DER++): {v1}/3 -> {'PASS' if v1 >= 2 else 'FAIL'}")
say(f"V2 DER++ hinders new exceptions more: dFWT {R.loc['derpp', 'dFWT']:+.3f} [{R.loc['derpp', 'dFWT_lo']:+.3f},{R.loc['derpp', 'dFWT_hi']:+.3f}] -> {'PASS' if v2 else 'FAIL'}")
say(f"V3 rule+exceptions buffer: exception BWT {gd:+.3f} [{g[0]:+.3f},{g[1]:+.3f}], overall BWT {P['derpp_ruleexc']['BWT_all'] - P['derpp']['BWT_all']:+.3f} "
    f"[{h[0]:+.3f},{h[1]:+.3f}] -> {'PASS' if v3 else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_clove_vlm.txt"), "w").write("\n".join(lines) + "\n")
