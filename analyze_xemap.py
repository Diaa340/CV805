"""Analysis of the exact additive projection (paper/preregistration_xemap.md). Writes paper/results_xemap.txt"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []; NB = 2000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
NAMES = {"idefics": "Idefics3-8B", "internvl": "InternVL3.5-8B", "llava": "LLaVA-OV-1.5-8B", "qwen": "Qwen2.5-VL-7B"}
U = pd.read_csv(os.path.join(ROOT, "runs", "vlm3", "selected2.csv")).set_index("cid"); UQ = dict(np.load(os.path.join(ROOT, "runs", "rev", "unique.npz")))
X = pd.read_csv(os.path.join(ROOT, "runs", "rev", "xemap_units.csv")); X["kind"] = X["kind"].fillna("null")   # pandas reads the label "null" as missing
C = pd.read_csv(os.path.join(ROOT, "runs", "rev", "xemap_cells.csv.gz"))
G = np.load(os.path.join(ROOT, "runs", "vlm3", "grids.npz"))
proj = lambda A: A.mean(1, keepdims=True) + A.mean(0, keepdims=True) - A.mean()

# 1. Where do the flips fall? exact projection vs EMAP, by cell status (stable exception / ambiguous / stable rule)
say("1. Share of flipped cells that are stable exceptions / ambiguous / stable rule cells")
lab = {cid: (UQ[f"exc__vlm__{cid}"].ravel().astype(bool), UQ[f"amb__vlm__{cid}"].ravel().astype(bool)) for cid in U.index}
X1 = {}
for m in NAMES:
    S = np.load(os.path.join(ROOT, "runs", "vlm3", f"scores_{m}.npz")); c_m = C[C.model == m]; out = {}
    ef = []
    for cid, cols in zip(U.index, U.cols):
        c = json.loads(cols); L = S[f"L__{U.key[cid]}"][:, c]; ef.append(((L > 0) != (proj(L) > 0)).ravel())
    ef = np.concatenate(ef); exc = np.concatenate([lab[c][0] for c in U.index]); amb = np.concatenate([lab[c][1] for c in U.index])
    order = np.concatenate([np.where(c_m.cid.values == c)[0][np.argsort(c_m.cell.values[c_m.cid.values == c])] for c in U.index])
    for nm, fl in (("EMAP", ef), ("exact", c_m.flip.values[order].astype(bool)), ("exact, confidence-weighted", c_m.flip_w.values[order].astype(bool))):
        out[nm] = (fl.mean(), (fl & exc).sum() / fl.sum(), (fl & amb).sum() / fl.sum(), (fl & ~exc & ~amb).sum() / fl.sum())
    X1[m] = out["exact"][1] + out["exact"][2]
    say(f"  {NAMES[m]:16s} " + " | ".join(f"{k}: flip rate {v[0]:.3f}, exceptions {v[1]:.2f} / ambiguous {v[2]:.2f} / rule {v[3]:.2f}" for k, v in out.items()))

# 2. Exact drop vs EMAP drop; floor check
M = X[X.kind == "model"].copy(); M["drop_exact"] = M.acc - M.acc_proj; M["drop_exact_w"] = M.acc - M.acc_proj_w; M["s"] = M.cid.map(U.s); M["s_low"] = M.cid.map(U.s_low)
nl = X[X.kind == "null"].assign(d=lambda z: z.acc - z.acc_proj).groupby(["model", "cid"]).d.mean().rename("drop_null_exact").reset_index()
M = M.merge(nl, on=["model", "cid"]); M["bench"] = M.cid.map(U.bench); M["pid"] = M.cid.map(U.broja); M["yes"] = M.cid.map(U.yes); M["key"] = M.cid
M["excess"] = M.drop_exact - M.drop_null_exact
say("\n2. Mean drop: exact projection / confidence-weighted / null scorer (exact); proven-optimal share; floor drop >= s - e")
for m in NAMES:
    x = M[M.model == m]
    say(f"  {NAMES[m]:16s} exact {x.drop_exact.mean():.3f} | weighted {x.drop_exact_w.mean():.3f} | null {x.drop_null_exact.mean():.3f} | proven {x.proven.mean():.2f} / {x.proven_w.mean():.2f} | "
        f"floor holds {(x.drop_exact >= x.s_low - (1 - x.acc) - 1e-9).mean():.3f}")


def wrank(x, g):
    o = np.zeros(len(x))
    for b in np.unique(g):
        i = g == b; r = rankdata(x[i]); o[i] = (r - r.mean()) / len(r)
    return o


def partial(d, a, b, ctrl):
    g = d.bench.values; R = {c: wrank(d[c].values.astype(float), g) for c in [a, b] + ctrl}; Z = np.c_[np.ones(len(d)), np.c_[[R[c] for c in ctrl]].T]
    res = lambda y: y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]; return np.corrcoef(res(R[a]), res(R[b]))[0, 1]


def boot(d, f):
    k = d.key.unique(); idx = {c: np.where(d.key.values == c)[0] for c in k}
    return np.nanpercentile([f(d.iloc[np.concatenate([idx[c] for c in rng.choice(k, len(k))])]) for _ in range(NB)], [2.5, 97.5])


say("\n3. Same tests as Study C and the null analysis, with the exact drop (controls: synergy, accuracy, yes-share)")
X3 = {}
for m in NAMES:
    x = M[M.model == m].reset_index(drop=True); res = {}
    for nm, a, b, ctrl in (("rho(drop, s)", "drop_exact", "s", ["pid", "acc", "yes"]), ("rho(drop, PID)", "drop_exact", "pid", ["s", "acc", "yes"]),
                           ("rho(null drop, s)", "drop_null_exact", "s", ["pid", "acc", "yes"]), ("rho(drop - null, s)", "excess", "s", ["pid", "acc", "yes"])):
        r = partial(x, a, b, ctrl); c = boot(x, lambda d: partial(d, a, b, ctrl)); res[nm] = (r, c)
    X3[m] = res["rho(drop - null, s)"]
    say(f"  {NAMES[m]:16s} " + " | ".join(f"{k} {v[0]:+.3f} [{v[1][0]:+.3f},{v[1][1]:+.3f}]" for k, v in res.items()))
say(f"\nX1 (exact flips on exceptions or ambiguous cells >= 0.60 for every model): {'PASS' if all(v >= 0.6 for v in X1.values()) else 'FAIL'} "
    + str({NAMES[m]: round(v, 2) for m, v in X1.items()}))
say(f"X2 (exact drop beyond the null tracks structure: rho(drop - null, s) lower bound > 0) for Idefics3: "
    f"{'PASS' if X3['idefics'][1][0] > 0 else 'FAIL'}; others: " + ", ".join(f"{NAMES[m]} {'lower>0' if X3[m][1][0] > 0 else 'includes 0'}" for m in NAMES if m != "idefics"))
M.round(5).to_csv(os.path.join(ROOT, "paper", "revision", "data", "16_exact_projection_units.csv"), index=False)
open(os.path.join(ROOT, "paper", "results_xemap.txt"), "w").write("\n".join(lines) + "\n")
