"""Null-scorer control for the VLM structure study (paper/preregistration_vlm3_null.md). Saved log-odds only; no GPU.
For each model m and full answer table u (99 tables, 10 x 100):
  - interaction strength IS = sum (L - Lhat)^2 / sum (L - mean L)^2, with L the log-odds clipped to [-10, 10] and
    Lhat the additive projection (row mean + column mean - grand mean);
  - EMAP drop Delta = Acc(L) - Acc(Lhat).
Null scorers: L0 = kappa_y * (2T - 1) + N(0, 1) per cell, with kappa_y = Phi^-1(acc_y) chosen so that the expected
accuracy on yes cells and on no cells (y = 1, 0) equals the VLM's on that table (answer-matched null; primary). A simple
null with one kappa from overall accuracy is reported. 200 draws per (model, table); null values are draw means.
Excess = VLM value - null value. Writes paper/results_vlm3_null.txt and paper/figs/vlm3_null_units.csv"""
import os
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata, spearmanr
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "vlm3"); lines = []
NB, ND = 2000, 200; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
MODELS = {"qwen": "Qwen2.5-VL-7B", "internvl": "InternVL3.5-8B", "llava": "LLaVA-OV-1.5-8B", "idefics": "Idefics3-8B"}; BENCH = ["clevr", "gqa", "clove", "pope"]


def proj(L):
    return L.mean(-1, keepdims=True) + L.mean(-2, keepdims=True) - L.mean((-2, -1), keepdims=True)


def stats(L, T):
    """L: (..., 10, 100) scores. Returns interaction strength and EMAP drop (arrays over leading dims)."""
    H = proj(L); c = L - L.mean((-2, -1), keepdims=True)
    IS = ((L - H) ** 2).sum((-2, -1)) / np.maximum((c ** 2).sum((-2, -1)), 1e-12)
    D = ((L > 0) == T).mean((-2, -1)) - ((H > 0) == T).mean((-2, -1)); return IS, D


def kappa(acc):
    return norm.ppf(np.clip(acc, 0.5 + 1e-3, 1 - 1e-3))


G = np.load(os.path.join(OUT, "grids.npz")); U = pd.read_csv(os.path.join(OUT, "units.csv")).set_index("key"); rows = []
for m in MODELS:
    S = np.load(os.path.join(OUT, f"scores_{m}.npz"))
    for k in U.index:
        T = G[f"T__{k}"].astype(int); L = np.clip(S[f"L__{k}"].astype(float), -10, 10); ok = (L > 0) == T
        IS, D = stats(L, T); sgn = 2 * T - 1
        ay = {y: ok[T == y].mean() if (T == y).any() else 0.75 for y in (0, 1)}
        K = np.where(T == 1, kappa(ay[1]), kappa(ay[0]))                       # answer-matched null
        N = K * sgn + rng.standard_normal((ND,) + T.shape); ISn, Dn = stats(N, T)
        N1 = kappa(ok.mean()) * sgn + rng.standard_normal((ND,) + T.shape); ISs, Ds = stats(N1, T)   # simple null
        rows.append(dict(model=m, key=k, bench=U.bench[k], s=U.s[k], acc=ok.mean(), IS=IS, IS_null=ISn.mean(), IS_simple=ISs.mean(),
                         D=D, D_null=Dn.mean(), D_simple=Ds.mean(), acc_null=((N > 0) == T).mean()))
D = pd.DataFrame(rows); D["IS_ex"] = D.IS - D.IS_null; D["D_ex"] = D.D - D.D_null
D.round(5).to_csv(os.path.join(ROOT, "paper", "figs", "vlm3_null_units.csv"), index=False)


def wrank(x, g):
    o = np.zeros(len(x))
    for b in np.unique(g):
        i = g == b; r = rankdata(x[i]); o[i] = (r - r.mean()) / len(r)
    return o


def pcor(d, a, b, ctrl):
    g = d.bench.values; R = {c: wrank(d[c].values.astype(float), g) for c in [a, b] + ctrl}
    X = np.c_[np.ones(len(d)), np.c_[[R[c] for c in ctrl]].T] if ctrl else np.ones((len(d), 1))
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; return np.corrcoef(res(R[a]), res(R[b]))[0, 1]


def boot(d, f):
    keys = d.key.unique(); idx = {k: np.where(d.key.values == k)[0] for k in keys}
    return np.nanpercentile([f(d.iloc[np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])]) for _ in range(NB)], [2.5, 97.5])


say(f"=== Null-scorer control: {D.key.nunique()} full tables x {len(MODELS)} models; {ND} null draws each ===")
say("Null accuracy check (answer-matched null vs VLM, mean over tables): " +
    ", ".join(f"{MODELS[m]} {D[D.model == m].acc_null.mean():.3f} vs {D[D.model == m].acc.mean():.3f}" for m in MODELS))
say("\nMeans by model: interaction strength VLM / null (answer-matched) / null (simple) | EMAP drop VLM / null / simple")
for m in MODELS:
    x = D[D.model == m]; say(f"  {MODELS[m]:16s} IS {x.IS.mean():.3f} / {x.IS_null.mean():.3f} / {x.IS_simple.mean():.3f} | "
                             f"Delta {x.D.mean():+.3f} / {x.D_null.mean():+.3f} / {x.D_simple.mean():+.3f}")
say("\nHow much of the structure-drop relation the null reproduces (pooled within-benchmark Spearman with s, no controls):")
for m in MODELS:
    x = D[D.model == m]; say(f"  {MODELS[m]:16s} rho(Delta_VLM, s) {pcor(x, 'D', 's', []):+.3f} | rho(Delta_null, s) {pcor(x, 'D_null', 's', []):+.3f} | "
                             f"rho(IS_VLM, s) {pcor(x, 'IS', 's', []):+.3f} | rho(IS_null, s) {pcor(x, 'IS_null', 's', []):+.3f}")
say("\nN1 (primary): excess interaction strength (VLM - answer-matched null) rises with s: within-benchmark partial rho(IS_ex, s | acc)")
say("N2: excess EMAP drop (VLM - answer-matched null) rises with s: within-benchmark partial rho(D_ex, s | acc)")
n1 = n2 = 0
for m in MODELS:
    x = D[D.model == m]
    a = pcor(x, "IS_ex", "s", ["acc"]); ca = boot(x, lambda d: pcor(d, "IS_ex", "s", ["acc"]))
    b = pcor(x, "D_ex", "s", ["acc"]); cb = boot(x, lambda d: pcor(d, "D_ex", "s", ["acc"]))
    n1 += bool(ca[0] > 0); n2 += bool(cb[0] > 0)
    say(f"  {MODELS[m]:16s} N1 {a:+.3f} [{ca[0]:+.3f},{ca[1]:+.3f}] | N2 {b:+.3f} [{cb[0]:+.3f},{cb[1]:+.3f}] | "
        f"mean excess IS {x.IS_ex.mean():+.3f}, excess Delta {x.D_ex.mean():+.3f}")
say("\nWithin each benchmark, Spearman(IS_ex, s) per model:")
for m in MODELS:
    x = D[D.model == m]; say(f"  {MODELS[m]:16s} " + " | ".join(f"{b} {spearmanr(x[x.bench == b].IS_ex, x[x.bench == b].s).correlation:+.2f}" for b in BENCH))
say(f"\nVERDICTS: N1 {'PASS' if n1 >= 3 else 'FAIL'} ({n1}/4 models with lower bound > 0) | N2 {'PASS' if n2 >= 3 else 'FAIL'} ({n2}/4)")
open(os.path.join(ROOT, "paper", "results_vlm3_null.txt"), "w").write("\n".join(lines) + "\n")
