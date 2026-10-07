"""Confirmatory test (paper/preregistration_vlm3_conf.md): on sub-tables where structure is uncorrelated with both PID
synergy and the yes share (runs/vlm3/selected2.csv), does structure predict a NEW VLM's interaction dependence, and does
PID add nothing once answer balance is held fixed? Confirmatory model: Idefics3-8B (unseen). Qwen2.5-VL, InternVL3.5 and
LLaVA-OV-1.5 are reported as exploratory. Writes paper/results_vlm3_conf.txt, paper/figs/vlm3_conf_units.csv"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "vlm3"); lines = []; NB = 2000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
MODELS = {"idefics": "Idefics3-8B (confirmatory)", "qwen": "Qwen2.5-VL-7B", "internvl": "InternVL3.5-8B", "llava": "LLaVA-OV-1.5-8B"}; BENCH = ["clevr", "gqa", "clove", "pope"]
G = np.load(os.path.join(OUT, "grids.npz")); U = pd.read_csv(os.path.join(OUT, "selected2.csv")).set_index("cid"); COLS = {c: json.loads(v) for c, v in U.cols.items()}
rows = []
for m in MODELS:
    S = np.load(os.path.join(OUT, f"scores_{m}.npz"))
    for k in U.index:
        g = U.key[k]; T, L = G[f"T__{g}"][:, COLS[k]], S[f"L__{g}"][:, COLS[k]]; E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean()
        a, ae = ((L > 0) == T).mean(), ((E > 0) == T).mean()
        rows.append(dict(model=m, key=k, bench=U.bench[k], s=U.s[k], s_low=U.s_low[k], pid=U.broja[k], yes=U.yes[k], acc=a, acc_emap=ae, delta=a - ae))
D = pd.DataFrame(rows); D.round(4).to_csv(os.path.join(ROOT, "paper", "figs", "vlm3_conf_units.csv"), index=False)


def wrank(x, g):
    o = np.zeros(len(x))
    for b in np.unique(g):
        i = g == b; r = rankdata(x[i]); o[i] = (r - r.mean()) / len(r)
    return o


def partial(d, a, b, ctrl):
    g = d.bench.values; R = {c: wrank(d[c].values.astype(float), g) for c in [a, b] + ctrl}; X = np.c_[np.ones(len(d)), np.c_[[R[c] for c in ctrl]].T]
    res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; return np.corrcoef(res(R[a]), res(R[b]))[0, 1]


def boot(d, f):
    keys = d.key.unique(); idx = {k: np.where(d.key.values == k)[0] for k in keys}
    return np.nanpercentile([f(d.iloc[np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])]) for _ in range(NB)], [2.5, 97.5])


def fit(d, terms):
    z = d.copy()
    for t in terms + ["delta"]:
        z[t] = (z[t] - z[t].mean()) / z[t].std()
    X = np.c_[np.ones(len(z)), np.c_[[z[t].values for t in terms] + [(z.bench == b).values.astype(float) for b in BENCH[1:]]].T]
    beta = np.linalg.lstsq(X, z.delta.values, rcond=None)[0]; r = z.delta.values - X @ beta; return beta[1:1 + len(terms)], 1 - r.var() / z.delta.var()


say(f"=== {len(U)} sub-tables (one per grid, 10 x 50); within-benchmark Spearman(s, BROJA) / (s, yes): " +
    ", ".join(f"{b} {spearmanr(x.s, x.broja).correlation:+.2f} / {spearmanr(x.s, x.yes).correlation:+.2f}" for b, x in U.groupby("bench")))
for m, nm in MODELS.items():
    x = D[D.model == m]; say(f"\n--- {nm} ---")
    say("  accuracy / EMAP drop by benchmark: " + " | ".join(f"{b} {x[x.bench == b].acc.mean():.3f} / {x[x.bench == b].delta.mean():+.3f}" for b in BENCH))
    say(f"  bound check (delta >= s_low - error): {(x.delta >= x.s_low - (1 - x.acc) - 1e-9).mean():.3f}")
    c1 = partial(x, "delta", "s", ["pid", "acc", "yes"]); c1i = boot(x, lambda d: partial(d, "delta", "s", ["pid", "acc", "yes"]))
    c2 = partial(x, "delta", "pid", ["s", "acc", "yes"]); c2i = boot(x, lambda d: partial(d, "delta", "pid", ["s", "acc", "yes"]))
    p1, p2 = bool(c1 >= 0.3 and c1i[0] > 0), bool(c2i[0] <= 0 <= c2i[1] and abs(c2) <= 0.15)
    say(f"  C1 rho(delta, s | PID, acc, yes)  {c1:+.3f} [{c1i[0]:+.3f},{c1i[1]:+.3f}]  (>= 0.3, lower > 0) -> {p1}")
    say(f"  C2 rho(delta, PID | s, acc, yes)  {c2:+.3f} [{c2i[0]:+.3f},{c2i[1]:+.3f}]  (interval contains 0, |rho| <= 0.15) -> {p2}")
    terms = ["s", "pid", "yes", "acc"]; bs, r2 = fit(x, terms); _, r2s = fit(x, ["pid", "yes", "acc"]); _, r2p = fit(x, ["s", "yes", "acc"])
    cb = boot(x, lambda d: fit(d, terms)[0][0]); p3 = bool(cb[0] > 0 and r2 - r2s >= 0.05 and r2 - r2p <= 0.01)
    say(f"  C3 regression (benchmark effects): beta_s {bs[0]:+.3f} [{cb[0]:+.3f},{cb[1]:+.3f}], beta_PID {bs[1]:+.3f}, beta_yes {bs[2]:+.3f}; "
        f"R2 added by s {r2 - r2s:+.3f} (>= 0.05), by PID {r2 - r2p:+.3f} (<= 0.01) -> {p3}")
    say("  within benchmark rho(delta, s) / rho(delta, PID): " + " | ".join(
        f"{b} {spearmanr(x[x.bench == b].delta, x[x.bench == b].s).correlation:+.2f} / {spearmanr(x[x.bench == b].delta, x[x.bench == b].pid).correlation:+.2f}" for b in BENCH))
    if m == "idefics":
        V = (p1, p2, p3)
say(f"\nVERDICT (confirmatory, Idefics3-8B): C1 {'PASS' if V[0] else 'FAIL'} | C2 {'PASS' if V[1] else 'FAIL'} | C3 {'PASS' if V[2] else 'FAIL'} -> "
    f"{'CONFIRMED' if V[0] and V[1] else 'NOT CONFIRMED'} (primary = C1 and C2)")
open(os.path.join(ROOT, "paper", "results_vlm3_conf.txt"), "w").write("\n".join(lines) + "\n")
