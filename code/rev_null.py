"""Revision R1/R2 (paper/preregistration_revision.md): how much of the Study-C structure effect is arithmetic?
Null scorer ("oracle + noise"): for each model and sub-table, logits l_ab = (2T_ab - 1) * mu_y + N(0, 1) with mu_y chosen
so that its expected accuracy on cells with answer y equals the model's accuracy on those cells (mu_y = Phi^-1(acc_y),
acc_y clipped to [0.001, 0.999]); 20 noise draws (seeds 0-19); its EMAP drop Delta_null is the mean over draws. Such a
scorer has the model's accuracy and answer bias but no structure in its errors beyond chance.
Reported per model (same within-benchmark partial rank correlations and unit bootstrap as Study C):
  rho_obs    = rho(Delta,          s | PID, acc, yes)        (Study C)
  rho_null   = rho(Delta_null,     s | PID, acc, yes)        (what accuracy alone forces)
  rho_excess = rho(Delta - Delta_null, s | PID, acc, yes)    (primary)
  rho_adj    = rho(Delta, s | PID, acc, yes, Delta_null)
  rho_slack  = rho(Delta - (s_low - e), s | PID, acc, yes)   (interaction use beyond the floor)
  rho_yes    = rho(Delta, yes | s, PID, acc)                 (R2: answer balance's own signal)
Writes paper/results_rev_null.txt and figures/rev_null_units.csv"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "vlm3"); lines = []; NB = 2000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
MODELS = {"idefics": "Idefics3-8B (confirmatory model of Study C)", "internvl": "InternVL3.5-8B", "llava": "LLaVA-OV-1.5-8B", "qwen": "Qwen2.5-VL-7B"}
G = np.load(os.path.join(OUT, "grids.npz")); U = pd.read_csv(os.path.join(OUT, "selected2.csv")).set_index("cid"); COLS = {c: json.loads(v) for c, v in U.cols.items()}


def emap_acc(L, T):
    E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean(); return ((L > 0) == T).mean(), ((E > 0) == T).mean()


rows = []
for m in MODELS:
    S = np.load(os.path.join(OUT, f"scores_{m}.npz"))
    for k in U.index:
        g = U.key[k]; T, L = G[f"T__{g}"][:, COLS[k]].astype(int), S[f"L__{g}"][:, COLS[k]]; a, ae = emap_acc(L, T)
        mu = {y: norm.ppf(np.clip(((L > 0) == T)[T == y].mean() if (T == y).any() else 0.5, 0.001, 0.999)) for y in (0, 1)}
        dn, an = [], []
        for d in range(20):
            Ln = (2 * T - 1) * np.where(T == 1, mu[1], mu[0]) + np.random.default_rng(d).standard_normal(T.shape); x, xe = emap_acc(Ln, T); dn.append(x - xe); an.append(x)
        rows.append(dict(model=m, key=k, bench=U.bench[k], s=U.s[k], s_low=U.s_low[k], pid=U.broja[k], yes=U.yes[k], acc=a, delta=a - ae,
                         delta_null=np.mean(dn), acc_null=np.mean(an)))
D = pd.DataFrame(rows); D["excess"] = D.delta - D.delta_null; D["slack"] = D.delta - (D.s_low - (1 - D.acc))
D.round(5).to_csv(os.path.join(ROOT, "figures", "rev_null_units.csv"), index=False)


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


C = ["pid", "acc", "yes"]
SPEC = [("rho_obs", "delta", "s", C), ("rho_null", "delta_null", "s", C), ("rho_excess", "excess", "s", C), ("rho_adj", "delta", "s", C + ["delta_null"]),
        ("rho_slack", "slack", "s", C), ("rho_yes", "delta", "yes", ["s", "pid", "acc"])]
say("Null scorer: accuracy- and answer-matched oracle + Gaussian logit noise, 20 draws, identical EMAP pipeline (Study C sub-tables).")
say(f"Check: mean |acc_null - acc| = {np.abs(D.acc_null - D.acc).mean():.4f}")
V = {}
for m, nm in MODELS.items():
    x = D[D.model == m]; say(f"\n--- {nm} ---  mean EMAP drop: model {x.delta.mean():+.3f}, null {x.delta_null.mean():+.3f}")
    for name, a, b, ctrl in SPEC:
        r = partial(x, a, b, ctrl); c = boot(x, lambda d: partial(d, a, b, ctrl)); V[(m, name)] = (r, c)
        say(f"  {name:10s} {r:+.3f} [{c[0]:+.3f},{c[1]:+.3f}]")
    ro, re_ = V[(m, "rho_obs")][0], V[(m, "rho_excess")][0]; say(f"  share of rho_obs surviving the null (rho_excess / rho_obs): {re_ / ro:.2f}")
r, c = V[("idefics", "rho_excess")]
say(f"\nR1 PRIMARY (Idefics3): rho_excess {r:+.3f} [{c[0]:+.3f},{c[1]:+.3f}] -> {'PASS' if c[0] > 0 else 'FAIL'} (lower bound > 0)")
say("R1 other models (reported): " + ", ".join(f"{m} {'lower>0' if V[(m, 'rho_excess')][1][0] > 0 else 'interval includes 0'}" for m in MODELS if m != "idefics"))
open(os.path.join(ROOT, "paper", "results_rev_null.txt"), "w").write("\n".join(lines) + "\n")
