"""Pre-registered analysis: does the structural score of an answer table predict the interaction dependence of
modern VLMs, and does PID synergy? (paper/preregistration_vlm3.md). Writes paper/results_vlm3.txt, figures/vlm3_units.csv,
figures/vlm3_scatter.png"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "vlm3"); lines = []; NB = 2000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
MODELS = {"qwen": "Qwen2.5-VL-7B", "internvl": "InternVL3.5-8B", "llava": "LLaVA-OV-1.5-8B"}; BENCH = ["clevr", "gqa", "clove", "pope"]
G = np.load(os.path.join(OUT, "grids.npz")); U = pd.read_csv(os.path.join(OUT, "units.csv")).set_index("key"); B = json.load(open(os.path.join(OUT, "broja.json")))
U["broja"] = U.index.map(B); U["pid"] = U.broja.fillna(U.mmi); U["s_low"] = 1 - U.bound   # s_low = proven lower bound on the minimum additive error
rows, cells = [], []
for m in MODELS:
    S = np.load(os.path.join(OUT, f"scores_{m}.npz"))
    for k in U.index:
        T, L = G[f"T__{k}"], S[f"L__{k}"]; E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean()   # EMAP: closest additive function
        full, emap = (L > 0) == T, (E > 0) == T
        rows.append(dict(model=m, key=k, bench=U.bench[k], s=U.s[k], s_low=U.s_low[k], pid=U.pid[k], mmi=U.mmi[k], acc=full.mean(), acc_emap=emap.mean(),
                         delta=full.mean() - emap.mean(), acc_exc=full[G[f"exc__{k}"] == 1].mean(), acc_rule=full[G[f"exc__{k}"] == 0].mean()))
        flip = ((L > 0) != (E > 0)).ravel(); exc = G[f"exc__{k}"].ravel(); mis = G[f"misfit__{k}"].ravel(); conf = -np.abs(L).ravel()
        cells.append(pd.DataFrame(dict(model=m, key=k, flip=flip, exc=exc, misfit=mis, lowconf=conf)))
D = pd.DataFrame(rows); C = pd.concat(cells); D.round(4).to_csv(os.path.join(ROOT, "figures", "vlm3_units.csv"), index=False)


def resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]; return y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]


def wrank(x, g):   # ranks within benchmark (centred), so pooled statistics are within-benchmark
    out = np.zeros(len(x))
    for b in np.unique(g):
        i = g == b; r = rankdata(x[i]); out[i] = (r - r.mean()) / len(r)
    return out


def partial(d, a, b, ctrl):
    """Spearman partial correlation of a and b given ctrl, all ranked within benchmark."""
    g = d.bench.values; R = {c: wrank(d[c].values.astype(float), g) for c in [a, b] + ctrl}; X = np.c_[[R[c] for c in ctrl]].T
    return np.corrcoef(resid(R[a], X), resid(R[b], X))[0, 1]


def boot(d, f):
    keys = d.key.unique(); idx = {k: np.where(d.key.values == k)[0] for k in keys}; v = []
    for _ in range(NB):
        s = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))]); v.append(f(d.iloc[s]))
    return np.nanpercentile(v, [2.5, 97.5])


say(f"=== {len(U)} units ({U.bench.value_counts().to_dict()}), 1,000 cells each; models: {', '.join(MODELS.values())} ===")
say(f"structural rules proven exact: {U.exact.sum()}/{len(U)}; BROJA available: {U.broja.notna().sum()}/{len(U)}")
say("\nmean accuracy | EMAP accuracy | EMAP drop | accuracy on exceptions / rule cells, by model and benchmark")
for m in MODELS:
    for b in BENCH:
        x = D[(D.model == m) & (D.bench == b)]
        say(f"  {MODELS[m]:16s} {b:6s} {x.acc.mean():.3f} | {x.acc_emap.mean():.3f} | {x.delta.mean():+.3f} | {x.acc_exc.mean():.3f} / {x.acc_rule.mean():.3f}")
# bound: EMAP's hard prediction is additive, so its error is at least the minimum additive error s; hence delta >= s - e
D["floor"] = D.s_low - (1 - D.acc); hold = (D.delta >= D.floor - 1e-9).mean()
say(f"\nBound check: delta >= s - e holds for {hold:.3f} of model-unit pairs (must be 1.000; s uses the proven lower bound)")

say("\nT1: within-benchmark Spearman rho(EMAP drop, s) [95% CI], per model")
t1 = {}
for m in MODELS:
    ok = 0; parts = []
    for b in BENCH:
        x = D[(D.model == m) & (D.bench == b)]; r = spearmanr(x.delta, x.s).correlation; c = boot(x, lambda y: spearmanr(y.delta, y.s).correlation)
        ok += bool(c[0] > 0); parts.append(f"{b} {r:+.2f} [{c[0]:+.2f},{c[1]:+.2f}]")
        rp = spearmanr(x.delta, x.pid).correlation; parts[-1] += f" (PID {rp:+.2f})"
    t1[m] = ok >= 3; say(f"  {MODELS[m]:16s} " + " | ".join(parts) + f"  -> {ok}/4")

say("\nT2: pooled within-benchmark partial Spearman, per model")
t2 = {}
for m in MODELS:
    x = D[D.model == m]; ps = partial(x, "delta", "s", ["pid", "acc"]); pp = partial(x, "delta", "pid", ["s", "acc"])
    cs = boot(x, lambda y: partial(y, "delta", "s", ["pid", "acc"])); cd = boot(x, lambda y: partial(y, "delta", "s", ["pid", "acc"]) - abs(partial(y, "delta", "pid", ["s", "acc"])))
    t2[m] = bool(ps >= 0.3 and cs[0] > 0 and ps - abs(pp) >= 0.2)
    say(f"  {MODELS[m]:16s} rho(delta, s | PID, acc) {ps:+.3f} [{cs[0]:+.3f},{cs[1]:+.3f}] | rho(delta, PID | s, acc) {pp:+.3f} | difference CI [{cd[0]:+.3f},{cd[1]:+.3f}] -> {t2[m]}")


def design(d, terms):
    X = [d[t].values for t in terms] + [(d.model == m).values.astype(float) for m in list(MODELS)[1:]] + [(d.bench == b).values.astype(float) for b in BENCH[1:]]
    return np.c_[X].T


def fit(d, terms):
    z = d.copy()
    for t in terms + ["delta"]:
        z[t] = (z[t] - z[t].mean()) / z[t].std()
    X = np.c_[np.ones(len(z)), design(z, terms)]; beta = np.linalg.lstsq(X, z.delta.values, rcond=None)[0]; r = z.delta.values - X @ beta
    return beta[1:1 + len(terms)], 1 - r.var() / z.delta.var()


bs, r2 = fit(D, ["s", "pid", "acc"]); _, r2_nos = fit(D, ["pid", "acc"]); _, r2_nop = fit(D, ["s", "acc"])
cb = boot(D, lambda d: fit(d, ["s", "pid", "acc"])[0][0]); cp_ = boot(D, lambda d: fit(d, ["s", "pid", "acc"])[0][1])
t3 = bool(cb[0] > 0 and (cp_[0] <= 0 <= cp_[1] or abs(bs[1]) < abs(bs[0]) / 3) and r2 - r2_nos >= 0.05 and r2 - r2_nop <= 0.01)
say(f"\nT3: regression delta ~ s + PID + acc + model + benchmark (standardised), all models pooled")
say(f"  beta_s {bs[0]:+.3f} [{cb[0]:+.3f},{cb[1]:+.3f}] | beta_PID {bs[1]:+.3f} [{cp_[0]:+.3f},{cp_[1]:+.3f}] | beta_acc {bs[2]:+.3f} | R2 {r2:.3f}")
say(f"  R2 added by s after PID: {r2 - r2_nos:+.3f} (need >= 0.05) | R2 added by PID after s: {r2 - r2_nop:+.3f} (need <= 0.01) -> {t3}")

say("\nT4: leave-one-benchmark-out: fit on three benchmarks, Spearman(predicted, actual drop) within the held-out one")
t4 = {}
for m in MODELS:
    ok = 0; parts = []
    for hb in BENCH:
        tr, te = D[(D.model == m) & (D.bench != hb)], D[(D.model == m) & (D.bench == hb)]
        pred = {}
        for nm, terms in (("s", ["s", "acc"]), ("PID", ["pid", "acc"])):
            X = np.c_[np.ones(len(tr)), tr[terms].values]; beta = np.linalg.lstsq(X, tr.delta.values, rcond=None)[0]
            pred[nm] = spearmanr(np.c_[np.ones(len(te)), te[terms].values] @ beta, te.delta).correlation
        ok += bool(pred["s"] > 0 and pred["s"] > pred["PID"]); parts.append(f"{hb} s {pred['s']:+.2f} / PID {pred['PID']:+.2f}")
    t4[m] = ok >= 3; say(f"  {MODELS[m]:16s} " + " | ".join(parts) + f"  -> {ok}/4")

say("\nCell level (secondary): AUC for predicting the cells whose answer EMAP flips")
for m in MODELS:
    x = C[C.model == m]; say(f"  {MODELS[m]:16s} exception label {roc_auc_score(x.flip, x.exc):.3f} | graded misfit {roc_auc_score(x.flip, x.misfit):.3f} | "
                             f"low confidence {roc_auc_score(x.flip, x.lowconf):.3f} | flipped share {x.flip.mean():.3f}")
say(f"\nVERDICTS: T1 {'PASS' if all(t1.values()) else 'FAIL'} {t1} | T2 {'PASS' if all(t2.values()) else 'FAIL'} {t2} | T3 {'PASS' if t3 else 'FAIL'} | "
    f"T4 {'PASS' if all(t4.values()) else 'FAIL'} {t4}")
open(os.path.join(ROOT, "paper", "results_vlm3.txt"), "w").write("\n".join(lines) + "\n")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True); col = dict(clevr="#2471a3", gqa="#c0392b", clove="#239b56", pope="#e67e22")
for j, m in enumerate(MODELS):
    x = D[D.model == m]
    for b in BENCH:
        y = x[x.bench == b]; ax[0, j].scatter(y.s, y.delta, s=18 + 60 * (y.acc - 0.5).clip(0), color=col[b], alpha=.7, label=b)
        ax[1, j].scatter(y.pid, y.delta, s=18 + 60 * (y.acc - 0.5).clip(0), color=col[b], alpha=.7)
    xs = np.linspace(0, x.s.max() * 1.05, 50); ax[0, j].plot(xs, xs, "k--", lw=.8, label="floor if the model were perfect (Δ ≥ s)")
    ax[0, j].set_title(f"{MODELS[m]}: structure"); ax[1, j].set_title(f"{MODELS[m]}: PID synergy")
    ax[0, j].set_xlabel("exception share s (task structure)"); ax[1, j].set_xlabel("PID synergy (bits)")
ax[0, 0].set_ylabel("EMAP accuracy drop Δ"); ax[1, 0].set_ylabel("EMAP accuracy drop Δ"); ax[0, 0].legend(fontsize=8)
fig.suptitle("Each point: one fully labelled answer-table unit (size = model accuracy). Δ ≥ s − error for every model.", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, .95)); plt.savefig(os.path.join(ROOT, "figures", "vlm3_scatter.png"), dpi=150)
