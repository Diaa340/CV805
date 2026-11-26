"""Analysis for experiments 3-5 exactly as pre-registered in paper/preregistration.md (Route Law of Multimodal Forgetting).
Reads runs/*_task2.csv and runs/*_corrupt.csv written by route_exps.py; writes figures to figures/ and a text
report to paper/results_route_law.txt."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
out_lines = []


def say(s=""):
    print(s); out_lines.append(s)


def law(p1, p2):
    p1, p2 = np.clip(p1, 0, 1), np.clip(p2, 0, 1)
    return 1 - (1 - p1) * (1 - p2), p1 * p2


def metrics(obs, pred):
    m = np.isfinite(obs) & np.isfinite(pred); o, p = obs[m], pred[m]
    if len(o) < 3:
        return dict(n=len(o))
    return dict(n=len(o), pearson=pearsonr(o, p)[0], spearman=spearmanr(o, p)[0],
                R2=1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum(), MAE=np.abs(o - p).mean(), RMSE=np.sqrt(((o - p) ** 2).mean()))


def fmt(d):
    return " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def valid(df, t, prefix="before"):
    return (df[f"n_{t}"] >= 30) & (df[f"{prefix}_{t}"] >= df.c + 0.3)


# ------------------------------------------------------------------------------------------ load
T2 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs", "*_task2.csv")) if "smoke" not in f], ignore_index=True)
CR = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs", "*_corrupt.csv")) if "smoke" not in f], ignore_index=True)
T2["unit"] = T2[["data", "snr", "shift", "class_split", "order", "depth", "width", "seed", "batch"]].astype(str).agg("|".join, axis=1)
ok = valid(T2, "U1") & valid(T2, "U2")
T2["p1"], T2["p2"] = T2.ret_U1.clip(0, 1), T2.ret_U2.clip(0, 1)
T2["predR"], T2["predS"] = law(T2.p1, T2.p2)
T2["predR_probe"], T2["predS_probe"] = law(T2.p1_probe, T2.p2_probe)
say(f"loaded {T2.unit.nunique()} units ({(T2.batch == 'heldout').groupby(T2.unit).first().sum()} held-out), {len(T2)} task-2 rows, {len(CR)} corruption rows")

# ------------------------------------------------------------------------------ experiment 3
say("\n==================== Experiment 3: held-out prediction (zero parameters) ====================")
H = T2[(T2.batch == "heldout") & ok]
for regime, sub in (("replay (control) [primary]", H[H.cond == "control"]), ("no replay [known failure regime]", H[H.cond == "none"])):
    say(f"\n--- {regime} ---")
    for t in ("R", "S"):
        s = sub[valid(sub, t)]
        say(f"  {t}: primary p   {fmt(metrics(s[f'ret_{t}'].values, s[f'pred{t}'].values))}")
        say(f"  {t}: label-free p {fmt(metrics(s[f'ret_{t}'].values, s[f'pred{t}_probe'].values))}")
    for t in ("U1", "U2"):
        s = sub[valid(sub, t)]; m = 1 if t == "U1" else 2
        say(f"  {t}: label-free p {fmt(metrics(s[f'ret_{t}'].values, s[f'p{m}_probe'].clip(0, 1).values))}")
    say("  by data source (primary p, R / S):")
    for dname, g in sub.groupby("data"):
        mr = metrics(g[valid(g, 'R')].ret_R.values, g[valid(g, 'R')].predR.values)
        ms = metrics(g[valid(g, 'S')].ret_S.values, g[valid(g, 'S')].predS.values)
        say(f"    {dname:8s} R: {fmt(mr)}\n    {'':8s} S: {fmt(ms)}")

say("\n--- baselines (linear regression fitted on the discovery batch, scored on held-out replay runs) ---")
D = T2[(T2.batch == "discovery") & ok & (T2.cond == "control")]
Hc = H[H.cond == "control"]
feats = {"(a) initial PID amount": lambda t: [f"amount_{t}"], "(b) accuracy before": lambda t: [f"before_{t}"],
         "(c) per-modality accuracy": lambda t: ["modacc1", "modacc2"], "(d) gradient norm": lambda t: ["grad_e1", "grad_e2"],
         "(e) parameter drift": lambda t: ["drift_e1", "drift_e2"], "(f) additive in p1, p2": lambda t: ["p1", "p2"],
         "(g) all of (a)-(f)": lambda t: [f"amount_{t}", f"before_{t}", "modacc1", "modacc2", "grad_e1", "grad_e2", "drift_e1", "drift_e2", "p1", "p2"]}
for t in ("R", "S"):
    d, h = D[valid(D, t)], Hc[valid(Hc, t)]
    say(f"  {t}: ROUTE LAW (0 params)       {fmt(metrics(h[f'ret_{t}'].values, h[f'pred{t}'].values))}")
    for name, f in feats.items():
        cols = f(t); dd, hh = d.dropna(subset=cols + [f"ret_{t}"]), h.dropna(subset=cols + [f"ret_{t}"])
        if len(dd) < len(cols) + 2:
            say(f"  {t}: {name:26s} (too few discovery rows)"); continue
        reg = LinearRegression().fit(dd[cols].values, dd[f"ret_{t}"].values)
        say(f"  {t}: {name:26s} {fmt(metrics(hh[f'ret_{t}'].values, reg.predict(hh[cols].values)))}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
for i, t in enumerate(("R", "S")):
    s = Hc[valid(Hc, t)]
    for dname, g in s.groupby("data"):
        ax[i].scatter(g[f"pred{t}"], g[f"ret_{t}"], s=14, alpha=0.7, label=dname)
    ax[i].plot([0, 1], [0, 1], "k--", lw=1); ax[i].set_xlim(-0.05, 1.05); ax[i].set_ylim(-0.05, 1.1)
    ax[i].set_xlabel(f"predicted ({'1-(1-p1)(1-p2)' if t == 'R' else 'p1 p2'})"); ax[i].set_ylabel("observed retention")
    ax[i].set_title({"R": "Redundant", "S": "Synergistic"}[t]); ax[i].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "exp3_scatter.png"), dpi=150); plt.close()

# ------------------------------------------------------------------------------ experiment 4
say("\n==================== Experiment 4: interventions on one route (replay regime) ====================")
C = T2[(T2.cond == "control") & ok].set_index("unit")
I = T2[~T2.cond.isin(["control", "none"]) & ok].copy()
I = I[I.unit.isin(C.index)]
for col in ("p1", "p2", "ret_R", "ret_S", "ret_U1", "ret_U2", "predR", "predS"):
    I[f"d_{col}"] = I[col].values - C.loc[I.unit, col].values
I["m"] = I.cond.str[-1].astype(int); I["kind"] = I.cond.str[:-2]
I["d_pm"] = np.where(I.m == 1, I.d_p1, I.d_p2); I["d_pother"] = np.where(I.m == 1, I.d_p2, I.d_p1)
I["d_Uother"] = np.where(I.m == 1, I.d_ret_U2, I.d_ret_U1)
say("per intervention kind (means; sign agreement = share of runs where observed change has the predicted sign):")
for kind, g in I.groupby("kind"):
    gs = g[valid(g, "S") & valid(g, "R")]
    say(f"  {kind:10s} n={len(g):3d} | d p_m {g.d_pm.mean():+.3f} (raised in {100 * (g.d_pm > 0).mean():.0f}%) | "
        f"|d p_other|<=0.1 in {100 * (g.d_pother.abs() <= 0.1).mean():.0f}% | d U_other {g.d_Uother.mean():+.3f}")
    for t in ("S", "R"):
        o, p = gs[f"d_ret_{t}"], gs[f"d_pred{t}"]
        say(f"  {'':10s}   d{t}: observed {o.mean():+.3f} predicted {p.mean():+.3f} | sign agreement "
            f"{100 * (np.sign(o) == np.sign(p)).mean():.0f}% | MAE of change {np.abs(o - p).mean():.3f} | r {np.corrcoef(o, p)[0, 1] if len(o) > 2 else np.nan:.2f}")
say("\nasymmetric target (one route ~0.9, the other ~0.4), all replay-regime runs:")
A = T2[T2.cond != "none"].copy(); A = A[ok.loc[A.index] & valid(A, "S")]
hi, lo = np.maximum(A.p1, A.p2), np.minimum(A.p1, A.p2)
sel = A[(hi.between(0.85, 0.95)) & (lo.between(0.35, 0.45))]
if len(sel):
    say(f"  n={len(sel)}: mean p_hi {np.maximum(sel.p1, sel.p2).mean():.2f}, p_lo {np.minimum(sel.p1, sel.p2).mean():.2f} | "
        f"observed S {sel.ret_S.mean():.3f} vs predicted {sel.predS.mean():.3f} | "
        f"observed U_hi {np.where(sel.p1 >= sel.p2, sel.ret_U1, sel.ret_U2).mean():.3f}")
else:
    say("  no run landed in the target window")

# ------------------------------------------------------------------------------ experiment 5
say("\n==================== Experiment 5: corrupt / remove one modality ====================")
CR = CR.copy()
CR["q1"], CR["q2"] = CR.ret_U1.clip(0, 1), CR.ret_U2.clip(0, 1)
CR["predR"], CR["predS"] = law(CR.q1, CR.q2)
okc = (CR.base_U1 >= CR.c + 0.3) & (CR.base_U2 >= CR.c + 0.3)
for model, g in CR[okc & ~((CR.a1 == 0) & (CR.a2 == 0))].groupby("model"):
    say(f"  model after {model}:")
    for t in ("R", "S"):
        gg = g[g[f"base_{t}"] >= g.c + 0.3]
        say(f"    {t}: {fmt(metrics(gg[f'ret_{t}'].values, gg[f'pred{t}'].values))}")
say("\n  removal pattern (a = 1), mean retention [pre-registered: remove m1 -> U1, S collapse; U2 stable; R partial]:")
for (a1, a2), g in CR[okc & CR.a1.isin([0, 1]) & CR.a2.isin([0, 1]) & ~((CR.a1 == 0) & (CR.a2 == 0))].groupby(["a1", "a2"]):
    lab = {(1.0, 0.0): "remove m1", (0.0, 1.0): "remove m2", (1.0, 1.0): "remove both"}[(a1, a2)]
    say(f"    {lab:11s} R {g.ret_R.mean():.2f} U1 {g.ret_U1.mean():.2f} U2 {g.ret_U2.mean():.2f} S {g.ret_S.mean():.2f}  (law: R {g.predR.mean():.2f}, S {g.predS.mean():.2f})")
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
g = CR[okc & ~((CR.a1 == 0) & (CR.a2 == 0))]
for i, t in enumerate(("R", "S")):
    gg = g[g[f"base_{t}"] >= g.c + 0.3]
    ax[i].scatter(gg[f"pred{t}"], gg[f"ret_{t}"].clip(-0.1, 1.2), s=6, alpha=0.3)
    ax[i].plot([0, 1], [0, 1], "k--", lw=1); ax[i].set_xlabel("predicted from corrupted-route survival"); ax[i].set_ylabel("observed retention")
    ax[i].set_title({"R": "Redundant under corruption", "S": "Synergistic under corruption"}[t])
plt.tight_layout(); plt.savefig(os.path.join(FIG, "exp5_scatter.png"), dpi=150); plt.close()
open(os.path.join(ROOT, "paper", "results_route_law.txt"), "w").write("\n".join(out_lines) + "\n")
