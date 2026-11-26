"""EXPLORATORY (not pre-registered): a graded version of the drift-composition law, from P3's per-sample dumps
(runs/p3/*.npz). Uses only samples the task-1 model classified correctly.
  E1  per-sample breakage under one modality's drift vs graded reliance on that modality
  E2  graded composition: within (reliance-1, reliance-2) bins, survival(both drifted) vs survival(1) x survival(2)
  E3  graded predictive law: survival curves learned on half the units (by seed), used to predict each type's retention
      under both drifts on the other half; compared with the discrete-type product law (the P2 form) on the same units
Writes paper/results_graded_exploratory.txt"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = []


def say(s=""):
    print(s); lines.append(s)


rows = []
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "p3", "p3_*.npz"))):
    z = np.load(f, allow_pickle=True); name = os.path.basename(f)[:-4]
    data = name.split("_")[1]; seed = int(name.split("_s")[-1].split("_")[0]); regime = name.rsplit("_", 1)[1]
    keep = z["before"].astype(bool)
    rows.append(pd.DataFrame(dict(unit=name.rsplit("_", 1)[0], data=data, seed=seed, regime=regime, type=z["types"][keep],
                                  d1=z["d1"][keep], d2=z["d2"][keep], s1=z["correct_s1"][keep], s2=z["correct_s2"][keep],
                                  sb=z["correct_sb"][keep], actual=z["correct_actual"][keep], c=float(z["c"]))))
S = pd.concat(rows, ignore_index=True)
say(f"EXPLORATORY graded analysis: {S.unit.nunique()} units, {len(S)} correctly-classified task-1 test samples per regime-pooled set")
edges = [-1.0, 0.02, 0.1, 0.25, 0.5, 1.01]
S["b1"] = pd.cut(S.d1, edges, labels=False); S["b2"] = pd.cut(S.d2, edges, labels=False)
for regime in ("none", "control"):
    X = S[S.regime == regime]
    say(f"\n=== {'NO REPLAY' if regime == 'none' else 'REPLAY'} ===")
    say("E1 survival under one modality's drift, by graded reliance on that modality (bins of reliance):")
    say("   bin(reliance)   " + "  ".join(f"[{edges[i]:.2f},{edges[i + 1]:.2f})" for i in range(len(edges) - 1)))
    say("   modality 1 (s1) " + "  ".join(f"{X[X.b1 == i].s1.mean():>13.2f}" for i in range(len(edges) - 1)))
    say("   modality 2 (s2) " + "  ".join(f"{X[X.b2 == i].s2.mean():>13.2f}" for i in range(len(edges) - 1)))
    cells = X.groupby(["data", "b1", "b2"]).agg(n=("sb", "size"), sb=("sb", "mean"), s1=("s1", "mean"), s2=("s2", "mean")).reset_index()
    cells = cells[cells.n >= 50]; pred = cells.s1 * cells.s2
    w = cells.n / cells.n.sum()
    r2 = 1 - (w * (cells.sb - pred) ** 2).sum() / (w * (cells.sb - np.average(cells.sb, weights=w)) ** 2).sum()
    say(f"E2 graded composition over {len(cells)} (dataset x reliance-bin) cells (n>=50): weighted R^2 {r2:.3f}, "
        f"weighted MAE {np.average(np.abs(cells.sb - pred), weights=w):.3f}")
    # E3: learn survival curves s_m(d_m) per dataset on even seeds, predict type-level sb retention on odd seeds
    out = []
    for d, G in X.groupby("data"):
        tr, te = G[G.seed % 2 == 0], G[G.seed % 2 == 1]
        c1 = tr.groupby("b1").s1.mean(); c2 = tr.groupby("b2").s2.mean()
        te = te.assign(p=te.b1.map(c1).fillna(tr.s1.mean()) * te.b2.map(c2).fillna(tr.s2.mean()))
        for u, U_ in te.groupby("unit"):
            # same-unit graded variant: survival rates by reliance bin inside this unit, multiplied per sample
            r1 = U_.groupby("b1").s1.mean(); r2 = U_.groupby("b2").s2.mean()
            U_ = U_.assign(q=U_.b1.map(r1) * U_.b2.map(r2))
            for t, g in U_.groupby("type"):
                if len(g) < 30 or t == "none":
                    continue
                disc = g.s1.mean() * g.s2.mean()  # P2's discrete-type product law (same unit, type-level factors)
                out.append(dict(data=d, unit=u, type=t, obs=g.sb.mean(), graded_heldout=g.p.mean(), graded_same_unit=g.q.mean(), discrete=disc))
    O = pd.DataFrame(out)
    say("E3 per-type survival under both drifts (fraction of correctly-learned samples), odd seeds:")
    say("   graded_heldout: reliance curves learned on even seeds | graded_same_unit and discrete: factors measured in the same unit")
    for t in ("R", "U1", "U2", "S"):
        o = O[O.type == t]
        if len(o) < 3:
            continue
        m = lambda p: f"R^2 {1 - ((o.obs - p) ** 2).sum() / ((o.obs - o.obs.mean()) ** 2).sum():.3f} MAE {np.abs(o.obs - p).mean():.3f}"
        say(f"   {t:2s} n={len(o):3d}  graded_heldout {m(o.graded_heldout)} | graded_same_unit {m(o.graded_same_unit)} | discrete {m(o.discrete)}")
open(os.path.join(ROOT, "paper", "results_graded_exploratory.txt"), "w").write("\n".join(lines) + "\n")
