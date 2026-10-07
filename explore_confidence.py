"""EXPLORATORY (not pre-registered; existing data): is "needs both" (S) knowledge still forgotten first at equal
old-model confidence? 140 natural units (P3: 5 datasets x 20; P4: CLEVR 20, VQA 20), both regimes.
Survival = still correct after task 2, among task-1-correct samples. Confidence = old model's probability of the true
class (p_full), binned into deciles of logit(p_full) within each unit. Adjusted gap = S survival minus R survival
re-weighted to S's confidence distribution. Writes paper/results_explore_confidence.txt"""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []
def say(s=""): print(s); lines.append(s)
rows = []
files = [f for f in sorted(glob.glob(os.path.join(ROOT, "runs", "p3", "*.npz"))) if "fv_cremad" not in f]
for f in files:
    z = np.load(f, allow_pickle=True); name = os.path.basename(f)[:-4]; regime = name.rsplit("_", 1)[1]
    data = name.split("_")[1]
    ok = z["before"].astype(bool); t = z["types"]; surv = z["correct_actual"].astype(float); p = np.clip(z["p_full"], 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)); edges = np.quantile(lg[ok], np.linspace(0, 1, 11)[1:-1]); b = np.digitize(lg, edges)
    r = dict(unit=name.rsplit("_", 1)[0], data=data, regime=regime)
    for ty in ("R", "U1", "U2", "S"):
        k = ok & (t == ty); r[f"n_{ty}"] = k.sum(); r[f"raw_{ty}"] = surv[k].mean() if k.sum() else np.nan
        r[f"conf_{ty}"] = np.median(p[k]) if k.sum() else np.nan
    # adjusted: within each confidence bin, S survival minus R survival, weighted by the number of S samples in the bin
    for ty in ("U1", "U2", "S"):
        num = den = 0.0; cov = 0
        for j in range(10):
            ks, kr = ok & (t == ty) & (b == j), ok & (t == "R") & (b == j)
            if ks.sum() >= 10 and kr.sum() >= 10:
                num += ks.sum() * (surv[ks].mean() - surv[kr].mean()); den += ks.sum()
        r[f"adj_{ty}"] = num / den if den else np.nan
        r[f"cover_{ty}"] = den / max(1, (ok & (t == ty)).sum())
    # finer check: 20 bins, only the S-vs-R comparison
    edges20 = np.quantile(lg[ok], np.linspace(0, 1, 21)[1:-1]); b20 = np.digitize(lg, edges20); num = den = 0.0
    for j in range(20):
        ks, kr = ok & (t == "S") & (b20 == j), ok & (t == "R") & (b20 == j)
        if ks.sum() >= 10 and kr.sum() >= 10:
            num += ks.sum() * (surv[ks].mean() - surv[kr].mean()); den += ks.sum()
    r["adj20_S"] = num / den if den else np.nan
    rows.append(r)
D = pd.DataFrame(rows); D["raw_gap_S"] = D.raw_S - D.raw_R
D["least"] = D[["raw_R", "raw_U1", "raw_U2", "raw_S"]].idxmin(1) == "raw_S"
say(f"EXPLORATORY: {D.unit.nunique()} units x {D.regime.nunique()} regimes ({len(D)} rows)")
for reg, x in D.groupby("regime"):
    v = x.dropna(subset=["raw_gap_S", "adj_S"])
    say(f"\n=== regime {reg} ({len(v)} units with S and R) ===")
    say(f"  median confidence p_full: R {x.conf_R.median():.3f} | U1 {x.conf_U1.median():.3f} | U2 {x.conf_U2.median():.3f} | S {x.conf_S.median():.3f}")
    say(f"  RAW:       S survival < R in {100 * (v.raw_gap_S < 0).mean():.0f}% of units; median gap {v.raw_gap_S.median():+.3f}; "
        f"S least retained in {100 * x.least.mean():.0f}%")
    say(f"  ADJUSTED (10 conf. bins, coverage of S samples {v.cover_S.median():.2f}): S < R in {100 * (v.adj_S < 0).mean():.0f}% of units; "
        f"median gap {v.adj_S.median():+.3f}; share of raw gap explained by confidence {1 - v.adj_S.median() / v.raw_gap_S.median():.2f}")
    say(f"  ADJUSTED (20 bins): S < R in {100 * (v.adj20_S < 0).mean():.0f}%; median gap {v.adj20_S.median():+.3f}")
    say(f"  unique vs R, adjusted: U1 {x.adj_U1.median():+.3f} (raw {(x.raw_U1 - x.raw_R).median():+.3f}), "
        f"U2 {x.adj_U2.median():+.3f} (raw {(x.raw_U2 - x.raw_R).median():+.3f})")
    say("  by dataset (median raw gap -> adjusted gap; % units adjusted < 0):")
    for d, y in v.groupby("data"):
        say(f"    {d:8s} n={len(y):2d} {y.raw_gap_S.median():+.3f} -> {y.adj_S.median():+.3f} ({100 * (y.adj_S < 0).mean():.0f}%)")
open(os.path.join(ROOT, "paper", "results_explore_confidence.txt"), "w").write("\n".join(lines) + "\n")
