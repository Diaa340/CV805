"""Pre-registered analysis for paper/preregistration_kinds_v2.md. Writes paper/results_kinds_v2.txt"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import t as tdist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []
def say(s=""): print(s); lines.append(s)
D = pd.read_csv(os.path.join(ROOT, "runs", (sys.argv[1] if len(sys.argv) > 1 else "kinds_v2") + ".csv"))
bad = D[(D.acc0 < 0.75) | (D.lin_acc > 0.60)][["seed", "kind"]].drop_duplicates()
say(f"seeds {sorted(D.seed.unique())}; units failing manipulation checks: {bad.values.tolist()}")
D = D.merge(bad.assign(x=1), how="left", on=["seed", "kind"]); D = D[D.x.isna()].drop(columns="x")
D["cond"] = D.cond.str.replace("_AB", "").str.replace("_BA", "")
D = D.groupby(["seed", "kind", "level", "cond"], as_index=False).mean(numeric_only=True)
D["E"] = D.k12 - D.k1 * D.k2; D["den"] = D[["k1", "k2"]].min(1) - D.k1 * D.k2; D["nest"] = D.E / D.den
D["den_sup"] = D[["k1_sup", "k2_sup"]].min(1) - D.k1_sup * D.k2_sup; D["nest_sup"] = (D.k12_sup - D.k1_sup * D.k2_sup) / D.den_sup
D["rec_sup"] = (D.k12_sup - D.k12) / (1 - D.k12); D["rec_free"] = (D.k12_free - D.k12) / (1 - D.k12)
cols = ["k1", "k2", "k12", "E", "nest", "C_all", "N_all", "C_y1", "N_y1", "C_y0", "N_y0", "rec_sup", "rec_free", "nest_sup", "retrieval", "drift1", "drift2"]
say("\nmeans over seeds:"); say(D.groupby(["cond", "kind", "level"])[cols].mean().round(3).to_string())
lo = lambda x: x.mean() - tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
hi = lambda x: x.mean() + tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
fmt = lambda x: f"{x.mean():+.3f} [{lo(x):+.3f}, {hi(x):+.3f}]"
mean = lambda cond, kind, col: D[(D.cond == cond) & (D.kind == kind)].groupby("level")[col].mean()
def informative(cond):
    m = D[D.cond == cond].groupby(["kind", "level"])[["k1", "k2"]].mean().min(1).unstack(0)
    return {l for l in m.index if all(0.1 <= m.loc[l, k] <= 0.9 for k in ("comparative", "conjunctive"))}
def nestok(cond, kind="comparative"):
    d = mean(cond, kind, "den"); return set(d[d >= 0.05].index)
def per_seed(cond, levels, col, kind=None):
    x = D[(D.cond == cond) & D.level.isin(levels)].groupby(["seed", "kind"])[col].mean().unstack()
    return x if kind is None else x[kind]
say("\n=== T0 replication (matched) ===")
Lm = informative("matched") & set((1, 3, 10)); M = per_seed("matched", Lm, "E"); g = M.comparative - M.conjunctive
t0 = M.comparative.mean() >= 0.05 and lo(M.comparative) > 0 and g.mean() >= 0.05 and lo(g) > 0
say(f"  levels {sorted(Lm)}: E comp {fmt(M.comparative)}, E conj {M.conjunctive.mean():+.3f}, comp - conj {fmt(g)} -> {'PASS' if t0 else 'FAIL'}")
say("\n=== mismatch check (retrieval on held-out pool-A items, comparative + conjunctive units) ===")
use = None
for b in ("misB1", "misB2"):
    ra, rb = D[D.cond == "matched"].groupby("level").retrieval.mean(), D[D.cond == b].groupby("level").retrieval.mean()
    passed = bool(((ra - rb) >= 0.10).all())
    say(f"  {b}: within-run {ra.round(3).to_dict()} mismatched {rb.round(3).to_dict()} -> {'PASS' if passed else 'FAIL'}")
    if passed and use is None:
        use = b
say(f"  using: {use}")
say("\n=== T1 mismatch (comparative) ===")
if use is None:
    say("  INCONCLUSIVE (no mismatch condition passed the check)")
else:
    L = Lm & informative(use) & nestok("matched") & nestok(use)
    if not L:
        say("  no eligible level -> INCONCLUSIVE")
    else:
        dN = per_seed("matched", L, "nest", "comparative") - per_seed(use, L, "nest", "comparative")
        dC = per_seed("matched", L, "C_all", "comparative") - per_seed(use, L, "C_all", "comparative")
        se = lambda x: tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        cancel = (dN.mean() >= 0.2 and lo(dN) > 0) or (dC.mean() >= 0.03 and lo(dC) > 0)
        weak = (dN.mean() - se(dN) >= -0.2 and dN.mean() + se(dN) <= 0.2) and (dC.mean() - se(dC) >= -0.03 and dC.mean() + se(dC) <= 0.03)
        say(f"  levels {sorted(L)}: nesting matched {per_seed('matched', L, 'nest', 'comparative').mean():.2f} vs {use} "
            f"{per_seed(use, L, 'nest', 'comparative').mean():.2f}; dNest {fmt(dN)}; C matched {per_seed('matched', L, 'C_all', 'comparative').mean():.3f} "
            f"vs {per_seed(use, L, 'C_all', 'comparative').mean():.3f}; dC {fmt(dC)}")
        say(f"  -> {'DRIFTS CANCEL' if cancel else ('SAME WEAK PAIRS FAIL' if weak else 'MIXED')}")
say("\n=== T2 independent, damage-matched ===")
ref = (D[(D.cond == "matched") & (D.kind == "comparative") & (D.level == 3)].eval("k1*k2")).mean()
ik = D[(D.cond == "independent") & (D.kind == "comparative")].assign(p=lambda x: x.k1 * x.k2).groupby("level").p.mean()
lm = int((ik - ref).abs().idxmin()); gap = abs(ik[lm] - ref)
say(f"  matched L3 comparative k1*k2 {ref:.3f}; independent by level {ik.round(3).to_dict()} -> level {lm} (gap {gap:.3f}); "
    f"independent comparative denominator at that level {mean('independent', 'comparative', 'den')[lm]:.3f}")
Ni = per_seed("independent", {lm}, "nest"); nc, nd = Ni.comparative, Ni.comparative - Ni.conjunctive
say(f"  nesting independent: comparative {fmt(nc)}, conjunctive {Ni.conjunctive.mean():+.3f}, comp - conj {fmt(nd)}; "
    f"coupled L3 comparative nesting {per_seed('matched', {3}, 'nest', 'comparative').mean():+.3f}")
if gap > 0.15:
    say("  -> INCONCLUSIVE (no damage match within 0.15)")
else:
    cancel = nc.mean() < 0.1 and hi(nc) < 0.2; weak = nc.mean() >= 0.2 and lo(nc) > 0 and nd.mean() >= 0.2 and lo(nd) > 0
    say(f"  -> {'DRIFTS CANCEL' if cancel else ('SAME WEAK PAIRS FAIL' if weak else 'MIXED')}")
say("\n=== T3 threshold refit (matched coupled, levels 3 and 10, supervised) ===")
R = per_seed("matched", {3, 10}, "rec_sup"); dr = R.comparative - R.conjunctive
t3 = R.comparative.mean() >= 0.25 and dr.mean() >= 0.15 and lo(dr) > 0
Rf = per_seed("matched", {3, 10}, "rec_free")
say(f"  recovery comparative {fmt(R.comparative)}, conjunctive {R.conjunctive.mean():+.3f}, difference {fmt(dr)} -> {'PASS' if t3 else 'FAIL'}")
say(f"  [reported] label-free: comparative {Rf.comparative.mean():+.3f}, conjunctive {Rf.conjunctive.mean():+.3f}, xor {Rf.xor.mean():+.3f}; "
    f"supervised xor {R.xor.mean():+.3f}")
say("\n[reported] label split, matched coupled, levels 1/3/10 (C and N pooled vs within label):")
say(D[D.cond == "matched"].groupby(["kind", "level"])[["C_all", "C_y1", "C_y0", "N_all", "N_y1", "N_y0"]].mean().round(3).to_string())
open(os.path.join(ROOT, "paper", "results_kinds_v2.txt"), "w").write("\n".join(lines) + "\n")
