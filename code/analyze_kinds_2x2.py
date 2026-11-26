"""Pre-registered analysis for paper/preregistration_kinds_2x2.md. Writes paper/results_kinds_2x2.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import t as tdist
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []
def say(s=""): print(s); lines.append(s)
D = pd.read_csv(os.path.join(ROOT, "runs", "kinds_2x2.csv"))
bad = D[(D.acc0 < 0.75) | (D.lin_acc > 0.60)][["seed", "kind"]].drop_duplicates()
say(f"seeds {sorted(D.seed.unique())}; units failing manipulation checks: {bad.values.tolist()}")
D = D.merge(bad.assign(x=1), how="left", on=["seed", "kind"]); D = D[D.x.isna()].drop(columns="x")
D["cond"] = D.cond.replace({"mismatched_AB": "mismatched", "mismatched_BA": "mismatched"})
D = D.groupby(["seed", "kind", "level", "cond"], as_index=False).mean(numeric_only=True)
say("\nmeans over seeds:")
say(D.groupby(["cond", "kind", "level"])[["k1", "k2", "k12", "E", "C", "N", "drift1", "drift2", "coordAB_e2"]].mean().round(3).to_string())
def informative(cond):
    m = D[D.cond == cond].groupby(["kind", "level"])[["k1", "k2"]].mean().min(1).unstack(0)
    return {l for l in m.index if all(0.1 <= m.loc[l, k] <= 0.9 for k in ("comparative", "conjunctive"))}
lo = lambda x: x.mean() - tdist.ppf(0.95, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
def Es(cond, levels):
    return D[(D.cond == cond) & D.level.isin(levels)].groupby(["seed", "kind"]).E.mean().unstack()
fmt = lambda x: f"{x.mean():+.3f} (lower {lo(x):+.3f})"
inf = {c: informative(c) & {1, 3, 10} for c in ("matched", "mismatched", "independent")}
say(f"\ninformative levels: { {c: sorted(v) for c, v in inf.items()} }")
say("\n=== T0 replication (matched) ===")
M = Es("matched", inf["matched"]); g = M.comparative - M.conjunctive
t0 = M.comparative.mean() >= 0.05 and lo(M.comparative) > 0 and g.mean() >= 0.05 and lo(g) > 0
say(f"  E comp {fmt(M.comparative)}, E conj {M.conjunctive.mean():+.3f}, comp - conj {fmt(g)} -> {'PASS' if t0 else 'FAIL'}")
say("\n=== T1 mismatch ===")
L = inf["matched"] & inf["mismatched"]
if not L:
    say("  no common informative level -> INCONCLUSIVE")
else:
    Mm, Mx = Es("matched", L), Es("mismatched", L); d1 = Mm.comparative - Mx.comparative
    co = d1.mean() >= 0.05 and lo(d1) > 0; ag = Mx.comparative.mean() >= 0.05 and lo(Mx.comparative) > 0 and d1.mean() < 0.05
    say(f"  levels {sorted(L)}: E comp matched {Mm.comparative.mean():+.3f}, mismatched {fmt(Mx.comparative)}, D1 {fmt(d1)}; "
        f"E conj mismatched {Mx.conjunctive.mean():+.3f}")
    say(f"  -> {'CO-DRIFT supported' if co else ('AGREEMENT-FEATURE supported' if ag else 'MIXED')}")
say("\n=== T2 2x2 interaction ===")
L = inf["matched"] & inf["independent"]
if not L:
    say("  no common informative level -> INCONCLUSIVE")
else:
    Mc, Mi = Es("matched", L), Es("independent", L); gc, gi = Mc.comparative - Mc.conjunctive, Mi.comparative - Mi.conjunctive
    inter = gc - gi
    only = inter.mean() >= 0.05 and lo(inter) > 0 and gi.mean() < 0.03; nocoup = gi.mean() >= 0.05 and lo(gi) > 0
    say(f"  levels {sorted(L)}: coupled comp {Mc.comparative.mean():+.3f} conj {Mc.conjunctive.mean():+.3f} | independent comp "
        f"{Mi.comparative.mean():+.3f} conj {Mi.conjunctive.mean():+.3f}")
    say(f"  G(coupled) {fmt(gc)}, G(independent) {fmt(gi)}, interaction {fmt(inter)}")
    say(f"  -> {'ONLY COUPLED-COMPARATIVE CELL DIFFERS' if only else ('COMPARATIVE ADVANTAGE WITHOUT COUPLING' if nocoup else 'MIXED')}")
say("\n=== T2m damage-matched (addendum) ===")
ref = D[(D.cond == "matched") & (D.kind == "comparative") & (D.level == 3)].k12.mean()
ik = D[(D.cond == "independent") & (D.kind == "comparative")].groupby("level").k12.mean()
lm = int((ik - ref).abs().idxmin()); Mi = Es("independent", {lm}); gi = Mi.comparative - Mi.conjunctive
Mc = Es("matched", {3}); gc = Mc.comparative - Mc.conjunctive
say(f"  matched level 3 comparative k12 {ref:.2f}; independent comparative k12 by level {ik.round(2).to_dict()} -> level {lm}")
say(f"  G(coupled, L3) {fmt(gc)} | G(independent, L{lm}) {fmt(gi)} (E comp {Mi.comparative.mean():+.3f}, conj {Mi.conjunctive.mean():+.3f}) | "
    f"difference {fmt(gc - gi)}")
say(f"  -> {'COMPARATIVE ADVANTAGE WITHOUT COUPLING at matched damage' if gi.mean() >= 0.05 and lo(gi) > 0 else ('COUPLING NEEDED at matched damage' if (gc - gi).mean() >= 0.05 and lo(gc - gi) > 0 else 'MIXED')}")
say("\n[reported] XOR E by condition (all levels): " + "; ".join(f"{c} {x.E.mean():+.3f}" for c, x in D[D.kind == 'xor'].groupby('cond')))
open(os.path.join(ROOT, "paper", "results_kinds_2x2.txt"), "w").write("\n".join(lines) + "\n")
