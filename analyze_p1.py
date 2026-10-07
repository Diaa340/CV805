"""Pre-registered analysis of P1, the semi-synthetic bridge (paper/preregistration_p1_bridge.md).
Same criteria as v3's synthetic thresholds, plus the learnability rule. Writes paper/results_p1_bridge.txt."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["R", "U1", "U2", "S"]
lines = []


def say(s=""):
    print(s); lines.append(s)


def m(o, p):
    k = np.isfinite(o) & np.isfinite(p); o, p = np.asarray(o)[k], np.asarray(p)[k]
    if len(o) < 3:
        return dict(n=len(o), r=np.nan, R2=np.nan, MAE=np.nan)
    return dict(n=len(o), r=pearsonr(o, p)[0], R2=1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum(), MAE=np.abs(o - p).mean())


f = lambda d: " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())
D = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "runs", "p1br_*_v3.csv"))], ignore_index=True)
valid = lambda d, t: (d[f"n_{t}"] >= 30) & (d[f"before_{t}"] >= d.c + 0.3)
for t in TYPES:
    for s in ("s1", "s2", "sb", "actual", "sf"):
        D[f"{s}_{t}"] = D[f"{s}_{t}"].clip(-0.2, 1.2)
    D[f"prod_{t}"] = D[f"s1_{t}"].clip(0, 1.2) * D[f"s2_{t}"].clip(0, 1.2)
units = D.drop_duplicates(["data", "shift", "depth", "width", "seed"])
learn = valid(units, "S").mean()
say(f"P1 bridge: {len(units)} units; synergy learned in task 1 in {100 * learn:.0f}% of units (learnability rule: >= 50%)")
say("  per dataset: " + ", ".join(f"{d}: {100 * valid(g, 'S').mean():.0f}% (task-1 acc S {g.before_S.mean():.2f}, chance {g.c.iloc[0]:.2f})"
                                  for d, g in units.groupby("data")))


def evaluate(sub, label):
    say(f"\n==================== {label} ====================")
    say("(a) composition sb_t = s1_t * s2_t (both regimes)")
    ok_a = True
    for t in TYPES:
        s = sub[valid(sub, t)]; res = m(s[f"sb_{t}"], s[f"prod_{t}"])
        nt = s[(s[f"s1_{t}"] <= 0.95) & (s[f"s2_{t}"] <= 0.95)]
        say(f"  {t:2s} {f(res)} | non-trivial subset {f(m(nt[f'sb_{t}'], nt[f'prod_{t}']))}")
        ok_a &= (res["R2"] >= 0.8 and res["MAE"] <= 0.06) if t != "R" else res["R2"] >= 0.5
    u1, u2, sS, sR = sub[valid(sub, "U1")], sub[valid(sub, "U2")], sub[valid(sub, "S")], sub[valid(sub, "R")]
    b1, b1b = (u1.s2_U1 >= 0.9).mean(), (u2.s1_U2 >= 0.9).mean()
    b2 = np.minimum(sR.s1_R, sR.s2_R).mean() - np.minimum(sS.s1_S, sS.s2_S).mean()
    say("(b) two-factor signature")
    say(f"  b1. U1 undamaged by modality-2 drift in {100 * b1:.0f}%; U2 undamaged by modality-1 drift in {100 * b1b:.0f}% (need >= 80%)")
    say(f"  b2. mean min-factor R {np.minimum(sR.s1_R, sR.s2_R).mean():.2f} vs S {np.minimum(sS.s1_S, sS.s2_S).mean():.2f} -> {b2:+.3f} (need >= 0.10)")
    say(f"      mean factors S: s1 {sS.s1_S.mean():.2f} s2 {sS.s2_S.mean():.2f} | R: s1 {sR.s1_R.mean():.2f} s2 {sR.s2_R.mean():.2f} | "
        f"U1: s1 {u1.s1_U1.mean():.2f} s2 {u1.s2_U1.mean():.2f} | U2: s1 {u2.s1_U2.mean():.2f} s2 {u2.s2_U2.mean():.2f}")
    ok_b = b1 >= 0.8 and b1b >= 0.8 and b2 >= 0.10
    say("(c) reproduction: actual vs swap")
    ok_c = False
    for regime in ("control", "none"):
        r = sub[sub.regime == regime]
        for t in TYPES:
            s = r[valid(r, t)]; res = m(s[f"actual_{t}"], s[f"sb_{t}"])
            say(f"  {'replay' if regime == 'control' else 'no replay':9s} {t:2s} {f(res)} (mean actual {s[f'actual_{t}'].mean():.2f}, swap {s[f'sb_{t}'].mean():.2f}, fusion-only {s[f'sf_{t}'].mean():.2f})")
            if regime == "control" and t == "S":
                ok_c = res["R2"] >= 0.7 and res["MAE"] <= 0.08
    return ok_a, ok_b, ok_c


a, b, c = evaluate(D, "ALL UNITS POOLED [pre-registered test]")
per = {d: evaluate(g, f"per dataset: {d} [reported]") for d, g in D.groupby("data")}
say("\n==================== PRE-REGISTERED VERDICT ====================")
if learn < 0.5:
    say("  INCONCLUSIVE: synergy learned in fewer than 50% of units")
say(f"  pooled: (a) composition {'PASS' if a else 'FAIL'} | (b) two-factor signature {'PASS' if b else 'FAIL'} | (c) reproduction {'PASS' if c else 'FAIL'}")
for d, (x, y_, z) in per.items():
    say(f"  {d:16s} (a) {'PASS' if x else 'FAIL'} | (b) {'PASS' if y_ else 'FAIL'} | (c) {'PASS' if z else 'FAIL'}   [reported, not a criterion]")
open(os.path.join(ROOT, "paper", "results_p1_bridge.txt"), "w").write("\n".join(lines) + "\n")
