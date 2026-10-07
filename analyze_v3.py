"""Pre-registered analysis of v3 (paper/preregistration_v3.md). Writes paper/results_v3.txt."""
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
D = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "runs", "v3_*_v3.csv"))], ignore_index=True)
valid = lambda d, t: (d[f"n_{t}"] >= 30) & (d[f"before_{t}"] >= d.c + 0.3)
for t in TYPES:
    for s in ("s1", "s2", "sb", "actual", "sf"):
        D[f"{s}_{t}"] = D[f"{s}_{t}"].clip(-0.2, 1.2)
    D[f"prod_{t}"] = D[f"s1_{t}"].clip(0, 1.2) * D[f"s2_{t}"].clip(0, 1.2)
verdict = {}
for src, sub in (("SYNTHETIC", D[(D.data == "synth") & (D.types == "est")]), ("REAL (probe types)", D[(D.data != "synth") & (D.types == "est")]),
                 ("CREMA-D HUMAN TYPES [cross-check]", D[D.types == "human"])):
    say(f"\n==================== {src}: {sub.groupby(['data', 'snr', 'shift', 'class_split', 'order', 'depth', 'width', 'seed'], dropna=False).ngroups} units ====================")
    real = not src.startswith("SYN")
    say("(a) composition sb_t = s1_t * s2_t  (both regimes)")
    ok_a = True
    for t in TYPES:
        s = sub[valid(sub, t)]; res = m(s[f"sb_{t}"], s[f"prod_{t}"])
        nt = s[(s[f"s1_{t}"] <= 0.95) & (s[f"s2_{t}"] <= 0.95)]; rn = m(nt[f"sb_{t}"], nt[f"prod_{t}"])
        say(f"  {t:2s} {f(res)} | non-trivial subset {f(rn)}")
        if t != "R":
            ok_a &= res["R2"] >= (0.6 if real else 0.8) and res["MAE"] <= (0.08 if real else 0.06)
        else:
            ok_a &= real or res["R2"] >= 0.5
    say("(b) two-factor signature")
    u1 = sub[valid(sub, "U1")]; u2 = sub[valid(sub, "U2")]
    b1 = (u1.s2_U1 >= 0.9).mean(); b1b = (u2.s1_U2 >= 0.9).mean()
    sS, sR = sub[valid(sub, "S")], sub[valid(sub, "R")]
    b2 = np.minimum(sR.s1_R, sR.s2_R).mean() - np.minimum(sS.s1_S, sS.s2_S).mean()
    say(f"  b1. U1 undamaged by modality-2 drift in {100 * b1:.0f}% ; U2 undamaged by modality-1 drift in {100 * b1b:.0f}% (need >= 80%)")
    say(f"  b2. mean min-factor: R {np.minimum(sR.s1_R, sR.s2_R).mean():.2f} vs S {np.minimum(sS.s1_S, sS.s2_S).mean():.2f} -> difference {b2:+.3f} (need >= 0.10)")
    say(f"      mean factors S: s1 {sS.s1_S.mean():.2f} s2 {sS.s2_S.mean():.2f} | R: s1 {sR.s1_R.mean():.2f} s2 {sR.s2_R.mean():.2f} | "
        f"U1: s1 {u1.s1_U1.mean():.2f} s2 {u1.s2_U1.mean():.2f} | U2: s1 {u2.s1_U2.mean():.2f} s2 {u2.s2_U2.mean():.2f}")
    ok_b = b1 >= 0.8 and b1b >= 0.8 and b2 >= 0.10
    say("(c) reproduction: actual a_t vs swap sb_t")
    ok_c = True
    for regime in ("control", "none"):
        r = sub[sub.regime == regime]
        for t in TYPES:
            s = r[valid(r, t)]; res = m(s[f"actual_{t}"], s[f"sb_{t}"])
            say(f"  {'replay' if regime == 'control' else 'no replay':9s} {t:2s} {f(res)}  (mean actual {s[f'actual_{t}'].mean():.2f}, swap {s[f'sb_{t}'].mean():.2f}, fusion-only swap {s[f'sf_{t}'].mean():.2f})")
            if regime == "control" and t == "S":
                ok_c = res["R2"] >= (0.5 if real else 0.7) and res["MAE"] <= (0.10 if real else 0.08)
    if not src.startswith("CREMA"):
        verdict[src] = (ok_a, ok_b, ok_c)
say("\n==================== PRE-REGISTERED VERDICT ====================")
for src, (a, b, c) in verdict.items():
    say(f"  {src:20s} (a) composition {'PASS' if a else 'FAIL'} | (b) two-factor signature {'PASS' if b else 'FAIL'} | (c) reproduction {'PASS' if c else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_v3.txt"), "w").write("\n".join(lines) + "\n")
