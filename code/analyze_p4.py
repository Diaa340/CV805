"""Pre-registered analysis of P4 (paper/preregistration_p4_synergy_rich.md). Writes paper/results_p4_synergy_rich.txt."""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
C = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "runs", "p4_*_p3.csv"))], ignore_index=True)
# per-sample dumps -> per-unit, per-regime, per-type retention under actual / swaps
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "runs", "p3", "p4_*.npz"))):
    z = np.load(p, allow_pickle=True); name = os.path.basename(p)[:-4]
    data = [d for d in ("fv_cremad", "vqa", "clevr") if f"_{d}_" in name][0]
    regime = name.rsplit("_", 1)[1]; c = float(z["c"])
    for t in TYPES:
        k = z["types"] == t
        if k.sum() < 30:
            continue
        b = z["before"][k].mean()
        if b - c < 0.3:
            continue
        r = dict(data=data, unit=name.rsplit("_", 1)[0], regime=regime, type=t, n=int(k.sum()))
        for x in ("actual", "s1", "s2", "sb"):
            r[x] = (z[f"correct_{x}"][k].mean() - c) / (b - c)
        rows.append(r)
P = pd.DataFrame(rows)
for x in ("actual", "s1", "s2", "sb"):
    P[x] = P[x].clip(-0.2, 1.2)
P["prod"] = P.s1.clip(0, 1.2) * P.s2.clip(0, 1.2)
say(f"P4: {C.drop_duplicates(['data', 'class_split', 'order', 'seed']).shape[0]} units")
say("\n=== manipulation check: share of the task-1 model's accuracy on S samples reached by a LINEAR readout (need median <= 0.6) ===")
passed = []
for d, g in C[C.arm == "task1_linear"].groupby("data"):
    med = g.ret_S.median(); ok = med <= 0.6; passed += [d] if ok else []
    say(f"  {d:10s} median {med:.2f} (n={len(g)}) -> {'SYNERGY-RICH' if ok else 'additive: not tested'}")
key = ["data", "class_split", "order", "seed", "regime"]
W = C[C.regime != "task1"].pivot_table(index=key, columns="arm", values=[f"ret_{t}" for t in TYPES])
for d in sorted(set(C.data)):
    say(f"\n==================== {d} {'[tested]' if d in passed else '[reported only]'} ====================")
    Pd = P[P.data == d]
    say("  per-type mean retention (no replay / replay) under actual; S factors s1, s2, sb:")
    for t in TYPES:
        g = Pd[Pd.type == t]
        if len(g):
            say(f"    {t:2s} n_units={g.unit.nunique():2d} actual {g[g.regime == 'none'].actual.mean():.2f} / {g[g.regime == 'control'].actual.mean():.2f} | "
                f"s1 {g.s1.mean():.2f} s2 {g.s2.mean():.2f} sb {g.sb.mean():.2f} prod {g['prod'].mean():.2f}")
    s = Pd[Pd.type == "S"]
    h1 = m(s.sb, s["prod"]); h2 = m(s[s.regime == "control"].actual, s[s.regime == "control"].sb)
    say(f"  H1 composition (S, both regimes): {f(h1)}")
    say(f"  H2 reproduction (S, replay):      {f(h2)}")
    w = W.xs(d, level="data").xs("none", level="regime")
    gap = w[("ret_S", "ceiling")] - w[("ret_S", "no_repair")]
    rec = lambda arm: ((w[("ret_S", arm)] - w[("ret_S", "no_repair")]) / gap)[gap >= 0.05]
    r_ro, r_re = rec("readout_mlp"), rec("encoder_realign")
    say(f"  H3 readout limit (no replay): median recovery_S readout_mlp {r_ro.median():.2f} (need <= 0.6), "
        f"realign - readout {(r_re - r_ro).median():+.2f} (need >= 0.2); units with gap>=0.05: {len(r_ro)}; "
        f"linear readout recovery {rec('readout_linear').median():.2f}")
    if d in passed:
        say(f"  VERDICT {d}: H1 {'PASS' if h1['R2'] >= 0.8 and h1['MAE'] <= 0.06 else 'FAIL'} | "
            f"H2 {'PASS' if h2['R2'] >= 0.7 and h2['MAE'] <= 0.08 else 'FAIL'} | "
            f"H3 {'PASS' if len(r_ro) and r_ro.median() <= 0.6 and (r_re - r_ro).median() >= 0.2 else 'FAIL'}")
P3c = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "runs", "p3_*_p3.csv"))], ignore_index=True)
say(f"\n[reported] no-replay S retention without repair: P3 benchmarks median {P3c[(P3c.arm == 'no_repair') & (P3c.regime == 'none')].ret_S.median():.2f} | "
    + " | ".join(f"{d} {C[(C.data == d) & (C.arm == 'no_repair') & (C.regime == 'none')].ret_S.median():.2f}" for d in sorted(set(C.data))))
open(os.path.join(ROOT, "paper", "results_p4_synergy_rich.txt"), "w").write("\n".join(lines) + "\n")
