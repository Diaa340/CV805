"""Pre-registered analysis of step 2 (paper/preregistration_step2.md). Writes paper/results_step2.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2"); lines = []
def say(s=""): print(s); lines.append(s)
U = pd.read_csv(os.path.join(OUT, "families.csv")); U = U[U.keep]
R = pd.read_csv(os.path.join(OUT, "train.csv")).groupby(["unit", "feat"], as_index=False).mean(numeric_only=True)
P = pd.read_csv(os.path.join(OUT, "pid.csv"))
W = R.pivot(index="unit", columns="feat", values=["acc_full", "acc_lin", "acc_maj", "ceil_test"])
U.columns = pd.MultiIndex.from_tuples([(c, "") for c in U.columns])  # bug fix (pandas join of 1- and 2-level columns)
D = U.set_index(("unit", "")).join(W, how="inner"); D.index.name = "unit"
D.columns = [c if c[1] else c[0] for c in D.columns]
for f in ("sym", "clip"):
    D[f"gap_{f}"] = D[("acc_full", f)] - D[("acc_lin", f)]
    D = D.join(P[P.feat == f].set_index("unit")[["syn_broja", "syn_ccs"]].add_suffix(f"_{f}"))
D["ceil"] = D[("ceil_test", "sym")]; D["dadd_test"] = 1 - D.ceil
say(f"units {len(D)} from {D.family.nunique()} families ({D.final.value_counts().to_dict()})")
say(f"collinearity: Spearman(dens, MMI synergy) {spearmanr(D.dens, D.syn_mmi_exact)[0]:+.2f}; (d_add, MMI) {spearmanr(D.d_add_hat, D.syn_mmi_exact)[0]:+.2f}")
V = {}
say("\n=== S2-A (primary): the table's additive ceiling predicts the linear readout (symbolic features) ===")
r = spearmanr(D.gap_sym, D.dadd_test)[0]
hi = D[D[("acc_full", "sym")] >= 0.90]; mae = (hi[("acc_lin", "sym")] - hi.ceil).abs().mean() if len(hi) else np.nan
say(f"  (i) Spearman(full - linear, 1 - ceiling) = {r:+.3f} (need >= 0.6)")
say(f"  (ii) units with full >= 0.90: {len(hi)}; MAE(linear, ceiling) = {mae:.3f} (need <= 0.05); mean linear {hi[('acc_lin', 'sym')].mean():.3f} vs ceiling {hi.ceil.mean():.3f}")
V["S2-A"] = "PASS" if r >= 0.6 and len(hi) >= 5 and mae <= 0.05 else "FAIL"
say(f"  -> {V['S2-A']}")
say("\n=== S2-B (primary): the tasks need fusion; frozen CLIP models learn only the additive part ===")
s_ok = (D[("acc_full", "sym")] >= D.ceil + 0.03).mean()
L = D[D[("acc_full", "clip")] - D[("acc_maj", "clip")] >= 0.05]; c_ok = (L[("acc_full", "clip")] <= L.ceil + 0.02).mean() if len(L) else np.nan
say(f"  symbolic full model above the additive ceiling (+0.03) in {100 * s_ok:.0f}% of units (need >= 70%)")
say(f"  CLIP: learnable units {len(L)}/{len(D)}; full model at or below the ceiling (+0.02) in {100 * c_ok:.0f}% (need >= 70%)")
say(f"  means: ceiling {D.ceil.mean():.3f} | symbolic full {D[('acc_full', 'sym')].mean():.3f} linear {D[('acc_lin', 'sym')].mean():.3f} | "
    f"CLIP full {D[('acc_full', 'clip')].mean():.3f} linear {D[('acc_lin', 'clip')].mean():.3f} | majority {D[('acc_maj', 'sym')].mean():.3f}")
V["S2-B"] = "PASS" if s_ok >= 0.7 and len(L) >= 5 and c_ok >= 0.7 else "FAIL"
say(f"  -> {V['S2-B']}")
say("\n[reported] S2-C: PID estimates vs exact structure (Spearman across units)")
for s in ("syn_mmi_exact", "syn_broja_sym", "syn_ccs_sym", "syn_broja_clip", "syn_ccs_clip"):
    say(f"  {s:15s} vs d_add {spearmanr(D[s], D.d_add_hat)[0]:+.2f} | vs density {spearmanr(D[s], D.dens)[0]:+.2f} | "
        f"vs symbolic gap {spearmanr(D[s], D.gap_sym)[0]:+.2f} | mean {D[s].mean():.3f}")
def pspear(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z); Zm = np.c_[np.ones(len(rz)), rz]
    res = lambda v: v - Zm @ np.linalg.lstsq(Zm, v, rcond=None)[0]
    return np.corrcoef(res(ry), res(rx))[0, 1]
say(f"  partial (symbolic gap): rho(gap, d_add | BROJA_sym) {pspear(D.gap_sym, D.d_add_hat, D.syn_broja_sym):+.2f}; "
    f"rho(gap, BROJA_sym | d_add) {pspear(D.gap_sym, D.syn_broja_sym, D.d_add_hat):+.2f} (high collinearity: not decisive)")
say("\n[reported] by final function:")
say(D.groupby("final")[["dens", "d_add_hat", "syn_mmi_exact", "ceil", "gap_sym", "gap_clip"]].mean().round(3).to_string())
say("\nVERDICTS: " + " | ".join(f"{k} {v}" for k, v in V.items()))
open(os.path.join(ROOT, "paper", "results_step2.txt"), "w").write("\n".join(lines) + "\n")
