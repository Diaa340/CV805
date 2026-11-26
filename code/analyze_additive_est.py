"""Pre-registered analysis of the additive-model estimator (paper/preregistration_additive_est.md). Writes paper/results_additive_est.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []
def say(s=""): print(s); lines.append(s)
A = pd.read_csv(os.path.join(ROOT, "runs", "additive_est_clevr.csv")).groupby(["unit", "feat"]).acc_add.mean().unstack()
T = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "train.csv")).groupby(["unit", "feat"]).mean(numeric_only=True)
ceil = T.ceil_test.unstack()["sym"]; full = T.acc_full.unstack()
say("=== E1 (primary): on CLEVR, the jointly trained additive model recovers the exact additive ceiling ===")
for f in ("sym", "clip"):
    m = (A[f] - ceil).abs().mean(); r = spearmanr(A[f], ceil)[0]
    say(f"  {f}: MAE(additive, exact ceiling) {m:.3f}; Spearman {r:+.3f}; mean additive {A[f].mean():.3f} vs ceiling {ceil.mean():.3f} vs full {full[f].mean():.3f}")
E1 = (A["sym"] - ceil).abs().mean() <= 0.03 and spearmanr(A["sym"], ceil)[0] >= 0.8
say(f"  -> E1 {'PASS' if E1 else 'FAIL'} (symbolic: MAE <= 0.03 and Spearman >= 0.8)")
say("\n=== E2 (descriptive): benchmarks, fusion gap of frozen-feature models vs Liang et al.'s PID estimate ===")
B = pd.read_csv(os.path.join(ROOT, "runs", "additive_est_bench.csv")).groupby("data").mean(numeric_only=True)
L = pd.read_csv(os.path.join(ROOT, "runs", "liang_bench.csv")).set_index("data"); B = B.join(L)
B["gap"] = B.acc_full - B.acc_add; B["gap_norm"] = B.gap / (B.acc_full - B.acc_maj); B["S_share"] = B.S / B.H_y
say(B[["classes", "acc_maj", "acc_uni1", "acc_uni2", "acc_add", "acc_full", "gap", "gap_norm", "R", "U1", "U2", "S", "S_share"]].round(3).to_string())
say(f"  Spearman across datasets: (gap_norm, S) {spearmanr(B.gap_norm, B.S)[0]:+.2f}; (gap_norm, S share) {spearmanr(B.gap_norm, B.S_share)[0]:+.2f} (n = {len(B)}, descriptive)")
say(f"\nVERDICT: E1 {'PASS' if E1 else 'FAIL'}")
open(os.path.join(ROOT, "paper", "results_additive_est.txt"), "w").write("\n".join(lines) + "\n")
