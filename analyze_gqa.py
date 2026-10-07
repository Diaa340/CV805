"""Pre-registered analysis of GQA (paper/preregistration_gqa.md). Writes paper/results_gqa.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "gqa"); lines = []
def say(s=""): print(s); lines.append(s)
U = pd.read_csv(os.path.join(OUT, "units.csv")).query("keep").set_index("unit"); V = {}
say(f"{len(U)} units from {U.family.nunique()} families; yes {U.yes.mean():.3f}; exact MMI synergy {U.syn_mmi_exact.mean():.3f} bits")
R = pd.read_csv(os.path.join(OUT, "train.csv")).groupby("unit").mean(numeric_only=True)
m = (R.acc_add - R.ceil).abs().mean(); V["G1"] = "PASS" if m <= 0.03 else "FAIL"
say(f"\n=== G1: trained additive model (CLIP) vs exact additive ceiling on test cells ===\n  additive {R.acc_add.mean():.3f} | exact ceiling {R.ceil.mean():.3f} | full {R.acc_full.mean():.3f} | "
    f"linear readout {R.acc_lin.mean():.3f} | majority {R.acc_maj.mean():.3f}; MAE(additive, ceiling) {m:.3f} -> {V['G1']}")
D = pd.read_csv(os.path.join(OUT, "vlm.csv")); rng = np.random.default_rng(0)
g = D.bal_vlm - D.bal_emap; bs = [g.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(5000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
V["G2"] = "VLM FUSES" if g.mean() >= 0.05 and lo > 0 else ("NO FUSION" if hi < 0.02 else "MIXED")
mae = (D.bal_emap - D.ceil_bal).abs().mean(); V["G3"] = "PASS" if mae <= 0.05 and spearmanr(D.bal_emap, D.ceil_bal)[0] >= 0.6 else "FAIL"
say(f"\n=== VLM (Qwen2.5-VL-7B), balanced accuracy on complete 10 x 100 tables ===\n" + D.groupby("family")[["bal_vlm", "bal_emap", "ceil_bal", "bal_emap_truth"]].mean().round(3).to_string())
say(f"  means: VLM {D.bal_vlm.mean():.3f} | EMAP(VLM) {D.bal_emap.mean():.3f} | exact balanced ceiling {D.ceil_bal.mean():.3f} | EMAP(truth) {D.bal_emap_truth.mean():.3f}")
say(f"  G2 (VLM uses interactions): VLM - EMAP(VLM) {g.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {V['G2']}")
say(f"  G3 (EMAP of the VLM recovers the exact ceiling): MAE {mae:.3f}, Spearman {spearmanr(D.bal_emap, D.ceil_bal)[0]:+.3f} -> {V['G3']}")
say(f"  [reported] plain accuracy: VLM {D.acc_vlm.mean():.3f}, EMAP {D.acc_emap.mean():.3f}, exact ceiling {D.ceil.mean():.3f}")
say("\nVERDICTS: " + " | ".join(f"{k} {v}" for k, v in V.items()))
open(os.path.join(ROOT, "paper", "results_gqa.txt"), "w").write("\n".join(lines) + "\n")
