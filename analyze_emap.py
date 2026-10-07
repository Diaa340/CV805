"""Pre-registered analysis of EMAP (paper/preregistration_emap.md). Writes paper/results_emap.txt"""
import os
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
ROOT = os.path.dirname(os.path.abspath(__file__)); lines = []
def say(s=""): print(s); lines.append(s)
V = {}
p = os.path.join(ROOT, "runs", "emap_vlm.csv")
if os.path.exists(p):
    D = pd.read_csv(p); rng = np.random.default_rng(0)
    say(f"=== VLM (Qwen2.5-VL-7B) on {len(D)} fully well-posed CLEVR units (10 questions x 100 held-out images) ===")
    say(D.groupby("final")[["acc_vlm", "acc_emap_vlm", "ceiling", "acc_emap_truth"]].mean().round(3).to_string())
    say(f"  means: VLM {D.acc_vlm.mean():.3f} | EMAP(VLM) {D.acc_emap_vlm.mean():.3f} | exact additive ceiling {D.ceiling.mean():.3f} | EMAP(truth) {D.acc_emap_truth.mean():.3f}")
    mae = (D.acc_emap_vlm - D.ceiling).abs().mean(); r = spearmanr(D.acc_emap_vlm, D.ceiling)[0]
    V["M1"] = "PASS" if mae <= 0.05 and r >= 0.6 else "FAIL"
    say(f"  M1 (EMAP of a strong model recovers the exact ceiling): MAE {mae:.3f} (<= 0.05), Spearman {r:+.3f} (>= 0.6) -> {V['M1']}")
    g = D.acc_vlm - D.acc_emap_vlm; bs = [g.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(5000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    V["M2"] = "PASS" if g.mean() >= 0.05 and lo > 0 else "FAIL"
    say(f"  M2 (the VLM uses interactions): VLM - EMAP(VLM) {g.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (>= 0.05, lower > 0) -> {V['M2']}")
    x = D[D.final != "equal_integer"]; say(f"  [reported] without equal_integer: VLM {x.acc_vlm.mean():.3f}, EMAP(VLM) {x.acc_emap_vlm.mean():.3f}, ceiling {x.ceiling.mean():.3f}")
p = os.path.join(ROOT, "runs", "emap_bench.csv")
if os.path.exists(p):
    B = pd.read_csv(p).groupby("data").mean(numeric_only=True); A = pd.read_csv(os.path.join(ROOT, "runs", "additive_est_bench.csv")).groupby("data").mean(numeric_only=True)
    B["emap_gap"] = B.acc_full - B.acc_emap; B["trained_gap"] = A.acc_full - A.acc_add
    say("\n=== Benchmarks: EMAP of the frozen-feature full model vs the trained additive model ===")
    say(B[["acc_full", "acc_emap", "emap_gap", "trained_gap"]].round(3).to_string())
    ok = ((B.emap_gap.abs() <= 0.02) & ((B.emap_gap - B.trained_gap).abs() <= 0.02)).sum()
    V["B1"] = "PASS" if ok >= 6 else "FAIL"; say(f"  B1 (|EMAP gap| <= 0.02 and within 0.02 of the trained gap) in {ok}/7 datasets (need >= 6) -> {V['B1']}")
p = os.path.join(ROOT, "runs", "emap_tables.csv")
if os.path.exists(p):
    S = np.load(os.path.join(ROOT, "runs", "tables", "selected.npz"))
    M = pd.DataFrame(dict(table=range(len(S["T"])), d_add=S["d_add"], dens=S["dens"])).merge(pd.read_csv(os.path.join(ROOT, "runs", "tables", "pid.csv")), on="table")
    E = pd.read_csv(p).groupby("table", as_index=False).mean(numeric_only=True); D = M.merge(E, on="table")
    D["emap_gap"] = (D.acc_full - D.acc_emap) / (D.acc_full - D.acc_maj)
    R = pd.read_csv(os.path.join(ROOT, "runs", "tables", "train.csv")).groupby("table", as_index=False).mean(numeric_only=True)
    D = D.merge(R[["table", "acc_lin"]].rename(columns={"acc_lin": "lin"}), on="table"); D["lin_gap"] = (D.acc_full - D.lin) / (D.acc_full - D.acc_maj)
    def ps(y, x, z):
        ry, rx, rz = rankdata(y), rankdata(x), rankdata(z); Zm = np.c_[np.ones(len(rz)), rz]; res = lambda v: v - Zm @ np.linalg.lstsq(Zm, v, rcond=None)[0]
        return np.corrcoef(res(ry), res(rx))[0, 1]
    a, b = ps(D.emap_gap, D.d_add, D.syn_broja), ps(D.emap_gap, D.syn_broja, D.d_add)
    V["T1"] = "PASS" if a >= 0.5 and a - abs(b) >= 0.3 else ("FAIL (synergy wins)" if b >= a else "MIXED")
    say(f"\n=== Label tables (63): EMAP gap of the full model ===\n  rho(EMAP gap, d_add | BROJA) {a:+.3f} | rho(EMAP gap, BROJA | d_add) {b:+.3f} -> T1 {V['T1']}")
    say(f"  [reported] Spearman(EMAP gap, linear-readout gap) {spearmanr(D.emap_gap, D.lin_gap)[0]:+.3f}; mean EMAP gap {D.emap_gap.mean():.3f}, readout gap {D.lin_gap.mean():.3f}; "
        f"tables with d_add = 0: EMAP gap {D[D.d_add == 0].emap_gap.mean():.3f}")
say("\nVERDICTS: " + " | ".join(f"{k} {v}" for k, v in V.items()))
open(os.path.join(ROOT, "paper", "results_emap.txt"), "w").write("\n".join(lines) + "\n")
