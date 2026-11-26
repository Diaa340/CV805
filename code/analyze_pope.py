"""Pre-registered analysis of POPE (paper/preregistration_pope.md). Writes paper/results_pope.txt"""
import os, re, subprocess
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from pid_utils import mmi
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "pope"); lines = []
def say(s=""): print(s); lines.append(s)
Z = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); T, Tp = Z["T"], Z["T_pope"]
say(f"=== A (descriptive): exact table ===\n  full table 80 x 500: yes {T.mean():.3f}, exact MMI synergy {mmi(T):.3f} bits\n"
    f"  POPE cells ({int((Tp >= 0).sum())} unique): yes {Tp[Tp >= 0].mean():.3f}, exact MMI synergy {mmi(Tp):.3f} bits (additive-fit error: see pope_build.py output)")
D = pd.read_csv(os.path.join(OUT, "vlm_pope.csv")); D["err"] = ((D.logodds > 0) != D.truth).astype(int)
say(f"\n=== VLM (Qwen2.5-VL-7B) on all 9,000 POPE questions ===\n  accuracy {1 - D.err.mean():.3f} | " + " | ".join(f"{s} {1 - x.err.mean():.3f}" for s, x in D.groupby("split")))
say(f"  error on additive / non-additive cells (POPE-cell fit): {D[D.nonadd_pope == 0].err.mean():.3f} / {D[D.nonadd_pope == 1].err.mean():.3f} (n non-additive {int(D.nonadd_pope.sum())})")
n = len(D); M = sparse.hstack([sparse.csr_matrix(D.nonadd_pope.values[:, None].astype(float)), sparse.csr_matrix((np.ones(n), (np.arange(n), D.obj))), sparse.csr_matrix((np.ones(n), (np.arange(n), D.img)))]).tocsr()
say(f"  [reported] error odds ratio on non-additive cells with object and image fixed effects: {np.exp(LogisticRegression(C=10.0, max_iter=3000).fit(M, D.err).coef_[0, 0]):.2f}")
S = np.load(os.path.join(OUT, "vlm_subtable.npz")); L, Ts = S["L"], S["T"]
bal = lambda pred, y: np.mean([(pred[y == c] == c).mean() for c in (0, 1)])
E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean(); Tt = np.where(Ts == 1, 1.0, -1.0); Et = Tt.mean(1, keepdims=True) + Tt.mean(0, keepdims=True) - Tt.mean()
from step2_train import additive_pred
A = additive_pred(Ts)
rng = np.random.default_rng(0); g = []
for _ in range(2000):
    r = rng.integers(0, L.shape[0], L.shape[0]); Lb, Tb = L[r], Ts[r]; Eb = Lb.mean(1, keepdims=True) + Lb.mean(0, keepdims=True) - Lb.mean()
    g.append(bal((Lb > 0).astype(int), Tb) - bal((Eb > 0).astype(int), Tb))
gap = bal((L > 0).astype(int), Ts) - bal((E > 0).astype(int), Ts); lo, hi = np.percentile(g, [2.5, 97.5])
say(f"\n=== B (primary): EMAP on the complete 80 x 100 sub-table (balanced accuracy) ===\n  VLM {bal((L > 0).astype(int), Ts):.3f} | EMAP(VLM) {bal((E > 0).astype(int), Ts):.3f} | "
    f"best additive fit {bal(A, Ts):.3f} | EMAP(truth) {bal((Et > 0).astype(int), Ts):.3f}  (plain accuracy: VLM {((L > 0) == Ts).mean():.3f}, EMAP(VLM) {((E > 0) == Ts).mean():.3f}, additive {(A == Ts).mean():.3f})")
VB = "VLM FUSES ON POPE" if gap >= 0.05 and lo > 0 else ("NO FUSION" if hi < 0.02 else "MIXED")
say(f"  VLM - EMAP(VLM) {gap:+.3f} [95% bootstrap over objects {lo:+.3f}, {hi:+.3f}] -> {VB}")
R = pd.read_csv(os.path.join(OUT, "train.csv")).mean(numeric_only=True)
VC = "PASS" if abs(R.acc_add - R.ceiling) <= 0.03 else "FAIL"
say(f"\n=== C: trained additive model (CLIP) vs exact ceiling on POPE test cells ===\n  additive {R.acc_add:.3f} | exact ceiling {R.ceiling:.3f} | full {R.acc_full:.3f} | "
    f"linear readout {R.acc_lin:.3f} | majority {R.acc_maj:.3f} -> |additive - ceiling| {abs(R.acc_add - R.ceiling):.3f} (<= 0.03): {VC}")
def pid(path):
    out = subprocess.run([os.path.join(ROOT, ".venv_xiu", "bin", "python"), "batch_vlm_final_pope_drop.py", "--directory", path, "--file_name", "qwen25_7.json"],
                         cwd=os.path.join(ROOT, "code", "xiu"), capture_output=True, text=True).stdout
    return {k: float(re.search(rf"{k}: ([0-9.eE+-]+)", out).group(1)) for k in ("Redundancy", "Unique1", "Unique2", "Synergy")}
say("\n=== D: decision-level PID of the VLM with Xiu et al.'s pipeline (BATCH; POPE adversarial, 3:1 split) ===")
pn, pr = pid("results/embeds/pope/"), pid("results/embeds_real/pope/")
for nm, p in (("Gaussian-noise removal (theirs)", pn), ("random-real-sample removal (ours)", pr)):
    tot = sum(p.values()); say(f"  {nm}: R {p['Redundancy']:.3f} U_vis {p['Unique1']:.3f} U_lang {p['Unique2']:.3f} S {p['Synergy']:.3f} bits (S share {p['Synergy'] / tot:.2f})")
dS = abs(pn["Synergy"] - pr["Synergy"]); sh = lambda p: p["Synergy"] / sum(p.values())
VD = "REMOVAL METHOD MATTERS" if dS > 0.1 or (sh(pn) - 0.5) * (sh(pr) - 0.5) < 0 else "ROBUST TO REMOVAL METHOD"
say(f"  |S difference| {dS:.3f} bits; S share {sh(pn):.2f} vs {sh(pr):.2f} -> {VD}")
if os.path.exists(os.path.join(OUT, "liang.csv")):
    Lg = pd.read_csv(os.path.join(OUT, "liang.csv")).iloc[0]; say(f"  [reported] task-level clustered PID (Liang, CLIP, k = 30): R {Lg.R:.3f} U_img {Lg.U_img:.3f} U_q {Lg.U_q:.3f} S {Lg.S:.3f}; exact MMI synergy of POPE cells {mmi(Tp):.3f}")
say(f"\nVERDICTS: B {VB} | C {VC} | D {VD}")
open(os.path.join(ROOT, "paper", "results_pope.txt"), "w").write("\n".join(lines) + "\n")
