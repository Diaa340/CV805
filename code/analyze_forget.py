"""Pre-registered analysis of the VLM forgetting test (paper/preregistration_vlm_forget.md). Writes paper/results_vlm_forget.txt"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2"); lines = []
def say(s=""): print(s); lines.append(s)
tag = sys.argv[1] if len(sys.argv) > 1 else "3b"
D = pd.read_csv(os.path.join(OUT, f"forget_{tag}.csv")); F = pd.read_csv(os.path.join(OUT, "families.csv")).set_index("unit")
D["final"] = D.unit.map(F.final); D["ok0"] = (D.lo_base > 0) == D.truth
levels = sorted(int(c[6:]) for c in D.columns if c.startswith("lo_ocr"))
for L in levels:
    D[f"ok{L}"] = (D[f"lo_ocr{L}"] > 0) == D.truth
arms = [c[7:] for c in D.columns if c.startswith("lo_rep_")]
for a in arms:
    D[f"okrep_{a}"] = (D[f"lo_rep_{a}"] > 0) == D.truth
C = D[D.ok0].copy(); C["conf_bin"] = pd.qcut(C.lo_base.abs().rank(method="first"), 10, labels=False)
say(f"[{tag}] cells {len(D)} (non-additive {int(D.nonadd.sum())}); base accuracy {D.ok0.mean():.3f} (additive {D[D.nonadd == 0].ok0.mean():.3f}, "
    f"non-additive {D[D.nonadd == 1].ok0.mean():.3f}); correct at base {len(C)}")


def matched(X, col):
    """Confidence-decile x question-type matched difference in `col` (non-additive minus additive), weights = n non-additive."""
    num = den = 0.0
    for _, s in X.groupby(["conf_bin", "final"]):
        a, b = s[s.nonadd == 1][col], s[s.nonadd == 0][col]
        if len(a) >= 5 and len(b) >= 5:
            num += len(a) * (a.mean() - b.mean()); den += len(a)
    return num / den if den else np.nan


def boot(X, col, n=500):
    rng = np.random.default_rng(0); ims = X.image_index.unique(); g = {k: v for k, v in X.groupby("image_index")}; out = []
    for _ in range(n):
        out.append(matched(pd.concat([g[k] for k in rng.choice(ims, len(ims))]), col))
    return np.nanpercentile(out, [2.5, 97.5])


P = C[C.final != "equal_integer"]
say("\nretention among cells correct at base (excluding equal_integer): overall | additive | non-additive | matched gap (non-add - add)")
ret = {}
for L in levels:
    ret[L] = P[f"ok{L}"].mean()
    say(f"  OCR {L:5d} steps: {ret[L]:.3f} | {P[P.nonadd == 0][f'ok{L}'].mean():.3f} | {P[P.nonadd == 1][f'ok{L}'].mean():.3f} | {matched(P, f'ok{L}'):+.3f}")
cand = [L for L in levels if 0.50 <= ret[L] <= 0.95]
if not cand:
    V = "INCONCLUSIVE (no drift level with retention in [0.50, 0.95])"
else:
    L = min(cand, key=lambda l: abs(ret[l] - 0.80)); g = matched(P, f"ok{L}"); lo, hi = boot(P, f"ok{L}")
    say(f"\n=== primary (level {L}): matched retention gap {g:+.3f} [95% image bootstrap {lo:+.3f}, {hi:+.3f}] ===")
    V = "FUSION FORGOTTEN FIRST" if g <= -0.03 and hi < 0 else ("KILL: fusion not forgotten first" if hi >= 0 and g > -0.03 else "MIXED")
say(f"  -> {V}")
Lmax = levels[-1]; lost = P[~P[f"ok{Lmax}"]]
say(f"\n=== repair (secondary): among cells lost at {Lmax} steps (n={len(lost)}, non-additive {int(lost.nonadd.sum())}), share regained ===")
for a in arms:
    if len(lost) >= 50:
        g = -matched(lost, f"okrep_{a}"); lo, hi = boot(lost, f"okrep_{a}"); lo, hi = -hi, -lo
        say(f"  {a}: additive {lost[lost.nonadd == 0][f'okrep_{a}'].mean():.3f} | non-additive {lost[lost.nonadd == 1][f'okrep_{a}'].mean():.3f} | "
            f"matched (add - non-add) {g:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'ADDITIVE RECOVERS MORE' if g >= 0.05 and lo > 0 else 'NOT SHOWN'}")
say("\n[reported] equal_integer separately, and raw retention by type at the largest drift:")
E = C[C.final == "equal_integer"]
for L in levels:
    say(f"  equal_integer OCR {L}: additive {E[E.nonadd == 0][f'ok{L}'].mean():.3f} | non-additive {E[E.nonadd == 1][f'ok{L}'].mean():.3f} | matched {matched(E, f'ok{L}'):+.3f}")
for f, x in C.groupby("final"):
    say(f"  {f:15s} retention at {Lmax}: additive {x[x.nonadd == 0][f'ok{Lmax}'].mean():.3f} | non-additive {x[x.nonadd == 1][f'ok{Lmax}'].mean():.3f}")
say(f"\nVERDICT: {V}")
open(os.path.join(ROOT, "paper", f"results_vlm_forget_{tag}.txt"), "w").write("\n".join(lines) + "\n")
