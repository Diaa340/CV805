"""Pre-registered analysis of the VLM scale test (paper/preregistration_vlm.md). Writes paper/results_vlm.txt"""
import os, sys
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2"); lines = []
def say(s=""): print(s); lines.append(s)
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen2.5-VL-7B-Instruct"
D = pd.read_csv(os.path.join(OUT, f"vlm_{name}.csv")); U = pd.read_csv(os.path.join(OUT, "families.csv")).set_index("unit")
D["final"] = D.unit.map(U.final); D["qid"] = D.unit + "_" + D.q.astype(str)
bal = np.mean([(D.pred[D.truth == c] == c).mean() for c in (0, 1)])
say(f"{name}: {len(D)} cells, {D.unit.nunique()} units; accuracy {1 - D.err.mean():.3f}, balanced {bal:.3f}; non-additive share {D.nonadd.mean():.3f}")
say(f"raw error: additive cells {D[D.nonadd == 0].err.mean():.3f} (n={int((D.nonadd == 0).sum())}), non-additive {D[D.nonadd == 1].err.mean():.3f} (n={int(D.nonadd.sum())})")


def fe_or(X):
    q = pd.factorize(X.qid)[0]; i = pd.factorize(X.image_index)[0]; n = len(X)
    M = sparse.hstack([sparse.csr_matrix(X.nonadd.values[:, None].astype(float)), sparse.csr_matrix((np.ones(n), (np.arange(n), q))),
                       sparse.csr_matrix((np.ones(n), (np.arange(n), i)))]).tocsr()
    m = LogisticRegression(C=10.0, max_iter=3000).fit(M, X.err.values)
    return float(np.exp(m.coef_[0, 0]))


if bal < 0.60:
    V = "INCONCLUSIVE (balanced accuracy < 0.60)"
else:
    o = fe_or(D); rng = np.random.default_rng(0); ims = D.image_index.unique(); bs = []
    for _ in range(200):
        pick = rng.choice(ims, len(ims)); X = pd.concat([D[D.image_index == k].assign(image_index=f"{k}_{n}") for n, k in enumerate(pick)])
        bs.append(fe_or(X))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    say(f"\n=== V1 (primary): odds ratio of an error on non-additive cells, with question and image fixed effects ===")
    say(f"  OR = {o:.2f} [95% image-bootstrap {lo:.2f}, {hi:.2f}]")
    V = ("ERRORS CONCENTRATE ON NON-ADDITIVE CELLS" if o >= 1.5 and lo > 1.0 else
         ("NO SPECIFIC FAILURE" if lo >= 0.67 and hi <= 1.5 else "MIXED"))
say(f"  -> {V}")
say("\n[reported] by final function (error additive / non-additive, n non-additive):")
for f, x in D.groupby("final"):
    say(f"  {f:15s} {x[x.nonadd == 0].err.mean():.3f} / {x[x.nonadd == 1].err.mean():.3f} (n={int(x.nonadd.sum())})")
say(f"\n[reported] share of non-additive-cell errors: the VLM gives the additive rule's answer by construction (binary); "
    f"mean |log-odds| additive {D[D.nonadd == 0].logodds_yes.abs().mean():.2f} vs non-additive {D[D.nonadd == 1].logodds_yes.abs().mean():.2f}")
say(f"\nVERDICT: {V}")
open(os.path.join(ROOT, "paper", f"results_vlm_{name}.txt"), "w").write("\n".join(lines) + "\n")
