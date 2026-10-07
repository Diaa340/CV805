"""Liang et al.'s PID estimate on the benchmarks, solved with SCS (ECOS did not finish on 24-101-class problems after
6.5 h). Same pipeline as liang_bench.py (normalize + PCA(5) + KMeans(30, seed 0) per modality, test split).
Usage: python liang_bench2.py <name> [<name> ...]; appends to runs/liang_bench_scs.csv"""
import contextlib, io, os, sys, time
import numpy as np, pandas as pd
import cvxpy as cp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import liang_cvx as L
ROOT = os.path.dirname(os.path.abspath(__file__))
_solve = cp.Problem.solve
def _scs(self, *a, **k):
    k["verbose"] = False; k["solver"] = cp.SCS; k["max_iters"] = 50000; return _solve(self, *a, **k)
cp.Problem.solve = _scs

for name in sys.argv[1:]:
    t0 = time.time(); Z = np.load(os.path.join(ROOT, "data", "feats", f"{name}.npz"), allow_pickle=True); s = Z["split"] == 1
    cl = lambda X: KMeans(n_clusters=30, random_state=0, n_init=10).fit(PCA(n_components=5).fit_transform(normalize(np.nan_to_num(X)))).labels_
    c1, c2 = cl(Z["f1"][s].astype(np.float32)), cl(Z["f2"][s].astype(np.float32)); y = pd.factorize(Z["y"][s])[0]
    P = np.zeros((30, 30, y.max() + 1)); np.add.at(P, (c1, c2, y), 1); P /= P.sum()
    with contextlib.redirect_stdout(io.StringIO()):
        Q = L.solve_Q_new(P)
    f = 1 / np.log(2); H = -(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(y) / len(y))
    r = dict(data=name, solver="SCS", H_y=H, R=L.CoI(Q) * f, U1=L.UI(Q, cond_id=1) * f, U2=L.UI(Q, cond_id=0) * f, S=L.CI(P, Q) * f, sec=time.time() - t0)
    out = os.path.join(ROOT, "runs", "liang_bench_scs.csv"); pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
