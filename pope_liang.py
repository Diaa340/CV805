"""Liang et al.'s clustered PID (their default: normalize + PCA(5) + KMeans(30), CVX via SCS) on POPE's 9,000
questions with CLIP features (task-level PID of the dataset). Run with .venv_pid. Writes runs/pope/liang.csv"""
import contextlib, io, os
import numpy as np, pandas as pd, cvxpy as cp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import liang_cvx as L
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "pope")
_s = cp.Problem.solve
cp.Problem.solve = lambda self, *a, **k: _s(self, *a, **{**k, "verbose": False, "solver": cp.SCS, "max_iters": 50000})
Z = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); C = Z["cells"]; F = np.load(os.path.join(OUT, "clip.npz"))
cl = lambda X, k: KMeans(n_clusters=k, random_state=0, n_init=10).fit(PCA(n_components=5).fit_transform(normalize(X))).labels_
ci, cq = cl(F["img"], 30), cl(F["q"], 30)
P = np.zeros((30, 30, 2)); np.add.at(P, (ci[C[:, 1]], cq[C[:, 0]], C[:, 2]), 1); P /= P.sum()
with contextlib.redirect_stdout(io.StringIO()):
    Q = L.solve_Q_new(P)
f = 1 / np.log(2); r = dict(R=L.CoI(Q) * f, U_img=L.UI(Q, cond_id=1) * f, U_q=L.UI(Q, cond_id=0) * f, S=L.CI(P, Q) * f)
pd.DataFrame([r]).to_csv(os.path.join(OUT, "liang.csv"), index=False); print(r)
