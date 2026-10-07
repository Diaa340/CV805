"""Robustness of the PID-estimator finding (paper/preregistration_estimator.md): Liang et al.'s released CVX estimator
(liang_cvx.py, verbatim; ECOS) on the 103 CLEVR units with their clustering pipeline (normalize + PCA(5) + KMeans, as in
their notebook) and variations: k in {5, 10, 30 (their default), 100} image clusters, PCA on / off, symbolic or CLIP
features. Questions (10 per unit) are clustered into min(k, 10). Joint distribution over all 10 x 1,000 cells.
Writes runs/clevr2/liang_sweep.csv"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clevr2")
Z = dict(np.load(os.path.join(OUT, "data.npz"))); UNITS = pd.read_csv(os.path.join(OUT, "families.csv")).query("keep").unit.tolist()


def clustering(X, pca, k):  # their clustering() with a fixed seed
    X = np.nan_to_num(X)
    if pca:
        X = PCA(n_components=5).fit_transform(normalize(X))
    return KMeans(n_clusters=k, random_state=0, n_init=10).fit(X).labels_


IMGC = {(f, p, k): clustering(Z["Ximg"] if f == "sym" else Z["Cimg"], p, k) for f in ("sym", "clip") for p in (True, False) for k in (5, 10, 30, 100)}


def build(a):
    u, f, p, k = a; T = Z[f"T__{u}"]; Xq = Z[f"Xq__{u}"] if f == "sym" else Z[f"Cq__{u}"]
    cq = clustering(Xq, p and Xq.shape[1] > 5, min(k, 10)) if k < 10 else np.arange(10); ci = IMGC[(f, p, k)]
    P = np.zeros((cq.max() + 1, ci.max() + 1, 2))
    for q in range(T.shape[0]):
        np.add.at(P, (cq[q], ci, T[q]), 1)
    return P / P.sum()


def job(item):
    import cvxpy as cp
    from liang_pid import pid_bits
    (u, f, p, k), P = item
    for solver in (cp.ECOS, cp.SCS):  # ECOS (their option set); fall back to SCS if ECOS fails; else record missing
        try:
            import liang_pid; cp.Problem.solve.__defaults__  # noqa
            r = pid_bits_with(P, solver); return dict(unit=u, feat=f, pca=p, k=k, solver=solver, **{a_: float(v) for a_, v in r.items()})
        except Exception:
            continue
    return dict(unit=u, feat=f, pca=p, k=k, solver="failed", R=np.nan, U1=np.nan, U2=np.nan, S=np.nan)


def pid_bits_with(P, solver):
    import contextlib, io
    import cvxpy as cp
    import liang_cvx as L
    orig = cp.Problem.solve
    def patched(self, *a, **k):
        k["verbose"] = False; k["solver"] = solver
        if solver == cp.SCS:
            k.pop("max_iters", None); k["max_iters"] = 50000
        return orig(self, *a, **k)
    import liang_pid  # noqa: F401  (ensures liang_cvx is importable)
    cp.Problem.solve = patched
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            Q = L.solve_Q_new(P)
    finally:
        cp.Problem.solve = orig
    f = 1 / np.log(2)
    return dict(R=L.CoI(Q) * f, U1=L.UI(Q, cond_id=1) * f, U2=L.UI(Q, cond_id=0) * f, S=L.CI(P, Q) * f)


if __name__ == "__main__":
    ks = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [5, 10, 30]
    jobs = [(u, f, p, k) for k in ks for f in ("sym", "clip") for p in (True, False) for u in UNITS]
    out = os.path.join(OUT, "liang_sweep.csv")
    done = set(map(tuple, pd.read_csv(out)[["unit", "feat", "pca", "k"]].astype(str).values)) if os.path.exists(out) else set()
    jobs = [a for a in jobs if tuple(map(str, a)) not in done]
    items = [(a, build(a)) for a in jobs]  # all clustering in the parent; workers only run the CVX solver (no fork-after-OpenMP)
    out = os.path.join(OUT, "liang_sweep.csv")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(12) as pool:
        for n, r in enumerate(pool.imap_unordered(job, items)):
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            if n % 100 == 0:
                print(n, len(items), flush=True)
