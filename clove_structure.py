"""Exact per-cell labels for the CLOVE-scene units (paper/preregistration_clove.md). Outcome-free.
Per unit: exact additive rule (MILP, 300 s) on the training table (10 x 150 train images) and on the test table
(10 x 50 val images). Writes runs/clove/structure.npz"""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))


def job(a):
    from exact_additive import exact_partial_fit
    key, T, W, tl = a; P, b, up, m = exact_partial_fit(T, W, tl)
    return key, (P != T).astype(int), b, up


if __name__ == "__main__":
    U = dict(np.load(os.path.join(ROOT, "runs", "clove", "units.npz"))); K = pd.read_csv(os.path.join(ROOT, "runs", "clove", "units.csv")).query("keep").unit.tolist()
    jobs = [(f"{sp}__{u}", U[f"T__{u}"][:, sl].astype(int), np.ones((10, n)), 300) for u in K for sp, sl, n in (("train", slice(0, 150), 150), ("test", slice(150, 200), 50))]
    out = {}
    with Pool(12) as p:
        for key, N, b, up in p.imap_unordered(job, jobs):
            out[f"nonadd__{key}"] = N; out[f"bounds__{key}"] = np.array([b, up]); print(key, round(b, 4), round(up, 4), flush=True)
    np.savez(os.path.join(ROOT, "runs", "clove", "structure.npz"), **out)
