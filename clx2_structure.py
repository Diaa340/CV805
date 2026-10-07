"""Exact per-cell additive labels on TRAINING cells for the CL-exceptions study (paper/preregistration_clx.md).
Outcome-free. 40 NEW CLEVR units (not used in the first study; stratified, seed 6); training cells = 10 questions x 700 train images
(step-2 split). Writes runs/clx/train_structure.npz"""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))
Z = dict(np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz"))); FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep")
prev = set(np.load(os.path.join(ROOT, "runs", "clx", "train_structure.npz"))["units"]); FU = FU[~FU.unit.isin(prev)]; rng = np.random.default_rng(6); sel = []
for f, x in FU.groupby("final"):
    n = max(1, round(40 * len(x) / len(FU))); sel += list(rng.choice(x.unit.values, min(n, len(x)), replace=False))
sel = sorted(sel)[:40] if len(sel) >= 40 else sorted(sel)
itr = np.sort(np.random.default_rng(14).permutation(1000)[:700])


def job(u):
    from exact_additive import exact_partial_fit
    T = Z[f"T__{u}"][:, itr].astype(int); P, b, up, m = exact_partial_fit(T, None, 300)
    return u, (P != T).astype(int), b, up


if __name__ == "__main__":
    out = {}
    with Pool(12) as p:
        for u, N, b, up in p.imap_unordered(job, sel):
            out[f"nonadd__{u}"] = N; out[f"bounds__{u}"] = np.array([b, up]); print(u, round(b, 4), round(up, 4), flush=True)
    np.savez(os.path.join(ROOT, "runs", "clx2", "train_structure.npz"), units=np.array(sel), train_imgs=itr, **out)
