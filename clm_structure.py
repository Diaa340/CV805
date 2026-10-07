"""Exact per-cell labels needed by the CL-metrics study (paper/preregistration_clm.md). Outcome-free.
- GQA: training cells of every unit (10 questions x 150 train images, seed-92 split).
- POPE task 1 training cells (POPE questions on the 375 train images, weighted by multiplicity).
- POPE task 2 (COCO cells POPE never asks): its test cells (125 test images) and training cells (375 train images).
Writes runs/clm/structure.npz"""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__))


def job(a):
    from exact_additive import exact_partial_fit
    key, T, W, tl = a; P, b, up, m = exact_partial_fit(T, W, tl)
    return key, np.where(T >= 0, (P != T).astype(int), -1), b, up


if __name__ == "__main__":
    jobs = []
    G = dict(np.load(os.path.join(ROOT, "runs", "gqa", "units.npz"))); GU = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep").unit.tolist()
    for u in GU:
        T = G[f"T__{u}"].astype(int); tr = np.sort(np.random.default_rng(92).permutation(T.shape[1])[:150]); jobs.append((f"gqa_train__{u}", T[:, tr], np.ones((10, 150)), 300))
    P = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); C, Tf = P["cells"], P["T"].astype(int)
    trset = set(np.random.default_rng(71).permutation(500)[:375]); trm = np.array([k in trset for k in range(500)])
    W = np.zeros(Tf.shape)
    for o, k, y in C:
        if trm[k]: W[o, k] += 1
    jobs.append(("pope_t1_train", np.where(W > 0, P["T_pope"], -1).astype(int), W, 1800))
    asked = np.zeros(Tf.shape, bool); asked[C[:, 0], C[:, 1]] = True
    for nm, m in (("pope_t2_test", ~trm), ("pope_t2_train", trm)):
        T2 = np.where(~asked & m[None, :], Tf, -1); Wb = np.where(T2 == 1, 0.5 / max(1, (T2 == 1).sum()), np.where(T2 == 0, 0.5 / max(1, (T2 == 0).sum()), 0))
        jobs.append((nm, T2, Wb, 1800))  # task 2 is trained balanced (50% yes), so its cells are weighted to balance
    out = {}
    with Pool(12) as p:
        for key, N, b, up in p.imap_unordered(job, jobs):
            out[f"nonadd__{key}"] = N; out[f"bounds__{key}"] = np.array([b, up]); print(key, round(b, 4), round(up, 4), flush=True)
    np.savez(os.path.join(ROOT, "runs", "clm", "structure.npz"), **out)
