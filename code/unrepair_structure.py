"""Per-cell structure labels for the unrepairability test (paper/preregistration_unrepair.md). Outcome-free.
non-additive cell = the exact best additive rule (MILP, identical columns merged) gets it wrong, computed on the cells
that are evaluated: CLEVR units (10 questions x 300 test images, step-2 split), GQA units (10 x 50 test images, seed 92),
POPE (POPE's questions on its 125 test images, seed 71, weighted by multiplicity). Also the local synergy-minus-
redundancy of each cell under the same cell distribution: log p(y) - log p(y | row) - log p(y | column).
Writes runs/unrepair/structure.npz"""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_synred(T, W):
    """Pointwise (i(x1x2;y) - i(x1;y) - i(x2;y)) for deterministic labels: log p(y) - log p(y|r) - log p(y|c)."""
    W = np.where(T >= 0, W, 0.0); tot = W.sum(); out = np.full(T.shape, np.nan)
    py = {y: W[T == y].sum() / tot for y in (0, 1)}
    pr = {y: (W * (T == y)).sum(1) / np.maximum(W.sum(1), 1e-12) for y in (0, 1)}; pc = {y: (W * (T == y)).sum(0) / np.maximum(W.sum(0), 1e-12) for y in (0, 1)}
    for y in (0, 1):
        m = T == y; r, c = np.where(m)
        out[r, c] = np.log2(py[y]) - np.log2(np.maximum(pr[y][r], 1e-12)) - np.log2(np.maximum(pc[y][c], 1e-12))
    return out


def job(a):
    from exact_additive import exact_partial_fit
    key, T, W, tl = a; P, best, up, msg = exact_partial_fit(T, W, tl)
    return key, np.where(T >= 0, (P != T).astype(int), -1), local_synred(T, W), best, up


if __name__ == "__main__":
    jobs = []
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); U = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep").unit.tolist()
    ite = np.sort(np.random.default_rng(14).permutation(1000)[700:])
    for u in U:
        T = Z[f"T__{u}"][:, ite].astype(int); jobs.append((f"clevr__{u}", T, np.ones(T.shape), 300))
    G = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz")); GU = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep").unit.tolist()
    for u in GU:
        T = G[f"T__{u}"].astype(int); te = np.sort(np.random.default_rng(92).permutation(T.shape[1])[150:]); jobs.append((f"gqa__{u}", T[:, te], np.ones((10, len(te))), 300))
    P = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); C = P["cells"]; tr = set(np.random.default_rng(71).permutation(500)[:375])
    W = np.zeros(P["T"].shape)
    for o, k, y in C:
        if k not in tr: W[o, k] += 1
    jobs.append(("pope__all", np.where(W > 0, P["T_pope"], -1).astype(int), W, 1800))
    out = {}
    with Pool(12) as p:
        for key, N, L, best, up in p.imap_unordered(job, jobs):
            out[f"nonadd__{key}"] = N; out[f"synred__{key}"] = L; out[f"bounds__{key}"] = np.array([best, up]); print(key, round(best, 4), round(up, 4), flush=True)
    np.savez(os.path.join(ROOT, "runs", "unrepair", "structure.npz"), clevr_test_imgs=ite, **out)
