"""Candidate sub-tables for the decorrelated test (paper/preregistration_vlm3_decor.md). Outcome-free: uses only the
answer tables and their exact labels, never model outputs.
From each 10 x 100 grid, 27 candidates of 50 images: image weights exp(a * z(exceptions in the column) + b * z(yes in
the column)) for a, b in {-2, 0, 2}, three seeds each (seed = 700 + 9 * grid index + 3 * tilt index + rep). Per
candidate: exact additive rule (MILP, 300 s; s = exception share; proven bound kept), BROJA (convex solver, SCS) and MMI.
Writes runs/vlm3/candidates.csv. BROJA needs .venv_pid; run as: python vlm3_subsets.py (main env) then
.venv_pid/bin/python vlm3_subsets.py --broja."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "vlm3")


def candidates():
    G = np.load(os.path.join(OUT, "grids.npz")); keys = sorted(k[3:] for k in G.files if k.startswith("T__")); out = []
    for gi, k in enumerate(keys):
        T, E = G[f"T__{k}"], G[f"exc__{k}"]; z = lambda v: (v - v.mean()) / (v.std() + 1e-9); ze, zy = z(E.sum(0).astype(float)), z(T.sum(0).astype(float))
        for ti, (a, b) in enumerate([(a, b) for a in (-2, 0, 2) for b in (-2, 0, 2)]):
            w = np.exp(a * ze + b * zy); w /= w.sum()
            for rep in range(3):
                cols = np.sort(np.random.default_rng(700 + 9 * gi + 3 * ti + rep).choice(100, 50, replace=False, p=w))
                out.append(dict(cid=f"{k}#{a}{b}{rep}", key=k, bench=k.split("__")[0], a=a, b=b, rep=rep, cols=cols.tolist()))
    return out, G


def milp(a):
    from exact_additive import exact_partial_fit
    from pid_utils import mmi
    cid, T = a; P, bst, up, m = exact_partial_fit(T, np.ones(T.shape), 300)
    return cid, float((P != T).mean()), 1 - up, bool(abs(up - bst) < 1e-6), mmi(T), float(T.mean())


def broja(a):
    import cvxpy as cp
    from clevr_liang_sweep import pid_bits_with
    cid, T = a; P = np.zeros(T.shape + (2,)); P[np.arange(T.shape[0])[:, None], np.arange(T.shape[1])[None], T] = 1.0 / T.size
    try:
        return cid, float(pid_bits_with(P, cp.SCS)["S"])
    except Exception:
        return cid, float("nan")


if __name__ == "__main__":
    C, G = candidates(); T = {c["cid"]: G[f"T__{c['key']}"][:, c["cols"]].astype(int) for c in C}
    if "--broja" in sys.argv:
        with Pool(12) as p:
            res = dict(p.map(broja, list(T.items()), chunksize=4))
        json.dump(res, open(os.path.join(OUT, "candidates_broja.json"), "w")); print(len(res), "BROJA;", sum(np.isnan(v) for v in res.values()), "failed")
    else:
        import pandas as pd
        with Pool(12) as p:
            R = {cid: r for cid, *r in p.imap_unordered(milp, list(T.items()), chunksize=4)}
        D = pd.DataFrame(C); D[["s", "s_low", "exact", "mmi", "yes"]] = [R[c] for c in D.cid]; D["cols"] = D.cols.map(json.dumps)
        D.to_csv(os.path.join(OUT, "candidates.csv"), index=False); print(len(D), "candidates; proven exact", D.exact.mean().round(3))
