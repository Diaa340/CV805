"""Identical-PID families (paper/preregistration_samepid.md). Outcome-free. For each of 12 random 10x10 tables, search
margin-preserving interchanges ([[1,0],[0,1]] <-> [[0,1],[1,0]]) to minimise and to maximise the additive distance
(exact MILP). All members of a family share every row and column label count, so their BROJA / MMI PID are identical.
Writes runs/tables/samepid.npz"""
import os
from multiprocessing import Pool
import numpy as np
from step1_build import nearest_ferrers
from tables_build import mmi_synergy, xor_density
ROOT = os.path.dirname(os.path.abspath(__file__))


def climb(args):
    T0, direction, seed = args; rng = np.random.default_rng(seed); T = T0.copy(); cur = nearest_ferrers(T.astype(np.int8))[1]
    for _ in range(600):
        r, c = rng.choice(10, 2, replace=False), rng.choice(10, 2, replace=False)
        a, b, cc, d = T[r[0], c[0]], T[r[0], c[1]], T[r[1], c[0]], T[r[1], c[1]]
        if a == d and b == cc and a != b:
            U = T.copy(); U[r[0], c[0]], U[r[0], c[1]], U[r[1], c[0]], U[r[1], c[1]] = b, a, d, cc
            n = nearest_ferrers(U.astype(np.int8))[1]
            if (n - cur) * direction >= 0:
                T, cur = U, n
    return T


def main():
    rng = np.random.default_rng(77); bases = []
    while len(bases) < 12:
        T = (rng.random((10, 10)) < rng.uniform(0.35, 0.65)).astype(int)
        if 0.3 <= T.mean() <= 0.7:
            bases.append(T)
    jobs = [(T, d, 100 * k + (d > 0)) for k, T in enumerate(bases) for d in (-1, +1)]
    with Pool(12) as p:
        res = p.map(climb, jobs)
    out, fam, kind = [], [], []
    for k, T in enumerate(bases):
        for nm, X in (("min", res[2 * k]), ("orig", T), ("max", res[2 * k + 1])):
            assert (X.sum(0) == T.sum(0)).all() and (X.sum(1) == T.sum(1)).all()
            out.append(X); fam.append(k); kind.append(nm)
    T = np.stack(out).astype(np.int8)
    d_add = np.array([nearest_ferrers(t)[1] for t in T]); dens = np.array([xor_density(t) for t in T]); syn = np.array([mmi_synergy(t) for t in T])
    np.savez(os.path.join(ROOT, "runs", "tables", "samepid.npz"), T=T, family=np.array(fam), kind=np.array(kind), d_add=d_add, dens=dens,
             syn_mmi=syn, F=np.stack([nearest_ferrers(t)[0] for t in T]))
    for k in range(12):
        m = np.array(fam) == k
        print(f"family {k:2d}: d_add {d_add[m].round(2).tolist()} density {dens[m].round(3).tolist()} MMI synergy {syn[m].round(3).tolist()}")


if __name__ == "__main__":
    main()
