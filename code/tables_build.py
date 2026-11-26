"""Label-table kill test, step 1 (paper/preregistration_tables.md): build a pool of 10x10 binary label tables over class
IDs, compute cheap structure/synergy predictors, select 60 tables that decorrelate synergy from XOR-square density.
Outcome-free (no training). Writes runs/tables/pool.npz and runs/tables/selected.npz"""
import itertools, os
import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); K = 10
PAIRS = [(r, c) for r in itertools.combinations(range(K), 2) for c in itertools.combinations(range(K), 2)]


def xor_density(T):
    """Share of 2x2 submatrices (2 rows x 2 cols) that are checkerboards (XOR squares)."""
    n = 0
    for (r1, r2), (c1, c2) in PAIRS:
        n += T[r1, c1] == T[r2, c2] != T[r1, c2] == T[r2, c1]
    return n / len(PAIRS)


def additive_distance(T, M=201.0):
    """Fewest cells to flip so that T = [a_i + b_j > 0] (a Ferrers / threshold-additive table): exact MILP."""
    nv = 2 * K + K * K; c = np.r_[np.zeros(2 * K), np.ones(K * K)]; A = np.zeros((K * K, nv)); lb = np.zeros(K * K); ub = np.zeros(K * K)
    for i in range(K):
        for j in range(K):
            r = i * K + j; A[r, i] = 1; A[r, K + j] = 1
            if T[i, j]:
                A[r, 2 * K + r] = M; lb[r], ub[r] = 1, np.inf       # a_i + b_j + M z >= 1
            else:
                A[r, 2 * K + r] = -M; lb[r], ub[r] = -np.inf, -1    # a_i + b_j - M z <= -1
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.r_[np.zeros(2 * K), np.ones(K * K)],
               bounds=Bounds(np.r_[-50 * np.ones(2 * K), np.zeros(K * K)], np.r_[50 * np.ones(2 * K), np.ones(K * K)]))
    return round(res.fun) / (K * K)


def mmi_synergy(T):
    """MMI-PID synergy (bits) under uniform independent class IDs: I(X1X2;Y) - max(I(X1;Y), I(X2;Y))."""
    H = lambda p: -sum(q * np.log2(q) for q in (p, 1 - p) if q > 0)
    p = T.mean(); I12 = H(p); I1 = H(p) - np.mean([H(r) for r in T.mean(1)]); I2 = H(p) - np.mean([H(r) for r in T.mean(0)])
    return I12 - max(I1, I2)


def pool(seed=2026):
    rng = np.random.default_rng(seed); out = []
    def add(T, fam):
        if 0.25 <= T.mean() <= 0.75 and len({tuple(r) for r in T}) > 1 and len({tuple(c) for c in T.T}) > 1:
            out.append((T.astype(np.int8), fam))
    def ferrers():
        a, b = rng.normal(size=K) * rng.uniform(0.3, 2), rng.normal(size=K) * rng.uniform(0.3, 2)
        s = a[:, None] + b[None]; return (s > np.quantile(s, rng.uniform(0.25, 0.75))).astype(int)
    for _ in range(300):
        add(ferrers(), "ferrers")
    for _ in range(100):  # AND / OR of class attributes (Ferrers)
        A, B = rng.random(K) < rng.uniform(0.3, 0.9), rng.random(K) < rng.uniform(0.3, 0.9)
        add((A[:, None] & B[None]) if rng.random() < 0.5 else (A[:, None] | B[None]), "and_or")
    for _ in range(600):  # Ferrers with m flipped cells
        T = ferrers(); m = rng.integers(1, 41); idx = rng.choice(K * K, m, replace=False); T.flat[idx] ^= 1; add(T, "ferrers_flip")
    for _ in range(200):
        add((rng.random((K, K)) < rng.uniform(0.3, 0.7)).astype(int), "random")
    for _ in range(400):  # XOR of attributes on a sub-block, Ferrers or constant elsewhere
        k1, k2 = rng.integers(2, K + 1, 2); R, C = rng.choice(K, k1, replace=False), rng.choice(K, k2, replace=False)
        base = ferrers() if rng.random() < 0.5 else np.full((K, K), rng.integers(0, 2))
        A, B = rng.integers(0, 2, K), rng.integers(0, 2, K); T = base.copy()
        T[np.ix_(R, C)] = A[R][:, None] ^ B[C][None]; add(T, "xor_block")
    return out


def main():
    P = pool(); T = np.stack([t for t, _ in P]); fam = np.array([f for _, f in P])
    dens = np.array([xor_density(t) for t in T]); syn = np.array([mmi_synergy(t) for t in T])
    np.savez(os.path.join(ROOT, "runs", "tables", "pool.npz"), T=T, fam=fam, dens=dens, syn_mmi=syn)
    print(f"pool {len(T)} tables; families {dict(zip(*np.unique(fam, return_counts=True)))}")
    # selection (primary set): the synergy band where zero-density and XOR-dense tables both exist (MMI synergy 0.40-0.75),
    # 4 equal-width synergy bins x 4 density bins (0, and terciles of the >0 part within the band); 4 tables per cell
    rng = np.random.default_rng(7); band = (syn >= 0.40) & (syn <= 0.75)
    sb = np.digitize(syn, [0.4875, 0.575, 0.6625]); pos = dens[band & (dens > 0)]
    db = np.where(dens == 0, 0, 1 + np.digitize(dens, np.quantile(pos, [1 / 3, 2 / 3])))
    sel = []
    for i in range(4):
        for j in range(4):
            cand = np.where(band & (sb == i) & (db == j))[0]
            sel += list(rng.choice(cand, min(4, len(cand)), replace=False))
    sel = np.array(sorted(sel)); dadd = np.array([additive_distance(t) for t in T[sel]])
    np.savez(os.path.join(ROOT, "runs", "tables", "selected.npz"), T=T[sel], fam=fam[sel], dens=dens[sel], syn_mmi=syn[sel], d_add=dadd, idx=sel)
    from scipy.stats import spearmanr
    print(f"selected {len(sel)}; cells filled: {len(set(zip(sb[sel], db[sel])))}/16; Spearman(syn_mmi, dens) selected "
          f"{spearmanr(syn[sel], dens[sel])[0]:+.2f} (pool {spearmanr(syn, dens)[0]:+.2f}); Spearman(dens, d_add) {spearmanr(dens[sel], dadd)[0]:+.2f}")
    print(f"zero-density tables: {(dens[sel] == 0).sum()} (d_add of those: {np.unique(dadd[dens[sel] == 0])})")


if __name__ == "__main__":
    main()
