"""Step 1 table sets (paper/preregistration_step1.md). Outcome-free.
- fresh: a new pool (seed 2027), same selection rule as tables_build.py (replicate set).
- nearest threshold table F for every table (for 1c).
- pair candidates for 1d: zero-density and XOR-dense (density >= 0.06) tables in the MMI band, near-equal MMI synergy.
Writes runs/tables/{fresh,orig_F,paircand}.npz"""
import os
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from tables_build import K, mmi_synergy, pool, xor_density
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TD = os.path.join(ROOT, "runs", "tables")


def nearest_ferrers(T, M=201.0):
    """Exact MILP: the threshold table [a_i + b_j > 0] closest to T (fewest flips). Returns (F, d_add)."""
    nv = 2 * K + K * K; c = np.r_[np.zeros(2 * K), np.ones(K * K)]; A = np.zeros((K * K, nv)); lb = np.zeros(K * K); ub = np.zeros(K * K)
    for i in range(K):
        for j in range(K):
            r = i * K + j; A[r, i] = 1; A[r, K + j] = 1
            if T[i, j]:
                A[r, 2 * K + r] = M; lb[r], ub[r] = 1, np.inf
            else:
                A[r, 2 * K + r] = -M; lb[r], ub[r] = -np.inf, -1
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=np.r_[np.zeros(2 * K), np.ones(K * K)],
               bounds=Bounds(np.r_[-50 * np.ones(2 * K), np.zeros(K * K)], np.r_[50 * np.ones(2 * K), np.ones(K * K)]))
    z = np.round(res.x[2 * K:]).reshape(K, K).astype(int)
    F = np.where(z == 1, 1 - T, T)
    assert xor_density(F) == 0
    return F.astype(np.int8), z.sum() / (K * K)


def select(T, dens, syn, rng, per_cell=4):
    band = (syn >= 0.40) & (syn <= 0.75); sb = np.digitize(syn, [0.4875, 0.575, 0.6625]); pos = dens[band & (dens > 0)]
    db = np.where(dens == 0, 0, 1 + np.digitize(dens, np.quantile(pos, [1 / 3, 2 / 3]))); sel = []
    for i in range(4):
        for j in range(4):
            cand = np.where(band & (sb == i) & (db == j))[0]
            sel += list(rng.choice(cand, min(per_cell, len(cand)), replace=False))
    return np.array(sorted(sel))


def main():
    old = np.load(os.path.join(TD, "selected.npz"))
    Fo = [nearest_ferrers(t) for t in old["T"]]
    assert np.allclose([d for _, d in Fo], old["d_add"])
    np.savez(os.path.join(TD, "orig_F.npz"), F=np.stack([f for f, _ in Fo]))
    P = pool(seed=2027); T = np.stack([t for t, _ in P]); fam = np.array([f for _, f in P])
    oldset = {t.tobytes() for t in old["T"]}; keep = np.array([t.tobytes() not in oldset for t in T]); T, fam = T[keep], fam[keep]
    dens = np.array([xor_density(t) for t in T]); syn = np.array([mmi_synergy(t) for t in T])
    sel = select(T, dens, syn, np.random.default_rng(8)); FF = [nearest_ferrers(t) for t in T[sel]]
    np.savez(os.path.join(TD, "fresh.npz"), T=T[sel], fam=fam[sel], dens=dens[sel], syn_mmi=syn[sel],
             d_add=np.array([d for _, d in FF]), F=np.stack([f for f, _ in FF]))
    print(f"fresh: {len(sel)} tables from pool {len(T)}")
    # pair candidates from the unselected tables of both pools
    P0 = pool(seed=2026); T0 = np.stack([t for t, _ in P0]); f0 = np.array([f for _, f in P0])
    used = oldset | {t.tobytes() for t in T[sel]}
    TA = np.concatenate([T0, T]); fA = np.concatenate([f0, fam]); uniq, cand = set(), []
    for k, t in enumerate(TA):
        b = t.tobytes()
        if b not in used and b not in uniq:
            uniq.add(b); cand.append(k)
    TA, fA = TA[cand], fA[cand]; dA = np.array([xor_density(t) for t in TA]); sA = np.array([mmi_synergy(t) for t in TA])
    band = (sA >= 0.40) & (sA <= 0.75); Z, X = np.where(band & (dA == 0))[0], np.where(band & (dA >= 0.06))[0]
    rng = np.random.default_rng(9); pick = set()
    for x in rng.permutation(X):
        d = np.abs(sA[Z] - sA[x]); z = [Z[i] for i in np.argsort(d) if d[i] <= 0.01 and Z[i] not in pick]
        if z:
            pick |= {x, z[0]}
        if len(pick) >= 60:
            break
    pick = np.array(sorted(pick)); FP = [nearest_ferrers(t) for t in TA[pick]]
    np.savez(os.path.join(TD, "paircand.npz"), T=TA[pick], fam=fA[pick], dens=dA[pick], syn_mmi=sA[pick],
             d_add=np.array([d for _, d in FP]), F=np.stack([f for f, _ in FP]))
    print(f"pair candidates: {len(pick)} ({(dA[pick] == 0).sum()} zero-density, {(dA[pick] >= 0.06).sum()} dense)")


if __name__ == "__main__":
    main()
