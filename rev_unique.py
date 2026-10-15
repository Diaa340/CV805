"""Revision R3 (paper/preregistration_revision.md): how unique are the exception cells?
Each table is re-solved K times with random tie-breaking: objective sum_cells w z + eps * r z with r ~ U(0, 1) per merged
cell and eps = 0.5 / (number of merged cells), so the perturbation is below one error and every solution found has the
minimum number of errors (when proven optimal). A solution is accepted if its error count equals the smallest count
found for that table. Cells that are exceptions in every accepted solution are stable exceptions, in none stable rule
cells, otherwise ambiguous. Random tie-breaking samples optimal rules, so the ambiguous share is a LOWER bound.
Tables: the 99 Study-C VLM sub-tables, and the test tables of the continual-learning studies (constructed tasks and
their task-2 table, CLEVR task-1 units and partners, GQA, CLOVE). Writes runs/rev/unique.npz and runs/rev/unique.csv"""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "rev"); os.makedirs(OUT, exist_ok=True)


def fit_tiebreak(T, seed, time_limit):
    """Exact additive fit of a complete 0/1 table (identical columns merged) with random tie-breaking."""
    pats, inv, cnt = np.unique(T.T, axis=0, return_inverse=True, return_counts=True); inv = np.ravel(inv); Tp = pats.T
    nq, ni = Tp.shape; qq, ii = np.meshgrid(np.arange(nq), np.arange(ni), indexing="ij"); qq, ii = qq.ravel(), ii.ravel(); yy = Tp[qq, ii]; n = len(yy)
    ww = cnt[ii].astype(float) + (0.5 / n) * np.random.default_rng(seed).random(n); M = 201.0
    A = lil_matrix((n, nq + ni + n)); lb = np.zeros(n); ub = np.zeros(n)
    for r in range(n):
        A[r, qq[r]] = 1; A[r, nq + ii[r]] = 1
        if yy[r]:
            A[r, nq + ni + r] = M; lb[r], ub[r] = 1, np.inf
        else:
            A[r, nq + ni + r] = -M; lb[r], ub[r] = -np.inf, -1
    res = milp(np.r_[np.zeros(nq + ni), ww], constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=np.r_[np.zeros(nq + ni), np.ones(n)],
               bounds=Bounds(np.r_[-50 * np.ones(nq + ni), np.zeros(n)], np.r_[50 * np.ones(nq + ni), np.ones(n)]), options={"time_limit": time_limit})
    u, w = res.x[:nq], res.x[nq:nq + ni]; P = ((u[:, None] + w[None]) > 0).astype(int)[:, inv]
    return (P != T).astype(np.int8), bool(res.status == 0)


def tables():
    out = []
    G = np.load(os.path.join(ROOT, "runs", "vlm3", "grids.npz")); U = pd.read_csv(os.path.join(ROOT, "runs", "vlm3", "selected2.csv"))
    for cid, key, cols in zip(U.cid, U.key, U.cols):
        out.append((f"vlm__{cid}", G[f"T__{key}"][:, eval(cols)].astype(int), 20, 120))
    def balanced_table(rng, K=10, swaps=5000):   # copied from kinds_kill.py, which loads data onto the GPU at import
        T = ((np.arange(K)[:, None] + np.arange(K)[None]) % K < K // 2).astype(int)
        for _ in range(swaps):
            r, c = rng.choice(K, 2, replace=False), rng.choice(K, 2, replace=False); sub = T[np.ix_(r, c)]
            if sub[0, 0] == sub[1, 1] != sub[0, 1] == sub[1, 0]:
                T[np.ix_(r, c)] = 1 - sub
        return T
    for setname in ("selected", "fresh"):
        S = np.load(os.path.join(ROOT, "runs", "tables", f"{setname}.npz"))
        for k in range(len(S["T"])):
            if S["d_add"][k] > 0:
                out.append((f"tables__{setname}_{k}", S["T"][k].astype(int), 20, 30))
    out.append(("tables__task2", balanced_table(np.random.default_rng(1023)).astype(int), 20, 30))
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz")); ite = ST["clevr_test_imgs"]
    FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep"); allU = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family))
    TS = np.load(os.path.join(ROOT, "runs", "clx", "train_structure.npz")); need = set()
    for u in TS["units"]:
        k = allU.index(u); need |= {u, next(allU[(k + s) % len(allU)] for s in range(len(allU) // 2, len(allU)) if fam[allU[(k + s) % len(allU)]] != fam[u])}
    for u in sorted(need):
        out.append((f"clevr__{u}", Z[f"T__{u}"][:, ite].astype(int), 10, 240))
    Q = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz")); ite = np.sort(np.random.default_rng(92).permutation(200)[150:])
    for u in pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep").unit:
        out.append((f"gqa__{u}", Q[f"T__{u}"][:, ite].astype(int), 20, 120))
    C = np.load(os.path.join(ROOT, "runs", "clove", "units.npz"))
    for u in pd.read_csv(os.path.join(ROOT, "runs", "clove", "units.csv")).query("keep").unit:
        out.append((f"clove__{u}", C[f"T__{u}"][:, 150:200].astype(int), 20, 120))
    return out


def job(a):
    key, T, d, tl = a; E, opt = fit_tiebreak(T, 1000 + d, tl); return key, d, E, opt


if __name__ == "__main__":
    TB = {k: (T, K, tl) for k, T, K, tl in tables()}; jobs = [(k, T, d, tl) for k, (T, K, tl) in TB.items() for d in range(K)]
    print(len(TB), "tables,", len(jobs), "solves", flush=True); sols = {}
    with Pool(12) as p:
        for i, (k, d, E, opt) in enumerate(p.imap_unordered(job, jobs, chunksize=1)):
            sols.setdefault(k, []).append((E, opt))
            if i % 200 == 0:
                print(i, flush=True)
    rows, out = [], {}
    for k, S in sols.items():
        cnt = [E.sum() for E, _ in S]; acc = [E for E, _ in S if E.sum() == min(cnt)]; St = np.stack(acc)
        exc, rule = St.all(0), (~St.astype(bool)).all(0); amb = ~(exc | rule)
        out[f"exc__{k}"], out[f"amb__{k}"] = exc.astype(np.int8), amb.astype(np.int8)
        rows.append(dict(key=k, bench=k.split("__")[0], cells=exc.size, min_errors=min(cnt), solves=len(S), accepted=len(acc), proven=sum(o for _, o in S),
                         ambiguous_share=amb.mean(), stable_exc_share=exc.mean(), any_exc_share=St.any(0).mean()))
    np.savez(os.path.join(OUT, "unique.npz"), **out); pd.DataFrame(rows).to_csv(os.path.join(OUT, "unique.csv"), index=False); print("DONE", flush=True)
