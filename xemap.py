"""Exact additive projection of a model (paper/preregistration_xemap.md). For a model's yes/no answers P on a complete
table, the exact projection is the additive rule that agrees with P on the most cells (MILP, as for answer tables);
"flips" are the cells where it disagrees with P, and the exact drop is Acc(P) - Acc(projection). A confidence-weighted
variant weights each cell by the model's |log-odds| (+0.01). For P = T (a perfect model) the flips are exactly an optimal
exception set and the drop equals s(T).
Computed on the Study C sub-tables (10 x 50) for the four VLMs, and (unweighted) for the accuracy-matched null scorer
(same construction as rev_null.py, 20 draws). Writes runs/rev/xemap_units.csv and runs/rev/xemap_cells.csv.gz"""
import json, os
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "rev")
MODELS = ("idefics", "internvl", "llava", "qwen")


def project(P, W, tl=120):
    from exact_additive import exact_partial_fit
    Q, best, up, m = exact_partial_fit(P.astype(int), W, tl); return Q, bool(abs(best - up) < 1e-6)


def job(a):
    kind, m, cid, T, L, d = a
    if kind == "model":
        P = (L > 0).astype(int); Qu, ou = project(P, np.ones(P.shape)); Qw, ow = project(P, np.abs(L) + 0.01)
        return dict(kind=kind, model=m, cid=cid, draw=-1, acc=(P == T).mean(), acc_proj=(Qu == T).mean(), acc_proj_w=(Qw == T).mean(), proven=ou, proven_w=ow,
                    flips=(Qu != P).ravel(), flips_w=(Qw != P).ravel(), conf=np.abs(L).ravel())
    mu = {y: norm.ppf(np.clip(((L > 0) == T)[T == y].mean(), 0.001, 0.999)) for y in (0, 1)}
    Ln = (2 * T - 1) * np.where(T == 1, mu[1], mu[0]) + np.random.default_rng(d).standard_normal(T.shape); P = (Ln > 0).astype(int); Qu, ou = project(P, np.ones(P.shape))
    return dict(kind=kind, model=m, cid=cid, draw=d, acc=(P == T).mean(), acc_proj=(Qu == T).mean(), proven=ou)


if __name__ == "__main__":
    G = np.load(os.path.join(ROOT, "runs", "vlm3", "grids.npz")); U = pd.read_csv(os.path.join(ROOT, "runs", "vlm3", "selected2.csv")); jobs = []
    for m in MODELS:
        S = np.load(os.path.join(ROOT, "runs", "vlm3", f"scores_{m}.npz"))
        for cid, key, cols in zip(U.cid, U.key, U.cols):
            c = json.loads(cols); T = G[f"T__{key}"][:, c].astype(int); L = S[f"L__{key}"][:, c]
            jobs.append(("model", m, cid, T, L, -1)); jobs += [("null", m, cid, T, L, d) for d in range(20)]
    print(len(jobs), "projections", flush=True)
    with Pool(12) as p:
        R = list(p.imap_unordered(job, jobs, chunksize=8))
    units = pd.DataFrame([{k: v for k, v in r.items() if k not in ("flips", "flips_w", "conf")} for r in R]); units.to_csv(os.path.join(OUT, "xemap_units.csv"), index=False)
    cells = []
    for r in R:
        if r["kind"] == "model":
            cells.append(pd.DataFrame(dict(model=r["model"], cid=r["cid"], cell=np.arange(len(r["flips"])), flip=r["flips"], flip_w=r["flips_w"], conf=r["conf"])))
    pd.concat(cells).to_csv(os.path.join(OUT, "xemap_cells.csv.gz"), index=False); print("DONE", flush=True)
