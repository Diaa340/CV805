"""Exact additive ceilings (MILP, identical columns merged) for the 103 CLEVR units: on the step-2 test images (300)
and on the full table (1,000 images). Replaces the logistic lower bound used so far. Writes runs/clevr2/exact_ceilings.csv"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2")
Z = dict(np.load(os.path.join(OUT, "data.npz"))); U = pd.read_csv(os.path.join(OUT, "families.csv")).query("keep").unit.tolist()
ite = np.random.default_rng(14).permutation(len(Z["imgs"]))[700:]


def job(u):
    from exact_additive import exact_table_ceiling
    T = Z[f"T__{u}"].astype(int); b1, u1, m1 = exact_table_ceiling(T, ite, 600); b2, u2, m2 = exact_table_ceiling(T, None, 600)
    return dict(unit=u, exact_test=b1, exact_test_upper=u1, status_test=m1[:30], exact_full=b2, exact_full_upper=u2, status_full=m2[:30])


if __name__ == "__main__":
    with Pool(12) as p:
        rows = []
        for r in p.imap_unordered(job, U):
            rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "exact_ceilings.csv"), index=False)
