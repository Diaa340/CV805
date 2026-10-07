"""Exact PID (BROJA, I_ccs) for any table set (run with .venv_dit): python tables_pid2.py <set> [procs].
Reads runs/tables/<set>.npz, writes runs/tables/pid_<set>.csv"""
import csv, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); SET = sys.argv[1]


def one(k):
    import dit
    from dit.pid import PID_BROJA, PID_CCS
    T = np.load(os.path.join(ROOT, "runs", "tables", f"{SET}.npz"))["T"][k]
    d = dit.Distribution([(i, j, int(T[i, j])) for i in range(10) for j in range(10)], [0.01] * 100)
    out = dict(table=k)
    for nm, P in (("broja", PID_BROJA), ("ccs", PID_CCS)):
        p = P(d, [[0], [1]], [2])
        out.update({f"syn_{nm}": float(p.get_pi(((0, 1),))), f"red_{nm}": float(p.get_pi(((0,), (1,))))})
    return out


if __name__ == "__main__":
    n = len(np.load(os.path.join(ROOT, "runs", "tables", f"{SET}.npz"))["T"])
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 12) as p:
        rows = p.map(one, range(n))
    with open(os.path.join(ROOT, "runs", "tables", f"pid_{SET}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{SET}: {n} tables done")
