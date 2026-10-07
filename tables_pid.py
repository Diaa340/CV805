"""Exact PID (BROJA, I_ccs) of the selected label tables under uniform independent class IDs. Run with .venv_dit.
Writes runs/tables/pid.csv"""
import os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))


def one(k):
    import dit
    from dit.pid import PID_BROJA, PID_CCS
    T = np.load(os.path.join(ROOT, "runs", "tables", "selected.npz"))["T"][k]
    d = dit.Distribution([(i, j, int(T[i, j])) for i in range(10) for j in range(10)], [0.01] * 100)
    out = dict(table=k)
    for nm, P in (("broja", PID_BROJA), ("ccs", PID_CCS)):
        p = P(d, [[0], [1]], [2])
        out.update({f"syn_{nm}": float(p.get_pi(((0, 1),))), f"red_{nm}": float(p.get_pi(((0,), (1,)))),
                    f"u1_{nm}": float(p.get_pi(((0,),))), f"u2_{nm}": float(p.get_pi(((1,),)))})
    print(k, {a: round(b, 3) for a, b in out.items()}, flush=True)
    return out


if __name__ == "__main__":
    n = len(np.load(os.path.join(ROOT, "runs", "tables", "selected.npz"))["T"])
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as p:
        rows = p.map(one, range(n))
    import csv
    with open(os.path.join(ROOT, "runs", "tables", "pid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
