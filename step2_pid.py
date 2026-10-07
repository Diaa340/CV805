"""BROJA / I_ccs synergy of the clustered joint distributions (Liang et al.-style estimate; k = 10 clusters per modality),
for symbolic and CLIP features. Run with .venv_dit. Writes runs/clevr2/pid.csv"""
import csv, os, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clevr2")


def one(key):
    import dit
    from dit.pid import PID_BROJA, PID_CCS
    J = np.load(os.path.join(OUT, "clusters.npz"))[key]
    outs = [(a, b, c) for a in range(10) for b in range(10) for c in range(2) if J[a, b, c] > 0]
    d = dit.Distribution(outs, [J[o] for o in outs])
    u, feat = key.rsplit("_", 1); r = dict(unit=u, feat=feat)
    for nm, P in (("broja", PID_BROJA), ("ccs", PID_CCS)):
        p = P(d, [[0], [1]], [2]); r[f"syn_{nm}"] = float(p.get_pi(((0, 1),))); r[f"red_{nm}"] = float(p.get_pi(((0,), (1,))))
    return r


if __name__ == "__main__":
    keys = list(np.load(os.path.join(OUT, "clusters.npz")).keys())
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as p:
        rows = p.map(one, keys)
    with open(os.path.join(OUT, "pid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} PID estimates")
