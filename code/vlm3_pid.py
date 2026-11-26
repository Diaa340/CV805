"""BROJA synergy (bits) of every grid's answer table under uniform cells (paper/preregistration_vlm3.md), with the convex
solver of Liang et al. (liang_cvx.py via clevr_liang_sweep.pid_bits_with, SCS). Run with .venv_pid.
`--check`: compares with dit's BROJA (.venv_dit, via a subprocess) on 10 x 20 sub-tables, where dit is fast.
Writes runs/vlm3/broja.json"""
import json, os, subprocess, sys
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "vlm3")


def joint(T):
    P = np.zeros(T.shape + (2,)); P[np.arange(T.shape[0])[:, None], np.arange(T.shape[1])[None], T] = 1.0 / T.size; return P


def cvx(T):
    import cvxpy as cp
    from clevr_liang_sweep import pid_bits_with
    try:
        return float(pid_bits_with(joint(T), cp.SCS)["S"])
    except Exception as e:
        print("failed", type(e).__name__, flush=True); return float("nan")


DIT = """import sys, json, dit
from dit.pid import PID_BROJA
T = json.loads(sys.stdin.read()); outs = [(f"{q}", f"{j}", f"{T[q][j]}") for q in range(len(T)) for j in range(len(T[0]))]
d = dit.Distribution(outs, [1.0 / len(outs)] * len(outs)); print(float(PID_BROJA(d, [[0], [1]], [2]).get_pi(((0, 1),))))"""


def dit_broja(T):
    r = subprocess.run([os.path.join(ROOT, ".venv_dit", "bin", "python"), "-c", DIT], input=json.dumps(T.tolist()), capture_output=True, text=True, timeout=1800)
    return float(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    G = np.load(os.path.join(OUT, "grids.npz")); keys = sorted(k[3:] for k in G.files if k.startswith("T__"))
    if "--check" in sys.argv:
        rng = np.random.default_rng(0); sub = [(k, G[f"T__{k}"][:, np.sort(rng.choice(100, 20, replace=False))].astype(int)) for k in rng.choice(keys, 6, replace=False)]
        with Pool(6) as p:
            a = p.map(cvx, [T for _, T in sub]); b = p.map(dit_broja, [T for _, T in sub])
        for (k, _), x, y in zip(sub, a, b):
            print(f"{k}: cvx {x:.4f} dit {y:.4f} diff {x - y:+.4f}")
    else:
        with Pool(12) as p:
            res = dict(zip(keys, p.map(cvx, [G[f"T__{k}"].astype(int) for k in keys])))
        json.dump(res, open(os.path.join(OUT, "broja.json"), "w")); print(len(res), "tables;", sum(np.isnan(v) for v in res.values()), "failed")
