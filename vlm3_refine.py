"""Re-solves, with a 1-hour limit, the grid tables whose exact additive rule was not proven within 600 s
(paper/preregistration_vlm3.md). Updates exc__ in runs/vlm3/grids.npz and s / ceiling / bound / exact in units.csv."""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
from vlm3_build import job as _job
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "vlm3")


def job(a):
    from exact_additive import exact_partial_fit
    key, T = a; P, b, up, m = exact_partial_fit(T, np.ones(T.shape), 3600); return key, (P != T).astype(np.int8), b, up


if __name__ == "__main__":
    U = pd.read_csv(os.path.join(OUT, "units.csv")); G = dict(np.load(os.path.join(OUT, "grids.npz"))); todo = U[~U.exact].key.tolist(); print("re-solving", todo, flush=True)
    with Pool(len(todo)) as p:
        for key, N, b, up in p.imap_unordered(job, [(k, G[f"T__{k}"].astype(int)) for k in todo]):
            G[f"exc__{key}"] = N; i = U.index[U.key == key][0]; U.loc[i, ["s", "ceiling", "bound", "exact"]] = [float(N.mean()), b, up, bool(abs(up - b) < 1e-6)]
            print(key, round(b, 4), round(up, 4), flush=True)
    np.savez(os.path.join(OUT, "grids.npz"), **G); U.to_csv(os.path.join(OUT, "units.csv"), index=False)
