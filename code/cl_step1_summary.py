"""Summarise the step-1 robustness checks of the 'what is forgotten first' finding (cl_step1/*_results.csv)."""
import glob
import os

import numpy as np
import pandas as pd

TYPES = ["R", "U1", "U2", "S"]
rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cl_step1", "*_results.csv"))):
    cfg = os.path.basename(f).replace("_results.csv", "")
    R = pd.read_csv(f)
    B = pd.read_csv(f.replace("_results.csv", "_confbins.csv"))
    # a type only counts in a seed where task 1 actually learned it (>= 50% before task 2; chance is 10%)
    learned = {k: set(R[R[f"before_{k}"] >= 0.5].seed) for k in TYPES}
    for cond, r in R.groupby("cond"):
        row = dict(check=cfg, cond=cond, seeds=r.seed.nunique(), seeds_S_learned=len(learned["S"]))
        for k in TYPES:
            rk = r[r.seed.isin(learned[k])]
            row[f"before_{k}"] = 100 * rk[f"before_{k}"].mean()
            row[f"kept_{k}"] = 100 * rk[f"retention_{k}"].mean()
        # confidence-matched ratio vs R: mean over (seed, bin) cells where both types have >= 30 samples
        b = B[B.cond == cond].pivot_table(index=["seed", "conf_bin"], columns="type", values="retained")
        for k in ("U1", "U2", "S"):
            ok = b[b.index.get_level_values("seed").isin(learned[k] & learned["R"])][["R", k]].dropna()
            ok = ok[ok.R > 0]
            row[f"matched_{k}/R"] = (ok[k] / ok.R).mean() if len(ok) else np.nan
        rows.append(row)
S = pd.DataFrame(rows).set_index(["check", "cond"])
pd.set_option("display.width", 250)
print("share of task-1 accuracy kept after task 2 (%), per type; before = task-1 accuracy before task 2 (%)")
print(S[[c for c in S.columns if c.startswith(("before_", "kept_"))]].round(1).to_string())
print("\nconfidence-matched retention relative to redundant (1.0 = forgotten as slowly as redundant knowledge)")
print(S[[c for c in S.columns if c.startswith("matched_")]].round(2).to_string())
