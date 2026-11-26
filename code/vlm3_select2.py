"""Fresh selection for the confirmatory test (paper/preregistration_vlm3_conf.md). Outcome-free (answer tables only).
From the same 2,673 candidates, one sub-table per grid, chosen per benchmark so that structure s is nearly uncorrelated
with BOTH BROJA synergy and the yes share. Objective: |Spearman(s, BROJA)| + |Spearman(s, yes)| +
max(0, 0.5 sd_pool - sd_sel) / sd_pool. Coordinate descent: random start (seed 900), 30 sweeps. Writes runs/vlm3/selected2.csv"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "vlm3")
D = pd.read_csv(os.path.join(OUT, "candidates.csv")); D["broja"] = D.cid.map(json.load(open(os.path.join(OUT, "candidates_broja.json"))))
D = D[D.broja.notna() & (D.yes > 0) & (D.yes < 1)]; rng = np.random.default_rng(900); out = []
for b, X in D.groupby("bench"):
    keys = sorted(X.key.unique()); sdp = X.s.std(); opts = {k: X[X.key == k].reset_index(drop=True) for k in keys}
    pick = {k: int(rng.integers(len(opts[k]))) for k in keys}
    def J(p):
        sel = pd.concat([opts[k].iloc[[p[k]]] for k in keys])
        return abs(spearmanr(sel.s, sel.broja).correlation) + abs(spearmanr(sel.s, sel.yes).correlation) + max(0.0, 0.5 * sdp - sel.s.std()) / sdp
    best = J(pick)
    for sweep in range(30):
        for k in rng.permutation(keys):
            for i in range(len(opts[k])):
                q = dict(pick); q[k] = i; v = J(q)
                if v < best - 1e-12:
                    best, pick = v, q
    sel = pd.concat([opts[k].iloc[[pick[k]]] for k in keys]); out.append(sel)
    print(f"{b}: {len(sel)} units | Spearman(s, BROJA) {spearmanr(sel.s, sel.broja).correlation:+.2f} | Spearman(s, yes) {spearmanr(sel.s, sel.yes).correlation:+.2f} | "
          f"Spearman(BROJA, yes) {spearmanr(sel.broja, sel.yes).correlation:+.2f} | sd(s) {sel.s.std():.3f} (pool {sdp:.3f})")
pd.concat(out).to_csv(os.path.join(OUT, "selected2.csv"), index=False)
