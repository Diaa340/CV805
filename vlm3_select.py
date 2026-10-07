"""Chooses one candidate sub-table per grid so that, within each benchmark, structure and PID synergy are nearly
uncorrelated (paper/preregistration_vlm3_decor.md). Outcome-free (answer tables only).
Objective per benchmark: |Spearman(s, BROJA)| + max(0, 0.5 sd_pool - sd_sel) / sd_pool, where sd_pool is the standard
deviation of s over all candidates of that benchmark (keeps a spread of s). Coordinate descent: random start (seed 800),
20 sweeps over the grids in random order, each grid set to its best candidate. Writes runs/vlm3/selected.csv"""
import json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "vlm3")
D = pd.read_csv(os.path.join(OUT, "candidates.csv")); D["broja"] = D.cid.map(json.load(open(os.path.join(OUT, "candidates_broja.json"))))
D = D[D.broja.notna() & (D.yes > 0) & (D.yes < 1)]; rng = np.random.default_rng(800); out = []
for b, X in D.groupby("bench"):
    keys = sorted(X.key.unique()); sdp = X.s.std(); opts = {k: X[X.key == k].reset_index(drop=True) for k in keys}
    pick = {k: int(rng.integers(len(opts[k]))) for k in keys}
    def J(p):
        sel = pd.concat([opts[k].iloc[[p[k]]] for k in keys]); r = spearmanr(sel.s, sel.broja).correlation
        return abs(r) + max(0.0, 0.5 * sdp - sel.s.std()) / sdp
    best = J(pick)
    for sweep in range(20):
        for k in rng.permutation(keys):
            for i in range(len(opts[k])):
                q = dict(pick); q[k] = i; v = J(q)
                if v < best - 1e-12:
                    best, pick = v, q
    sel = pd.concat([opts[k].iloc[[pick[k]]] for k in keys]); out.append(sel)
    print(f"{b}: {len(sel)} units | Spearman(s, BROJA) {spearmanr(sel.s, sel.broja).correlation:+.2f} (all candidates {spearmanr(X.s, X.broja).correlation:+.2f}) | "
          f"sd(s) {sel.s.std():.3f} (pool {sdp:.3f}) | Spearman(s, yes) {spearmanr(sel.s, sel.yes).correlation:+.2f} | Spearman(BROJA, yes) {spearmanr(sel.broja, sel.yes).correlation:+.2f}")
S = pd.concat(out); S.to_csv(os.path.join(OUT, "selected.csv"), index=False)
print(f"pooled within-benchmark: Spearman(s, BROJA) of benchmark-centred ranks "
      f"{np.corrcoef(*[S.groupby('bench')[c].rank(pct=True) - .5 for c in ('s', 'broja')])[0, 1]:+.2f}")
