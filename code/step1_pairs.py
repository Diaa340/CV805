"""1d: choose matched pairs (zero-density vs density >= 0.06) with |BROJA synergy difference| <= 0.03, greedy nearest,
no reuse, at most 12 pairs. Outcome-free. Writes runs/tables/pairs.npz"""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TD = os.path.join(ROOT, "runs", "tables")
C = np.load(os.path.join(TD, "paircand.npz")); P = pd.read_csv(os.path.join(TD, "pid_paircand.csv")).set_index("table")
Z = [k for k in range(len(C["T"])) if C["dens"][k] == 0]; X = [k for k in range(len(C["T"])) if C["dens"][k] >= 0.06]
out, used = [], set()
for z in sorted(Z, key=lambda k: P.syn_broja[k]):
    c = sorted([(abs(P.syn_broja[x] - P.syn_broja[z]), x) for x in X if x not in used])
    if c and c[0][0] <= 0.03:
        used.add(c[0][1]); out.append((z, c[0][1]))
    if len(out) >= 12:
        break
idx = [k for p in out for k in p]
np.savez(os.path.join(TD, "pairs.npz"), **{k: C[k][idx] for k in ("T", "fam", "dens", "syn_mmi", "d_add", "F")},
         pair=np.repeat(np.arange(len(out)), 2), role=np.tile(["zero", "dense"], len(out)))
P.loc[idx].reset_index(drop=True).assign(table=range(len(idx))).to_csv(os.path.join(TD, "pid_pairs.csv"), index=False)
print(f"{len(out)} pairs; BROJA differences {[round(abs(P.syn_broja[a] - P.syn_broja[b]), 3) for a, b in out]}")
