"""GQA with Qwen2.5-VL-7B (paper/preregistration_gqa.md): per unit, a complete 10-question x 100-image table (the first
100 of the unit's 200 images after a fixed permutation, seed 91); log-odds Yes vs No; EMAP; exact additive ceilings
(plain and balanced accuracy). Writes runs/gqa/vlm.csv and runs/gqa/vlm_tables.npz"""
import os, zipfile
import numpy as np, pandas as pd
from PIL import Image
from exact_additive import exact_partial_ceiling
from vlm_forget import VLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "gqa"); IMD = os.path.join(ROOT, "data", "raw", "gqa", "images")
Z = np.load(os.path.join(OUT, "units.npz")); U = pd.read_csv(os.path.join(OUT, "units.csv")).query("keep").unit.tolist()
vlm = VLM("Qwen/Qwen2.5-VL-7B-Instruct"); bal = lambda p, y: np.mean([(p[y == c] == c).mean() for c in (0, 1)]); store = {}
def bal_w(T):
    return np.where(T == 1, 0.5 / max(1, (T == 1).sum()), 0.5 / max(1, (T == 0).sum()))
done = set(pd.read_csv(os.path.join(OUT, "vlm.csv")).unit) if os.path.exists(os.path.join(OUT, "vlm.csv")) else set()
for u in U:
    if u in done:
        continue
    T = Z[f"T__{u}"]; ims = Z[f"imgs__{u}"]; sel = np.sort(np.random.default_rng(91).permutation(T.shape[1])[:100]); Ts = T[:, sel]
    cache = {k: Image.open(os.path.join(IMD, f"{ims[k]}.jpg")).convert("RGB") for k in sel}
    cells = [(q, k) for q in range(10) for k in sel]; L = vlm.logodds([cache[k] for _, k in cells], [f"{Z[f'text__{u}'][q]} Answer with yes or no." for q, _ in cells]).reshape(10, 100)
    E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean(); Tt = np.where(Ts == 1, 1.0, -1.0); Et = Tt.mean(1, keepdims=True) + Tt.mean(0, keepdims=True) - Tt.mean()
    cp, cpu, _ = exact_partial_ceiling(Ts.astype(int), None, 300); cb, cbu, _ = exact_partial_ceiling(Ts.astype(int), bal_w(Ts), 300)
    r = dict(unit=u, family=u.rsplit("_", 1)[0], yes=Ts.mean(), acc_vlm=((L > 0) == Ts).mean(), acc_emap=((E > 0) == Ts).mean(), bal_vlm=bal((L > 0).astype(int), Ts),
             bal_emap=bal((E > 0).astype(int), Ts), bal_emap_truth=bal((Et > 0).astype(int), Ts), ceil=cp, ceil_upper=cpu, ceil_bal=cb, ceil_bal_upper=cbu)
    store[u] = L; pd.DataFrame([r]).to_csv(os.path.join(OUT, "vlm.csv"), mode="a", header=not os.path.exists(os.path.join(OUT, "vlm.csv")), index=False); print(r, flush=True)
np.savez(os.path.join(OUT, "vlm_tables.npz"), **store)
