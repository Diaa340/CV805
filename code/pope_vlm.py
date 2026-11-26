"""POPE with Qwen2.5-VL-7B (paper/preregistration_pope.md):
 (B) all 9,000 POPE questions (random / popular / adversarial), log-odds of Yes vs No; per-cell additive / non-additive
     label from the best additive fit of POPE's own cells (T_pope) and of the full table.
 (C) EMAP on a complete sub-table: all 80 objects x 100 images (fixed seed), well-posed everywhere.
Writes runs/pope/vlm_pope.csv and runs/pope/vlm_subtable.npz"""
import os
import numpy as np, pandas as pd, torch
from PIL import Image
from step2_train import additive_pred
from vlm_forget import VLM
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = os.path.join(ROOT, "data", "raw", "pope"); OUT = os.path.join(ROOT, "runs", "pope"); os.makedirs(OUT, exist_ok=True)
Z = np.load(os.path.join(P, "pope_table.npz")); T, Tp, O, IM, C, S = Z["T"], Z["T_pope"], Z["objects"], Z["images"], Z["cells"], Z["split"]
img = lambda k: Image.open(os.path.join(P, "images", IM[k])).convert("RGB")
q = lambda o: f"Is there a {O[o]} in the image? Answer with yes or no."
vlm = VLM("Qwen/Qwen2.5-VL-7B-Instruct")
Af = additive_pred(T)  # additive fit of the full table
# exact additive fit restricted to POPE's own cells
import torch.nn.functional as F
qi, ii = np.where(Tp >= 0); y = torch.tensor(Tp[qi, ii], device="cuda").float()
u = torch.zeros(Tp.shape[0], device="cuda", requires_grad=True); w = torch.zeros(Tp.shape[1], device="cuda", requires_grad=True); b = torch.zeros(1, device="cuda", requires_grad=True)
opt = torch.optim.Adam([u, w, b], lr=0.1); qt, it = torch.tensor(qi, device="cuda"), torch.tensor(ii, device="cuda")
for _ in range(3000):
    opt.zero_grad(); F.binary_cross_entropy_with_logits(u[qt] + w[it] + b, y).backward(); opt.step()
with torch.no_grad():
    Ap = ((u[:, None] + w[None] + b) > 0).long().cpu().numpy()
D = pd.DataFrame(dict(split=S, obj=C[:, 0], img=C[:, 1], truth=C[:, 2]))
D["nonadd_pope"] = (Ap[D.obj, D.img] != D.truth).astype(int); D["nonadd_full"] = (Af[D.obj, D.img] != D.truth).astype(int)
ims = [img(k) for k in D.img]; D["logodds"] = vlm.logodds(ims, [q(o) for o in D.obj])
D.to_csv(os.path.join(OUT, "vlm_pope.csv"), index=False); del ims
print(f"POPE accuracy {((D.logodds > 0) == D.truth).mean():.3f} | " + " | ".join(f"{s} {((x.logodds > 0) == x.truth).mean():.3f}" for s, x in D.groupby("split")), flush=True)
sub = np.sort(np.random.default_rng(61).choice(T.shape[1], 100, replace=False))
cells = [(o, k) for o in range(T.shape[0]) for k in sub]; L = np.zeros(len(cells)); cache = {k: img(k) for k in sub}
for s in range(0, len(cells), 256):
    B = cells[s:s + 256]; L[s:s + len(B)] = vlm.logodds([cache[k] for _, k in B], [q(o) for o, _ in B])
np.savez(os.path.join(OUT, "vlm_subtable.npz"), L=L.reshape(T.shape[0], len(sub)), sub=sub, T=T[:, sub])
print("SUBTABLE DONE", flush=True)
