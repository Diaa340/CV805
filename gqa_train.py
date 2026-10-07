"""GQA trained estimators (paper/preregistration_gqa.md): CLIP features of the unit's 200 images and 10 questions;
images split 150 / 50 (seed 92); full model, jointly trained additive model, linear readout; seeds 23-25; exact
additive ceiling on the test cells (MILP). Writes runs/gqa/train.csv"""
import os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from additive_estimator import Additive, acc, fit
from exact_additive import exact_table_ceiling
from kinds_kill import linear_check
from route_exps import Net
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "gqa"); IMD = os.path.join(ROOT, "data", "raw", "gqa", "images"); dev = "cuda"
sys.path.insert(0, ROOT); from extract_features import clip_images, clip_texts
Z = np.load(os.path.join(OUT, "units.npz")); U = pd.read_csv(os.path.join(OUT, "units.csv")).query("keep").unit.tolist(); out = os.path.join(OUT, "train.csv")
for u in U:
    T = Z[f"T__{u}"].astype(int); ims = Z[f"imgs__{u}"]
    Xi = clip_images(np.stack([np.asarray(Image.open(os.path.join(IMD, f"{m}.jpg")).convert("RGB").resize((224, 224), Image.BICUBIC)) for m in ims])).astype(np.float32)
    Xq = clip_texts(list(Z[f"text__{u}"])).astype(np.float32); nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6)
    Xi, Xq = torch.tensor(nz(Xi), device=dev), torch.tensor(nz(Xq), device=dev)
    perm = np.random.default_rng(92).permutation(T.shape[1]); itr, ite = perm[:150], perm[150:]; ceil, ceil_up, _ = exact_table_ceiling(T, ite, 300)
    for seed in (23, 24, 25):
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        def cells(cols, n):
            q, i = rng.integers(0, 10, n), cols[rng.integers(0, len(cols), n)]; t = lambda a: torch.tensor(a, device=dev)
            return Xi[t(i)], Xq[t(q)], t(T[q, i])
        tr, te = cells(itr, 50000), cells(ite, 20000)
        full = fit(Net(Xi.shape[1], Xq.shape[1], 2).to(dev), tr).eval(); add = fit(Additive(Xi.shape[1], Xq.shape[1], 2).to(dev), tr).eval()
        with torch.no_grad():
            ztr = torch.cat([full.e1(tr[0][:20000]), full.e2(tr[1][:20000])], 1); zte = torch.cat([full.e1(te[0]), full.e2(te[1])], 1)
        p = te[2].float().mean().item()
        r = dict(unit=u, seed=seed, acc_full=acc(full, te), acc_add=acc(add, te), acc_lin=linear_check(ztr, tr[2][:20000], zte, te[2]), ceil=ceil, ceil_upper=ceil_up, acc_maj=max(p, 1 - p))
        pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); print(r, flush=True)
