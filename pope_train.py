"""POPE trained estimators (paper/preregistration_pope.md): CLIP ViT-B/32 image features (500 POPE images) and question
text features (80 questions). Data = POPE's 9,000 questions; images split 3:1 at random (seed 71). Full two-branch model,
jointly trained additive model, linear readout on the full model's codes; seeds 23-25. Exact ceiling = best additive
fit of POPE's cells (all images), evaluated on the test cells. Writes runs/pope/train.csv and runs/pope/clip.npz"""
import os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from additive_estimator import Additive, acc, fit
from kinds_kill import linear_check
from route_exps import Net
ROOT = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(ROOT, "data", "raw", "pope"); OUT = os.path.join(ROOT, "runs", "pope"); dev = "cuda"
Z = np.load(os.path.join(P, "pope_table.npz")); Tp, O, IM, C = Z["T_pope"], Z["objects"], Z["images"], Z["cells"]
cf = os.path.join(OUT, "clip.npz")
if not os.path.exists(cf):
    sys.path.insert(0, ROOT); from extract_features import clip_images, clip_texts
    arr = np.stack([np.asarray(Image.open(os.path.join(P, "images", f)).convert("RGB").resize((224, 224), Image.BICUBIC)) for f in IM])
    np.savez(cf, img=clip_images(arr).astype(np.float32), q=clip_texts([f"Is there a {o} in the image?" for o in O]).astype(np.float32))
F_ = np.load(cf); nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6); Xi, Xq = torch.tensor(nz(F_["img"]), device=dev), torch.tensor(nz(F_["q"]), device=dev)
perm = np.random.default_rng(71).permutation(len(IM)); tr_im = set(perm[:375])
import torch.nn.functional as F
qi, ii = np.where(Tp >= 0); y = torch.tensor(Tp[qi, ii], device=dev).float()
u = torch.zeros(Tp.shape[0], device=dev, requires_grad=True); w = torch.zeros(Tp.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
opt = torch.optim.Adam([u, w, b], lr=0.1); qt, it = torch.tensor(qi, device=dev), torch.tensor(ii, device=dev)
for _ in range(3000):
    opt.zero_grad(); F.binary_cross_entropy_with_logits(u[qt] + w[it] + b, y).backward(); opt.step()
with torch.no_grad():
    A = ((u[:, None] + w[None] + b) > 0).long().cpu().numpy()
te = np.array([k not in tr_im for k in C[:, 1]]); ceil = float((A[C[te, 0], C[te, 1]] == C[te, 2]).mean())
rows = []
for seed in (23, 24, 25):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); trc = C[~te]; idx = trc[rng.integers(0, len(trc), 50000)]
    t = lambda a: torch.tensor(a, device=dev)
    tr = (Xi[t(idx[:, 1])], Xq[t(idx[:, 0])], t(idx[:, 2])); tst = (Xi[t(C[te, 1])], Xq[t(C[te, 0])], t(C[te, 2]))
    full = fit(Net(Xi.shape[1], Xq.shape[1], 2).to(dev), tr); add = fit(Additive(Xi.shape[1], Xq.shape[1], 2).to(dev), tr)
    with torch.no_grad():
        z = lambda d: torch.cat([full.e1(d[0]), full.e2(d[1])], 1)
        ztr, zte = z((tr[0][:20000], tr[1][:20000])), z(tst)
    lin = linear_check(ztr, tr[2][:20000], zte, tst[2])
    p = tst[2].float().mean().item()
    rows.append(dict(seed=seed, n_test=int(te.sum()), acc_full=acc(full.eval(), tst), acc_add=acc(add.eval(), tst), acc_lin=lin, ceiling=ceil, acc_maj=max(p, 1 - p)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "train.csv"), index=False)
