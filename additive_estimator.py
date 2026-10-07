"""Additive-model estimator of the fusion ceiling (paper/preregistration_additive_est.md).
Additive model = two unimodal networks (encoder + MLP head each) whose class logits are summed, trained jointly; full model
= the two-branch model with a joint MLP head (route_exps.Net). Same optimiser and budget for both.
 --part clevr: the 103 CLEVR units (validation against the exact additive ceiling; step-2 data and split);
 --part bench: seven frozen-feature benchmarks (all classes, their train / test split), plus unimodal models.
Output: runs/additive_est_<part>.csv"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from route_exps import Net
dev = "cuda"; ROOT = os.path.dirname(os.path.abspath(__file__))


class Additive(nn.Module):
    def __init__(self, d1, d2, C, depth=2, width=256, emb=128):
        super().__init__(); n = Net(d1, d2, C, depth, width, emb)
        self.e1, self.e2 = n.e1, n.e2
        self.h1 = nn.Sequential(nn.Linear(emb, width), nn.ReLU(), nn.Linear(width, C)); self.h2 = nn.Sequential(nn.Linear(emb, width), nn.ReLU(), nn.Linear(width, C))
    def forward(self, x1, x2):
        return self.h1(self.e1(x1)) + self.h2(self.e2(x2))


class Uni(nn.Module):
    def __init__(self, d, C, m, width=256, emb=128):
        super().__init__(); self.m = m; self.f = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, emb), nn.Linear(emb, width), nn.ReLU(), nn.Linear(width, C))
    def forward(self, x1, x2):
        return self.f(x1 if self.m == 1 else x2)


def fit(model, tr, epochs=30, bs=256, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5); n = len(tr[2])
    for _ in range(epochs * max(1, n // bs)):
        i = torch.randint(0, n, (bs,), device=dev); opt.zero_grad(); F.cross_entropy(model(tr[0][i], tr[1][i]), tr[2][i]).backward(); opt.step()
    return model


@torch.no_grad()
def acc(model, te):
    return float(sum((model(te[0][b:b + 4096], te[1][b:b + 4096]).argmax(1) == te[2][b:b + 4096]).sum().item() for b in range(0, len(te[2]), 4096)) / len(te[2]))


def part_clevr(seeds, out):
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); U = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep")
    split = np.random.default_rng(14).permutation(len(Z["imgs"])); itr, ite = split[:700], split[700:]
    for u in U.unit:
        T = Z[f"T__{u}"].astype(int)
        for feat in ("sym", "clip"):
            Xi = torch.tensor(Z["Ximg"] if feat == "sym" else Z["Cimg"], device=dev); Xq = torch.tensor(Z[f"Xq__{u}"] if feat == "sym" else Z[f"Cq__{u}"], device=dev)
            if feat == "clip":
                Xi = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-6); Xq = (Xq - Xq.mean(0)) / (Xq.std(0) + 1e-6)
            for seed in seeds:  # same cell sampling as step2_train.py
                rng = np.random.default_rng(seed); torch.manual_seed(seed)
                def cells(imgset, n):
                    q, i = rng.integers(0, T.shape[0], n), imgset[rng.integers(0, len(imgset), n)]
                    return Xi[torch.tensor(i, device=dev)], Xq[torch.tensor(q, device=dev)], torch.tensor(T[q, i], device=dev)
                tr, te = cells(itr, 50000), cells(ite, 20000)
                a = acc(fit(Additive(Xi.shape[1], Xq.shape[1], 2).to(dev), tr), te)
                pd.DataFrame([dict(unit=u, feat=feat, seed=seed, acc_add=a)]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        print(u, flush=True)


def part_bench(seeds, out):
    for name in ("avmnist", "cifar2v", "clevr", "cremad", "food101", "n24news", "vqa"):
        Z = np.load(os.path.join(ROOT, "data", "feats", f"{name}.npz"), allow_pickle=True); s = Z["split"]
        f1, f2 = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32); y = pd.factorize(Z["y"])[0]
        nz = lambda f: (f - f[s == 0].mean(0)) / (f[s == 0].std(0) + 1e-6)
        X1, X2, Y = torch.tensor(nz(f1), device=dev), torch.tensor(nz(f2), device=dev), torch.tensor(y, device=dev); C = int(y.max()) + 1
        tr = tuple(v[torch.tensor(s == 0, device=dev)] for v in (X1, X2, Y)); te = tuple(v[torch.tensor(s == 1, device=dev)] for v in (X1, X2, Y))
        for seed in seeds:
            torch.manual_seed(seed); t0 = time.time()
            r = dict(data=name, seed=seed, n_train=len(tr[2]), n_test=len(te[2]), classes=C,
                     acc_full=acc(fit(Net(X1.shape[1], X2.shape[1], C).to(dev), tr), te),
                     acc_add=acc(fit(Additive(X1.shape[1], X2.shape[1], C).to(dev), tr), te),
                     acc_uni1=acc(fit(Uni(X1.shape[1], C, 1).to(dev), tr), te), acc_uni2=acc(fit(Uni(X2.shape[1], C, 2).to(dev), tr), te))
            p = np.bincount(te[2].cpu().numpy(), minlength=C) / len(te[2]); r["acc_maj"] = float(p.max())
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(r, f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25])
    ap.add_argument("--out", default=None); a = ap.parse_args(); out = os.path.join(ROOT, "runs", a.out or f"additive_est_{a.part}.csv")
    (part_clevr if a.part == "clevr" else part_bench)(a.seeds, out)
