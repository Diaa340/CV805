"""
Multi-dataset study: WHEN is cross-modal (synergistic) knowledge forgotten first in continual learning?

Frozen features (extract_features.py) -> per-modality MLP encoders + non-additive fusion MLP (+ optional unimodal
auxiliary heads). Class-incremental: classes split in half (two random class splits); task-1 accuracy is measured with
logits masked to task-1 classes (feature/fusion forgetting, not new-class output bias).

Per-sample interaction type of every task-1 TEST sample, from calibrated probes trained on task-1 TRAIN data
(80% fit / 20% temperature calibration): unimodal q1(y|x1), q2(y|x2), fusion q12(y|x1,x2) -> local i1, i2, i12 (bits),
I_ccs redundancy (Ince 2017), atoms red / u1 / u2 / syn; type = argmax atom, 'unrec' if i12 <= 0.
CREMA-D additionally carries human types (crowd ratings) for validation.

Candidate explanatory dimensions, measured per run:
  reliance   : task-1 accuracy of the fused model with modality 2 zeroed (-> uses m1) and with m1 zeroed (-> uses m2);
               balance = (smaller above-chance accuracy) / (larger above-chance accuracy), in [0, 1]
  similarity : linear CKA between modality-1 and modality-2 features (task-1 train), in [0, 1]
  mix        : fraction of task-1 test samples of each estimated type
Outcome: retention (fraction of correctly-learned task-1 test samples still correct after task 2) per type, and the
confidence-matched ratio syn/red, u1/red, u2/red (within bins of task-1 confidence; 1.0 = no type effect).
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["red", "u1", "u2", "syn"]


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self, d1, d2, C, h=256, e=128):
        super().__init__()
        self.e1, self.e2 = mlp(d1, h, e), mlp(d2, h, e)
        self.head = nn.Sequential(nn.Linear(2 * e, h), nn.ReLU(), nn.Linear(h, C))
        self.aux1, self.aux2 = nn.Linear(e, C), nn.Linear(e, C)

    def forward(self, x1, x2, drop=None):
        h1, h2 = self.e1(x1), self.e2(x2)
        if drop == 1:
            h1 = torch.zeros_like(h1)
        if drop == 2:
            h2 = torch.zeros_like(h2)
        return self.head(torch.cat([h1, h2], 1)), h1, h2


def linear_cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    hsic = (a.T @ b).pow(2).sum(); return (hsic / ((a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt())).item()


def fit(net, X1, X2, Y, idx, epochs, lr, aux, bs=256, buf=None, rb=32):
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    n = len(idx); spe = max(1, n // bs)
    for _ in range(epochs * spe):
        j = idx[torch.randint(0, n, (bs,), device=dev)]
        if buf is not None:
            j = torch.cat([j, buf[torch.randint(0, len(buf), (rb,), device=dev)]])
        out, h1, h2 = net(X1[j], X2[j])
        loss = F.cross_entropy(out, Y[j])
        if aux:
            loss = loss + F.cross_entropy(net.aux1(h1), Y[j]) + F.cross_entropy(net.aux2(h2), Y[j])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def masked_eval(net, X1, X2, Y, idx, classes, drop=None):
    net.eval()
    z = net(X1[idx], X2[idx], drop=drop)[0]
    mask = torch.full((z.shape[1],), float("-inf"), device=dev); mask[classes] = 0
    z = z + mask; y = Y[idx]
    corr = (z.argmax(1) == y).cpu().numpy(); p = F.softmax(z, 1).gather(1, y[:, None])[:, 0].cpu().numpy()
    net.train(); return corr, p


def probe_types(X1, X2, Y, tr, te, classes, steps, seed):
    """Calibrated probes on task-1 train -> local MIs and I_ccs atoms on task-1 test (labels among task-1 classes)."""
    torch.manual_seed(1000 + seed)
    perm = tr[torch.randperm(len(tr), device=dev)]; cut = int(0.8 * len(perm)); fitx, cal = perm[:cut], perm[cut:]
    remap = torch.full((int(Y.max()) + 1,), -1, device=dev, dtype=torch.long)
    remap[torch.tensor(classes, device=dev)] = torch.arange(len(classes), device=dev)
    C = len(classes); lp = {}
    for k in ("1", "2", "12"):
        if k == "12":
            m = Net(X1.shape[1], X2.shape[1], C).to(dev); f = lambda j: m(X1[j], X2[j])[0]
        else:
            X = X1 if k == "1" else X2; m = nn.Sequential(mlp(X.shape[1], 256, 256), nn.ReLU(), nn.Linear(256, C)).to(dev)
            f = lambda j, X=X: m(X[j])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        for _ in range(steps):
            j = fitx[torch.randint(0, len(fitx), (256,), device=dev)]
            loss = F.cross_entropy(f(j), remap[Y[j]]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            zc = f(cal)
        logT = torch.zeros(1, device=dev, requires_grad=True); o = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

        def closure():
            o.zero_grad(); l = F.cross_entropy(zc / logT.exp(), remap[Y[cal]]); l.backward(); return l
        o.step(closure)
        with torch.no_grad():
            lp[k] = F.log_softmax(f(te) / logT.exp(), 1).gather(1, remap[Y[te]][:, None])[:, 0]
    prior = torch.bincount(remap[Y[tr]], minlength=C).float() / len(tr)
    lpy = prior.log()[remap[Y[te]]]
    i1, i2, i12 = [((lp[k] - lpy) / math.log(2)).cpu().numpy() for k in ("1", "2", "12")]
    c = i1 + i2 - i12
    same = (np.sign(i1) == np.sign(i2)) & (np.sign(i2) == np.sign(i12)) & (np.sign(i12) == np.sign(c))
    red = np.where(same, c, 0.0)
    A = dict(i1=i1, i2=i2, i12=i12, red=red, u1=i1 - red, u2=i2 - red, syn=i12 - i1 - i2 + red)
    t = np.array(TYPES)[np.stack([A[k] for k in TYPES], 1).argmax(1)].astype(object)
    t[i12 <= 0] = "unrec"
    return A, t


def matched_ratio(corr0, p0, corr1, types, cls=None, min_n=10):
    """Mean over confidence bins (x class, if given) of retention(type)/retention(red)."""
    edges = [0.0, 0.5, 0.8, 0.95, 0.99, 1.01]; out = {}
    groups = [None] if cls is None else np.unique(cls)
    for t in ("u1", "u2", "syn"):
        r = []
        for g in groups:
            gm = np.ones(len(corr0), bool) if g is None else (cls == g)
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = corr0 & (p0 >= lo) & (p0 < hi) & gm
                mr, mt = b & (types == "red"), b & (types == t)
                if mr.sum() >= min_n and mt.sum() >= min_n and corr1[mr].mean() > 0:
                    r.append(corr1[mt].mean() / corr1[mr].mean())
        out[f"{t}/red"] = np.mean(r) if r else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cifar2v,avmnist,cremad")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--splits", type=int, default=2)
    ap.add_argument("--recipes", default="plain,aux")
    ap.add_argument("--epochs1", type=int, default=30)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-3)
    ap.add_argument("--buffer_frac", type=float, default=0.02)
    ap.add_argument("--probe_steps", type=int, default=3000)
    ap.add_argument("--out", default="feat_cl")
    args = ap.parse_args()
    rows = []
    for ds in args.datasets.split(","):
        Z = np.load(os.path.join(ROOT, "data", "feats", f"{ds}.npz"), allow_pickle=True)
        f1, f2, y, split = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32), Z["y"].astype(int), Z["split"]
        trm = split == 0
        norm = lambda f: (f - f[trm].mean(0)) / (f[trm].std(0) + 1e-6)
        X1, X2 = torch.tensor(norm(f1), device=dev), torch.tensor(norm(f2), device=dev)
        Y = torch.tensor(y, device=dev); C = int(y.max()) + 1
        human = Z["human_type"] if "human_type" in Z.files else None
        for sp in range(args.splits):
            perm = np.random.default_rng(500 + sp).permutation(C); c1, c2 = sorted(perm[:C // 2]), sorted(perm[C // 2:])
            m1, m2 = np.isin(y, c1), np.isin(y, c2)
            T = lambda m: torch.tensor(np.where(m)[0], device=dev)
            tr1, te1, tr2 = T(m1 & trm), T(m1 & ~trm), T(m2 & trm)
            te1_np = te1.cpu().numpy()
            cka = linear_cka(X1[tr1], X2[tr1])
            A, types = probe_types(X1, X2, Y, tr1, te1, c1, args.probe_steps, sp)
            mix = {f"frac_{t}": (types == t).mean() for t in TYPES + ["unrec"]}
            hinfo = ""
            if human is not None:
                h = human[te1_np]; hmap = {"red": "red", "u_a": "u1", "u_v": "u2", "syn": "syn"}
                hh = np.array([hmap.get(v, "unrec") for v in h])
                hinfo = f" | agreement est-vs-human type {np.mean(hh == types):.2f}"
            print(f"== {ds} split {sp}: C1={len(c1)} tr1={len(tr1)} te1={len(te1)} tr2={len(tr2)} | CKA={cka:.3f} | "
                  + " ".join(f"{k[5:]}={v:.2f}" for k, v in mix.items()) + hinfo, flush=True)
            for recipe in args.recipes.split(","):
                for seed in range(args.seeds):
                    torch.manual_seed(seed)
                    net = Net(X1.shape[1], X2.shape[1], C).to(dev)
                    fit(net, X1, X2, Y, tr1, args.epochs1, args.lr1, recipe == "aux")
                    corr0, p0 = masked_eval(net, X1, X2, Y, te1, c1)
                    only1 = masked_eval(net, X1, X2, Y, te1, c1, drop=2)[0].mean()
                    only2 = masked_eval(net, X1, X2, Y, te1, c1, drop=1)[0].mean()
                    ch = 1 / len(c1); a1, a2 = max(only1 - ch, 0), max(only2 - ch, 0)
                    balance = min(a1, a2) / max(a1, a2, 1e-6)
                    g = torch.Generator(device="cpu").manual_seed(100 + seed)
                    buf = tr1[torch.randperm(len(tr1), generator=g)[:max(200, int(args.buffer_frac * len(tr1)))].to(dev)]
                    state = {k: v.clone() for k, v in net.state_dict().items()}
                    for cond in ("none", "replay"):
                        net.load_state_dict(state)
                        fit(net, X1, X2, Y, tr2, args.epochs2, args.lr2, recipe == "aux", buf=buf if cond == "replay" else None)
                        corr1, _ = masked_eval(net, X1, X2, Y, te1, c1)
                        row = dict(dataset=ds, split=sp, recipe=recipe, seed=seed, cond=cond, cka=cka, acc_before=corr0.mean(),
                                   acc_after=corr1.mean(), only_m1=only1, only_m2=only2, balance=balance, **mix)
                        for t in TYPES:
                            mt = (types == t) & corr0
                            row[f"kept_{t}"] = corr1[mt].mean() if mt.sum() else np.nan
                        row.update(matched_ratio(corr0, p0, corr1, types))
                        if human is not None:
                            row.update({f"human_{k}": v for k, v in matched_ratio(corr0, p0, corr1, hh, cls=y[te1_np], min_n=5).items()})
                        rows.append(row)
                        print(f"   {recipe:5s} s{seed} {cond:6s} acc {corr0.mean():.3f}->{corr1.mean():.3f} | uses m1 {only1:.2f} m2 {only2:.2f} "
                              f"(balance {balance:.2f}) | kept " + " ".join(f"{t}={row['kept_' + t]:.2f}" for t in TYPES)
                              + f" | matched syn/red {row['syn/red']:.2f} u1/red {row['u1/red']:.2f} u2/red {row['u2/red']:.2f}", flush=True)
                        pd.DataFrame(rows).to_csv(os.path.join(ROOT, f"{args.out}_results.csv"), index=False)
    R = pd.DataFrame(rows)
    print("\n=== per dataset x recipe x condition (means over splits and seeds) ===")
    cols = ["cka", "balance", "frac_syn", "acc_before", "acc_after", "kept_red", "kept_syn", "syn/red", "u1/red", "u2/red"]
    print(R.groupby(["dataset", "recipe", "cond"])[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
