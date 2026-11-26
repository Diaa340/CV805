"""
Fusion-Interface Anchoring (FIA) and baselines (paper/preregistration_v4_fia.md). Per unit: task 1 once, then task 2
trained under each method from the same task-1 model. Records task-1 retention by type and by interaction-share
quintile, task-2 accuracy, average accuracy, BWT, and time per step.
Output (appended): runs/<tag>_fia.csv
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import jvp

from p3_readout import refit
from route_exps import Net, load_unit, train
from v3_drift import model_types
from v4_measure import profile

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["R", "U1", "U2", "S"]
LAMS = [0.1, 0.3, 1.0, 3.0, 10.0]


def masked_logits(z, cls):
    mask = torch.full((z.shape[1],), float("-inf"), device=dev); mask[cls] = 0
    return z + mask


def kl_old_new(lo, ln, cls, T=2.0):
    """KL(softmax(old/T) || softmax(new/T)) over the task-1 classes."""
    lo, ln = lo[:, cls] / T, ln[:, cls] / T
    return F.kl_div(F.log_softmax(ln, 1), F.log_softmax(lo, 1), log_target=True, reduction="batchmean") * T * T


def fisher(net, data, n=2000, bs=32):
    for p in net.parameters():
        p.requires_grad_(True)
    F_ = {k: torch.zeros_like(p) for k, p in net.named_parameters()}; x1, x2, y = data
    idx = torch.randperm(len(y), device=dev)[:n]
    for i in range(0, n, bs):
        j = idx[i:i + bs]; net.zero_grad(); F.cross_entropy(net(x1[j], x2[j]), y[j]).backward()
        for k, p in net.named_parameters():
            F_[k] += p.grad.detach() ** 2 * len(j) / n
    net.zero_grad(); return F_


def train_task2(net, old, U, method, lam, buf=None, buf_logits=None, epochs=10, bs=256):
    """One task-2 run. `old` = frozen task-1 model (encoders + head)."""
    c1 = U["c1"]; x1, x2, y = U["tr2"]; n = len(y); steps = epochs * max(1, n // bs)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    if method == "ewc":
        Fi = fisher(copy.deepcopy(old), U["tr1"]); theta0 = {k: p.detach().clone() for k, p in old.named_parameters()}
    kind = method.split("+")[0]
    t0 = time.time()
    for _ in range(steps):
        i = torch.randint(0, n, (bs,), device=dev); a, b, t = x1[i], x2[i], y[i]
        loss = F.cross_entropy(net(a, b), t)
        if kind == "fia_mixed":  # cross-modal mixed anchor pairs
            b = b[torch.randperm(bs, device=dev)]
        if lam and kind in ("lwf", "l2", "jac") or (lam and kind.startswith("fia")):
            with torch.no_grad():
                o1, o2 = old.e1(a), old.e2(b); lo = old.head(torch.cat([o1, o2], 1))
            n1, n2 = net.e1(a), net.e2(b)
            if kind == "lwf":
                loss = loss + lam * kl_old_new(lo, net.head(torch.cat([n1, n2], 1)), c1)
            elif kind == "l2":
                loss = loss + lam * ((n1 - o1).pow(2).sum(1).mean() + (n2 - o2).pow(2).sum(1).mean()) / o1.shape[1]
            elif kind == "jac":
                h = lambda z1, z2: old.head(torch.cat([z1, z2], 1))[:, c1]
                j1 = jvp(lambda z: h(z, o2), (o1,), (n1 - o1,))[1]; j2 = jvp(lambda z: h(o1, z), (o2,), (n2 - o2,))[1]
                loss = loss + lam * (j1.pow(2).sum(1).mean() + j2.pow(2).sum(1).mean())
            else:
                g = lambda z1, z2: old.head(torch.cat([z1, z2], 1))
                use_swap = kind in ("fia", "fia_swap", "fia_nobeta", "fia_mixed")
                use_joint = kind in ("fia", "fia_joint", "fia_nobeta", "fia_mixed")
                use_beta = kind in ("fia", "fia_mixed")
                if use_swap:
                    loss = loss + lam * (kl_old_new(lo, g(n1, o2), c1) + kl_old_new(lo, g(o1, n2), c1))
                if use_joint:
                    loss = loss + lam * kl_old_new(lo, g(n1, n2), c1)
                if use_beta:
                    p = torch.randperm(bs, device=dev)
                    beta = lambda z1, z2: (g(z1, z2) - g(z1, z2[p]) - g(z1[p], z2) + g(z1[p], z2[p]))[:, c1]
                    with torch.no_grad():
                        bo = beta(o1, o2)
                    loss = loss + lam * (beta(n1, n2) - bo).pow(2).mean()
        if method == "ewc" and lam:
            loss = loss + lam * sum((Fi[k] * (p - theta0[k]) ** 2).sum() for k, p in net.named_parameters())
        if buf is not None:
            j = torch.randint(0, len(buf[2]), (min(bs, len(buf[2])),), device=dev)
            loss = loss + F.cross_entropy(net(buf[0][j], buf[1][j]), buf[2][j])
            if kind == "derpp":
                loss = loss + 0.5 * F.mse_loss(net(buf[0][j], buf[1][j]), buf_logits[j])
        opt.zero_grad(); loss.backward(); opt.step()
    return (time.time() - t0) / steps


@torch.no_grad()
def evaluate(net, U, types, quint, before, c):
    x1, x2, y = U["te1"]; ok = (masked_logits(net(x1, x2), U["c1"]).argmax(1) == y).cpu().numpy()
    a1 = ok.mean(); X1, X2, Y2 = U["te2"]
    a2 = (masked_logits(net(X1, X2), U["c2"]).argmax(1) == Y2).float().mean().item()
    out = dict(task1_acc=a1, task2_acc=a2, avg_acc=(a1 + a2) / 2)
    for t in TYPES:
        k = types == t
        if k.sum() >= 30 and before[k].mean() - c >= 0.3:
            out[f"ret_{t}"] = (ok[k].mean() - c) / (before[k].mean() - c)
    for q in range(5):
        k = (quint == q) & before
        out[f"keep_q{q + 1}"] = ok[k].mean() if k.sum() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bridge_cifar2v")
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--class_split", type=int, default=0)
    ap.add_argument("--order", default="normal")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[23])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--quick", action="store_true", help="smoke test: one lambda, one task-2 epoch")
    args = ap.parse_args()
    lams, ep2 = ([1.0], 1) if args.quick else (LAMS, 10)
    cfg = dict(data=args.data, class_split=args.class_split, order=args.order)
    for seed in args.seeds:
        t0 = time.time(); U = load_unit(args, seed); c1 = U["c1"]; c = 1 / len(c1)
        torch.manual_seed(seed)
        d1, d2 = U["tr1"][0].shape[1], U["tr1"][1].shape[1]
        net1 = Net(d1, d2, U["C"], args.depth, args.width).to(dev); train(net1, U["tr1"], 30, 1e-3, w=U["w1"])
        for p in net1.parameters():
            p.requires_grad_(False)
        exact = args.data in ("synth", "fv_cremad") or args.data.startswith("bridge_")
        types = U["types"] if exact else model_types(net1, U["te1"], c1, seed)
        x1, x2, y = U["te1"]
        with torch.no_grad():
            z1, z2 = net1.e1(x1), net1.e2(x2)
            before = (masked_logits(net1(x1, x2), c1).argmax(1) == y).cpu().numpy()
        g = torch.Generator(device=dev).manual_seed(9000 + seed)
        perms = [(torch.randperm(len(y), generator=g, device=dev), torch.randperm(len(y), generator=g, device=dev)) for _ in range(3)]
        pr = profile(net1.head, z1, z2, y, c1, perms)
        share = (pr["beta"].abs() / (pr["a1"].abs() + pr["a2"].abs() + pr["beta"].abs() + 1e-8)).cpu().numpy()
        quint = np.digitize(share, np.quantile(share, [0.2, 0.4, 0.6, 0.8]))
        # replay buffers: candidates from task-1 train; priority random / needs-both / beta-weighted
        n1 = len(U["tr1"][2]); m = max(200, int(0.02 * n1)); gc = torch.Generator(device="cpu").manual_seed(100 + seed)
        cand = torch.randperm(n1, generator=gc)[:min(n1, 20 * m)].to(dev); CT = tuple(v[cand] for v in U["tr1"])
        with torch.no_grad():
            cz1, cz2 = net1.e1(CT[0]), net1.e2(CT[1])
        gp = torch.Generator(device=dev).manual_seed(9500 + seed)
        cperms = [(torch.randperm(len(cand), generator=gp, device=dev), torch.randperm(len(cand), generator=gp, device=dev)) for _ in range(3)]
        cpr = profile(net1.head, cz1, cz2, CT[2], c1, cperms)
        cshare = (cpr["beta"].abs() / (cpr["a1"].abs() + cpr["a2"].abs() + cpr["beta"].abs() + 1e-8)).cpu().numpy()
        rng = np.random.default_rng(seed)
        pick = {"random": rng.choice(len(cand), m, replace=False),
                "beta": rng.choice(len(cand), m, replace=False, p=cshare / cshare.sum())}
        # needs-both (evidence definition, all datasets): neither modality alone suffices, both together do
        e = {k: v.cpu().numpy() for k, v in cpr.items()}
        nb = np.where((e["b"] + e["a1"] < 0) & (e["b"] + e["a2"] < 0) & (e["b"] + e["a1"] + e["a2"] + e["beta"] > 0))[0]
        pick["needsboth"] = rng.choice(nb, m, replace=len(nb) < m) if len(nb) else pick["random"]
        bufs = {k: tuple(v[torch.tensor(ix, device=dev)] for v in CT) for k, ix in pick.items()}
        with torch.no_grad():
            buf_logits = net1(bufs["random"][0], bufs["random"][1])
        runs = [("finetune", 0.0, None)]
        for meth in ("ewc", "lwf", "l2", "jac", "fia", "fia_joint", "fia_swap", "fia_nobeta", "fia_mixed"):
            runs += [(meth, lam, None) for lam in lams]
        runs += [("replay", 0.0, "random"), ("derpp", 0.0, "random")] + [("fia+buf", 1.0, k) for k in ("random", "needsboth", "beta")]
        rows = []
        for meth, lam, bk in runs:
            torch.manual_seed(1000 + seed); net = copy.deepcopy(net1)
            for p in net.parameters():
                p.requires_grad_(True)
            tps = train_task2(net, net1, U, meth, lam, buf=bufs[bk] if bk else None, buf_logits=buf_logits, epochs=ep2)
            rows.append(dict(**cfg, seed=seed, method=meth, lam=lam, buffer=bk or "", sec_per_step=tps, **evaluate(net, U, types, quint, before, c)))
        # joint training (upper bound) and post-hoc oracle encoder re-alignment
        torch.manual_seed(1000 + seed); jn = Net(d1, d2, U["C"], args.depth, args.width).to(dev)
        J = tuple(torch.cat([a, b]) for a, b in zip(U["tr1"], U["tr2"])); train(jn, J, 30, 1e-3)
        rows.append(dict(**cfg, seed=seed, method="joint", lam=0.0, buffer="", sec_per_step=np.nan, **evaluate(jn, U, types, quint, before, c)))
        ft = copy.deepcopy(net1)
        for p in ft.parameters():
            p.requires_grad_(True)
        train_task2(ft, net1, U, "finetune", 0.0, epochs=ep2)
        orc = copy.deepcopy(ft); orc.head = copy.deepcopy(net1.head); orc = refit(orc, [orc.e1, orc.e2], U["tr1"], 3000)
        rows.append(dict(**cfg, seed=seed, method="oracle_realign", lam=0.0, buffer="", sec_per_step=np.nan, **evaluate(orc, U, types, quint, before, c)))
        for r in rows:
            r["bwt"] = r["task1_acc"] - before.mean(); r["task1_before"] = before.mean()
        p = os.path.join(ROOT, "runs", f"{args.tag}_fia.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) {len(rows)} runs", flush=True)


if __name__ == "__main__":
    main()
