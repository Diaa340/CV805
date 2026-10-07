"""
v4 measurements (paper/preregistration_v4_fia.md + preregistration_v4_addendum.md).
Per unit: task 1 -> per-sample evidence profile (b, alpha1, alpha2, beta) through the task-1 head, fusion-sensitive
subspaces P_m; per regime (none / control): task 2, then per sample the route retention rho_m (feature geometry),
survival S1 / S2 / S12 through the task-1 head, actual correctness, and linear-readout recovery; per unit the
interface distortions kappa, kappa_1, kappa_2 and the compensation index kappa_c (finite-difference / Harsanyi form).
Output: runs/v4/<tag>_s<seed>_<regime>.npz
"""
import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from p3_readout import LinearReadout, refit
from route_exps import Net, load_unit, train
from v3_drift import model_types

dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
EXACT = ("synth", "fv_cremad")


def margin(head, a, b, y, c1, bs=8192):
    out = []
    for i in range(0, len(y), bs):
        z = head(torch.cat([a[i:i + bs], b[i:i + bs]], 1))
        mask = torch.full((z.shape[1],), float("-inf"), device=dev); mask[c1] = 0; z = z + mask
        yy = y[i:i + bs]; t = z.gather(1, yy[:, None])[:, 0]
        z = z.scatter(1, yy[:, None], float("-inf")); out.append(t - z.max(1).values)
    return torch.cat(out)


@torch.no_grad()
def profile(head, z1, z2, y, c1, perms, maps=(None, None)):
    """Per-sample b, alpha1, alpha2, beta averaged over replacement draws; maps apply the linear drift to features."""
    f1 = (lambda v: v) if maps[0] is None else maps[0]
    f2 = (lambda v: v) if maps[1] is None else maps[1]
    acc = {k: torch.zeros(len(y), device=dev) for k in ("b", "a1", "a2", "beta")}
    for p1, p2 in perms:
        Z1, Z2, B1, B2 = f1(z1), f2(z2), f1(z1[p1]), f2(z2[p2])
        s12, s1, s2, s0 = margin(head, Z1, Z2, y, c1), margin(head, Z1, B2, y, c1), margin(head, B1, Z2, y, c1), margin(head, B1, B2, y, c1)
        acc["b"] += s0; acc["a1"] += s1 - s0; acc["a2"] += s2 - s0; acc["beta"] += s12 - s1 - s2 + s0
    return {k: (v / len(perms)) for k, v in acc.items()}


def sensitive_subspace(head, z1, z2, y, c1, k=16, n=5000):
    a, b = z1[:n].clone().requires_grad_(True), z2[:n].clone().requires_grad_(True)
    s = margin(head, a, b, y[:n], c1).sum(); g1, g2 = torch.autograd.grad(s, (a, b))
    return [torch.linalg.svd(g, full_matrices=False)[2][:k].T for g in (g1, g2)]  # (d, k) each


@torch.no_grad()
def linmap(old, new, lam=1e-2):
    """Ridge regression new ~ [old, 1] B; returns a function applying the fitted linear drift."""
    X = torch.cat([old, torch.ones(len(old), 1, device=dev)], 1)
    B = torch.linalg.solve(X.T @ X + lam * len(X) * torch.eye(X.shape[1], device=dev), X.T @ new)
    return lambda v: torch.cat([v, torch.ones(len(v), 1, device=dev)], 1) @ B


@torch.no_grad()
def correct(head, a, b, y, c1):
    return (margin(head, a, b, y, c1) > 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth")
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--class_split", type=int, default=0)
    ap.add_argument("--order", default="normal")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[23])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "runs", "v4"), exist_ok=True)
    for seed in args.seeds:
        t0 = time.time()
        U = load_unit(args, seed); c1 = U["c1"]
        torch.manual_seed(seed)
        d1, d2 = U["tr1"][0].shape[1], U["tr1"][1].shape[1]
        net1 = Net(d1, d2, U["C"], args.depth, args.width).to(dev)
        train(net1, U["tr1"], 30, 1e-3, w=U["w1"])
        exact = args.data in EXACT or args.data.startswith("bridge_")
        types = U["types"] if exact else model_types(net1, U["te1"], c1, seed)
        x1, x2, y = U["te1"]; t1, t2, _ = U["tr1"]; cap = 20000
        g = torch.Generator(device=dev).manual_seed(9000 + seed)
        perms = [(torch.randperm(len(y), generator=g, device=dev), torch.randperm(len(y), generator=g, device=dev)) for _ in range(3)]
        with torch.no_grad():
            z1, z2 = net1.e1(x1), net1.e2(x2); o1, o2 = net1.e1(t1[:cap]), net1.e2(t2[:cap])
        prof = profile(net1.head, z1, z2, y, c1, perms)
        P1, P2 = sensitive_subspace(net1.head, z1, z2, y, c1)
        mu1, mu2 = o1.mean(0), o2.mean(0)
        before = correct(net1.head, z1, z2, y, c1)
        g2 = torch.Generator(device="cpu").manual_seed(100 + seed)
        n1 = len(U["tr1"][2]); bi = torch.randperm(n1, generator=g2)[:max(1000 if args.data == "synth" else 200, int(0.02 * n1))].to(dev)
        buf = tuple(v[bi] for v in U["tr1"])
        for regime in ("none", "control"):
            net = copy.deepcopy(net1)
            train(net, U["tr2"], 10, 1e-3, buf=None if regime == "none" else buf, cond="control")
            with torch.no_grad():
                n1_, n2_ = net.e1(x1), net.e2(x2); m1_tr, m2_tr = net.e1(t1[:cap]), net.e2(t2[:cap])

                def rho(new, old, P, mu):
                    a, b = (new - mu) @ P, (old - mu) @ P
                    return ((a * b).sum(1) / (b * b).sum(1).clamp(min=1e-8)).clamp(0, 1).cpu().numpy()
                r1, r2 = rho(n1_, z1, P1, mu1), rho(n2_, z2, P2, mu2)
                S1, S2, S12 = correct(net1.head, n1_, z2, y, c1), correct(net1.head, z1, n2_, y, c1), correct(net1.head, n1_, n2_, y, c1)
                actual = (margin(net.head, n1_, n2_, y, c1) > 0).cpu().numpy()
                M1, M2 = linmap(o1, m1_tr), linmap(o2, m2_tr)
                base = prof["beta"].abs().mean()
                kj = (profile(net1.head, z1, z2, y, c1, perms, (M1, M2))["beta"] - prof["beta"]).abs().mean() / base
                k1 = (profile(net1.head, z1, z2, y, c1, perms, (M1, None))["beta"] - prof["beta"]).abs().mean() / base
                k2 = (profile(net1.head, z1, z2, y, c1, perms, (None, M2))["beta"] - prof["beta"]).abs().mean() / base
            torch.manual_seed(4000 + seed)
            lin = LinearReadout(copy.deepcopy(net), U["C"]).to(dev); lin = refit(lin, [lin.head], U["tr1"], 3000)
            with torch.no_grad():
                zz = lin(x1, x2); mask = torch.full((zz.shape[1],), float("-inf"), device=dev); mask[c1] = 0
                lin_ok = ((zz + mask).argmax(1) == y).cpu().numpy()
            np.savez_compressed(os.path.join(ROOT, "runs", "v4", f"{args.tag}_s{seed}_{regime}.npz"),
                                types=types, before=before, S1=S1, S2=S2, S12=S12, actual=actual, lin=lin_ok, rho1=r1, rho2=r2,
                                **{k: v.cpu().numpy() for k, v in prof.items()}, kappa=kj.item(), kappa1=k1.item(), kappa2=k2.item(),
                                kappa_c=(0.5 * (k1 + k2) - kj).item(), data=args.data)
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) done", flush=True)


if __name__ == "__main__":
    main()
