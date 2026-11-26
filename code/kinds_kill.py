"""Kill test (paper/preregistration_kinds_kill.md): comparative vs conjunctive task-1 knowledge under COUPLED task-2 drift
(cross-modal InfoNCE on task-2 classes, task-1 head frozen). Output (appended): runs/kinds_kill.csv"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from route_exps import Net, train
dev = "cuda"; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = np.load(os.path.join(ROOT, "data", "feats", "cifar2v.npz"), allow_pickle=True)
f1, f2, Y, SPL = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32), Z["y"].astype(int), Z["split"]
nz = lambda f: (f - f[SPL == 0].mean(0)) / (f[SPL == 0].std(0) + 1e-6)
F1, F2 = torch.tensor(nz(f1), device=dev), torch.tensor(nz(f2), device=dev)
perm = np.random.default_rng(900).permutation(Y.max() + 1); POOL1, POOL2 = perm[:10], perm[10:20]
ITEMS = {s: {c: np.where((Y == c) & (SPL == s))[0] for c in np.unique(Y)} for s in (0, 1)}


def balanced_table(rng, K=10, swaps=5000):
    T = ((np.arange(K)[:, None] + np.arange(K)[None]) % K < K // 2).astype(int)
    for _ in range(swaps):  # checkerboard swaps keep every row and column sum at K/2
        r, c = rng.choice(K, 2, replace=False), rng.choice(K, 2, replace=False)
        sub = T[np.ix_(r, c)]
        if sub[0, 0] == sub[1, 1] != sub[0, 1] == sub[1, 0]:
            T[np.ix_(r, c)] = 1 - sub
    return T


def pairs(kind, n, sp, rng, T, attr):
    K = 10
    ci = rng.integers(0, K, n)
    if kind == "comparative":
        pos = rng.random(n) < 0.5; cj = np.where(pos, ci, (ci + rng.integers(1, K, n)) % K); y = pos.astype(int)
    else:
        cj = rng.integers(0, K, n); y = T[ci, cj] if kind == "conjunctive" else attr[ci] ^ attr[cj]
    ia = np.array([rng.choice(ITEMS[sp][POOL1[c]]) for c in ci]); ib = np.array([rng.choice(ITEMS[sp][POOL1[c]]) for c in cj])
    while (bad := ia == ib).any():  # never two views of the same item
        ib[bad] = [rng.choice(ITEMS[sp][POOL1[c]]) for c in cj[bad]]
    t = lambda a: torch.tensor(a, device=dev)
    return F1[t(ia)], F2[t(ib)], t(y)


def coupled_task2(net, epochs, seed, bs=256, tau=0.1):
    """Symmetric cross-modal InfoNCE on task-2 items (both views of the same item = positive). Head untouched."""
    idx = torch.tensor(np.concatenate([ITEMS[0][c] for c in POOL2]), device=dev)
    opt = torch.optim.Adam(list(net.e1.parameters()) + list(net.e2.parameters()), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(epochs * (50000 // bs)):
        i = idx[torch.randint(0, len(idx), (bs,), device=dev, generator=g)]
        a, b = F.normalize(net.e1(F1[i]), dim=1), F.normalize(net.e2(F2[i]), dim=1)
        lo = a @ b.T / tau; t = torch.arange(bs, device=dev)
        loss = (F.cross_entropy(lo, t) + F.cross_entropy(lo.T, t)) / 2
        opt.zero_grad(); loss.backward(); opt.step()


def linear_check(z_tr, y_tr, z_te, y_te, steps=500):
    lin = torch.nn.Linear(z_tr.shape[1], 2).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(lin(z_tr), y_tr).backward(); opt.step()
    with torch.no_grad():
        return (lin(z_te).argmax(1) == y_te).float().mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25, 26, 27])
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 10, 30]); ap.add_argument("--out", default="kinds_kill")
    args = ap.parse_args(); out = os.path.join(ROOT, "runs", f"{args.out}.csv")
    for seed in args.seeds:
        rng = np.random.default_rng(seed); T = balanced_table(rng); attr = rng.permutation([0] * 5 + [1] * 5)
        for kind in ("comparative", "conjunctive", "xor"):
            t0 = time.time(); torch.manual_seed(seed)
            tr, te = pairs(kind, 50000, 0, rng, T, attr), pairs(kind, 20000, 1, rng, T, attr)
            net1 = Net(512, 512, 2).to(dev); train(net1, tr, 30, 1e-3)
            for p in net1.parameters():
                p.requires_grad_(False)
            x1, x2, y = te
            with torch.no_grad():
                o1, o2 = net1.e1(x1), net1.e2(x2); ok0 = net1.head(torch.cat([o1, o2], 1)).argmax(1) == y
                ztr = torch.cat([net1.e1(tr[0][:20000]), net1.e2(tr[1][:20000])], 1)
            lin = linear_check(ztr, tr[2][:20000], torch.cat([o1, o2], 1), y)
            acc0 = ok0.float().mean().item()
            for lev in args.levels:
                net = Net(512, 512, 2).to(dev); net.load_state_dict(net1.state_dict())
                for p in net.head.parameters():
                    p.requires_grad_(False)
                coupled_task2(net, lev, 5000 + seed)
                with torch.no_grad():
                    n1, n2 = net.e1(x1), net.e2(x2); h = lambda a, b: net1.head(torch.cat([a, b], 1)).argmax(1) == y
                    s1, s2, s12 = h(n1, o2), h(o1, n2), h(n1, n2)
                    cos = lambda a, b: F.cosine_similarity(a, b, 1).mean().item()
                acc = lambda s: s.float().mean().item(); k = lambda s: (acc(s) - 0.5) / (acc0 - 0.5)
                m = ok0
                r = dict(seed=seed, kind=kind, level=lev, acc0=acc0, lin_acc=lin, acc1=acc(s1), acc2=acc(s2), acc12=acc(s12),
                         k1=k(s1), k2=k(s2), k12=k(s12), E=k(s12) - k(s1) * k(s2),
                         P1=acc(s1[m]), P2=acc(s2[m]), r12=acc(s12[m]), P1n2=acc((s1 & s2)[m]),
                         drift1=1 - cos(n1, o1), drift2=1 - cos(n2, o2))
                r["C"] = r["P1n2"] - r["P1"] * r["P2"]; r["N"] = r["r12"] - r["P1n2"]
                pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"s{seed} {kind:11s} L{lev:2d} acc0 {acc0:.3f} lin {lin:.3f} | k1 {r['k1']:.2f} k2 {r['k2']:.2f} k12 {r['k12']:.2f} "
                      f"E {r['E']:+.3f} | N {r['N']:+.3f} | drift {r['drift1']:.2f}/{r['drift2']:.2f}", flush=True)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
