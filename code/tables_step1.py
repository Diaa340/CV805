"""Step 1 (paper/preregistration_step1.md), per table set / table / seed:
 (P1-rep) readout gap on frozen task-1 codes;
 (1a) repair after drift: task 2 = a random balanced table on task-2 classes (whole model trained 10 epochs); then on the
      frozen drifted encoders, a linear head vs an MLP head is retrained on task-1 data;
 (1c) survival under single / joint encoder drift (old head), split into removable cells (T = nearest threshold table F)
      and non-removable cells (T != F);
 (1b) compositional OOD: a separate model trained on 80% of the (i, j) cells, tested on the 20% held-out cells.
Output (appended): runs/tables/step1_<set>.csv"""
import argparse, copy, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from kinds_kill import F1, F2, ITEMS, balanced_table, linear_check, perm
from route_exps import Net, train
dev = "cuda"; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TD = os.path.join(ROOT, "runs", "tables")
POOL1, POOL2 = perm[:10], perm[10:20]


def table_pairs(T, n, sp, rng, pool=POOL1, cells=None):
    """View 1 of an item of class i, view 2 of a different item of class j, label T[i, j]. cells: allowed (i, j) list."""
    if cells is None:
        ci, cj = rng.integers(0, 10, n), rng.integers(0, 10, n)
    else:
        c = np.array(cells)[rng.integers(0, len(cells), n)]; ci, cj = c[:, 0], c[:, 1]
    ia = np.array([rng.choice(ITEMS[sp][pool[c]]) for c in ci]); ib = np.array([rng.choice(ITEMS[sp][pool[c]]) for c in cj])
    while (bad := ia == ib).any():
        ib[bad] = [rng.choice(ITEMS[sp][pool[c]]) for c in cj[bad]]
    t = lambda a: torch.tensor(a, device=dev)
    return F1[t(ia)], F2[t(ib)], t(T[ci, cj]), t(ci), t(cj)


def mlp_head(ztr, ytr, zte, yte, steps=2000, bs=512):
    h = nn.Sequential(nn.Linear(ztr.shape[1], 256), nn.ReLU(), nn.Linear(256, 2)).to(dev); opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    for _ in range(steps):
        i = torch.randint(0, len(ytr), (bs,), device=dev); opt.zero_grad(); F.cross_entropy(h(ztr[i]), ytr[i]).backward(); opt.step()
    with torch.no_grad():
        return (h(zte).argmax(1) == yte).float().mean().item()


def bal_acc(pred, y):
    return float(np.mean([(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()]))


def holdout_cells(rng, n=20, cap=3):
    while True:
        H = [divmod(int(k), 10) for k in rng.choice(100, n, replace=False)]
        r, c = np.bincount([h[0] for h in H], minlength=10), np.bincount([h[1] for h in H], minlength=10)
        if r.max() <= cap and c.max() <= cap:
            return H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25])
    ap.add_argument("--tables", type=int, nargs="*"); ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--out", default=None); args = ap.parse_args()
    S = np.load(os.path.join(TD, f"{args.set}.npz")); Fs = np.load(os.path.join(TD, "orig_F.npz"))["F"] if args.set == "selected" else S["F"]
    out = os.path.join(TD, f"step1_{args.out or args.set}.csv"); ks = args.tables if args.tables else range(len(S["T"]))
    ks = [k for n, k in enumerate(ks) if n % args.shard[1] == args.shard[0]]
    for k in ks:
        T, Fn = S["T"][k].astype(int), Fs[k].astype(int)
        for seed in args.seeds:
            t0 = time.time(); rng = np.random.default_rng(seed); torch.manual_seed(seed)
            tr, te = table_pairs(T, 50000, 0, rng), table_pairs(T, 20000, 1, rng)
            net1 = Net(512, 512, 2).to(dev); train(net1, tr[:3], 30, 1e-3)
            for p in net1.parameters():
                p.requires_grad_(False)
            y = te[2]; p1 = y.float().mean().item(); r = dict(set=args.set, table=k, seed=seed, acc_maj=max(p1, 1 - p1))
            with torch.no_grad():
                o1, o2, q1, q2 = net1.e1(te[0]), net1.e2(te[1]), net1.e1(tr[0][:20000]), net1.e2(tr[1][:20000])
                ok0 = net1.head(torch.cat([o1, o2], 1)).argmax(1) == y
            r["acc_before"] = ok0.float().mean().item(); r["acc_lin"] = linear_check(torch.cat([q1, q2], 1), tr[2][:20000], torch.cat([o1, o2], 1), y)
            # 1a / 1c: task-2 drift
            T2 = balanced_table(np.random.default_rng(1000 + seed)); tr2 = table_pairs(T2, 50000, 0, np.random.default_rng(2000 + seed), POOL2)
            net2 = copy.deepcopy(net1)
            for p in net2.parameters():
                p.requires_grad_(True)
            train(net2, tr2[:3], 10, 1e-3)
            with torch.no_grad():
                n1, n2, m1, m2 = net2.e1(te[0]), net2.e2(te[1]), net2.e1(tr[0][:20000]), net2.e2(tr[1][:20000])
                h = lambda a, b: net1.head(torch.cat([a, b], 1)).argmax(1) == y
                s1, s2, s12 = h(n1, o2), h(o1, n2), h(n1, n2)
            r["acc_drift"] = s12.float().mean().item()
            zr_tr, zr_te = torch.cat([m1, m2], 1), torch.cat([n1, n2], 1)
            r["acc_rep_lin"] = linear_check(zr_tr, tr[2][:20000], zr_te, y); r["acc_rep_mlp"] = mlp_head(zr_tr, tr[2][:20000], zr_te, y)
            r["acc_mlp_orig"] = mlp_head(torch.cat([q1, q2], 1), tr[2][:20000], torch.cat([o1, o2], 1), y)
            rem = torch.tensor(T == Fn, device=dev)[te[3], te[4]]
            for g, m in (("rem", ok0 & rem), ("non", ok0 & ~rem)):
                a = lambda s: s[m].float().mean().item() if m.any() else np.nan
                r.update({f"n_{g}": int(m.sum()), f"P1_{g}": a(s1), f"P2_{g}": a(s2), f"r12_{g}": a(s12), f"P1n2_{g}": a(s1 & s2)})
            # 1b: compositional OOD
            H = holdout_cells(np.random.default_rng(5000 + seed)); seen = [(i, j) for i in range(10) for j in range(10) if (i, j) not in H]
            trs, tes_h = table_pairs(T, 50000, 0, rng, cells=seen), table_pairs(T, 20000, 1, rng, cells=H)
            tes_s = table_pairs(T, 20000, 1, rng, cells=seen)
            neto = Net(512, 512, 2).to(dev); train(neto, trs[:3], 30, 1e-3)
            with torch.no_grad():
                zo = lambda d: torch.cat([neto.e1(d[0]), neto.e2(d[1])], 1)
                pred_h = neto.head(zo(tes_h)).argmax(1).cpu().numpy(); pred_s = neto.head(zo(tes_s)).argmax(1).cpu().numpy()
            yh, ys = tes_h[2].cpu().numpy(), tes_s[2].cpu().numpy()
            r["ood_bal"] = bal_acc(pred_h, yh); r["seen_bal"] = bal_acc(pred_s, ys); r["ood_acc"] = (pred_h == yh).mean()
            lin = nn.Linear(256, 2).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
            with torch.no_grad():
                ztr_o = zo((trs[0][:20000], trs[1][:20000]))
            for _ in range(500):
                opt.zero_grad(); F.cross_entropy(lin(ztr_o), trs[2][:20000]).backward(); opt.step()
            with torch.no_grad():
                r["ood_lin_bal"] = bal_acc(lin(zo(tes_h)).argmax(1).cpu().numpy(), yh)
            r["ood_labels_both"] = int(len(np.unique(yh)) == 2)
            # ideal additive completion: one-hot logistic on seen cells, evaluated on held-out cells
            Xs = torch.tensor([[float(i == a) for a in range(10)] + [float(j == b) for b in range(10)] for i, j in seen], device=dev)
            Xh = torch.tensor([[float(i == a) for a in range(10)] + [float(j == b) for b in range(10)] for i, j in H], device=dev)
            ysc = torch.tensor([T[i, j] for i, j in seen], device=dev); lo = nn.Linear(20, 2).to(dev); opt = torch.optim.Adam(lo.parameters(), lr=5e-2)
            for _ in range(2000):
                opt.zero_grad(); F.cross_entropy(lo(Xs), ysc).backward(); opt.step()
            with torch.no_grad():
                r["ood_onehot_cells"] = (lo(Xh).argmax(1).cpu().numpy() == np.array([T[i, j] for i, j in H])).mean()
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"[{args.set}] table {k:2d} s{seed} before {r['acc_before']:.3f} lin {r['acc_lin']:.3f} drift {r['acc_drift']:.3f} "
                  f"rep lin {r['acc_rep_lin']:.3f} mlp {r['acc_rep_mlp']:.3f} | ood {r['ood_bal']:.3f} seen {r['seen_bal']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
