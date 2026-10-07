"""Label-table kill test, step 2 (paper/preregistration_tables.md): for each selected table and seed, train the task-1
model on two-view CIFAR pairs (view 1 of an item of class i, view 2 of an item of class j, label T[i, j]), then fit a
linear readout on the frozen codes. Output (appended): runs/tables/train.csv"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from kinds_kill import F1, F2, ITEMS, POOL1, linear_check
from route_exps import Net, train
dev = "cuda"; ROOT = os.path.dirname(os.path.abspath(__file__))


def table_pairs(T, n, sp, rng):
    """As kinds_kill.pairs(\"conjunctive\"), also returning the class IDs."""
    ci, cj = rng.integers(0, 10, n), rng.integers(0, 10, n)
    ia = np.array([rng.choice(ITEMS[sp][POOL1[c]]) for c in ci]); ib = np.array([rng.choice(ITEMS[sp][POOL1[c]]) for c in cj])
    while (bad := ia == ib).any():
        ib[bad] = [rng.choice(ITEMS[sp][POOL1[c]]) for c in cj[bad]]
    t = lambda a: torch.tensor(a, device=dev)
    return F1[t(ia)], F2[t(ib)], t(T[ci, cj]), t(ci), t(cj)


def onehot_check(tr, te, steps=2000):
    """Additive readout on the TRUE class IDs (one-hot i, one-hot j): the ideal-code version of the linear readout."""
    oh = lambda c: torch.cat([F.one_hot(c[0], 10), F.one_hot(c[1], 10)], 1).float()
    lin = torch.nn.Linear(20, 2).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=5e-2)
    Xtr, Xte = oh(tr), oh(te)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(lin(Xtr), tr[2]).backward(); opt.step()
    with torch.no_grad():
        return (lin(Xte).argmax(1) == te[2]).float().mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25])
    ap.add_argument("--tables", type=int, nargs="*"); ap.add_argument("--out", default="train")
    args = ap.parse_args(); S = np.load(os.path.join(ROOT, "runs", "tables", "selected.npz"))
    out = os.path.join(ROOT, "runs", "tables", f"{args.out}.csv")
    for k in (args.tables if args.tables else range(len(S["T"]))):
        T = S["T"][k].astype(int)
        for seed in args.seeds:
            t0 = time.time(); rng = np.random.default_rng(seed); torch.manual_seed(seed)
            tr, te = table_pairs(T, 50000, 0, rng), table_pairs(T, 20000, 1, rng)
            net = Net(512, 512, 2).to(dev); train(net, tr[:3], 30, 1e-3)
            with torch.no_grad():
                ztr = torch.cat([net.e1(tr[0][:20000]), net.e2(tr[1][:20000])], 1); zte = torch.cat([net.e1(te[0]), net.e2(te[1])], 1)
                acc_full = (net.head(zte).argmax(1) == te[2]).float().mean().item()
            acc_lin = linear_check(ztr, tr[2][:20000], zte, te[2])
            p1 = te[2].float().mean().item(); acc_maj = max(p1, 1 - p1)
            r = dict(table=k, seed=seed, acc_full=acc_full, acc_lin=acc_lin, acc_maj=acc_maj,
                     acc_onehot=onehot_check((tr[3], tr[4], tr[2]), (te[3], te[4], te[2])))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"table {k:2d} s{seed} full {acc_full:.3f} lin {acc_lin:.3f} onehot {r['acc_onehot']:.3f} maj {acc_maj:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
