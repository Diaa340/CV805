"""Step 2 training (paper/preregistration_step2.md): per CLEVR unit (10 questions x 1,000 images, set-lifted answers),
feature type (sym / clip) and seed: train the two-branch model on (image, question) cells of 700 train images, test on the
300 held-out images; linear readout on frozen codes; additive ceiling = accuracy of the best additive fit of the table
([u_q + w_img > 0], logistic, fitted on the full table) on the test cells. Output: runs/clevr2/train.csv"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from kinds_kill import linear_check
from route_exps import Net, train
dev = "cuda"; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2")


def additive_pred(T, steps=3000):
    q, i = np.meshgrid(np.arange(T.shape[0]), np.arange(T.shape[1]), indexing="ij"); y = torch.tensor(T, device=dev).float()
    u = torch.zeros(T.shape[0], device=dev, requires_grad=True); w = torch.zeros(T.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True); opt = torch.optim.Adam([u, w, b], lr=0.1)
    for _ in range(steps):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(u[:, None] + w[None] + b, y).backward(); opt.step()
    with torch.no_grad():
        return ((u[:, None] + w[None] + b) > 0).long().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25])
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1]); ap.add_argument("--out", default="train"); args = ap.parse_args()
    Z = np.load(os.path.join(OUT, "data.npz")); U = pd.read_csv(os.path.join(OUT, "families.csv")); U = U[U.keep].unit.tolist()
    U = [u for n, u in enumerate(U) if n % args.shard[1] == args.shard[0]]
    split = np.random.default_rng(14).permutation(len(Z["imgs"])); itr, ite = split[:700], split[700:]
    out = os.path.join(OUT, f"{args.out}.csv")
    for u in U:
        T = Z[f"T__{u}"].astype(int); A = additive_pred(T); ceil_test = float((A[:, ite] == T[:, ite]).mean())
        for feat in ("sym", "clip"):
            Xi = torch.tensor(Z["Ximg"] if feat == "sym" else Z["Cimg"], device=dev)
            Xq = torch.tensor(Z[f"Xq__{u}"] if feat == "sym" else Z[f"Cq__{u}"], device=dev)
            if feat == "clip":
                Xi = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-6); Xq = (Xq - Xq.mean(0)) / (Xq.std(0) + 1e-6)
            for seed in args.seeds:
                t0 = time.time(); rng = np.random.default_rng(seed); torch.manual_seed(seed)
                def cells(imgset, n):
                    q, i = rng.integers(0, T.shape[0], n), imgset[rng.integers(0, len(imgset), n)]
                    return Xi[torch.tensor(i, device=dev)], Xq[torch.tensor(q, device=dev)], torch.tensor(T[q, i], device=dev)
                tr, te = cells(itr, 50000), cells(ite, 20000)
                net = Net(Xi.shape[1], Xq.shape[1], 2).to(dev); train(net, tr, 30, 1e-3)
                with torch.no_grad():
                    ztr = torch.cat([net.e1(tr[0][:20000]), net.e2(tr[1][:20000])], 1); zte = torch.cat([net.e1(te[0]), net.e2(te[1])], 1)
                    acc_full = (net.head(zte).argmax(1) == te[2]).float().mean().item()
                acc_lin = linear_check(ztr, tr[2][:20000], zte, te[2]); p = te[2].float().mean().item()
                r = dict(unit=u, feat=feat, seed=seed, acc_full=acc_full, acc_lin=acc_lin, acc_maj=max(p, 1 - p), ceil_test=ceil_test)
                pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                print(f"{u:6s} {feat:4s} s{seed} full {acc_full:.3f} lin {acc_lin:.3f} ceiling {ceil_test:.3f} maj {r['acc_maj']:.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
