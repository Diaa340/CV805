"""
Route law v2 (paper/preregistration_v2.md): content-specific routes, synthetic test.
Per unit (config, seed): task 1 -> task 2 under none / control / freeze_m / distill1_m / replayonly_m; for every
(modality m, type t) three survival measures of the content modality m carries for type t:
  qp (primary)   : linear decodability of code c_m from encoder m, re-fitted after task 2 (chance-corrected retention)
  qf (secondary) : the readout fitted before task 2 applied unchanged after task 2
  qr (tertiary)  : ridge recoverability of the old encoder-m embedding from the new one (R^2), no labels or codes
Output (appended): runs/<tag>_v2.csv
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from route_exps import Net, per_type, train

dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["R", "U1", "U2", "S"]
K = 10


def make_codes(n, d, snr, mix, emb, rng):
    """stage0 generator (type-specific code tables), also returning the codes c1, c2."""
    t = rng.choice(4, n, p=mix); y = rng.integers(0, K, n)
    c1, c2 = y.copy(), y.copy()
    r1, r2 = rng.integers(0, K, n), rng.integers(0, K, n)
    c2[t == 1] = r2[t == 1]; c1[t == 2] = r1[t == 2]
    a = r1[t == 3]; c1[t == 3], c2[t == 3] = a, (y[t == 3] - a) % K
    x1 = snr[0] * emb[0][t, c1] + rng.standard_normal((n, d)); x2 = snr[1] * emb[1][t, c2] + rng.standard_normal((n, d))
    T = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
    L = lambda v: torch.tensor(v, device=dev)
    return dict(x1=T(x1), x2=T(x2), y=L(y), t=t, c1=L(c1), c2=L(c2))


def fit_lin(h, y, steps=300):
    lin = nn.Linear(h.shape[1], K).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        i = torch.randint(0, len(y), (min(1024, len(y)),), device=dev)
        loss = F.cross_entropy(lin(h[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return lin


@torch.no_grad()
def lin_acc(lin, h, y):
    return (lin(h).argmax(1) == y).float().mean().item()


@torch.no_grad()
def ridge_r2(xtr, ytr, xte, yte, lam=1e-2):
    X = torch.cat([xtr, torch.ones(len(xtr), 1, device=dev)], 1)
    W = torch.linalg.solve(X.T @ X + lam * len(X) * torch.eye(X.shape[1], device=dev), X.T @ ytr)
    P = torch.cat([xte, torch.ones(len(xte), 1, device=dev)], 1) @ W
    return (1 - ((yte - P) ** 2).sum() / ((yte - yte.mean(0)) ** 2).sum()).item()


def route_measures(net_old, net_new, tr, te, readouts, cap=8000):
    """qp, qf, qr for every (m, t)."""
    out = {}
    for m in (1, 2):
        enc_o, enc_n = (net_old.e1, net_new.e1) if m == 1 else (net_old.e2, net_new.e2)
        xk, ck = f"x{m}", f"c{m}"
        for j, t in enumerate(TYPES):
            itr = torch.tensor(np.where(tr["t"] == j)[0][:cap], device=dev); ite = torch.tensor(np.where(te["t"] == j)[0], device=dev)
            with torch.no_grad():
                ho_tr, ho_te = enc_o(tr[xk][itr]), enc_o(te[xk][ite]); hn_tr, hn_te = enc_n(tr[xk][itr]), enc_n(te[xk][ite])
            lin0, acc0 = readouts[(m, t)]
            acc_re = lin_acc(fit_lin(hn_tr, tr[ck][itr]), hn_te, te[ck][ite])
            acc_fx = lin_acc(lin0, hn_te, te[ck][ite])
            den = max(acc0 - 1 / K, 1e-6)
            out[f"qp_{m}{t}"] = float(np.clip((acc_re - 1 / K) / den, 0, 1))
            out[f"qf_{m}{t}"] = float(np.clip((acc_fx - 1 / K) / den, 0, 1))
            out[f"qr_{m}{t}"] = float(np.clip(ridge_r2(hn_tr, ho_tr, hn_te, ho_te), 0, 1))
            out[f"code_acc0_{m}{t}"] = acc0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--batch", default="heldout")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    conds = ["none", "control"] + [f"{k}_{m}" for m in (1, 2) for k in ("freeze", "distill1", "replayonly")]
    cfg = dict(snr=f"{args.snr[0]}/{args.snr[1]}", shift=args.shift, depth=args.depth, width=args.width, batch=args.batch)
    mix = [0.4, 0.15, 0.15, 0.3]; c = 1 / K
    for seed in args.seeds:
        t0 = time.time(); rng = np.random.default_rng(seed)
        mk = lambda: [e / np.linalg.norm(e, axis=2, keepdims=True) for e in (rng.standard_normal((4, K, 64)) for _ in range(2))]
        emb1, new = mk(), mk()
        emb2 = [new[0] if "1" in args.shift else emb1[0], new[1] if "2" in args.shift else emb1[1]]
        tr1, te1, tr2 = make_codes(50000, 64, args.snr, mix, emb1, rng), make_codes(20000, 64, args.snr, mix, emb1, rng), make_codes(50000, 64, args.snr, mix, emb2, rng)
        pack = lambda d: (d["x1"], d["x2"], d["y"])
        types = np.array(TYPES)[te1["t"]]
        w1 = torch.tensor(1 + 0.75 * np.array([-1.0, 0.0, 0.0, 1.0])[tr1["t"]], device=dev, dtype=torch.float32)
        torch.manual_seed(seed)
        net1 = Net(64, 64, K, args.depth, args.width).to(dev)
        train(net1, pack(tr1), 30, 1e-3, w=w1)
        before = per_type(net1, pack(te1), list(range(K)), types)
        readouts = {}
        for m in (1, 2):
            enc = net1.e1 if m == 1 else net1.e2
            for j, t in enumerate(TYPES):
                itr = torch.tensor(np.where(tr1["t"] == j)[0][:8000], device=dev); ite = torch.tensor(np.where(te1["t"] == j)[0], device=dev)
                with torch.no_grad():
                    h_tr, h_te = enc(tr1[f"x{m}"][itr]), enc(te1[f"x{m}"][ite])
                lin = fit_lin(h_tr, tr1[f"c{m}"][itr]); readouts[(m, t)] = (lin, lin_acc(lin, h_te, te1[f"c{m}"][ite]))
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        bi = torch.randperm(50000, generator=g)[:1000].to(dev); buf = tuple(v[bi] for v in pack(tr1))
        rows = []
        for cond in conds:
            net = copy.deepcopy(net1)
            train(net, pack(tr2), 10, 1e-3, buf=None if cond == "none" else buf, cond=cond, old=net1)
            after = per_type(net, pack(te1), list(range(K)), types)
            row = dict(**cfg, seed=seed, cond=cond, c=c)
            for t in TYPES:
                row[f"before_{t}"] = before[t]; row[f"after_{t}"] = after[t]
                row[f"ret_{t}"] = (after[t] - c) / (before[t] - c) if before[t] - c > 1e-6 else np.nan
            row.update(route_measures(net1, net, tr1, te1, readouts))
            rows.append(row)
        p = os.path.join(ROOT, "runs", f"{args.tag}_v2.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        r = pd.DataFrame(rows).set_index("cond").loc["control"]
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) before " + " ".join(f"{t}={before[t]:.2f}" for t in TYPES)
              + f" | control ret S {r.ret_S:.2f} qp1S {r.qp_1S:.2f} qp2S {r.qp_2S:.2f} (pred {r.qp_1S * r.qp_2S:.2f}) | qr1S {r.qr_1S:.2f} qr2S {r.qr_2S:.2f}", flush=True)


if __name__ == "__main__":
    main()
