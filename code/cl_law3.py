"""
Route law, three-modality test. Synthetic data with 3 modalities and 7 known per-sample interaction types:
  R123 : all three modalities encode y               (redundant across 3 routes)
  R12  : m1 and m2 encode y, m3 uninformative         (redundant across 2 routes)
  U1, U2, U3 : only that modality encodes y          (unique)
  S12  : m1 = a, m2 = y - a, m3 uninformative        (pairwise synergy)
  S123 : m1 = a, m2 = b, m3 = y - a - b  (mod K)      (three-way synergy)
Type-specific code tables (types identifiable from the input; uninformative modalities are uninformative).
Task 1 learned (synergistic types up-weighted so that they are actually learned), then task 2 = new code tables for a
SUBSET of modalities (7 shift patterns), with / without replay.

Route law (zero parameters), with p_m = chance-corrected retention of U_m:
  R123 = 1 - (1-p1)(1-p2)(1-p3)   R12 = 1 - (1-p1)(1-p2)   S12 = p1 p2   S123 = p1 p2 p3
Shared-bottleneck version: every composite type additionally multiplied by h (one parameter per run, fitted by least
squares on the four composite types), with the unique types themselves = h * q_m (so q_m = p_m / h).
"""
import argparse
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

dev = "cuda"
TYPES = ["R123", "R12", "U1", "U2", "U3", "S12", "S123"]


def tables(rng, K, d):
    e = rng.standard_normal((3, len(TYPES), K, d))
    return e / np.linalg.norm(e, axis=3, keepdims=True)


def make(n, K, d, snr, mix, emb, rng):
    t = rng.choice(len(TYPES), n, p=mix); y = rng.integers(0, K, n)
    c = [rng.integers(0, K, n) for _ in range(3)]  # default: uninformative random codes
    for m in range(3):
        c[m] = c[m].copy()
    sel = lambda name: t == TYPES.index(name)
    for m in range(3):
        c[m][sel("R123")] = y[sel("R123")]
    for m in (0, 1):
        c[m][sel("R12")] = y[sel("R12")]
    for m, name in enumerate(("U1", "U2", "U3")):
        c[m][sel(name)] = y[sel(name)]
    s = sel("S12"); c[1][s] = (y[s] - c[0][s]) % K
    s = sel("S123"); c[2][s] = (y[s] - c[0][s] - c[1][s]) % K
    x = [snr[m] * emb[m][t, c[m]] + rng.standard_normal((n, d)) for m in range(3)]
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    return dict(x=[T(v) for v in x], y=torch.tensor(y, device=dev), t=t)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Net(nn.Module):
    def __init__(self, d, K, h=256, e=128):
        super().__init__()
        self.enc = nn.ModuleList([mlp(d, h, e) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(3 * e, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, K))

    def forward(self, x):
        return self.head(torch.cat([enc(v) for enc, v in zip(self.enc, x)], 1))


def train(net, data, epochs, bs, lr, w=None, buf=None, rb=64, onecycle=True):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    n = len(data["y"]); total = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total, pct_start=0.25) if onecycle else None
    for _ in range(total):
        i = torch.randint(0, n, (bs,), device=dev)
        l = F.cross_entropy(net([v[i] for v in data["x"]]), data["y"][i], reduction="none")
        loss = (l * w[i] / w[i].mean()).mean() if w is not None else l.mean()
        if buf is not None:
            j = torch.randint(0, len(buf["y"]), (rb,), device=dev)
            loss = loss + F.cross_entropy(net([v[j] for v in buf["x"]]), buf["y"][j])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()


@torch.no_grad()
def per_type_acc(net, data):
    net.eval(); corr = (net(data["x"]).argmax(1) == data["y"]).cpu().numpy(); net.train()
    return {k: corr[data["t"] == j].mean() for j, k in enumerate(TYPES)}


def predict(p, h=1.0):
    p1, p2, p3 = [min(max(v, 0.0), 1.0) for v in p]
    return dict(R123=h * (1 - (1 - p1) * (1 - p2) * (1 - p3)), R12=h * (1 - (1 - p1) * (1 - p2)),
                S12=h * p1 * p2, S123=h * p1 * p2 * p3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--n_test", type=int, default=30000)
    ap.add_argument("--snr", type=float, nargs=3, default=[4.5, 4.5, 4.5])
    ap.add_argument("--mix", type=float, nargs=7, default=[0.15, 0.10, 0.10, 0.10, 0.10, 0.20, 0.25])
    ap.add_argument("--epochs1", type=int, default=60)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--lr1", type=float, default=0.05)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shifts", default="123,1,2,3,12,13,23")
    ap.add_argument("--out", default="cl_law3")
    args = ap.parse_args()
    K, d, c = args.K, args.d, 1.0 / args.K
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        emb1 = [tables(rng, K, d)[m] for m in range(3)]
        tr1 = make(args.n_train, K, d, args.snr, args.mix, emb1, rng); te1 = make(args.n_test, K, d, args.snr, args.mix, emb1, rng)
        s_up = np.array([-1, 0, 0, 0, 0, 1, 1], float)[tr1["t"]]
        w1 = torch.tensor(1 + 0.75 * s_up, device=dev, dtype=torch.float32)
        torch.manual_seed(seed)
        net1 = Net(d, K).to(dev); train(net1, tr1, args.epochs1, args.bs, args.lr1, w=w1)
        before = per_type_acc(net1, te1)
        print(f"seed {seed}: task 1 learned " + " ".join(f"{k}={v:.3f}" for k, v in before.items()), flush=True)
        g = np.random.default_rng(100 + seed); bi = torch.tensor(g.choice(args.n_train, args.buffer, replace=False), device=dev)
        buf = dict(x=[v[bi] for v in tr1["x"]], y=tr1["y"][bi])
        new = [tables(rng, K, d)[m] for m in range(3)]
        for shift in args.shifts.split(","):
            emb2 = [new[m] if str(m + 1) in shift else emb1[m] for m in range(3)]
            tr2 = make(args.n_train, K, d, args.snr, args.mix, emb2, rng)
            for cond in ("none", "replay"):
                net = copy.deepcopy(net1)
                train(net, tr2, args.epochs2, args.bs, args.lr2, buf=buf if cond == "replay" else None, onecycle=False)
                after = per_type_acc(net, te1)
                kept = {k: (after[k] - c) / (before[k] - c) for k in TYPES}  # chance-corrected retention
                p = [kept["U1"], kept["U2"], kept["U3"]]
                pred0 = predict(p)
                # shared bottleneck: unique = h*q, composites = h*f(q); fit h in (0, 1] by grid least squares
                best = None
                for h in np.linspace(0.05, 1.0, 96):
                    q = [min(v / h, 1.0) for v in p]; pr = predict(q, h)
                    err = sum((pr[k] - kept[k]) ** 2 for k in pr)
                    if best is None or err < best[0]:
                        best = (err, h, pr)
                row = dict(seed=seed, shift=shift, cond=cond, h=best[1],
                           **{f"before_{k}": before[k] for k in TYPES}, **{f"kept_{k}": kept[k] for k in TYPES},
                           **{f"pred0_{k}": v for k, v in pred0.items()}, **{f"predh_{k}": v for k, v in best[2].items()})
                rows.append(row)
                print(f"  shift m{shift:3s} {cond:6s} p=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) h={best[1]:.2f} | "
                      + " ".join(f"{k} {kept[k]:.2f}/{pred0[k]:.2f}" for k in ("R123", "R12", "S12", "S123"))
                      + "   (observed/zero-param prediction)", flush=True)
            pd.DataFrame(rows).to_csv(f"{args.out}_results.csv", index=False)
    R = pd.DataFrame(rows)
    learned = R[(R.before_S123 > 0.5) & (R.before_S12 > 0.5)]
    print(f"\nruns with both synergy types learned in task 1: {len(learned)}/{len(R)}")
    for name, sub in (("all", learned), ("replay", learned[learned.cond == "replay"]), ("none", learned[learned.cond == "none"])):
        print(f"\n=== {name} (n={len(sub)}): observed vs predicted, per composite type ===")
        for k in ("R123", "R12", "S12", "S123"):
            o = sub[f"kept_{k}"].clip(-0.2, 1.2)
            for tag in ("pred0", "predh"):
                pr = sub[f"{tag}_{k}"]
                print(f"  {k:5s} {tag}: r={np.corrcoef(o, pr)[0, 1]:.2f}  mean |err|={np.abs(o - pr).mean():.3f}  "
                      f"(mean obs {o.mean():.2f}, pred {pr.mean():.2f})")
    print("\nmean chance-corrected retention by shift pattern (replay):")
    cols = [f"kept_{k}" for k in TYPES] + ["h"]
    print(learned[learned.cond == "replay"].groupby("shift")[cols].mean().round(2).to_string())


if __name__ == "__main__":
    main()
