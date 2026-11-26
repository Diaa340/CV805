"""
P3 (paper/preregistration_p3_readout.md): does retraining the readout recover synergy on natural data?
Per unit and regime (none = no replay [primary], control = replay): model-functional types (P2), then repairs trained on
the FULL task-1 training set: readout_mlp (fresh fusion head on drifted encoders), ceiling (fresh head on task-1 encoders),
encoder_realign (task-1 head frozen, drifted encoders trained), readout_linear (linear readout on concatenated drifted
embeddings; additive, cannot represent synergy).
Exploratory per-sample dump (runs/p3/<tag>_s<seed>_<regime>.npz): graded reliance d1, d2 on each modality (task-1
model), and correctness under the actual model and drift swaps s1, s2, sb after task 2.
Output (appended): runs/<tag>_p3.csv
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

from route_exps import Net, load_unit, per_type, train
from v3_drift import hybrid, model_types

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["R", "U1", "U2", "S"]


class LinearReadout(nn.Module):
    def __init__(self, net, C):
        super().__init__(); self.e1, self.e2 = net.e1, net.e2
        self.head = nn.Linear(2 * net.head[0].in_features // 2, C)

    def forward(self, x1, x2):
        return self.head(torch.cat([self.e1(x1), self.e2(x2)], 1))


def refit(model, trainable, data, steps=3000, bs=256):
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for m in trainable for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=1e-3); x1, x2, y = data
    for _ in range(steps):
        i = torch.randint(0, len(y), (bs,), device=dev)
        loss = F.cross_entropy(model(x1[i], x2[i]), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


@torch.no_grad()
def masked(net, x1, x2, c1):
    z = net(x1, x2); mask = torch.full((z.shape[1],), float("-inf"), device=dev); mask[c1] = 0
    return z + mask


@torch.no_grad()
def reliance(net, te, c1, seed, draws=3):
    """Graded reliance: drop in correct-class probability when modality m is replaced by a random real sample's."""
    net.eval(); x1, x2, y = te; n = len(y); g = torch.Generator(device=dev).manual_seed(7000 + seed)
    pc = lambda a, b: F.softmax(masked(net, a, b, c1), 1).gather(1, y[:, None])[:, 0]
    p = pc(x1, x2); d1 = torch.zeros(n, device=dev); d2 = torch.zeros(n, device=dev)
    for _ in range(draws):
        pm = torch.randperm(n, generator=g, device=dev)
        d1 += (p - pc(x1[pm], x2)) / draws; d2 += (p - pc(x1, x2[pm])) / draws
    net.train()
    return p.cpu().numpy(), d1.cpu().numpy(), d2.cpu().numpy()


@torch.no_grad()
def correct(net, te, c1):
    net.eval(); r = (masked(net, te[0], te[1], c1).argmax(1) == te[2]).cpu().numpy(); net.train(); return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cremad")
    ap.add_argument("--class_split", type=int, default=0)
    ap.add_argument("--order", default="normal")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--seeds", type=int, nargs="+", default=[18])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "runs", "p3"), exist_ok=True)
    cfg = dict(data=args.data, class_split=args.class_split, order=args.order)
    for seed in args.seeds:
        t0 = time.time()
        U = load_unit(args, seed); c = 1 / len(U["c1"]); c1 = U["c1"]
        torch.manual_seed(seed)
        d1_, d2_ = U["tr1"][0].shape[1], U["tr1"][1].shape[1]
        net1 = Net(d1_, d2_, U["C"], args.depth, args.width).to(dev)
        train(net1, U["tr1"], 30, 1e-3, w=U["w1"])
        # P4: face-voice verification is synergistic by construction; otherwise model-functional types (P2)
        types = U["types"] if args.data == "fv_cremad" else model_types(net1, U["te1"], c1, seed)
        p_full, rel1, rel2 = reliance(net1, U["te1"], c1, seed)
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        n1 = len(U["tr1"][2]); bi = torch.randperm(n1, generator=g)[:max(200, int(0.02 * n1))].to(dev)
        buf = tuple(v[bi] for v in U["tr1"])
        rows = []
        # P4 manipulation check: how much of the task-1 model's accuracy a LINEAR readout on the frozen task-1 encoders
        # reaches (additive evidence -> close to 1; genuinely synergistic knowledge -> well below 1)
        torch.manual_seed(3000 + seed)
        lin1 = LinearReadout(copy.deepcopy(net1), U["C"]).to(dev)
        lin1 = refit(lin1, [lin1.head], U["tr1"], args.steps)
        b1, l1 = per_type(net1, U["te1"], c1, types), per_type(lin1, U["te1"], c1, types)
        row = dict(**cfg, seed=seed, regime="task1", arm="task1_linear", c=c)
        for t in TYPES:
            row[f"n_{t}"] = int((types == t).sum())
            row[f"ret_{t}"] = (l1[t] - c) / (b1[t] - c) if b1[t] - c > 1e-6 else np.nan
        rows.append(row)
        for regime in ("none", "control"):
            net = copy.deepcopy(net1)
            train(net, U["tr2"], 10, 1e-3, buf=None if regime == "none" else buf, cond="control")
            swaps = {"actual": net, "s1": hybrid(net1, net.e1, net1.e2, net1.head), "s2": hybrid(net1, net1.e1, net.e2, net1.head),
                     "sb": hybrid(net1, net.e1, net.e2, net1.head)}
            np.savez_compressed(os.path.join(ROOT, "runs", "p3", f"{args.tag}_s{seed}_{regime}.npz"), types=types, p_full=p_full,
                                d1=rel1, d2=rel2, before=correct(net1, U["te1"], c1), c=c,
                                **{f"correct_{k}": correct(m, U["te1"], c1) for k, m in swaps.items()})
            torch.manual_seed(2000 + seed)
            arms = {"no_repair": net}
            m = hybrid(net1, net.e1, net.e2, Net(d1_, d2_, U["C"], args.depth, args.width).to(dev).head)
            arms["readout_mlp"] = refit(m, [m.head], U["tr1"], args.steps)
            m = hybrid(net1, net1.e1, net1.e2, Net(d1_, d2_, U["C"], args.depth, args.width).to(dev).head)
            arms["ceiling"] = refit(m, [m.head], U["tr1"], args.steps)
            m = hybrid(net1, net.e1, net.e2, net1.head)
            arms["encoder_realign"] = refit(m, [m.e1, m.e2], U["tr1"], args.steps)
            m = LinearReadout(copy.deepcopy(net), U["C"]).to(dev)
            arms["readout_linear"] = refit(m, [m.head], U["tr1"], args.steps)
            before = per_type(net1, U["te1"], c1, types)
            for arm, model in arms.items():
                acc = per_type(model, U["te1"], c1, types)
                row = dict(**cfg, seed=seed, regime=regime, arm=arm, c=c)
                for t in TYPES:
                    row[f"n_{t}"] = int((types == t).sum())
                    row[f"ret_{t}"] = (acc[t] - c) / (before[t] - c) if before[t] - c > 1e-6 else np.nan
                rows.append(row)
        p = os.path.join(ROOT, "runs", f"{args.tag}_p3.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        r = pd.DataFrame(rows); r = r[r.regime == "none"].set_index("arm")
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) no-replay retention S/R: " +
              " ".join(f"{a}={r.loc[a, 'ret_S']:.2f}/{r.loc[a, 'ret_R']:.2f}" for a in r.index), flush=True)


if __name__ == "__main__":
    main()
