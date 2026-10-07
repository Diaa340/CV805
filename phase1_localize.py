"""
Phase 1 (paper/preregistration_phase1.md): is synergy erased or disconnected after task 2?
Per unit: task 1 -> task 2 (replay 'control' and no-replay 'none'), then repair arms with the same budget
(2000 Adam steps, batches of 64 replay-buffer + 64 task-2 samples):
  no_repair | fusion_refit (encoders frozen, fusion trained further) | encoder_refit (fusion frozen)
  additive_refit (encoders frozen, new additive head: cannot represent synergy)
  fresh_encoders (random frozen encoders + new fusion: what the buffer alone teaches)
  task1_encoders (task-1 encoders frozen + new fusion: ceiling)
  fusion_fresh [exploratory] (drifted encoders frozen + NEW fusion layer)
Exploratory Phase-2 drift measures are logged per regime (before any repair).
Output (appended): runs/<tag>_phase1.csv
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
from route_v2 import make_codes

dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["R", "U1", "U2", "S"]
K, EMB = 10, 128


class Additive(nn.Module):
    """Encoders + one linear readout per modality, summed (no cross-modal interaction possible)."""
    def __init__(self, net):
        super().__init__(); self.e1, self.e2 = net.e1, net.e2
        self.l1, self.l2 = nn.Linear(EMB, K), nn.Linear(EMB, K)

    def forward(self, x1, x2):
        return self.l1(self.e1(x1)) + self.l2(self.e2(x2))


def repair(model, trainable, buf, tr2, steps=2000, bs=64):
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for m in trainable for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=1e-3)
    for _ in range(steps):
        i = torch.randint(0, len(buf[2]), (bs,), device=dev); j = torch.randint(0, len(tr2[2]), (bs,), device=dev)
        x1 = torch.cat([buf[0][i], tr2[0][j]]); x2 = torch.cat([buf[1][i], tr2[1][j]]); y = torch.cat([buf[2][i], tr2[2][j]])
        loss = F.cross_entropy(model(x1, x2), y); opt.zero_grad(); loss.backward(); opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


@torch.no_grad()
def cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return ((a.T @ b).pow(2).sum() / ((a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt())).item()


@torch.no_grad()
def drift_measures(net1, net, te):
    x1, x2 = te[0][:5000], te[1][:5000]
    o1, o2, n1, n2 = net1.e1(x1), net1.e2(x2), net.e1(x1), net.e2(x2)
    hid = lambda m, a, b: m.head[1](m.head[0](torch.cat([a, b], 1)))
    W0, W = net1.head[0].weight, net.head[0].weight
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    return dict(cka_fusion_hidden=cka(hid(net1, o1, o2), hid(net, n1, n2)), cka_emb1=cka(o1, n1), cka_emb2=cka(o2, n2),
                wdrift_fusion_m1=rel(W[:, :EMB], W0[:, :EMB]), wdrift_fusion_m2=rel(W[:, EMB:], W0[:, EMB:]),
                crossmodal_cka_before=cka(o1, o2), crossmodal_cka_after=cka(n1, n2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    mix = [0.4, 0.15, 0.15, 0.3]; c = 1 / K
    cfg = dict(snr=f"{args.snr[0]}/{args.snr[1]}", shift=args.shift)
    for seed in args.seeds:
        t0 = time.time(); rng = np.random.default_rng(seed)
        mk = lambda: [e / np.linalg.norm(e, axis=2, keepdims=True) for e in (rng.standard_normal((4, K, 64)) for _ in range(2))]
        emb1, new = mk(), mk()
        emb2 = [new[0] if "1" in args.shift else emb1[0], new[1] if "2" in args.shift else emb1[1]]
        tr1, te1 = make_codes(50000, 64, args.snr, mix, emb1, rng), make_codes(20000, 64, args.snr, mix, emb1, rng)
        tr2, te2 = make_codes(50000, 64, args.snr, mix, emb2, rng), make_codes(10000, 64, args.snr, mix, emb2, rng)
        pk = lambda d: (d["x1"], d["x2"], d["y"])
        types = np.array(TYPES)[te1["t"]]; allc = list(range(K))
        w1 = torch.tensor(1 + 0.75 * np.array([-1.0, 0.0, 0.0, 1.0])[tr1["t"]], device=dev, dtype=torch.float32)
        torch.manual_seed(seed)
        net1 = Net(64, 64, K).to(dev); train(net1, pk(tr1), 30, 1e-3, w=w1)
        before = per_type(net1, pk(te1), allc, types)
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        bi = torch.randperm(50000, generator=g)[:1000].to(dev); buf = tuple(v[bi] for v in pk(tr1))
        rows = []
        for regime in ("control", "none"):
            net = copy.deepcopy(net1)
            train(net, pk(tr2), 10, 1e-3, buf=None if regime == "none" else buf, cond="control")
            dm = drift_measures(net1, net, pk(te1))
            torch.manual_seed(1000 + seed)
            arms = {"no_repair": net}
            fr = copy.deepcopy(net); arms["fusion_refit"] = repair(fr, [fr.head], buf, pk(tr2))
            er = copy.deepcopy(net); arms["encoder_refit"] = repair(er, [er.e1, er.e2], buf, pk(tr2))
            ad = Additive(copy.deepcopy(net)).to(dev); arms["additive_refit"] = repair(ad, [ad.l1, ad.l2], buf, pk(tr2))
            fe = Net(64, 64, K).to(dev); arms["fresh_encoders"] = repair(fe, [fe.head], buf, pk(tr2))
            t1 = copy.deepcopy(net1); t1.head = Net(64, 64, K).to(dev).head; arms["task1_encoders"] = repair(t1, [t1.head], buf, pk(tr2))
            ff = copy.deepcopy(net); ff.head = Net(64, 64, K).to(dev).head; arms["fusion_fresh"] = repair(ff, [ff.head], buf, pk(tr2))
            for arm, model in arms.items():
                acc = per_type(model, pk(te1), allc, types); a2 = per_type(model, pk(te2), allc, np.array(TYPES)[te2["t"]])["all"]
                row = dict(**cfg, seed=seed, regime=regime, arm=arm, c=c, task2_acc=a2, **dm)
                for t in TYPES:
                    row[f"before_{t}"] = before[t]; row[f"acc_{t}"] = acc[t]
                    row[f"ret_{t}"] = (acc[t] - c) / (before[t] - c) if before[t] - c > 1e-6 else np.nan
                rows.append(row)
        p = os.path.join(ROOT, "runs", f"{args.tag}_phase1.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        r = pd.DataFrame(rows); r = r[r.regime == "control"].set_index("arm")
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) before S={before['S']:.2f} | control S retention: "
              + " ".join(f"{a}={r.loc[a, 'ret_S']:.2f}" for a in r.index), flush=True)


if __name__ == "__main__":
    main()
