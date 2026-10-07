"""
Phase 1b (paper/preregistration_phase1b.md): does the task-1 fusion layer keep the interaction while the encoders drift?
Per unit and regime (control = replay, none = no replay), after task 2:
  trained arms (6000 steps, 64 buffer + 64 task-2 per batch): ceiling, fusion_refit, encoder_refit,
      anchor_1 / anchor_2 / anchor_both (TASK-1 fusion frozen; train encoder 1 / 2 / both)
  training-free swaps: swap_both, swap_1, swap_2 (task-1 fusion with drifted encoder(s))
Output (appended): runs/<tag>_phase1b.csv
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from phase1_localize import cka, repair
from route_exps import Net, per_type, train
from route_v2 import make_codes

dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["R", "U1", "U2", "S"]
K = 10


def hybrid(e1, e2, head):
    m = Net(64, 64, K).to(dev)
    m.e1, m.e2, m.head = copy.deepcopy(e1), copy.deepcopy(e2), copy.deepcopy(head)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--steps", type=int, default=6000)
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
        types = np.array(TYPES)[te1["t"]]; types2 = np.array(TYPES)[te2["t"]]; allc = list(range(K))
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
            with torch.no_grad():
                x1, x2 = te1["x1"][:5000], te1["x2"][:5000]
                d1, d2 = cka(net1.e1(x1), net.e1(x1)), cka(net1.e2(x2), net.e2(x2))
            torch.manual_seed(1000 + seed)
            fresh_head = lambda: Net(64, 64, K).to(dev).head
            arms = {"no_repair": net,
                    "swap_both": hybrid(net.e1, net.e2, net1.head), "swap_1": hybrid(net.e1, net1.e2, net1.head),
                    "swap_2": hybrid(net1.e1, net.e2, net1.head)}
            m = hybrid(net1.e1, net1.e2, fresh_head()); arms["ceiling"] = repair(m, [m.head], buf, pk(tr2), args.steps)
            m = copy.deepcopy(net); arms["fusion_refit"] = repair(m, [m.head], buf, pk(tr2), args.steps)
            m = copy.deepcopy(net); arms["encoder_refit"] = repair(m, [m.e1, m.e2], buf, pk(tr2), args.steps)
            for name, which in (("anchor_1", (1,)), ("anchor_2", (2,)), ("anchor_both", (1, 2))):
                m = hybrid(net.e1, net.e2, net1.head)
                arms[name] = repair(m, [m.e1 if k == 1 else m.e2 for k in which], buf, pk(tr2), args.steps)
            for arm, model in arms.items():
                acc = per_type(model, pk(te1), allc, types); a2 = per_type(model, pk(te2), allc, types2)["all"]
                row = dict(**cfg, seed=seed, regime=regime, arm=arm, c=c, task2_acc=a2, cka_emb1=d1, cka_emb2=d2)
                for t in TYPES:
                    row[f"before_{t}"] = before[t]; row[f"acc_{t}"] = acc[t]
                    row[f"ret_{t}"] = (acc[t] - c) / (before[t] - c) if before[t] - c > 1e-6 else np.nan
                rows.append(row)
        p = os.path.join(ROOT, "runs", f"{args.tag}_phase1b.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        r = pd.DataFrame(rows); r = r[r.regime == "none"].set_index("arm")
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) no-replay S: " + " ".join(f"{a}={r.loc[a, 'ret_S']:.2f}" for a in r.index), flush=True)


if __name__ == "__main__":
    main()
