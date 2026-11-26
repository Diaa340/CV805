"""
v3 (paper/preregistration_v3.md): forgetting as composed per-modality encoder drift at a fixed interface.
Per unit and regime (replay 'control' / no replay 'none'): actual retention a_t and training-free swaps under the
task-1 fusion layer: s1 (encoder 1 drifted), s2 (encoder 2 drifted), sb (both), plus the exploratory fusion-only swap.
Data and training exactly as route_exps.py. Output (appended): runs/<tag>_v3.csv
"""
import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from route_exps import Net, load_unit, per_type, train

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["R", "U1", "U2", "S"]
HMAP = {"red": "R", "u_a": "U1", "u_v": "U2", "syn": "S"}


@torch.no_grad()
def model_types(net, te, c1, seed):
    """P2 model-functional types on the task-1 model: modality m 'removed' = replaced by the same modality of a random
    other test sample (3 draws, majority). R: correct with either alone; U1/U2: only with that one; S: only with both."""
    x1, x2, y = te; n = len(y)
    mask = torch.full((net.head[-1].out_features,), float("-inf"), device=dev); mask[c1] = 0
    net.eval()
    ok = lambda a, b: ((net(a, b) + mask).argmax(1) == y)
    full = ok(x1, x2); g = torch.Generator(device=dev).manual_seed(5000 + seed)
    a1 = torch.zeros(n, device=dev); a2 = torch.zeros(n, device=dev)
    for _ in range(3):
        pm = torch.randperm(n, generator=g, device=dev)
        a1 += ok(x1, x2[pm]).float(); a2 += ok(x1[pm], x2).float()
    net.train()
    full, a1, a2 = full.cpu().numpy(), (a1 >= 2).cpu().numpy(), (a2 >= 2).cpu().numpy()
    t = np.full(n, "none", dtype=object)
    t[full & a1 & a2] = "R"; t[full & a1 & ~a2] = "U1"; t[full & ~a1 & a2] = "U2"; t[full & ~a1 & ~a2] = "S"
    return t.astype(str)


def hybrid(like, e1, e2, head):
    m = copy.deepcopy(like); m.e1, m.e2, m.head = copy.deepcopy(e1), copy.deepcopy(e2), copy.deepcopy(head)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth")
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12")
    ap.add_argument("--class_split", type=int, default=0)
    ap.add_argument("--order", default="normal")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_types", action="store_true", help="P2: add model-functional types (preregistration_p2)")
    args = ap.parse_args()
    cfg = dict(data=args.data, snr=f"{args.snr[0]}/{args.snr[1]}" if args.data == "synth" else "", shift=args.shift if (args.data == "synth" or args.data.startswith("bridge_")) else "",
               class_split=args.class_split, order=args.order, depth=args.depth, width=args.width)
    human = None
    if args.data == "cremad":
        human = np.load(os.path.join(ROOT, "data", "feats", "cremad.npz"), allow_pickle=True)["human_type"]
    for seed in args.seeds:
        t0 = time.time()
        U = load_unit(args, seed); c = 1 / len(U["c1"]); types = U["types"]
        typesets = {"est": types}
        if human is not None:
            typesets["human"] = np.array([HMAP.get(v, "unrec") for v in human[U["idx_te1"]]])
        torch.manual_seed(seed)
        d1, d2 = U["tr1"][0].shape[1], U["tr1"][1].shape[1]
        net1 = Net(d1, d2, U["C"], args.depth, args.width).to(dev)
        train(net1, U["tr1"], 30, 1e-3, w=U["w1"])
        agree = np.nan
        if args.model_types:
            typesets["model"] = model_types(net1, U["te1"], U["c1"], seed)
            typed = typesets["model"] != "none"
            agree = float(np.mean(typesets["model"][typed] == types[typed])) if typed.any() else np.nan
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        n1 = len(U["tr1"][2]); bi = torch.randperm(n1, generator=g)[:max(1000 if args.data == "synth" else 200, int(0.02 * n1))].to(dev)
        buf = tuple(v[bi] for v in U["tr1"])
        rows = []
        for regime in ("control", "none"):
            net = copy.deepcopy(net1)
            train(net, U["tr2"], 10, 1e-3, buf=None if regime == "none" else buf, cond="control")
            models = {"actual": net, "s1": hybrid(net1, net.e1, net1.e2, net1.head), "s2": hybrid(net1, net1.e1, net.e2, net1.head),
                      "sb": hybrid(net1, net.e1, net.e2, net1.head), "sf": hybrid(net1, net1.e1, net1.e2, net.head)}
            for tset, ty in typesets.items():
                before = per_type(net1, U["te1"], U["c1"], ty)
                row = dict(**cfg, seed=seed, regime=regime, types=tset, c=c, agree_est_model=agree)
                for t in TYPES:
                    row[f"n_{t}"] = int((ty == t).sum()); row[f"before_{t}"] = before[t]
                for name, m in models.items():
                    acc = per_type(m, U["te1"], U["c1"], ty)
                    for t in TYPES:
                        row[f"{name}_{t}"] = (acc[t] - c) / (before[t] - c) if before[t] - c > 1e-6 else np.nan
                rows.append(row)
        p = os.path.join(ROOT, "runs", f"{args.tag}_v3.csv")
        pd.DataFrame(rows).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        r = pd.DataFrame(rows); r = r[(r.regime == "control") & (r.types == "est")].iloc[0]
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) replay S: s1 {r.s1_S:.2f} s2 {r.s2_S:.2f} "
              f"prod {r.s1_S * r.s2_S:.2f} sb {r.sb_S:.2f} actual {r.actual_S:.2f}", flush=True)


if __name__ == "__main__":
    main()
