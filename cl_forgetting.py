"""
Option A, first test: "What is forgotten first?" — catastrophic forgetting split by per-sample interaction type.

Uses the Stage 0 synthetic two-modality data (stage0_synthetic.py), where every sample's type is known:
R (redundant), U1 / U2 (unique to modality 1 / 2; modality 1 is the stronger one), S (synergistic).
Task 1 is learned, then the model keeps training on task 2; we track task-1 test accuracy per type.

Scenarios
  dil : domain-incremental — task 2 has the same classes and rules but new code tables (inputs look different);
        --shift m1 / m2 changes only that modality's tables (tests the 'number of routes' explanation)
  cil : class-incremental  — task 2 has 10 new classes (own tables); head has 20 outputs. Task-1 accuracy is measured
        with logits masked to task-1 classes, so it reflects feature forgetting, not the new-class output bias.

Task 1 is trained with Stage 0's synergy up-weighting so that all four types are actually learned before task 2.
Difficulty control: forgetting per type is also reported within bins of the task-1 model's confidence in the correct
class, so "S is forgotten more" cannot be explained by S simply being less confidently learned.

Replay (1000 task-1 samples, 64 per batch, mixed into every task-2 batch), replay-loss weights w = 1 + beta*s:
  none            no replay (pure forgetting measurement)
  replay_uniform  s = 0
  replay_syn      s = +1 S, -1 R, 0 U      (true types: oracle)
  replay_syncand  s = rank(-max(i1, i2))  (estimated 'neither modality suffices', PID-motivated)
  replay_hard_mi  s = rank(-i12)          (MI-only 'hard' control)
  replay_lowconf  s = rank(-p_correct)    (model-confidence 'hard' control)
"""
import argparse
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from stage0_synthetic import TYPES, Fusion, make_data, pid_estimate, rank_scale

dev = "cuda"


def new_tables(rng, K, d):
    e = [rng.standard_normal((4, K, d)) for _ in range(2)]
    return [x / np.linalg.norm(x, axis=2, keepdims=True) for x in e]


@torch.no_grad()
def eval_task(model, te, lo, hi):
    """Accuracy with logits restricted to classes [lo, hi) and probability of the correct class."""
    model.eval()
    z = model(te["x1"], te["x2"])[:, lo:hi]
    p = F.softmax(z, 1).gather(1, (te["y"] - lo)[:, None])[:, 0]
    corr = (z.argmax(1) + lo == te["y"]).cpu().numpy()
    model.train()
    return corr, p.cpu().numpy()


def per_type(corr, t):
    return {name: corr[t == j].mean() for j, name in enumerate(TYPES)}


def train_task1(tr, K_out, d, epochs, bs, lr, seed, w=None):
    torch.manual_seed(seed)
    model = Fusion(d, K_out).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    n = len(tr["y"]); total = epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total, pct_start=0.25)
    for _ in range(total):
        i = torch.randint(0, n, (bs,), device=dev)
        l = F.cross_entropy(model(tr["x1"][i], tr["x2"][i]), tr["y"][i], reduction="none")
        loss = (l * w[i] / w[i].mean()).mean() if w is not None else l.mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def train_task2(model, tr2, te1, t1_lo, t1_hi, epochs, bs, lr, buf=None, buf_w=None, rb=64, n_ckpt=10):
    """Fine-tune on task 2 (constant lr, SGD+momentum); optional weighted replay. Returns task-1 correctness at
    checkpoints (list of arrays)."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    n = len(tr2["y"]); total = epochs * (n // bs); every = max(1, total // n_ckpt)
    curve = []
    for step in range(total):
        i = torch.randint(0, n, (bs,), device=dev)
        loss = F.cross_entropy(model(tr2["x1"][i], tr2["x2"][i]), tr2["y"][i])
        if buf is not None:
            j = torch.randint(0, len(buf["y"]), (rb,), device=dev)
            lr_ = F.cross_entropy(model(buf["x1"][j], buf["x2"][j]), buf["y"][j], reduction="none")
            w = buf_w[j]; loss = loss + (lr_ * w / w.mean()).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % every == 0:
            curve.append(eval_task(model, te1, t1_lo, t1_hi)[0])
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--mix", type=float, nargs=4, default=[0.4, 0.15, 0.15, 0.3])
    ap.add_argument("--epochs1", type=int, default=40)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--lr1", type=float, default=0.05)
    ap.add_argument("--lr2", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--buffer", type=int, default=1000)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--probe_steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--scenarios", default="dil,cil")
    ap.add_argument("--conditions", default="none,replay_uniform,replay_syn,replay_syncand,replay_hard_mi,replay_lowconf")
    ap.add_argument("--out", default="cl_forgetting")
    ap.add_argument("--shift", choices=["both", "m1", "m2"], default="both",
                    help="dil only: which modality gets new code tables in task 2 (the other keeps task 1's)")
    args = ap.parse_args()
    K, d = args.K, args.d
    rows, bins_rows, curves = [], [], []
    for seed in range(args.seeds):
        for sc in args.scenarios.split(","):
            rng = np.random.default_rng(seed)
            emb1, emb2 = new_tables(rng, K, d), new_tables(rng, K, d)
            if sc == "dil" and args.shift == "m1":
                emb2[1] = emb1[1]
            elif sc == "dil" and args.shift == "m2":
                emb2[0] = emb1[0]
            tr1 = make_data(args.n_train, K, d, args.snr, args.mix, emb1, rng)
            te1 = make_data(args.n_test, K, d, args.snr, args.mix, emb1, rng)
            tr2 = make_data(args.n_train, K, d, args.snr, args.mix, emb2, rng)
            te2 = make_data(args.n_test, K, d, args.snr, args.mix, emb2, rng)
            K_out = K if sc == "dil" else 2 * K
            if sc == "cil":
                tr2["y"] = tr2["y"] + K; te2["y"] = te2["y"] + K
            t2_lo, t2_hi = (0, K) if sc == "dil" else (K, 2 * K)
            # task 1 is trained with the (oracle) synergy up-weighting from Stage 0 round 2, which reliably learns all
            # four types; plain training often never learns S (then its forgetting cannot be measured)
            w1 = torch.tensor(1 + 0.75 * np.array([-1.0, 0.0, 0.0, 1.0])[tr1["t"]], device=dev, dtype=torch.float32)
            model1 = train_task1(tr1, K_out, d, args.epochs1, args.bs, args.lr1, seed, w=w1)
            corr0, p0 = eval_task(model1, te1, 0, K)
            acc0 = per_type(corr0, te1["t"])
            print(f"seed {seed} {sc}: task-1 acc after task 1: " + " ".join(f"{k}={v:.3f}" for k, v in acc0.items()), flush=True)
            # replay buffer: random task-1 training samples; weights from true types / estimates / model confidence
            g = np.random.default_rng(100 + seed)
            bidx = torch.tensor(g.choice(args.n_train, args.buffer, replace=False), device=dev)
            buf = {k: tr1[k][bidx] for k in ("x1", "x2", "y")}; bt = tr1["t"][bidx.cpu().numpy()]
            scores = {"replay_uniform": np.zeros(args.buffer), "replay_syn": np.array([-1.0, 0.0, 0.0, 1.0])[bt]}
            if {"replay_syncand", "replay_hard_mi"} & set(args.conditions.split(",")):
                A = pid_estimate(tr1, K, d, args.probe_steps, seed)
                b = bidx.cpu().numpy()
                scores["replay_syncand"] = rank_scale(-np.maximum(A["i1"], A["i2"])[b]).cpu().numpy()
                scores["replay_hard_mi"] = rank_scale(-A["i12"][b]).cpu().numpy()
            _, pb = eval_task(model1, buf, 0, K)
            scores["replay_lowconf"] = rank_scale(-pb).cpu().numpy()
            for cond in args.conditions.split(","):
                model = copy.deepcopy(model1)
                if cond == "none":
                    curve = train_task2(model, tr2, te1, 0, K, args.epochs2, args.bs, args.lr2)
                else:
                    w = torch.tensor(1 + args.beta * scores[cond], device=dev, dtype=torch.float32)
                    curve = train_task2(model, tr2, te1, 0, K, args.epochs2, args.bs, args.lr2, buf=buf, buf_w=w)
                corr1 = curve[-1]
                acc1 = per_type(corr1, te1["t"])
                acc2 = eval_task(model, te2, t2_lo, t2_hi)[0].mean()
                row = dict(seed=seed, scenario=sc, cond=cond, task2_acc=acc2)
                for k in TYPES:
                    row[f"before_{k}"] = acc0[k]; row[f"after_{k}"] = acc1[k]
                    row[f"retention_{k}"] = acc1[k] / max(acc0[k], 1e-6)
                rows.append(row)
                for ci, c in enumerate(curve):
                    for j, k in enumerate(TYPES):
                        curves.append(dict(seed=seed, scenario=sc, cond=cond, ckpt=ci + 1, type=k, acc=c[te1["t"] == j].mean()))
                # difficulty control: retention per type within bins of task-1 confidence (correct before only)
                edges = [0.0, 0.5, 0.8, 0.95, 0.99, 1.01]
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = corr0 & (p0 >= lo) & (p0 < hi)
                    for j, k in enumerate(TYPES):
                        mm = m & (te1["t"] == j)
                        if mm.sum() >= 30:
                            bins_rows.append(dict(seed=seed, scenario=sc, cond=cond, conf_bin=f"[{lo:.2f},{min(hi, 1):.2f})", type=k,
                                                  n=int(mm.sum()), retained=corr1[mm].mean()))
                print(f"   {cond:15s} task-1 after: " + " ".join(f"{k}={acc1[k]:.3f}" for k in TYPES)
                      + f" | retention " + " ".join(f"{k}={row['retention_' + k]:.2f}" for k in TYPES)
                      + f" | task-2 acc {acc2:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(f"{args.out}_results.csv", index=False)
            pd.DataFrame(bins_rows).to_csv(f"{args.out}_confbins.csv", index=False)
            pd.DataFrame(curves).to_csv(f"{args.out}_curves.csv", index=False)
    R = pd.DataFrame(rows); B = pd.DataFrame(bins_rows)
    for sc in args.scenarios.split(","):
        r = R[R.scenario == sc]
        print(f"\n===== {sc}: task-1 accuracy per type (%), mean over seeds =====")
        cols = [f"before_{k}" for k in TYPES] + [f"after_{k}" for k in TYPES] + [f"retention_{k}" for k in TYPES] + ["task2_acc"]
        print((r.groupby("cond")[cols].mean() * 100).round(1).to_string())
        for cond in B.cond.unique():
            b = B[(B.scenario == sc) & (B.cond == cond)]
            print(f"\n----- {sc}, {cond}: fraction retained per type within task-1 confidence bins (difficulty control) -----")
            print((b.groupby(["conf_bin", "type"]).apply(lambda x: np.average(x.retained, weights=x.n), include_groups=False)
                   .unstack() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
