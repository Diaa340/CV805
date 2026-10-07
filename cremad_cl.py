"""
Option A, step 2: is cross-modal (synergistic) knowledge forgotten first on REAL audio-visual data?

CREMA-D with the per-clip human interaction type from crowd ratings (voice-only / face-only / audio-visual votes,
see cremad_human_pid.csv): red (both suffice), u_a (voice only), u_v (face only), syn (only the combination).

Scenarios (a model learns task 1, then keeps training on task 2):
  cil_a : task 1 = {NEU, HAP, SAD}, task 2 = {FEA, DIS, ANG}
  cil_b : task 1 = {HAP, FEA, ANG}, task 2 = {NEU, SAD, DIS}
          (task-1 accuracy with logits masked to task-1 classes -> feature forgetting, not new-class bias)
  dil   : same 6 emotions, task 1 = one half of the actors, task 2 = the other half
Split: our own stratified 70/30 train/test split (larger test set than OGM-GE's 744 clips, for per-type power).
Model: ResNet18 audio + ResNet18 visual, MLP fusion, plus unimodal auxiliary heads/losses in both tasks so that both
branches actually learn (plain joint training leaves the visual branch at chance on CREMA-D).
Conditions: none (no replay) / replay (200 task-1 clips, 16 per batch).
Analysis: task-1 accuracy per human type before/after task 2, and retention relative to redundant clips within bins
of the model's task-1 confidence (difficulty control), overall and within each emotion class (class confound control).
"""
import argparse
import copy
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cremad_pid import AV, amp, audio_batch, load, visual_batch

dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
TYPES = ["red", "u_a", "u_v", "syn"]
CLS = {"NEU": 0, "HAP": 1, "SAD": 2, "FEA": 3, "DIS": 4, "ANG": 5}


def fit(model, heads, D, idx, epochs, lr, bs=64, lam=1.0, buf=None, rb=16, step_decay=True):
    params = list(model.parameters()) + list(heads.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    spe = math.ceil(len(idx) / bs)
    sched = torch.optim.lr_scheduler.StepLR(opt, int(0.7 * epochs) * spe if step_decay else 10 ** 9, 0.1)
    scaler = torch.amp.GradScaler("cuda"); y = D["y"]
    for ep in range(epochs):
        model.train(); perm = idx[torch.randperm(len(idx), device=dev)]
        for it in range(spe):
            j = perm[it * bs:(it + 1) * bs]
            if buf is not None:
                j = torch.cat([j, buf[torch.randint(0, len(buf), (rb,), device=dev)]])
            with amp():
                out, ea, ev = model(audio_batch(D["A"], j), visual_batch(D["V"], j, True))
                loss = F.cross_entropy(out.float(), y[j]) + lam * (
                    F.cross_entropy(heads[0](ea).float(), y[j]) + F.cross_entropy(heads[1](ev).float(), y[j]))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()


@torch.no_grad()
def eval_masked(model, D, idx, classes, bs=256):
    """Correctness and p(correct) with logits restricted to `classes`; also audio-/visual-ablated accuracy."""
    model.eval(); outs = {"av": [], "a": [], "v": []}
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        with amp():
            ea, ev = model.embed(audio_batch(D["A"], j), visual_batch(D["V"], j, False))
            outs["av"].append(model.head(ea, ev).float())
            outs["a"].append(model.head(ea, torch.zeros_like(ev)).float())
            outs["v"].append(model.head(torch.zeros_like(ea), ev).float())
    mask = torch.full((6,), float("-inf"), device=dev); mask[classes] = 0
    y = D["y"][idx]; res = {}
    for k, z in outs.items():
        z = torch.cat(z) + mask
        res[k] = (z.argmax(1) == y).cpu().numpy()
        if k == "av":
            p = F.softmax(z, 1).gather(1, y[:, None])[:, 0].cpu().numpy()
    return res["av"], p, res["a"].mean(), res["v"].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="cil_a,cil_b,dil")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs1", type=int, default=60)
    ap.add_argument("--epochs2", type=int, default=20)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-3)
    ap.add_argument("--buffer", type=int, default=200)
    ap.add_argument("--out", default="cremad_cl")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    D = load(); meta = D["meta"]; H = D["H"]
    N = len(meta); y_np = meta.label.values
    ht = H.htype.values; actor = meta.name.str[:4].values
    rng = np.random.default_rng(0)  # one fixed stratified 70/30 split for all seeds/scenarios
    test_mask = np.zeros(N, bool)
    for c in range(6):
        ids = np.where(y_np == c)[0]; test_mask[rng.choice(ids, int(0.3 * len(ids)), replace=False)] = True
    actors = np.array(sorted(set(actor))); half = set(rng.choice(actors, len(actors) // 2, replace=False))
    in_a = np.array([a in half for a in actor])
    T = lambda m: torch.tensor(np.where(m)[0], device=dev)
    scen = {
        "cil_a": ([CLS[c] for c in ("NEU", "HAP", "SAD")], [CLS[c] for c in ("FEA", "DIS", "ANG")], None),
        "cil_b": ([CLS[c] for c in ("HAP", "FEA", "ANG")], [CLS[c] for c in ("NEU", "SAD", "DIS")], None),
        "dil": (list(range(6)), list(range(6)), in_a),
    }
    rows, bins = [], []
    for sc in args.scenarios.split(","):
        c1, c2, grp = scen[sc]
        m1 = np.isin(y_np, c1) & (grp if grp is not None else True)
        m2 = np.isin(y_np, c2) & (~grp if grp is not None else True)
        tr1, te1, tr2 = T(m1 & ~test_mask), T(m1 & test_mask), T(m2 & ~test_mask)
        te1_np = te1.cpu().numpy()
        print(f"== {sc}: task-1 train {len(tr1)} test {len(te1)} | task-2 train {len(tr2)} | task-1 test types "
              f"{pd.Series(ht[te1_np]).value_counts().to_dict()}", flush=True)
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = AV("mlp").to(dev).to(memory_format=torch.channels_last)
            heads = nn.ModuleList([nn.Linear(512, 6), nn.Linear(512, 6)]).to(dev)
            fit(model, heads, D, tr1, args.epochs1, args.lr1)
            corr0, p0, a0, v0 = eval_masked(model, D, te1, c1)
            print(f"  seed {seed}: task 1 learned: acc {corr0.mean():.3f} (audio-branch {a0:.3f}, visual-branch {v0:.3f}) | "
                  + " ".join(f"{t}={corr0[ht[te1_np] == t].mean():.3f}" for t in TYPES), flush=True)
            g = torch.Generator(device="cpu").manual_seed(100 + seed)
            buf = tr1[torch.randperm(len(tr1), generator=g)[:args.buffer].to(dev)]
            for cond in ("none", "replay"):
                m = copy.deepcopy(model); h = copy.deepcopy(heads)
                fit(m, h, D, tr2, args.epochs2, args.lr2, buf=buf if cond == "replay" else None, step_decay=False)
                corr1, _, a1, v1 = eval_masked(m, D, te1, c1)
                row = dict(scenario=sc, seed=seed, cond=cond, before=corr0.mean(), after=corr1.mean(),
                           audio_before=a0, audio_after=a1, visual_before=v0, visual_after=v1)
                for t in TYPES:
                    mt = ht[te1_np] == t
                    row[f"before_{t}"] = corr0[mt].mean(); row[f"after_{t}"] = corr1[mt].mean()
                    row[f"kept_{t}"] = corr1[mt & corr0].mean()  # of clips correct before, fraction still correct
                rows.append(row)
                edges = [0.0, 0.5, 0.8, 0.95, 1.01]
                for lo, hi in zip(edges[:-1], edges[1:]):
                    for cls in c1:
                        for t in TYPES:
                            mm = corr0 & (p0 >= lo) & (p0 < hi) & (ht[te1_np] == t) & (y_np[te1_np] == cls)
                            if mm.sum() >= 5:
                                bins.append(dict(scenario=sc, seed=seed, cond=cond, conf_bin=f"[{lo:.2f},{min(hi, 1):.2f})",
                                                 cls=cls, type=t, n=int(mm.sum()), kept=corr1[mm].mean()))
                print(f"    {cond:6s}: task-1 acc {corr1.mean():.3f} (audio-branch {a1:.3f}, visual-branch {v1:.3f}) | kept "
                      + " ".join(f"{t}={row['kept_' + t]:.2f}" for t in TYPES), flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(ROOT, f"{args.out}_results.csv"), index=False)
                pd.DataFrame(bins).to_csv(os.path.join(ROOT, f"{args.out}_bins.csv"), index=False)
    R = pd.DataFrame(rows); B = pd.DataFrame(bins)
    print("\n=== share of correctly-learned task-1 clips still correct after task 2 (%), by human type ===")
    print((R.groupby(["scenario", "cond"])[[f"kept_{t}" for t in TYPES]].mean() * 100).round(1).to_string())
    print("\n=== matched retention relative to redundant clips (same confidence bin AND same emotion; 1.0 = no type effect) ===")
    out = []
    for (sc, cond), b in B.groupby(["scenario", "cond"]):
        piv = b.pivot_table(index=["seed", "conf_bin", "cls"], columns="type", values="kept")
        r = dict(scenario=sc, cond=cond)
        for t in ("u_a", "u_v", "syn"):
            ok = piv[["red", t]].dropna() if {"red", t} <= set(piv.columns) else pd.DataFrame()
            ok = ok[ok.red > 0] if len(ok) else ok
            r[f"{t}/red"] = (ok[t] / ok.red).mean() if len(ok) else np.nan
            r[f"n_cells_{t}"] = len(ok)
        out.append(r)
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
