"""
Experiments 3-5 of paper/preregistration.md (Route Law of Multimodal Forgetting). One unit = (data config, seed):
  1. train task 1; record per-type accuracy, label-free route readouts, and (exp 5 control) the corruption grid
  2. task 2 from the same task-1 model under: none (no replay), control (replay), and 10 single-route interventions
     (freeze / lowlr / distill1 / distill10 / replayonly, for modality 1 and 2), all in the replay regime
  3. log baselines for exp 3 (gradient norm and relative drift per encoder, per-modality accuracy after task 2)
  4. exp 5: corruption grid a1, a2 in {0, .25, .5, .75, 1} on the task-1, none and control models
Outputs (appended per unit): runs/<tag>_task2.csv, runs/<tag>_corrupt.csv
Data sources: 'synth' (2-modality generator of stage0_synthetic.py, ground-truth types, domain-incremental: task 2 has
new code tables for the modalities in --shift) or a frozen-feature dataset (class-incremental; probe-estimated types).
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from feat_cl import probe_types
from stage0_synthetic import make_data

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TYPES = ["R", "U1", "U2", "S"]
PROBE2T = {"red": "R", "u1": "U1", "u2": "U2", "syn": "S"}
GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, d1, d2, C, depth=2, width=256, emb=128):
        super().__init__()

        def enc(d):
            layers, i = [], d
            for _ in range(depth):
                layers += [nn.Linear(i, width), nn.ReLU()]; i = width
            return nn.Sequential(*layers, nn.Linear(i, emb))
        self.e1, self.e2 = enc(d1), enc(d2)
        self.head = nn.Sequential(nn.Linear(2 * emb, width), nn.ReLU(), nn.Linear(width, C))

    def forward(self, x1, x2):
        return self.head(torch.cat([self.e1(x1), self.e2(x2)], 1))


# ---------------------------------------------------------------------------------------------- data
BRIDGE_K = {"cifar2v": 10, "food101": 10, "n24news": 3}


def load_bridge(args, seed, n_train=50000, n_test=20000, mix=(0.4, 0.15, 0.15, 0.3)):
    """P1 semi-synthetic bridge (paper/preregistration_p1_bridge.md): modality 1 from one real item, modality 2 from a
    DIFFERENT real item; exact interaction types via disjoint class pools per (type, modality); task 2 = fresh pools
    for the shifted modalities. Domain-incremental (same K labels in both tasks)."""
    name = args.data[len("bridge_"):]; K = BRIDGE_K[name]
    Z = np.load(os.path.join(ROOT, "data", "feats", f"{name}.npz"), allow_pickle=True)
    f1, f2, y, split = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32), Z["y"].astype(int), Z["split"]
    trm = split == 0
    nz = lambda f: (f - f[trm].mean(0)) / (f[trm].std(0) + 1e-6)
    F1, F2 = torch.tensor(nz(f1), device=dev), torch.tensor(nz(f2), device=dev)
    perm = np.random.default_rng(900).permutation(int(y.max()) + 1)
    pools = lambda off: {k: perm[off + i * K: off + (i + 1) * K] for i, k in enumerate(("R", "U1", "U2", "S"))}
    P1, P2 = pools(0), pools(4 * K)  # task-1 pools, fresh task-2 pools (same pool per type for both modalities' items)
    items = {s: {c: np.where((y == c) & (split == s))[0] for c in np.unique(y)} for s in (0, 1)}
    rng = np.random.default_rng(seed)

    def pairs(n, sp, pa, pb):
        t = rng.choice(4, n, p=list(mix)); lab = rng.integers(0, K, n)
        ca, cb = np.zeros(n, int), np.zeros(n, int); rnd = rng.integers(0, K, (2, n))
        for j, k in enumerate(("R", "U1", "U2", "S")):
            m = t == j
            if k == "R":
                ca[m], cb[m] = pa["R"][lab[m]], pb["R"][lab[m]]
            elif k == "U1":
                ca[m], cb[m] = pa["U1"][lab[m]], pb["U1"][rnd[1][m]]
            elif k == "U2":
                ca[m], cb[m] = pa["U2"][rnd[0][m]], pb["U2"][lab[m]]
            else:
                a = rnd[0][m]; ca[m], cb[m] = pa["S"][a], pb["S"][(lab[m] - a) % K]
        ia = np.array([rng.choice(items[sp][c]) for c in ca]); ib = np.array([rng.choice(items[sp][c]) for c in cb])
        return (F1[torch.tensor(ia, device=dev)], F2[torch.tensor(ib, device=dev)], torch.tensor(lab, device=dev)), t

    tr1, t_tr1 = pairs(n_train, 0, P1, P1); te1, t_te1 = pairs(n_test, 1, P1, P1)
    pa2 = P2 if "1" in args.shift else P1; pb2 = P2 if "2" in args.shift else P1
    tr2, _ = pairs(n_train, 0, pa2, pb2); te2, _ = pairs(n_test, 1, pa2, pb2)
    types = np.array(TYPES)[t_te1]
    w1 = torch.tensor(1 + 0.75 * np.array([-1.0, 0.0, 0.0, 1.0])[t_tr1], device=dev, dtype=torch.float32)
    return dict(tr1=tr1, te1=te1, tr2=tr2, te2=te2, c1=list(range(K)), c2=list(range(K)), C=K, types=types, w1=w1,
                pid_amount={t: float((types == t).mean()) for t in TYPES})


def load_fv_cremad(args, seed, n_train=50000, n_test=20000):
    """P4: CREMA-D face-voice verification. Pair = (voice of clip i, face of clip j), label 1 iff same actor (j != i),
    50/50 balanced, so neither modality alone carries information about the label: every sample is synergistic by
    construction. Domain-incremental: task 1 = one half of the actors, task 2 = the other half (partition by
    --class_split, order by --order). Clip-level 70/30 split as in cremad.npz."""
    Z = np.load(os.path.join(ROOT, "data", "feats", "cremad.npz"), allow_pickle=True)
    meta = pd.read_csv(os.path.join(ROOT, "data", "cremad_meta.csv"))
    f1, f2, split = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32), Z["split"]
    trm = split == 0
    nz = lambda f: (f - f[trm].mean(0)) / (f[trm].std(0) + 1e-6)
    F1, F2 = torch.tensor(nz(f1), device=dev), torch.tensor(nz(f2), device=dev)  # f1 = voice (wav2vec2), f2 = face (CLIP)
    actor = meta.name.str[:4].values; actors = np.array(sorted(set(actor)))
    part = np.random.default_rng(800 + args.class_split).permutation(actors)
    gA, gB = set(part[:len(part) // 2]), set(part[len(part) // 2:])
    if args.order == "swap":
        gA, gB = gB, gA
    rng = np.random.default_rng(seed)

    def pairs(group, sp, n):
        clips = {a: np.where((actor == a) & (split == sp))[0] for a in group}
        clips = {a: c for a, c in clips.items() if len(c) >= 2}; act = np.array(list(clips))
        ai = rng.choice(act, n); pos = rng.random(n) < 0.5
        aj = np.where(pos, ai, [rng.choice(act[act != a]) for a in ai])
        i = np.array([rng.choice(clips[a]) for a in ai])
        j = np.array([rng.choice(clips[b][clips[b] != ii]) if b == a else rng.choice(clips[b]) for a, b, ii in zip(ai, aj, i)])
        return (F1[torch.tensor(i, device=dev)], F2[torch.tensor(j, device=dev)], torch.tensor(pos.astype(int), device=dev))

    tr1, te1, tr2, te2 = pairs(gA, 0, n_train), pairs(gA, 1, n_test), pairs(gB, 0, n_train), pairs(gB, 1, n_test)
    types = np.array(["S"] * n_test)
    return dict(tr1=tr1, te1=te1, tr2=tr2, te2=te2, c1=[0, 1], c2=[0, 1], C=2, types=types, w1=None,
                pid_amount={t: float(t == "S") for t in TYPES})


def load_unit(args, seed):
    """Returns dict with X1, X2 (task-1 and task-2 train/test tensors), classes, types of task-1 test samples."""
    if args.data == "fv_cremad":
        return load_fv_cremad(args, seed)
    if args.data.startswith("bridge_"):
        return load_bridge(args, seed)
    if args.data == "synth":
        rng = np.random.default_rng(seed)
        mk = lambda: [e / np.linalg.norm(e, axis=2, keepdims=True) for e in (rng.standard_normal((4, 10, 64)) for _ in range(2))]
        emb1, new = mk(), mk()
        emb2 = [new[0] if "1" in args.shift else emb1[0], new[1] if "2" in args.shift else emb1[1]]
        mix = [0.4, 0.15, 0.15, 0.3]
        tr1 = make_data(50000, 10, 64, args.snr, mix, emb1, rng); te1 = make_data(20000, 10, 64, args.snr, mix, emb1, rng)
        tr2 = make_data(50000, 10, 64, args.snr, mix, emb2, rng); te2 = make_data(20000, 10, 64, args.snr, mix, emb2, rng)
        types = np.array(TYPES)[te1["t"]]
        w1 = torch.tensor(1 + 0.75 * np.array([-1.0, 0.0, 0.0, 1.0])[tr1["t"]], device=dev, dtype=torch.float32)
        pack = lambda d: (d["x1"], d["x2"], d["y"])
        return dict(tr1=pack(tr1), te1=pack(te1), tr2=pack(tr2), te2=pack(te2), c1=list(range(10)), c2=list(range(10)), C=10, types=types, w1=w1,
                    pid_amount={t: float((types == t).mean()) for t in TYPES})
    Z = np.load(os.path.join(ROOT, "data", "feats", f"{args.data}.npz"), allow_pickle=True)
    f1, f2, y, split = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32), Z["y"].astype(int), Z["split"]
    trm = split == 0
    nz = lambda f: (f - f[trm].mean(0)) / (f[trm].std(0) + 1e-6)
    X1, X2, Y = torch.tensor(nz(f1), device=dev), torch.tensor(nz(f2), device=dev), torch.tensor(y, device=dev)
    C = int(y.max()) + 1
    if args.data == "clevr":  # P4: tasks = question families with disjoint answers ('query' vs count/exist/compare)
        q = np.array([f.startswith("query") for f in Z["family"]])
        c1, c2 = tuple(sorted(set(y[q]))), tuple(sorted(set(y[~q])))
        assert not set(c1) & set(c2), "CLEVR family answer sets must be disjoint"
    else:
        halves, s = [], 0
        while len(halves) <= args.class_split:  # distinct class splits (not equal to an earlier split or its complement)
            perm = np.random.default_rng(500 + s).permutation(C); a = tuple(sorted(perm[:C // 2])); b = tuple(sorted(perm[C // 2:]))
            if all(a != h[0] and a != h[1] for h in halves):
                halves.append((a, b))
            s += 1
        c1, c2 = halves[args.class_split]
    if args.order == "swap":
        c1, c2 = c2, c1
    c1, c2 = list(c1), list(c2)
    T = lambda m: torch.tensor(np.where(m)[0], device=dev)
    i_tr1, i_te1, i_tr2, i_te2 = T(np.isin(y, c1) & trm), T(np.isin(y, c1) & ~trm), T(np.isin(y, c2) & trm), T(np.isin(y, c2) & ~trm)
    A, pt = probe_types(X1, X2, Y, i_tr1, i_te1, c1, 3000, 1000 * args.class_split + (args.order == "swap"))
    types = np.array([PROBE2T.get(v, "unrec") for v in pt])
    pack = lambda i: (X1[i], X2[i], Y[i])
    amount = {"R": A["red"].mean(), "U1": A["u1"].mean(), "U2": A["u2"].mean(), "S": A["syn"].mean()}
    return dict(tr1=pack(i_tr1), te1=pack(i_te1), tr2=pack(i_tr2), te2=pack(i_te2), c1=c1, c2=c2, C=C, types=types, w1=None, pid_amount=amount,
                idx_te1=i_te1.cpu().numpy())


# ----------------------------------------------------------------------------------------- training
def train(net, data, epochs, lr, w=None, buf=None, rb=32, bs=256, cond="control", old=None):
    x1, x2, y = data; n = len(y)
    m = int(cond.split("_")[1]) if "_" in cond else None
    kind = cond.split("_")[0]
    groups = []
    for name, mod in (("e1", net.e1), ("e2", net.e2), ("head", net.head)):
        g_lr = lr
        if m and name == f"e{m}" and kind == "lowlr":
            g_lr = lr * 0.1
        if m and name == f"e{m}" and kind == "freeze":
            for p in mod.parameters():
                p.requires_grad_(False)
            continue
        groups.append({"params": list(mod.parameters()), "lr": g_lr, "name": name})
    opt = torch.optim.Adam(groups, lr=lr, weight_decay=1e-5)
    gn = {"e1": 0.0, "e2": 0.0}; steps = epochs * max(1, n // bs)
    lam = {"distill1": 1.0, "distill10": 10.0}.get(kind, 0.0)
    for _ in range(steps):
        i = torch.randint(0, n, (bs,), device=dev)
        a, b, t = x1[i], x2[i], y[i]
        l = F.cross_entropy(net(a, b), t, reduction="none")
        loss = (l * w[i] / w[i].mean()).mean() if w is not None else l.mean()
        if lam:
            enc, oenc, xm = (net.e1, old.e1, a) if m == 1 else (net.e2, old.e2, b)
            with torch.no_grad():
                target = oenc(xm)
            loss = loss + lam * F.mse_loss(enc(xm), target)
        if buf is not None:
            j = torch.randint(0, len(buf[2]), (rb,), device=dev)
            ra, rbb = buf[0][j], buf[1][j]
            if kind == "replayonly":  # replay with only modality m present (other set to its mean, i.e. 0)
                ra, rbb = (ra, torch.zeros_like(rbb)) if m == 1 else (torch.zeros_like(ra), rbb)
            loss = loss + F.cross_entropy(net(ra, rbb), buf[2][j])
        opt.zero_grad(); loss.backward()
        for name, mod in (("e1", net.e1), ("e2", net.e2)):
            gs = [p.grad for p in mod.parameters() if p.grad is not None]
            if gs:
                gn[name] += torch.sqrt(sum((g ** 2).sum() for g in gs)).item() / steps
        opt.step()
    for p in net.parameters():
        p.requires_grad_(True)
    return gn


@torch.no_grad()
def per_type(net, data, c1, types, a1=0.0, a2=0.0, zero=None, seed=0):
    net.eval(); x1, x2, y = data
    g = torch.Generator(device=dev).manual_seed(seed)
    if a1:
        x1 = (1 - a1) * x1 + a1 * torch.randn(x1.shape, generator=g, device=dev)
    if a2:
        x2 = (1 - a2) * x2 + a2 * torch.randn(x2.shape, generator=g, device=dev)
    if zero == 1:
        x1 = torch.zeros_like(x1)
    if zero == 2:
        x2 = torch.zeros_like(x2)
    z = net(x1, x2); mask = torch.full((z.shape[1],), float("-inf"), device=dev); mask[c1] = 0
    corr = ((z + mask).argmax(1) == y).cpu().numpy(); net.train()
    out = {t: corr[types == t].mean() if (types == t).sum() else np.nan for t in TYPES}
    out["all"] = corr.mean(); return out


def fit_readout(net, data, c1, m, steps=500):
    """Linear readout on encoder-m embeddings (task-1 train), fixed afterwards: the label-free route measurement."""
    x1, x2, y = data; enc = net.e1 if m == 1 else net.e2; x = x1 if m == 1 else x2
    remap = torch.full((int(y.max()) + 1,), -1, device=dev, dtype=torch.long); remap[torch.tensor(c1, device=dev)] = torch.arange(len(c1), device=dev)
    with torch.no_grad():
        h = enc(x)
    lin = nn.Linear(h.shape[1], len(c1)).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(steps):
        i = torch.randint(0, len(y), (1024,), device=dev)
        loss = F.cross_entropy(lin(h[i]), remap[y[i]]); opt.zero_grad(); loss.backward(); opt.step()
    return lin, remap


@torch.no_grad()
def readout_acc(net, lin, remap, data, m):
    x1, x2, y = data; enc = net.e1 if m == 1 else net.e2
    return (lin(enc(x1 if m == 1 else x2)).argmax(1) == remap[y]).float().mean().item()


@torch.no_grad()
def drift(new, old):
    num = sum(((a - b) ** 2).sum() for a, b in zip(new.parameters(), old.parameters()))
    den = sum((b ** 2).sum() for b in old.parameters())
    return math.sqrt((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synth")
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--shift", default="12", help="synth: modalities that get new tables in task 2")
    ap.add_argument("--class_split", type=int, default=0)
    ap.add_argument("--order", default="normal", choices=["normal", "swap"])
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs1", type=int, default=30)
    ap.add_argument("--epochs2", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", default="heldout", choices=["heldout", "discovery"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    conds = ["none", "control"] + [f"{k}_{m}" for m in (1, 2) for k in ("freeze", "lowlr", "distill1", "distill10", "replayonly")]
    cfg = dict(data=args.data, snr=f"{args.snr[0]}/{args.snr[1]}" if args.data == "synth" else "", shift=args.shift if args.data == "synth" else "",
               class_split=args.class_split, order=args.order, depth=args.depth, width=args.width, batch=args.batch)
    for seed in args.seeds:
        t0 = time.time()
        U = load_unit(args, seed)
        c = 1 / len(U["c1"]); types = U["types"]
        torch.manual_seed(seed)
        d1, d2 = U["tr1"][0].shape[1], U["tr1"][1].shape[1]
        net1 = Net(d1, d2, U["C"], args.depth, args.width).to(dev)
        train(net1, U["tr1"], args.epochs1, args.lr, w=U["w1"])
        before = per_type(net1, U["te1"], U["c1"], types)
        readouts = {m: fit_readout(net1, U["tr1"], U["c1"], m) for m in (1, 2)}
        ro_before = {m: readout_acc(net1, *readouts[m], U["te1"], m) for m in (1, 2)}
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        n1 = len(U["tr1"][2]); bi = torch.randperm(n1, generator=g)[:max(1000 if args.data == "synth" else 200, int(0.02 * n1))].to(dev)
        buf = tuple(v[bi] for v in U["tr1"])
        rows, crows, models = [], [], {"task1": net1}
        counts = {t: int((types == t).sum()) for t in TYPES}
        for cond in conds:
            net = copy.deepcopy(net1)
            gn = train(net, U["tr2"], args.epochs2, args.lr, buf=None if cond == "none" else buf, cond=cond, old=net1)
            after = per_type(net, U["te1"], U["c1"], types)
            only1 = per_type(net, U["te1"], U["c1"], types, zero=2)["all"]; only2 = per_type(net, U["te1"], U["c1"], types, zero=1)["all"]
            only1_b = per_type(net1, U["te1"], U["c1"], types, zero=2)["all"]; only2_b = per_type(net1, U["te1"], U["c1"], types, zero=1)["all"]
            row = dict(**cfg, seed=seed, cond=cond, c=c, grad_e1=gn["e1"], grad_e2=gn["e2"], drift_e1=drift(net.e1, net1.e1),
                       drift_e2=drift(net.e2, net1.e2), drift_head=drift(net.head, net1.head),
                       modacc1=(only1 - c) / max(only1_b - c, 1e-6), modacc2=(only2 - c) / max(only2_b - c, 1e-6))
            for m in (1, 2):
                ra = readout_acc(net, *readouts[m], U["te1"], m)
                row[f"p{m}_probe"] = (ra - c) / max(ro_before[m] - c, 1e-6)
            for t in TYPES:
                row[f"n_{t}"] = counts[t]; row[f"before_{t}"] = before[t]; row[f"after_{t}"] = after[t]
                row[f"ret_{t}"] = (after[t] - c) / (before[t] - c) if before[t] - c > 1e-6 else np.nan
                row[f"amount_{t}"] = U["pid_amount"][t]
            row["acc_before"], row["acc_after"] = before["all"], after["all"]
            rows.append(row)
            if cond in ("none", "control"):
                models[cond] = net
        for mname, net in models.items():
            base = per_type(net, U["te1"], U["c1"], types)
            for a1 in GRID:
                for a2 in GRID:
                    acc = base if (a1 == 0 and a2 == 0) else per_type(net, U["te1"], U["c1"], types, a1, a2, seed=seed)
                    cr = dict(**cfg, seed=seed, model=mname, a1=a1, a2=a2, c=c)
                    for t in TYPES:
                        cr[f"base_{t}"] = base[t]; cr[f"acc_{t}"] = acc[t]
                        cr[f"ret_{t}"] = (acc[t] - c) / (base[t] - c) if base[t] - c > 1e-6 else np.nan
                    crows.append(cr)
        for name, df in (("task2", rows), ("corrupt", crows)):
            p = os.path.join(ROOT, "runs", f"{args.tag}_{name}.csv")
            pd.DataFrame(df).to_csv(p, mode="a", header=not os.path.exists(p), index=False)
        r = pd.DataFrame(rows).set_index("cond")
        print(f"[{args.tag}] seed {seed} ({time.time() - t0:.0f}s) before " + " ".join(f"{t}={before[t]:.2f}(n={counts[t]})" for t in TYPES)
              + " | control ret " + " ".join(f"{t}={r.loc['control', 'ret_' + t]:.2f}" for t in TYPES), flush=True)


if __name__ == "__main__":
    main()
