"""
Stage 0: synthetic two-modality benchmark with KNOWN per-sample interaction type, to decide cheaply whether a
PID-guided multimodal curriculum is worth building on real audio-visual data (CREMA-D).

Data: K classes. Each modality m observes a discrete latent code c_m through a fixed random embedding plus noise,
x_m = snr_m * E_m[t, c_m] + N(0, I), with a SEPARATE code table per interaction type t, so the type is identifiable
from the input and an uninformative modality is uninformative (i = 0), not misleading. (v1 shared one table; then
U1/U2/S samples with the same inputs had conflicting labels, so per-type accuracies traded off against each other.)
Per sample the (c_1, c_2) -> y relation is one of
  R  : c1 = y,  c2 = y                        (redundant)
  U1 : c1 = y,  c2 ~ uniform                  (unique to modality 1)
  U2 : c1 ~ uniform, c2 = y                   (unique to modality 2)
  S  : c1 = a,  c2 = b,  y = (a + b) mod K    (synergistic: each modality alone is uninformative)
snr_1 > snr_2 reproduces modality imbalance (modality 1 dominant).

Model: per-modality MLP encoders, concat, MLP head (non-additive, so synergy is representable).

Pointwise PID estimate (per sample, full resolution): 2-fold cross-fitted probes q1(y|x1), q2(y|x2), q12(y|x1,x2)
give local i1, i2, i12 (bits); redundancy = I_ccs (Ince 2017): local co-information c = i1 + i2 - i12 if
sign(i1) = sign(i2) = sign(i12) = sign(c) else 0; u_m = i_m - red; syn = i12 - i1 - i2 + red.

Methods (same number of SGD steps):
  random            uniform sampling
  random_paced      competence pacing with a random order (pacing-only control)
  oracle_paced      pacing, true type order R -> U1/U2 -> S          (upper bound for a hard curriculum)
  oracle_paced_rev  pacing, S -> U -> R
  pid_paced         pacing, estimated type order (argmax atom) R -> U -> S
  mi_paced          pacing, by -i12 (MI-only control)
  oracle_soft       no pacing; per-type loss weights move from R-heavy to S-heavy over training
  pid_soft          same with estimated atoms (weights from normalised atoms, continuous)
  mi_soft           same schedule driven by i12 only (MI-only control)
  -- round 2 (synergy-first; motivated by round 1: oracle S->U->R beat R->U->S by ~14 pts) --
  pid_paced_rev     pacing, estimated type order S -> U -> R
  syncand_paced     pacing, 'neither modality suffices' first: order by max(i1, i2) ascending (needs only the
                    unimodal probes, which remain reliable when the fusion probe has not learned the synergy)
  mi_paced_rev      pacing, by i12 ascending (hard-first MI-only control)
  oracle_up         no pacing; static loss weight 1 + beta*s, s = +1 for S, -1 for R, 0 for U
  syncand_up        same, s = rank-scaled -max(i1, i2) in [-1, 1]
  mi_up_rev         same, s = rank-scaled -i12 in [-1, 1] (MI-only control)
  -- round 3 (label noise: 'hard' != 'synergistic', so PID and MI-only weighting can diverge) --
  pid2_up           two-stage: re-estimate atoms with a synergy-up-weighted fusion probe, then s = rank(syn),
                    and s = -1 for noise candidates (i12 < 0)
  mi2_filter        MI-only control on the SAME second-stage probe: s = -1 for i12 < 0, else 0
  mi_filter         MI-only noise filter from the first-stage probe
  noise_oracle      s = -1 for truly mislabeled samples (reference)
  ogm_ge            OGM-GE (Peng et al. CVPR 2022) gradient modulation + Gaussian enhancement
  mod_dropout       modality dropout (p=0.25 per modality, never both)
Outputs: stage0_results.csv (overall and per-type test accuracy), estimator diagnostics in stdout.
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

dev = "cuda"
TYPES = ["R", "U1", "U2", "S"]


# --------------------------------------------------------------------------------------------- data
def make_data(n, K, d, snr, mix, emb, rng):
    t = rng.choice(4, n, p=mix)
    y = rng.integers(0, K, n)
    c1, c2 = y.copy(), y.copy()
    rnd1, rnd2 = rng.integers(0, K, n), rng.integers(0, K, n)
    c2[t == 1] = rnd2[t == 1]
    c1[t == 2] = rnd1[t == 2]
    a = rnd1[t == 3]
    c1[t == 3], c2[t == 3] = a, (y[t == 3] - a) % K
    x1 = snr[0] * emb[0][t, c1] + rng.standard_normal((n, d))
    x2 = snr[1] * emb[1][t, c2] + rng.standard_normal((n, d))
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    return dict(x1=T(x1), x2=T(x2), y=torch.tensor(y, device=dev), t=t)


# -------------------------------------------------------------------------------------------- model
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Fusion(nn.Module):
    def __init__(self, d, K, h=256, e=128):
        super().__init__()
        self.e1, self.e2 = mlp(d, h, e), mlp(d, h, e)
        self.head = mlp(2 * e, h, K)

    def forward(self, x1, x2, drop1=None, drop2=None, feats=False):
        h1, h2 = self.e1(x1), self.e2(x2)
        if drop1 is not None:
            h1 = h1 * (~drop1)[:, None]
        if drop2 is not None:
            h2 = h2 * (~drop2)[:, None]
        out = self.head(torch.cat([h1, h2], 1))
        return (out, h1, h2) if feats else out


class Uni(nn.Module):
    def __init__(self, d, K, h=256):
        super().__init__(); self.f = mlp(d, h, K)

    def forward(self, x):
        return self.f(x)


# ------------------------------------------------------------------------------------ PID estimation
def fit_probe(model, inputs, y, steps, lr=1e-3, bs=512, w=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)
    for _ in range(steps):
        i = torch.randint(0, n, (bs,), device=dev)
        l = F.cross_entropy(model(*[a[i] for a in inputs]), y[i], reduction="none")
        loss = (l * w[i] / w[i].mean()).mean() if w is not None else l.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def calibrate(model, inputs, y):
    """Temperature scaling (Guo et al. 2017) on held-out data; local MIs need calibrated probabilities."""
    with torch.no_grad():
        z = model(*inputs)
    logT = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad(); l = F.cross_entropy(z / logT.exp(), y); l.backward(); return l
    opt.step(closure)
    model.T = logT.exp().item()
    return model


@torch.no_grad()
def logp_true(model, inputs, y):
    return F.log_softmax(model(*inputs) / getattr(model, "T", 1.0), 1).gather(1, y[:, None])[:, 0]


def pid_estimate(tr, K, d, steps, seed, w12=None):
    """2-fold cross-fitted local MIs and I_ccs atoms (bits) for every training sample."""
    torch.manual_seed(seed)
    n = len(tr["y"]); perm = torch.randperm(n, device=dev); folds = (perm[:n // 2], perm[n // 2:])
    lp = {k: torch.zeros(n, device=dev) for k in ("1", "2", "12")}
    for f in range(2):
        a, b = folds[f], folds[1 - f]
        a, c = a[:int(0.8 * len(a))], a[int(0.8 * len(a)):]  # fit / calibration split of the training fold
        ya, yc = tr["y"][a], tr["y"][c]
        m1 = calibrate(fit_probe(Uni(d, K).to(dev), [tr["x1"][a]], ya, steps), [tr["x1"][c]], yc)
        m2 = calibrate(fit_probe(Uni(d, K).to(dev), [tr["x2"][a]], ya, steps), [tr["x2"][c]], yc)
        m12 = calibrate(fit_probe(Fusion(d, K).to(dev), [tr["x1"][a], tr["x2"][a]], ya, steps,
                                  w=None if w12 is None else w12[a]),
                        [tr["x1"][c], tr["x2"][c]], yc)
        lp["1"][b] = logp_true(m1, [tr["x1"][b]], tr["y"][b])
        lp["2"][b] = logp_true(m2, [tr["x2"][b]], tr["y"][b])
        lp["12"][b] = logp_true(m12, [tr["x1"][b], tr["x2"][b]], tr["y"][b])
    prior = torch.bincount(tr["y"], minlength=K).float() / n
    lpy = prior.log()[tr["y"]]
    L2 = math.log(2)
    i1, i2, i12 = [((lp[k] - lpy) / L2).cpu().numpy() for k in ("1", "2", "12")]
    c = i1 + i2 - i12
    same = (np.sign(i1) == np.sign(i2)) & (np.sign(i2) == np.sign(i12)) & (np.sign(i12) == np.sign(c))
    red = np.where(same, c, 0.0)
    return dict(i1=i1, i2=i2, i12=i12, red=red, u1=i1 - red, u2=i2 - red, syn=i12 - i1 - i2 + red)


def est_type(A):
    """Hard type per sample = argmax over (red, u1, u2, syn)."""
    return np.stack([A["red"], A["u1"], A["u2"], A["syn"]], 1).argmax(1)


# ----------------------------------------------------------------------------------------- training
def competence(step, total, c0=0.2, frac=0.5):
    return min(1.0, math.sqrt(c0 ** 2 + (1 - c0 ** 2) * step / (frac * total)))


def order_from(rank, seed):
    rng = np.random.default_rng(seed)
    return torch.tensor(np.lexsort((rng.random(len(rank)), rank)), device=dev)


def soft_weights(stage_score, frac, beta=1.5):
    """stage_score in [-1, 1]: -1 = 'early' (R-like), +1 = 'late' (S-like). Weight moves from favouring early
    samples to favouring late samples as training progresses (frac 0 -> 1); mean-normalised per batch."""
    w = (1 + beta * (frac - 0.5) * 2 * stage_score).clamp(min=0.05)
    return w / w.mean()


def rank_scale(v):
    """Map scores to [-1, 1] by rank (robust to the heavy tails of local MI)."""
    r = np.argsort(np.argsort(v))
    return torch.tensor(2 * r / (len(v) - 1) - 1, device=dev, dtype=torch.float32)


def train(tr, te, K, d, method, seed, epochs, bs, lr, order=None, stage_score=None, static_score=None, beta=0.75):
    torch.manual_seed(seed)
    model = Fusion(d, K).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    n = len(tr["y"]); spe = n // bs; total = epochs * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total, pct_start=0.25)
    for step in range(total):
        if order is not None:
            m = math.ceil(competence(step, total) * n)
            idx = order[torch.randint(0, m, (bs,), device=dev)]
        else:
            idx = torch.randint(0, n, (bs,), device=dev)
        x1, x2, y = tr["x1"][idx], tr["x2"][idx], tr["y"][idx]
        if method == "mod_dropout":
            r = torch.rand(bs, device=dev)
            logits = model(x1, x2, drop1=r < 0.25, drop2=(r >= 0.25) & (r < 0.5))
            loss = F.cross_entropy(logits, y)
        elif method == "ogm_ge":
            logits, h1, h2 = model(x1, x2, feats=True)
            loss = F.cross_entropy(logits, y)
        else:
            logits = model(x1, x2)
            l = F.cross_entropy(logits, y, reduction="none")
            if stage_score is not None:
                loss = (l * soft_weights(stage_score[idx], step / total)).mean()
            elif static_score is not None:
                w = 1 + beta * static_score[idx]
                loss = (l * w / w.mean()).mean()
            else:
                loss = l.mean()
        opt.zero_grad(); loss.backward()
        if method == "ogm_ge":
            # OGM-GE: modality scores via ablation of the other modality's embedding, alpha = 0.5
            with torch.no_grad():
                s1 = F.softmax(model.head(torch.cat([h1, torch.zeros_like(h2)], 1)), 1).gather(1, y[:, None]).sum()
                s2 = F.softmax(model.head(torch.cat([torch.zeros_like(h1), h2], 1)), 1).gather(1, y[:, None]).sum()
                r1, r2 = s1 / s2.clamp(min=1e-8), s2 / s1.clamp(min=1e-8)
                k1 = 1 - torch.tanh(0.5 * F.relu(r1 - 1)); k2 = 1 - torch.tanh(0.5 * F.relu(r2 - 1))
                for enc, k in ((model.e1, k1), (model.e2, k2)):
                    for p in enc.parameters():
                        if p.grad is not None:
                            p.grad.mul_(k).add_(torch.randn_like(p.grad) * p.grad.std().nan_to_num(0) * (k < 1))
        opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        pred = model(te["x1"], te["x2"]).argmax(1).cpu().numpy()
    corr = pred == te["y"].cpu().numpy()
    out = dict(acc=corr.mean())
    for j, name in enumerate(TYPES):
        out[f"acc_{name}"] = corr[te["t"] == j].mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--snr", type=float, nargs=2, default=[4.5, 3.0])
    ap.add_argument("--mix", type=float, nargs=4, default=[0.4, 0.15, 0.15, 0.3], help="R U1 U2 S")
    ap.add_argument("--label_noise", type=float, default=0.0, help="symmetric train-label noise (test stays clean)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--probe_steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--methods", default="random,random_paced,oracle_paced,oracle_paced_rev,pid_paced,mi_paced,"
                                         "oracle_soft,pid_soft,mi_soft,ogm_ge,mod_dropout")
    ap.add_argument("--out", default="stage0_results.csv")
    args = ap.parse_args()
    methods = args.methods.split(",")
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        emb = [rng.standard_normal((4, args.K, args.d)) for _ in range(2)]
        emb = [e / np.linalg.norm(e, axis=2, keepdims=True) for e in emb]  # unit-norm codes; snr = code norm
        tr = make_data(args.n_train, args.K, args.d, args.snr, args.mix, emb, rng)
        te = make_data(args.n_test, args.K, args.d, args.snr, args.mix, emb, rng)
        noisy = rng.random(args.n_train) < args.label_noise
        yn = tr["y"].cpu().numpy().copy(); yn[noisy] = (yn[noisy] + rng.integers(1, args.K, noisy.sum())) % args.K
        tr["y"] = torch.tensor(yn, device=dev)
        t0 = time.time(); A = pid_estimate(tr, args.K, args.d, args.probe_steps, seed); t_pid = time.time() - t0
        tt = tr["t"]; et = est_type(A)
        if seed == 0:
            print(f"PID estimate ({t_pid:.0f}s). Mean atoms by TRUE type (bits):")
            print(pd.DataFrame({k: [A[k][tt == j].mean() for j in range(4)] for k in ("i1", "i2", "i12", "red", "u1", "u2", "syn")},
                               index=TYPES).round(2).to_string())
        diag = {f"auroc_{k}_{TYPES[j]}": roc_auc_score(tt == j, A[k]) for k, j in (("red", 0), ("u1", 1), ("u2", 2), ("syn", 3))}
        diag["auroc_syncand_S"] = roc_auc_score(tt == 3, -np.maximum(A["i1"], A["i2"]))
        diag["type_acc"] = (et == tt).mean()
        print(f"seed {seed}: estimated-type accuracy {diag['type_acc']:.3f}; " + ", ".join(f"{k}={v:.3f}" for k, v in diag.items() if k != "type_acc"))
        rank_true = np.array([0, 1, 1, 2])[tt]; rank_est = np.array([0, 1, 1, 2])[et]
        stage_true = torch.tensor(np.array([-1.0, 0.0, 0.0, 1.0])[tt], device=dev, dtype=torch.float32)
        # continuous stage score from estimated atoms: (syn - red) / (|red| + |u1| + |u2| + |syn|), in [-1, 1]
        tot = np.abs(A["red"]) + np.abs(A["u1"]) + np.abs(A["u2"]) + np.abs(A["syn"]) + 1e-6
        stage_pid = torch.tensor(np.clip((A["syn"] - A["red"]) / tot, -1, 1), device=dev, dtype=torch.float32)
        z = (A["i12"] - A["i12"].mean()) / (A["i12"].std() + 1e-6)
        stage_mi = torch.tensor(np.clip(-z / 2, -1, 1), device=dev, dtype=torch.float32)
        cfg = dict(random={}, random_paced=dict(order=order_from(np.zeros(len(tt)), seed)),
                   oracle_paced=dict(order=order_from(rank_true, seed)),
                   oracle_paced_rev=dict(order=order_from(-rank_true, seed)),
                   pid_paced=dict(order=order_from(rank_est, seed)),
                   mi_paced=dict(order=order_from(-A["i12"], seed)),
                   oracle_soft=dict(stage_score=stage_true), pid_soft=dict(stage_score=stage_pid),
                   mi_soft=dict(stage_score=stage_mi), ogm_ge={}, mod_dropout={},
                   pid_paced_rev=dict(order=order_from(-rank_est, seed)),
                   syncand_paced=dict(order=order_from(np.maximum(A["i1"], A["i2"]), seed)),
                   mi_paced_rev=dict(order=order_from(A["i12"], seed)),
                   oracle_up=dict(static_score=stage_true),
                   syncand_up=dict(static_score=rank_scale(-np.maximum(A["i1"], A["i2"]))),
                   mi_up_rev=dict(static_score=rank_scale(-A["i12"])))
        if any(m in methods for m in ("pid2_up", "mi2_filter")):
            # two-stage: re-estimate with a fusion probe trained under 'neither modality suffices' up-weighting, so
            # that it learns the synergistic interaction; then synergy (high syn) and noise (i12 < 0) separate
            w12 = 1 + 0.75 * rank_scale(-np.maximum(A["i1"], A["i2"]))
            A2 = pid_estimate(tr, args.K, args.d, args.probe_steps, seed + 100, w12=w12)
            clean = ~noisy
            diag["auroc_syn2_S_clean"] = roc_auc_score(tt[clean] == 3, A2["syn"][clean])
            if noisy.any():
                diag["auroc_noise_negi12"] = roc_auc_score(noisy, -A["i12"])
                diag["auroc_noise_negi12_2"] = roc_auc_score(noisy, -A2["i12"])
                diag["auroc_noise_syncand"] = roc_auc_score(noisy, -np.maximum(A["i1"], A["i2"]))
            print("    two-stage diag: " + ", ".join(f"{k}={v:.3f}" for k, v in diag.items() if "2" in k or "noise" in k))
            neg2 = A2["i12"] < 0
            s2 = rank_scale(A2["syn"]).cpu().numpy(); s2[neg2] = -1.0
            cfg["pid2_up"] = dict(static_score=torch.tensor(s2, device=dev, dtype=torch.float32))
            cfg["mi2_filter"] = dict(static_score=torch.tensor(np.where(neg2, -1.0, 0.0), device=dev, dtype=torch.float32))
        cfg["mi_filter"] = dict(static_score=torch.tensor(np.where(A["i12"] < 0, -1.0, 0.0), device=dev, dtype=torch.float32))
        cfg["noise_oracle"] = dict(static_score=torch.tensor(np.where(noisy, -1.0, 0.0), device=dev, dtype=torch.float32))
        for m in methods:
            t0 = time.time()
            r = train(tr, te, args.K, args.d, m, seed, args.epochs, args.bs, args.lr, **cfg[m])
            rows.append(dict(method=m, seed=seed, **r, **diag, secs=time.time() - t0))
            print(f"  {m:17s} acc {r['acc']:.4f} | " + " ".join(f"{t}={r['acc_' + t]:.3f}" for t in TYPES), flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    R = pd.DataFrame(rows)
    cols = ["acc"] + [f"acc_{t}" for t in TYPES]
    S = (R.groupby("method")[cols].mean() * 100).round(2).sort_values("acc", ascending=False)
    S["acc_std"] = (R.groupby("method").acc.std() * 100).round(2)
    print("\n=== test accuracy (%), mean over seeds ===\n" + S.to_string())
    base = R[R.method == "random"].set_index("seed").acc
    print("\npaired Δacc vs random (pts):")
    for m in S.index:
        d = (R[R.method == m].set_index("seed").acc - base) * 100
        h = 2.776 * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  {m:17s} {d.mean():+.2f} ± {h:.2f}")


if __name__ == "__main__":
    main()
