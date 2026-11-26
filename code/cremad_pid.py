"""
CREMA-D (audio-visual emotion, 6 classes, OGM-GE split 6698/744): PID-guided multimodal curriculum.

Protocol follows OGM-GE (GeWu-Lab/OGM-GE_CVPR2022): ResNet18 audio (1-ch log-STFT 257x188) + ResNet18 visual
(1 frame, 224px) from scratch, SGD lr 1e-3 momentum 0.9 wd 1e-4, bs 64, StepLR(0.1) at 70% of training.
Data is preprocessed by preprocess_cremad.py and kept on the GPU; augmentation (RandomResizedCrop 224 + flip,
random 1-of-3 frames) is done on the GPU. Test uses the middle frame (t = 1 s) resized to 224.

Fusion: 'concat' = Linear(1024 -> 6) as in OGM-GE (additive across modalities: cannot represent synergy);
        'mlp'    = Linear(1024, 512)-ReLU-Linear(512, 6) (non-additive: synergy representable).

Per-sample PID (cross-fitted, calibrated probes; I_ccs redundancy), see estimate_pid(). Human per-clip atoms from the
CREMA-D crowd ratings (voice-only / face-only / audio-visual votes) are used as a reference ('human_*' methods are
oracles and for evaluation strata only).

Methods (same number of SGD steps; static weights w = 1 + beta*s, s in [-1, 1] rank-scaled, batch-normalised):
  random        standard training
  ogm_ge        OGM-GE, their implementation (alpha 0.8, modulation in first 50% of epochs)
  mi_up_rev     s = rank(-i_av)                    (MI-only control: up-weight low joint information)
  syncand_up    s = rank(-max(i_a, i_v))           (neither modality alone suffices)
  syn_up        s = rank(syn atom)
  human_syn_up  s = +1 human-synergistic, -1 human-redundant, 0 otherwise   (oracle)
  audio_only / visual_only   unimodal references (same encoder + head)
  -- unique-information rebalancing (human-oracle test): fused loss + lam * per-sample-weighted unimodal aux losses
  uni_aux       aux weights uniform                                  (known baseline: unimodal auxiliary losses)
  human_i_aux   aux weights from human i_a / i_v (modality alone informative; MI-only control)
  human_u_aux   aux weights from human u_a / u_v (information ONLY that modality carries; the PID claim)
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.stats import spearmanr

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
HTYPES = ["red", "u_a", "u_v", "syn", "unrecognized"]


# --------------------------------------------------------------------------------------------- data
def load():
    A = torch.tensor(np.load(os.path.join(ROOT, "data", "cremad_audio.npy")), device=dev)          # N,257,188 f16
    V = torch.tensor(np.load(os.path.join(ROOT, "data", "cremad_frames.npy")), device=dev)         # N,3,256,256,3 u8
    meta = pd.read_csv(os.path.join(ROOT, "data", "cremad_meta.csv"))
    H = pd.read_csv(os.path.join(ROOT, "data", "cremad_human_pid.csv"), index_col=0).loc[meta.name]
    y = torch.tensor(meta.label.values, device=dev)
    tr = torch.tensor(np.where(meta.split == "train")[0], device=dev)
    te = torch.tensor(np.where(meta.split == "test")[0], device=dev)
    return dict(A=A, V=V, y=y, tr=tr, te=te, meta=meta, H=H)


def visual_batch(V, idx, train):
    if train:
        f = V[idx, torch.randint(0, V.shape[1], (len(idx),), device=dev)]                    # B,256,256,3
        x = f.permute(0, 3, 1, 2).float() / 255
        # RandomResizedCrop(224, scale=(0.08, 1), ratio=(3/4, 4/3)) + horizontal flip, per sample
        B = len(idx); area = torch.empty(B, device=dev).uniform_(0.08, 1.0)
        logr = torch.empty(B, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3)); r = logr.exp()
        w = (area * r).sqrt().clamp(max=1.0); h = (area / r).sqrt().clamp(max=1.0)
        cx = torch.rand(B, device=dev) * (1 - w) + w / 2; cy = torch.rand(B, device=dev) * (1 - h) + h / 2
        flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = w * flip; theta[:, 0, 2] = cx * 2 - 1; theta[:, 1, 1] = h; theta[:, 1, 2] = cy * 2 - 1
        grid = F.affine_grid(theta, (B, 3, 224, 224), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
    else:
        x = V[idx, 1].permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False, antialias=True)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def audio_batch(A, idx):
    return A[idx].float().unsqueeze(1).contiguous(memory_format=torch.channels_last)


# -------------------------------------------------------------------------------------------- model
def resnet18(in_ch):
    m = torchvision.models.resnet18(weights=None)
    if in_ch != 3:
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    m.fc = nn.Identity()
    return m


class AV(nn.Module):
    def __init__(self, fusion="concat", nc=6, use_a=True, use_v=True):
        super().__init__()
        self.use_a, self.use_v, self.fusion = use_a, use_v, fusion
        self.audio_net = resnet18(1) if use_a else None
        self.visual_net = resnet18(3) if use_v else None
        d = 512 * (use_a + use_v)
        self.fc_out = nn.Linear(d, nc) if fusion == "concat" else nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, nc))

    def embed(self, a, v):
        return (self.audio_net(a) if self.use_a else None, self.visual_net(v) if self.use_v else None)

    def head(self, ea, ev):
        return self.fc_out(torch.cat([e for e in (ea, ev) if e is not None], 1))

    def forward(self, a, v):
        ea, ev = self.embed(a, v)
        return self.head(ea, ev), ea, ev


def amp():
    return torch.autocast("cuda", dtype=torch.float16)


@torch.no_grad()
def predict(model, D, idx, bs=256, ablate=None):
    model.eval(); out = []
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        with amp():
            ea, ev = model.embed(audio_batch(D["A"], j) if model.use_a else None,
                                 visual_batch(D["V"], j, False) if model.use_v else None)
            if ablate == "a":
                ea = torch.zeros_like(ea)
            if ablate == "v":
                ev = torch.zeros_like(ev)
            out.append(model.head(ea, ev).float())
    return torch.cat(out)


# ----------------------------------------------------------------------------------------- training
def train(D, idx_train, epochs, seed, fusion="concat", method="random", weights=None, use_a=True, use_v=True,
          bs=64, lr=1e-3, alpha=0.8, log_every=0, aux=None, lam=1.0):
    torch.manual_seed(seed)
    model = AV(fusion, use_a=use_a, use_v=use_v).to(dev).to(memory_format=torch.channels_last)
    params = list(model.parameters())
    if aux is not None:  # created after the model so its initialisation matches the other methods exactly
        aux_heads = nn.ModuleList([nn.Linear(512, 6), nn.Linear(512, 6)]).to(dev)
        params += list(aux_heads.parameters())
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    n = len(idx_train); spe = math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.StepLR(opt, int(0.7 * epochs) * spe, 0.1)
    scaler = torch.amp.GradScaler("cuda")
    y = D["y"]
    for ep in range(epochs):
        model.train()
        perm = idx_train[torch.randperm(n, device=dev)]
        for it in range(spe):
            j = perm[it * bs:(it + 1) * bs]
            with amp():
                out, ea, ev = model(audio_batch(D["A"], j) if use_a else None, visual_batch(D["V"], j, True) if use_v else None)
            l = F.cross_entropy(out.float(), y[j], reduction="none")
            if weights is not None:
                w = weights[j]; l = l * (w / w.mean())
            loss = l.mean()
            if aux is not None:
                for e, head, wm in ((ea, aux_heads[0], aux[0]), (ev, aux_heads[1], aux[1])):
                    la = F.cross_entropy(head(e.float()), y[j], reduction="none")
                    w = wm[j]; loss = loss + lam * (la * w / w.mean()).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if method == "ogm_ge" and ep <= int(0.5 * epochs):
                scaler.unscale_(opt)
                with torch.no_grad():  # OGM-GE, as implemented in their main.py
                    W = model.fc_out.weight if fusion == "concat" else None
                    if fusion == "concat":
                        out_a = ea.float() @ W[:, :512].T + model.fc_out.bias / 2
                        out_v = ev.float() @ W[:, 512:].T + model.fc_out.bias / 2
                    else:
                        with amp():
                            out_a = model.head(ea, torch.zeros_like(ev)).float(); out_v = model.head(torch.zeros_like(ea), ev).float()
                    sa = F.softmax(out_a, 1).gather(1, y[j][:, None]).sum(); sv = F.softmax(out_v, 1).gather(1, y[j][:, None]).sum()
                    rv = sv / sa.clamp(min=1e-8); ra = 1 / rv.clamp(min=1e-8)
                    ca, cv = (1.0, 1 - torch.tanh(alpha * F.relu(rv))) if rv > 1 else (1 - torch.tanh(alpha * F.relu(ra)), 1.0)
                    for net, c in ((model.audio_net, ca), (model.visual_net, cv)):
                        for p in net.parameters():
                            if p.grad is not None and p.grad.dim() == 4:
                                p.grad.mul_(c).add_(torch.randn_like(p.grad) * (p.grad.std() + 1e-8))
            scaler.step(opt); scaler.update(); sched.step()
        if log_every and (ep + 1) % log_every == 0:
            acc = (predict(model, D, D["te"]).argmax(1) == y[D["te"]]).float().mean().item()
            print(f"      [{method}/{fusion}] ep {ep + 1}/{epochs} test acc {acc:.4f} {time.strftime('%H:%M:%S')}", flush=True)
    return model


def evaluate(model, D):
    te = D["te"]; y = D["y"][te]
    pred = predict(model, D, te).argmax(1)
    out = dict(acc=(pred == y).float().mean().item())
    bimodal = model.use_a and model.use_v
    out["acc_audio_only"] = (predict(model, D, te, ablate="v").argmax(1) == y).float().mean().item() if bimodal else float("nan")
    out["acc_visual_only"] = (predict(model, D, te, ablate="a").argmax(1) == y).float().mean().item() if bimodal else float("nan")
    ht = D["H"].htype.values[te.cpu().numpy()]
    corr = (pred == y).cpu().numpy()
    for t in HTYPES:
        out[f"acc_h_{t}"] = corr[ht == t].mean() if (ht == t).any() else float("nan")
    return out


# ------------------------------------------------------------------------------------ PID estimation
def calibrate_T(logits, y):
    logT = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad(); l = F.cross_entropy(logits / logT.exp(), y); l.backward(); return l
    opt.step(closure)
    return logT.exp().item()


def estimate_pid(D, epochs, seed, fusion="mlp"):
    """2-fold cross-fitted, temperature-calibrated probes on the training set -> local i_a, i_v, i_av (bits) and
    I_ccs atoms for every training clip."""
    tr = D["tr"]; y = D["y"]
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    perm = tr[torch.randperm(len(tr), generator=g).to(dev)]
    folds = (perm[:len(perm) // 2], perm[len(perm) // 2:])
    lp = {k: torch.zeros(len(y), device=dev) for k in ("a", "v", "av")}
    for f in range(2):
        fit, score = folds[f], folds[1 - f]
        cut = int(0.85 * len(fit)); fit, cal = fit[:cut], fit[cut:]
        for k, ua, uv in (("a", True, False), ("v", False, True), ("av", True, True)):
            m = train(D, fit, epochs, seed * 10 + f, fusion=fusion, use_a=ua, use_v=uv)
            T = calibrate_T(predict(m, D, cal), y[cal])
            lp[k][score] = F.log_softmax(predict(m, D, score) / T, 1).gather(1, y[score][:, None])[:, 0]
            print(f"    probe fold {f} {k:2s}: T={T:.2f} held-out acc {(predict(m, D, score).argmax(1) == y[score]).float().mean():.3f}", flush=True)
            del m
    prior = torch.bincount(y[tr], minlength=6).float() / len(tr)
    lpy = prior.log()[y]
    i_a, i_v, i_av = [((lp[k] - lpy) / math.log(2)).cpu().numpy() for k in ("a", "v", "av")]
    c = i_a + i_v - i_av
    same = (np.sign(i_a) == np.sign(i_v)) & (np.sign(i_v) == np.sign(i_av)) & (np.sign(i_av) == np.sign(c))
    red = np.where(same, c, 0.0)
    P = dict(i_a=i_a, i_v=i_v, i_av=i_av, red=red, u_a=i_a - red, u_v=i_v - red, syn=i_av - i_a - i_v + red)
    trn = tr.cpu().numpy(); H = D["H"]
    print("  model vs human pointwise quantities (Spearman over training clips): " + ", ".join(
        f"{k}={spearmanr(P[k][trn], H[k].values[trn])[0]:+.3f}" for k in ("i_a", "i_v", "i_av", "red", "u_a", "u_v", "syn")))
    return P


def rank_scale(v, idx):
    s = np.zeros(len(v)); r = np.argsort(np.argsort(v[idx])); s[idx] = 2 * r / (len(idx) - 1) - 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--probe_epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--fusions", default="concat,mlp")
    ap.add_argument("--methods", default="random,ogm_ge,mi_up_rev,syncand_up,syn_up,human_syn_up,audio_only,visual_only")
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the unimodal auxiliary losses")
    ap.add_argument("--log_every", type=int, default=0)
    ap.add_argument("--out", default="cremad_results.csv")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    D = load()
    print(f"train {len(D['tr'])} test {len(D['te'])}; human type counts (test): "
          f"{pd.Series(D['H'].htype.values[D['te'].cpu().numpy()]).value_counts().to_dict()}", flush=True)
    methods = args.methods.split(",")
    out_path = os.path.join(ROOT, args.out)
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    done = {(r["method"], r["fusion"], r["seed"]) for r in rows}
    trn = D["tr"].cpu().numpy(); H = D["H"]
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        P = None
        if any(m in ("mi_up_rev", "syncand_up", "syn_up") for m in methods):
            cache = os.path.join(ROOT, f"cremad_pid_s{seed}_pe{args.probe_epochs}.npz")
            if os.path.exists(cache):
                P = dict(np.load(cache))
            elif any((m, f, seed) not in done for m in ("mi_up_rev", "syncand_up", "syn_up") if m in methods
                     for f in args.fusions.split(",")):
                t0 = time.time(); P = estimate_pid(D, args.probe_epochs, seed); print(f"  PID estimated in {time.time() - t0:.0f}s")
                np.savez(cache, **P)
        score = dict(
            mi_up_rev=lambda: rank_scale(-P["i_av"], trn), syncand_up=lambda: rank_scale(-np.maximum(P["i_a"], P["i_v"]), trn),
            syn_up=lambda: rank_scale(P["syn"], trn),
            human_syn_up=lambda: np.select([H.htype.values == "syn", H.htype.values == "red"], [1.0, -1.0], 0.0))
        for fusion in args.fusions.split(","):
            T = lambda a: torch.tensor(a, device=dev, dtype=torch.float32)
            aux_w = dict(uni_aux=lambda: (T(np.ones(len(H))), T(np.ones(len(H)))),
                         human_i_aux=lambda: (T(1 + args.beta * rank_scale(H.i_a.values, trn)),
                                              T(1 + args.beta * rank_scale(H.i_v.values, trn))),
                         human_u_aux=lambda: (T(1 + args.beta * rank_scale(H.u_a.values, trn)),
                                              T(1 + args.beta * rank_scale(H.u_v.values, trn))))
            for m in methods:
                if (m, fusion, seed) in done or (m in ("audio_only", "visual_only") and fusion != args.fusions.split(",")[0]):
                    continue
                w = None
                if m in score:
                    w = torch.tensor(1 + args.beta * score[m](), device=dev, dtype=torch.float32)
                t0 = time.time()
                model = train(D, D["tr"], args.epochs, seed, fusion=fusion, method=m, weights=w, log_every=args.log_every,
                              use_a=m != "visual_only", use_v=m != "audio_only",
                              aux=aux_w[m]() if m in aux_w else None, lam=args.lam)
                r = evaluate(model, D)
                rows.append(dict(method=m, fusion=fusion, seed=seed, secs=time.time() - t0, **r))
                print(f"seed {seed} {fusion:6s} {m:13s} acc {r['acc']:.4f} (audio-only {r['acc_audio_only']:.3f}, visual-only "
                      f"{r['acc_visual_only']:.3f}) | " + " ".join(f"{t}={r['acc_h_' + t]:.3f}" for t in HTYPES)
                      + f" ({time.time() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                del model


if __name__ == "__main__":
    main()
