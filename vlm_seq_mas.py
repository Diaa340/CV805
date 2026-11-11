"""Memory Aware Synapses on CLOVE-scene (guide_online_mas.md). Same cells, model, LoRA, test sample, 300 steps x 8 per task,
lr 1e-4 and shared task-1 model as vlm_seq / vlm_seq_eap (imported unchanged). F(x) is the yes/no log-odds score s. Arms:
  lam in LAMS = 0.5, 1, 2, 5 for every MAS kind (the authors used 1 for object recognition; our output scale differs)
  mas_<lam>      ECCV MAS at task boundaries: Omega = mean over 200 random training cells of |d s^2/d theta| (one cell per
                 backward), summed over tasks; penalty lam * sum Omega (theta - theta*)^2
  masexc_<lam>   the same with Omega on (up to) 200 training exceptions
  tfmas_<mu>     task-free MAS (Aljundi et al. 2019): each step trains on the new batch of 8 plus a hard buffer of 8; the
                 hard buffer keeps the 8 highest-loss cells of both; on a loss plateau Omega = sum over the hard buffer of
                 |d |s| / d theta|, running-averaged over plateaus, theta* reset; penalty mu/2 * sum Omega (theta - theta*)^2.
                 Task ids and boundaries are never used.
  hardbuf        the task-free learner with mu = 0 (hard-buffer replay only)
Plateau thresholds come from the shared task-1 run, before any arm: mean threshold = median, variance threshold = 90th
percentile, of the window statistics over the last 100 task-1 steps. The window holds the new-batch loss only.
Saves runs/clove_vlm_mas/order_<o>_seed_<s>[_<arms>].csv.gz (columns as vlm_seq_eap), _omega.csv and _steps.csv.
Usage: python vlm_seq_mas.py <order index> [--seed 31] [--arms a,b] [--smoke]"""
import argparse, os, time, types
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import vlm_seq_eap   # noqa: F401  (patches the CLEVR zip import of vlm_forget)
import vlm_seq as VS
from vlm_seq import Learner, dev
OUT = os.path.join(VS.ROOT, "runs", "clove_vlm_mas"); os.makedirs(OUT, exist_ok=True)
LAMS = (0.5, 0.75, 1, 2, 5)
ARMS = {f"{k}_{l:g}": (k, float(l)) for k in ("mas", "masexc", "tfmas") for l in LAMS} | {"hardbuf": ("tfmas", 0.0)}
W, HB = 20, 8   # loss window (steps), hard buffer size


def omega(L, C, sq):
    """Sum over cells of |d F / d theta|, one cell per backward; F = s^2 (sq, ECCV MAS) or |s| (task-free MAS)."""
    L.v.model.eval(); O = {k: torch.zeros_like(p) for k, p in L.P.items()}
    for i in range(len(C)):
        L.v.model.zero_grad(); s = L.score(C.img.values[i:i + 1], C.text.values[i:i + 1])
        (s ** 2 if sq else s.abs()).sum().backward()
        for k, p in L.P.items():
            O[k] += p.grad.detach().abs()
    L.v.model.zero_grad(); return O


def sample(D, rng, n=200, exc_only=False):
    E = D[D.nonadd == 1] if exc_only else D; return E.loc[rng.permutation(E.index)[:n]]


def penalty(L, O, star):
    return sum((O[k] * (p - star[k]) ** 2).sum() for k, p in L.P.items())


def ostat(name, upd, O, C, **kw):
    v = torch.cat([o.flatten() for o in O.values()])
    return dict(arm=name, update=upd, omega_mean=v.mean().item(), omega_max=v.max().item(), cells=len(C),
                exc_share=round(C.nonadd.mean(), 3), yes_share=round(C.y.mean(), 3), **kw)


def bce(L, d, reduction="mean"):
    return F.binary_cross_entropy_with_logits(L.score(d.img.values, d.text.values), torch.tensor(d.y.values, device=dev).float(), reduction=reduction)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("order", type=int); ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    a = ap.parse_args(); arms = a.arms.split(","); order, seed = VS.ORDERS[a.order], a.seed
    tasks = VS.cells(); r51 = np.random.default_rng(51); te = {}
    arch = set(pd.read_csv(os.path.join(VS.ROOT, "runs", "clove", "test_exceptions_archived.csv")).cell)
    for tk in order:   # test sample exactly as vlm_seq_eap (archived test labels)
        E = tasks[tk]["test"]; E["nonadd"] = E.cell.isin(arch).astype(int); rule = E[E.nonadd == 0]; te[tk] = pd.concat([E[E.nonadd == 1], rule.loc[r51.choice(rule.index, VS.NRULE, replace=False)]]).reset_index(drop=True)
        tasks[tk]["train"] = tasks[tk]["train"].reset_index(drop=True)
        assert not set(tasks[tk]["train"].img) & set(te[tk].img), "train and test images overlap"
        print(f"task {tk}: {len(tasks[tk]['train'])} training cells, exception share {tasks[tk]['train'].nonadd.mean():.3f}", flush=True)
    if a.smoke:
        VS.STEPS = 6; te = {tk: d.iloc[:40] for tk, d in te.items()}
    w = min(W, VS.STEPS); tag = f"order_{a.order}_seed_{seed}{'_smoke' if a.smoke else ''}" + (f"_{'-'.join(arms)}" if arms != list(ARMS) else "")
    L = Learner(a.model); rows = {tk: te[tk][["unit", "nonadd", "y", "cell"]].assign(order=a.order, seed=seed, task=tk, pos=order.index(tk) + 1) for tk in order}
    t0 = time.time(); olog, slog = [], []
    for tk in order:
        rows[tk]["zeroshot"] = (L.scores(te[tk]) > 0) == te[tk].y.values

    # Shared task 1, exactly vlm_seq's L.train; the per-step loss is recorded through a pass-through shim of its F
    D1 = tasks[order[0]]["train"]; rec = []
    VS.F = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("_")})
    VS.F.binary_cross_entropy_with_logits = lambda *x, **k: (lambda l: (rec.append(l.item()), l)[1])(F.binary_cross_entropy_with_logits(*x, **k))
    L.load(L.init); L.train(D1, np.random.default_rng(seed), log=f"task1 {order[0]}"); VS.F = F; s1 = L.state()
    sc1 = L.scores(te[order[0]]); own1 = (sc1 > 0) == te[order[0]].y.values; assert len(rec) == VS.STEPS
    win = [np.mean(rec[i - w + 1:i + 1]) for i in range(w - 1, VS.STEPS)], [np.var(rec[i - w + 1:i + 1]) for i in range(w - 1, VS.STEPS)]
    last = slice(-min(100, len(win[0])), None); mthr, vthr = np.median(win[0][last]), np.percentile(win[1][last], 90)
    if a.smoke:   # make the plateau path run in the smoke test
        mthr = vthr = np.inf
    print(f"task 1 done ({time.time() - t0:.0f}s); plateau thresholds: window mean < {mthr:.4f}, variance < {vthr:.4f}", flush=True)
    idx1 = np.random.default_rng(seed).integers(0, len(D1), (VS.STEPS, VS.BSZ))   # task-1 batches, as drawn inside L.train
    C1 = D1.iloc[np.unique(idx1[-min(30, VS.STEPS):])]
    with torch.no_grad():
        l1 = np.concatenate([bce(L, C1.iloc[b:b + 32], "none").cpu().numpy() for b in range(0, len(C1), 32)])
    H1 = C1.iloc[np.argsort(-l1)[:HB]]
    teimg = set().union(*[set(te[t].img) for t in order])

    for name in arms:
        kind, lam = ARMS[name]; L.load(s1); rng = np.random.default_rng(seed + 1)
        rows[order[0]][f"{name}__own"] = own1; rows[order[0]][f"{name}__ownscore"] = sc1; star = L.state()
        if kind in ("mas", "masexc"):
            C = sample(D1, np.random.default_rng(seed * 100 + 99), exc_only=kind == "masexc"); O = omega(L, C, True)
            O = {k: v / len(C) for k, v in O.items()}; olog.append(ostat(name, order[0], O, C))
        else:   # end of task 1 = first plateau; its hard buffer = the 8 highest-loss cells of task 1's last 30 batches
            H = H1.copy(); O = omega(L, H, False); n_upd = 1; olog.append(ostat(name, 1, O, H, step=0))
            window = list(rec[-w:]); last_m, last_v = np.mean(window), np.var(window); peak = bool(a.smoke); gstep = 0
        for pos, tk in enumerate(order[1:], start=2):
            D = tasks[tk]["train"]; idx = rng.integers(0, len(D), (VS.STEPS, VS.BSZ))
            L.v.model.train(); opt = torch.optim.AdamW(list(L.P.values()), lr=VS.LR, weight_decay=0.0); t1 = time.time()
            for s in range(VS.STEPS):
                d = D.iloc[idx[s]]
                if kind in ("mas", "masexc"):
                    lo = bce(L, d); pen = lam * penalty(L, O, star); loss = lo + pen
                    opt.zero_grad(); loss.backward(); opt.step()
                    if s % 50 == 0:
                        slog.append(dict(arm=name, task=tk, step=s, bce=lo.item(), pen=pen.item()))
                else:
                    B = pd.concat([d, H]); lv = bce(L, B, "none"); lo = lv[:len(d)].mean()
                    pen = lam / 2 * penalty(L, O, star) if lam > 0 else torch.zeros((), device=dev); loss = lv.mean() + pen
                    opt.zero_grad(); loss.backward(); opt.step(); gstep += 1
                    window = (window + [lo.item()])[-w:]; m, v = np.mean(window), np.var(window)
                    if not peak and m > last_m + np.sqrt(last_v):
                        peak = True
                    plateau = m < mthr and v < vthr and peak
                    H = B.iloc[np.argsort(-lv.detach().cpu().numpy(), kind="stable")[:HB]]   # pre-update losses, as in the reference
                    assert not set(H.img) & teimg, "hard buffer holds a test image"
                    if plateau:
                        n_upd += 1; last_m, last_v, peak = m, v, False
                        On = omega(L, H, False); O = {k: On[k] / n_upd + (1 - 1 / n_upd) * O[k] for k in O}; star = L.state(); L.v.model.train()
                        olog.append(ostat(name, n_upd, O, H, step=gstep))
                    slog.append(dict(arm=name, task=tk, step=s, bce=lo.item(), pen=pen.item(), win_mean=m, win_var=v, plateau=int(plateau),
                                     hb_exc=H.nonadd.mean(), hb_yes=H.y.mean()))
                if s % 100 == 0:
                    print(f"  {name} {tk} step {s}/{VS.STEPS} bce {lo.item():.3f} pen {pen.item():.4f} ({time.time() - t1:.0f}s)", flush=True)
            L.v.model.eval()
            if kind in ("mas", "masexc"):   # Omega of the finished task, added to the stored Omega; theta* moves
                C = sample(D, np.random.default_rng(seed * 100 + 99 + pos), exc_only=kind == "masexc"); On = omega(L, C, True)
                O = {k: O[k] + On[k] / len(C) for k in O}; star = L.state(); olog.append(ostat(name, tk, O, C))
            sc = L.scores(te[tk]); rows[tk][f"{name}__own"] = (sc > 0) == te[tk].y.values; rows[tk][f"{name}__ownscore"] = sc
        for tk in order:
            rows[tk][f"{name}__final"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        print(f"{name} done ({time.time() - t0:.0f}s){f', {n_upd} importance updates' if kind == 'tfmas' else ''}", flush=True)
        pd.concat(rows.values()).to_csv(os.path.join(OUT, f"{tag}.csv.gz"), index=False)
        pd.DataFrame(olog).to_csv(os.path.join(OUT, f"{tag}_omega.csv"), index=False); pd.DataFrame(slog).to_csv(os.path.join(OUT, f"{tag}_steps.csv"), index=False)
    print("DONE", a.order, seed, flush=True)


if __name__ == "__main__":
    main()
