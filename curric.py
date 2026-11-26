"""Exception curricula, single task (paper/preregistration_curric.md). Same benchmarks, units, seeds, network and budget
as the first task of cl_metrics.py (constructed tasks, CLEVR, GQA; 50,000 training draws, batch 256, 30 epochs, Adam
1e-3). Six ways of choosing training samples, each from the same initialisation:
  random       uniform over the training draws (as in the CL study)
  misfit       sampling weight 1 + 4 q, q = percentile of the cell's graded misfit in the TRAINING answer table
               (rebuilt from the training draws; model-free)
  easy_first   only cells with misfit percentile <= c(t), c rising linearly from 0.5 to 1 over the first half of training
  hard_first   only cells with misfit percentile >= 1 - c(t), same schedule
  focal        uniform sampling, focal loss (gamma 2)
  loss_select  1,024 uniform candidates per step, 256 kept with probability proportional to their current loss
Saved per test cell: answer, saved exception label, correct under each condition.
Usage: python curric.py <tables|clevr|gqa> <shard> <n_shards> [--limit]"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import cl_metrics as CM
from route_exps import Net
from unrepair_structure import local_synred
OUT = os.path.join(CM.ROOT, "runs", "curric"); os.makedirs(OUT, exist_ok=True); dev = "cuda"; BS, E = 256, 30
CONDS = ["random", "misfit", "easy_first", "hard_first", "focal", "loss_select"]


def misfit_percentile(rows, cols, y):
    """Graded misfit of each training draw's cell in the empirical training table (cells seen at least once)."""
    r, c = np.asarray(rows), np.asarray(cols); R, C = r.max() + 1, c.max() + 1
    T = -np.ones((R, C), int); T[r, c] = y; W = (T >= 0).astype(float); M = local_synred(T, W)
    m = M[r, c]; q = pd.Series(m).rank(pct=True, method="average").values; return q


def train(cond, tr, q, seed, d1, d2):
    torch.manual_seed(seed); net = Net(d1, d2, 2).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(tr[2]); steps = E * (n // BS); qt = torch.tensor(q, device=dev).float(); w = 1 + 4 * qt
    for s in range(steps):
        if cond == "misfit":
            i = torch.multinomial(w, BS, replacement=True)
        elif cond in ("easy_first", "hard_first"):
            c = min(1.0, 0.5 + 0.5 * s / (steps / 2)); ok = (qt <= c) if cond == "easy_first" else (qt >= 1 - c)
            i = torch.multinomial(ok.float(), BS, replacement=True)
        elif cond == "loss_select":
            cand = torch.randint(0, n, (4 * BS,), device=dev)
            with torch.no_grad():
                l = F.cross_entropy(net(tr[0][cand], tr[1][cand]), tr[2][cand], reduction="none")
            i = cand[torch.multinomial(l + 1e-8, BS, replacement=False)]
        else:
            i = torch.randint(0, n, (BS,), device=dev)
        out = net(tr[0][i], tr[1][i])
        if cond == "focal":
            lp = F.log_softmax(out, 1).gather(1, tr[2][i][:, None])[:, 0]; loss = (-(1 - lp.exp()) ** 2 * lp).mean()
        else:
            loss = F.cross_entropy(out, tr[2][i])
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    t0 = time.time(); r, c = tr1[3], tr1[4]
    r = r.cpu().numpy() if torch.is_tensor(r) else np.asarray(r); c = c.cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
    q = misfit_percentile(r, c, tr1[2].cpu().numpy())
    D = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy()))
    for cond in CONDS:
        net = train(cond, tr1, q, seed, tr1[0].shape[1], tr1[1].shape[1])
        with torch.no_grad():
            D[cond] = (net(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()
    print(name, f"({time.time() - t0:.0f}s)", flush=True); return D, D


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    if "--limit" in sys.argv:
        orig = CM.run_unit
        def one(*a, **k):
            r = orig(*a, **k); r[0].to_csv(os.path.join(OUT, f"smoke_{bn}.csv.gz"), index=False); raise SystemExit
        CM.run_unit = one
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns))
    pd.concat([x[0] for x in res]).to_csv(os.path.join(OUT, f"test_{bn}_{s}.csv.gz"), index=False); print("DONE", bn, s)
