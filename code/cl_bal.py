"""Class-balance control for the rule+exceptions buffer (paper/preregistration_strat.md), two-task benchmarks
(tables, clevr, gqa; same units, seeds and task-1 training as cl_metrics.py). From the task-1 model, DER++ (memory
1000) learns task 2 with four buffers: random and rule+exceptions (the same buffers as cl_metrics, same draws), class-
balanced random (500 yes, 500 no), and class-balanced rule+exceptions (the same exception half; the other half chosen
so the whole buffer is 50% yes). Saves per test sample: answer, type, correct before / after each arm, task-2-only model.
Usage: python cl_bal.py <tables|clevr|gqa> <shard> <n_shards>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_metrics as CM
from cl_exceptions import buffers, task2, POOL
from route_exps import Net
OUT = os.path.join(CM.ROOT, "runs", "clbal"); os.makedirs(OUT, exist_ok=True); dev = "cuda"; M = 1000
ARMS = ["random", "ruleexc", "classbal", "ruleexc_bal"]


def exc_pick(tr, nonadd_tr, m, rng):   # one exception cell after another, as in cl_exceptions.buffers
    ex = np.where(nonadd_tr[:POOL] == 1)[0]; cells = {}
    for j in rng.permutation(ex):
        cells.setdefault(int(tr[3][j]) * 100000 + int(tr[4][j]), []).append(j)
    pick, lists = [], list(cells.values())
    while len(pick) < m and any(lists):
        for v in lists:
            if v and len(pick) < m:
                pick.append(v.pop())
    return np.array(pick, int)


def balanced_fill(y, have, n, rng):
    """n more indices from the pool (not in have) so that have + new is as close to 50% yes as possible."""
    rest = np.setdiff1d(np.arange(POOL), have); want_yes = int(np.clip((len(have) + n) // 2 - y[have].sum(), 0, n))
    ry, rn = rest[y[rest] == 1], rest[y[rest] == 0]; ky = min(want_yes, len(ry)); kn = min(n - ky, len(rn)); ky = n - kn
    return np.concatenate([rng.choice(ry, ky, replace=False), rng.choice(rn, kn, replace=False)])


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, _ = CM.train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy(), before=(net1(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); r2 = np.random.default_rng(seed + 7); y = tr1[2][:POOL].cpu().numpy()
    half = exc_pick(tr1, nonadd_tr, M // 2, r2)
    bufs = {"random": B[("random", M)], "ruleexc": B[("ruleexc", M)], "classbal": balanced_fill(y, np.array([], int), M, r2),
            "ruleexc_bal": np.concatenate([half, balanced_fill(y, half, M - len(half), r2)])}
    D2 = pd.DataFrame(dict(unit=name, nonadd=type2, y=te2[2].cpu().numpy()))
    for a in ARMS:
        net = task2(net1, tr1, tr2, "derpp", 0.0, bufs[a], None)
        with torch.no_grad():
            D1[a] = (net(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy(); D2[a] = (net(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
        D1[f"{a}__yes_share"] = y[bufs[a]].mean()
    sn = CM.scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
    with torch.no_grad():
        D2["scratch"] = (sn(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
    print(name, f"({time.time() - t0:.0f}s)", flush=True); return D1, D2


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns))
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}.csv.gz"), index=False); print("DONE", bn, s)
