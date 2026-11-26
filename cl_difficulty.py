"""Revision R4 (paper/preregistration_revision.md): continual-learning runs that also save each test cell's difficulty.
Same benchmarks, units, seeds and task-1 training as cl_metrics.py (constructed tasks, CLEVR, GQA). Methods: finetune,
LwF (0.3, 3), feature distillation (0.3, 3), ER and DER++ (random buffer 1000), DER-expansion. Saved per task-1 test
cell: answer, type, correct before task 2, the task-1 model's signed margin for the true answer (logit_true -
logit_other), correct after each method. Per task-2 test cell: answer, type, the task-2-only model's correctness and
signed margin, correct after each method. Usage: python cl_difficulty.py <tables|clevr|gqa> <shard> <n_shards>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_metrics as CM
from cl_exceptions import buffers, fisher_diag
from route_exps import Net
OUT = os.path.join(CM.ROOT, "runs", "rev", "difficulty"); os.makedirs(OUT, exist_ok=True); dev = "cuda"
KEEP = ("finetune", "lwf_0.3", "lwf_3", "l2_0.3", "l2_3", "er_random_1000", "derpp_random_1000", "der_expand_1000")
RUNS = [r for r in CM.RUNS if r[0] in KEEP]


def margin(logits, y):
    return (logits.gather(1, y[:, None])[:, 0] - logits.gather(1, (1 - y)[:, None])[:, 0]).cpu().numpy()


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, Om = CM.train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        lo = net1(te1[0], te1[1])
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy(), before=(lo.argmax(1) == te1[2]).cpu().numpy(), margin_before=margin(lo, te1[2])))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); Fi = fisher_diag(net1, tr1); D2 = pd.DataFrame(dict(unit=name, nonadd=type2, y=te2[2].cpu().numpy()))
    for key, kind, lam, bk in RUNS:
        f1, f2, g = CM.task2x(net1, tr1, tr2, kind, lam, B[bk] if bk else None, Fi, Om)
        with torch.no_grad():
            D1[key] = (f1(te1[0], te1[1]) == te1[2]).cpu().numpy(); D2[key] = (f2(te2[0], te2[1]) == te2[2]).cpu().numpy()
    sn = CM.scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
    with torch.no_grad():
        ls = sn(te2[0], te2[1]); D2["scratch"] = (ls.argmax(1) == te2[2]).cpu().numpy(); D2["margin_scratch"] = margin(ls, te2[2])
    print(name, f"({time.time() - t0:.0f}s)", flush=True); return D1, D2


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    if "--limit" in sys.argv:   # smoke test: stop after one unit
        orig = CM.run_unit
        def one(*a, **k):
            r = orig(*a, **k); pd.concat([r[0]]).to_csv(os.path.join(OUT, f"smoke_task1_{bn}.csv.gz"), index=False); pd.concat([r[1]]).to_csv(os.path.join(OUT, f"smoke_task2_{bn}.csv.gz"), index=False); raise SystemExit
        CM.run_unit = one
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns))
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}.csv.gz"), index=False); print("DONE", bn, s)
