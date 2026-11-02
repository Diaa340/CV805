"""RPSnet memory variants (rps_core.py) on the two-task benchmarks (tables = Constructed, CLEVR, GQA). Imports
cl_metrics_eap unchanged: same units, seeds, shared task-1 model (net1), buffers. All memories hold 1000 task-1 cells:
  rps               random 1000 (as cl_metrics_rps.py; re-run here so GQA arms share one process)
  rps_ruleexc       cl_exceptions "ruleexc" 1000: half exception cells (one per cell in turn), half random
  rps_herding       cl_exceptions "herding" 1000: iCaRL herding per answer on the task-1 model's features
  rps_poollogit_sm  storage-matched exception pool as cl_metrics_eap_sm: pool = first min(available, 500) cells of the
                    exc_pool draw, held at the task-1 logits (0.5 MSE, RB draws per step); memory = first 1000 - |pool|
                    cells of the random buffer (replay and path validation)
Also derpp_random_1000 (same-process reference) and scratch / margins for the gap and difficulty metrics.
Writes runs/clm_eap/{task1,task2}_<bench>_<shard>_rps2.csv.gz and rps2_<bench>_<shard>_paths.csv.
Usage: python cl_metrics_rps2.py <tables|clevr|gqa> <shard> <nshards>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_metrics_eap as CE
from cl_metrics_eap import CM, Net, buffers, task2, exc_pool, margin, NPOOL, OUT, t, dev
from cl_exceptions import E2, BS
import rps_core as RP

TOTAL, PMAX = 1000, 500
PLOG = []


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, Om = CM.train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy(), before=(net1(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); D2 = pd.DataFrame(dict(unit=name, nonadd=type2, y=te2[2].cpu().numpy())); rb = B[("random", 1000)]
    pidx, ncell = exc_pool(tr1, nonadd_tr, NPOOL, seed); k = min(len(pidx), PMAX); psm, bsm = pidx[:k], rb[:TOTAL - k]
    mem = lambda idx: dict(x1=tr1[0][t(idx)], x2=tr1[1][t(idx)], y=tr1[2][t(idx)])
    with torch.no_grad():
        pool = dict(x1=tr1[0][t(psm)], x2=tr1[1][t(psm)], logit=net1(tr1[0][t(psm)], tr1[1][t(psm)]))
    nets = {"derpp_random_1000": task2(net1, tr1, tr2, "derpp", 0.0, rb)}; steps = E2 * (len(tr2[2]) // BS)
    for key, m, pl in (("rps", mem(rb), None), ("rps_ruleexc", mem(B[("ruleexc", 1000)]), None),
                       ("rps_herding", mem(B[("herding", 1000)]), None), ("rps_poollogit_sm", mem(bsm), pool)):
        r0 = RP.build(net1, seed)
        nets[key], _, log = RP.session(r0, r0.infer.copy(), 1, tr2[0], tr2[1], tr2[2], m, steps, seed, BS, pool=pl)
        PLOG.extend(dict(unit=name, arm=key, **r) for r in log)
    for key, net in nets.items():
        with torch.no_grad():
            D1[key] = (net(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy(); D2[key] = (net(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
    sn = CM.scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
    with torch.no_grad():
        D1["m_before"] = margin(net1, te1); D2["scratch"] = (sn(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy(); D2["m_scratch"] = margin(sn, te2)
    print(name, "pool", len(psm), "buf", len(bsm), "sec", round(time.time() - t0), flush=True); return D1, D2


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns))
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}_rps2.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}_rps2.csv.gz"), index=False)
    pd.DataFrame(PLOG).to_csv(os.path.join(OUT, f"rps2_{bn}_{s}_paths.csv"), index=False); print("DONE", bn, s)
