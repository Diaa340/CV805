"""RPSnet (rps_core.py) on the two-task benchmarks (tables = Constructed, CLEVR, GQA). Imports cl_metrics_eap unchanged:
same units, seeds, shared task-1 model (net1), buffers. RPS uses DER++'s random 1000-cell buffer as its exemplar memory
and E2 * (n // BS) steps like every other task-2 method. One process per shard also runs finetune and derpp_random_1000
(sanity check against the earlier reruns) and scratch, so all rows share the task-1 model (needed on GQA).
Writes runs/clm_eap/{task1,task2}_<bench>_<shard>_rps.csv.gz (with m_before / m_scratch for difficulty matching) and
rps_<bench>_<shard>_paths.csv (candidate paths and validation accuracy).
Usage: python cl_metrics_rps.py <tables|clevr|gqa> <shard> <nshards>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_metrics_eap as CE
from cl_metrics_eap import CM, Net, buffers, task2, margin, OUT, t, dev
from cl_exceptions import E2, BS
import rps_core as RP

PLOG = []


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, Om = CM.train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy(), before=(net1(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); D2 = pd.DataFrame(dict(unit=name, nonadd=type2, y=te2[2].cpu().numpy())); rb = B[("random", 1000)]
    nets = {"finetune": task2(net1, tr1, tr2, "finetune", 0.0, None), "derpp_random_1000": task2(net1, tr1, tr2, "derpp", 0.0, rb)}
    r0 = RP.build(net1, seed); mem = dict(x1=tr1[0][t(rb)], x2=tr1[1][t(rb)], y=tr1[2][t(rb)])
    nets["rps"], _, log = RP.session(r0, r0.infer.copy(), 1, tr2[0], tr2[1], tr2[2], mem, E2 * (len(tr2[2]) // BS), seed, BS)
    for key, net in nets.items():
        with torch.no_grad():
            D1[key] = (net(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy(); D2[key] = (net(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
    sn = CM.scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
    with torch.no_grad():
        D1["m_before"] = margin(net1, te1); D2["scratch"] = (sn(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy(); D2["m_scratch"] = margin(sn, te2)
    PLOG.extend(dict(unit=name, **r) for r in log)
    print(name, "best val", max(r["val_acc"] for r in log), "sec", round(time.time() - t0), flush=True); return D1, D2


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns))
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}_rps.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}_rps.csv.gz"), index=False)
    pd.DataFrame(PLOG).to_csv(os.path.join(OUT, f"rps_{bn}_{s}_paths.csv"), index=False); print("DONE", bn, s)
