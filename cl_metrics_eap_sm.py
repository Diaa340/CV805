"""Storage-matched exception-pool arms on the two-task benchmarks (tables, CLEVR, GQA). Imports cl_metrics_eap unchanged.
derpp_eappool_0.6 and derpp_poollogit store DER++'s 1000-cell random buffer PLUS a pool of up to 1000 task-1 training
exception cells. The _sm arms keep total memory at 1000: pool = the first min(available, 500) cells of the same pool draw,
random buffer = the first 1000 - |pool| cells of DER++'s random-1000 buffer (both are random draws, so prefixes are random
subsets). Losses as in cl_metrics_eap.derpp_pool:
  derpp_eappool_sm   DER++ on the reduced buffer + (mu/2) Brier to the label on the pool, mu = 0.6
  derpp_poollogit_sm DER++ on the reduced buffer + 0.5 MSE to the task-1 logits on the pool
One process per shard runs finetune, derpp_random_1000, derpp_ruleexc_1000 (probe), derpp_eappool_0.6, derpp_poollogit,
the two _sm arms and scratch, so every arm shares the task-1 model (needed on GQA, whose CLIP features are recomputed
per process). Writes runs/clm_eap/{task1,task2}_<bench>_<shard>_sm.csv.gz and pool_<bench>_<shard>_sm.csv.
Usage: python cl_metrics_eap_sm.py <tables|clevr|gqa> <shard> <nshards>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_metrics_eap as CE
from cl_metrics_eap import CM, Net, buffers, task2, derpp_pool, exc_pool, margin, BASE, NPOOL, OUT, t, dev

TOTAL, PMAX = 1000, 500
PLOG = []


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    """cl_metrics_eap.run_unit (--full branch, logit arm) plus the storage-matched arms; same seeds and draws."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, Om = CM.train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy(), before=(net1(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); D2 = pd.DataFrame(dict(unit=name, nonadd=type2, y=te2[2].cpu().numpy()))
    pidx, ncell = exc_pool(tr1, nonadd_tr, NPOOL, seed); rb = B[("random", 1000)]
    k = min(len(pidx), PMAX); psm, bsm = pidx[:k], rb[:TOTAL - k]
    nets = {key: task2(net1, tr1, tr2, kind, 0.0, B[bk] if bk else None) for key, kind, bk in BASE}
    nets["derpp_eappool_0.6"] = derpp_pool(net1, tr1, tr2, rb, pidx, seed)
    nets["derpp_poollogit"] = derpp_pool(net1, tr1, tr2, rb, pidx, seed, 1.0, "logit")
    nets["derpp_eappool_sm"] = derpp_pool(net1, tr1, tr2, bsm, psm, seed)
    nets["derpp_poollogit_sm"] = derpp_pool(net1, tr1, tr2, bsm, psm, seed, 1.0, "logit")
    for key, net in nets.items():
        with torch.no_grad():
            D1[key] = (net(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy(); D2[key] = (net(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
    sn = CM.scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
    with torch.no_grad():
        D1["m_before"] = margin(net1, te1); D2["scratch"] = (sn(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy(); D2["m_scratch"] = margin(sn, te2)
    na = np.asarray(nonadd_tr)
    PLOG.append(dict(unit=name, exc_cells=ncell, pool=len(pidx), pool_sm=len(psm), buf_sm=len(bsm), derpp_buf_exc=int(na[rb].sum()),
                     buf_sm_exc=int(na[bsm].sum()), sec=round(time.time() - t0)))
    print(name, PLOG[-1], flush=True); return D1, D2


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns))
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}_sm.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}_sm.csv.gz"), index=False)
    pd.DataFrame(PLOG).to_csv(os.path.join(OUT, f"pool_{bn}_{s}_sm.csv"), index=False); print("DONE", bn, s)
