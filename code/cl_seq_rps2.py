"""RPSnet memory variants on the six-task CLOVE-scene fusion net (as cl_seq_rps.py; cl_seq / cl_seq_eap imported unchanged).
Memory 1000 split equally over past tasks; prefixes stay valid when a task's share shrinks.
  rps_ruleexc       Run.memory "ruleexc" (half exception cells, one per cell in turn, half random, interleaved)
  rps_herding       iCaRL herding per answer (as Run.memory "herding") on the RPS model's features (RPSNet.feat)
  rps_poollogit_sm  storage-matched: random memory 500 + exception pool 500 (cl_seq_eap.exc_pool), each split equally
                    over past tasks; the pool is held at the logits of the model right after its task (0.5 MSE)
rps (random) is in run_<r>_rps.csv.gz; task-1 model, test cells and scratch are deterministic per run (merge asserts it).
Saves runs/clove_cl_eap/run_<r>_rps2.csv.gz and run_<r>_rps2_paths.csv.
Usage: python cl_seq_rps2.py <run index 0-11>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_seq as CS, cl_seq_eap as CSE
from cl_seq import Run, ORDERS, MEM, POOL, E2, BS, t
import rps_core as RP
OUT = os.path.join(CS.ROOT, "runs", "clove_cl_eap")
ARMS = ("rps_ruleexc", "rps_herding", "rps_poollogit_sm")


def memory(R, tk, kind, m, net):
    if kind != "herding":
        return R.memory(tk, kind, m, net)
    x1, x2, y = R.tr[tk][:3]
    with torch.no_grad():
        h = CS.herd(net.feat(x1[:POOL], x2[:POOL]), y[:POOL], m // 2).cpu().numpy()
    idx = t(CS.interleave(h[:len(h) // 2], h[len(h) // 2:]))
    with torch.no_grad():
        lo = net(x1[idx], x2[idx])
    return dict(x1=x1[idx], x2=x2[idx], y=y[idx], logit=lo, task=tk)


def rps(R, net1, arm):
    kind = {"rps_ruleexc": "ruleexc", "rps_herding": "herding", "rps_poollogit_sm": "random"}[arm]
    mb = MEM // 2 if arm == "rps_poollogit_sm" else MEM
    net = RP.build(net1, R.seed); fixed = net.infer.copy(); own, mem, pl, plog = {}, [], [], []

    def add_memory(tk, k):
        nonlocal mem, pl
        mem = [dict({q: v[:mb // k] for q, v in m.items() if q != "task"}, task=m["task"]) for m in mem] + [memory(R, tk, kind, mb // k, net)]
        if arm == "rps_poollogit_sm":
            p = CSE.exc_pool(R, tk, mb // k)
            with torch.no_grad():
                p["logit"] = net(p["x1"], p["x2"])
            pl = [{q: v[:mb // k] for q, v in m.items() if q != "n_cells"} for m in pl] + [{q: v for q, v in p.items() if q != "n_cells"}]
    add_memory(R.order[0], 1); own[R.order[0]] = R.predictor(net, "rps", mem, None, R.order[0])
    for pos, tk in enumerate(R.order[1:], start=2):
        M = {q: torch.cat([m[q] for m in mem]) for q in ("x1", "x2", "y")}; x1, x2, y = R.tr[tk][:3]
        P = {q: torch.cat([m[q] for m in pl]) for q in ("x1", "x2", "logit")} if pl else None
        net, fixed, log = RP.session(net, fixed, pos - 1, x1, x2, y, M, E2 * (len(y) // BS), R.seed * 10 + pos, BS, pool=P)
        plog += [dict(arm=arm, task=tk, pos=pos, buffer=len(M["y"]), pool=0 if P is None else len(P["logit"]), **r) for r in log]
        add_memory(tk, pos); own[tk] = R.predictor(net, "rps", mem, None, tk)
    final = {tk: R.predictor(net, "rps", mem, None, tk) for tk in R.order}
    return own, final, plog


if __name__ == "__main__":
    r = int(sys.argv[1]); order, seed = ORDERS[r % 6], 31 + r // 6; R = Run(order, seed); t0 = time.time()
    net1, om1, fi1 = R.task1(); rows = {tk: R.tasks[tk]["test"][["unit", "nonadd", "y"]].assign(run=r, task=tk, pos=order.index(tk) + 1) for tk in order}
    for tk in order:
        rows[tk]["scratch"] = R.scratch(tk)
    logs = []
    for arm in ARMS:
        own, final, plog = rps(R, net1, arm); logs += plog
        for tk in order:
            rows[tk][f"{arm}__own"], rows[tk][f"{arm}__final"] = own[tk], final[tk]
        print(arm, "done", round(time.time() - t0), flush=True)
    pd.concat(rows.values()).to_csv(os.path.join(OUT, f"run_{r}_rps2.csv.gz"), index=False)
    pd.DataFrame(logs).assign(run=r).to_csv(os.path.join(OUT, f"run_{r}_rps2_paths.csv"), index=False)
    print("DONE", r, "sec", round(time.time() - t0), flush=True)
