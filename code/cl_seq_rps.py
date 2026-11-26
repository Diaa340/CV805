"""RPSnet (rps_core.py) on the six-task CLOVE-scene fusion net. Imports cl_seq unchanged: same data, task order and
seed per run, shared task-1 model (Run.task1), 50,000 training draws per task, 10 epochs per task, memory 1000 split
equally over past tasks (Run.memory "random", as ER), evaluation through Run.predictor (own = right after its task,
final = after task 6), scratch per task. RPS sessions 1..5 with jump 1: a new random path per task, controller
cx = 1 at task 2, then 2.5 * (task index - 2).
Saves runs/clove_cl_eap/run_<r>_rps.csv.gz (columns as cl_seq_eap: unit, nonadd, y, run, task, pos, scratch,
rps__own, rps__final) and run_<r>_rps_paths.csv.
Usage: python cl_seq_rps.py <run index 0-11>"""
import os, sys, time
import numpy as np, pandas as pd, torch
import cl_seq as CS
from cl_seq import Run, ORDERS, MEM, E2, BS
import rps_core as RP
OUT = os.path.join(CS.ROOT, "runs", "clove_cl_eap")


def rps(R, net1):
    net = RP.build(net1, R.seed); fixed = net.infer.copy(); own, mem, plog = {}, [], []

    def add_memory(tk, k):   # as cl_seq.Run.method for ER: random, split equally over the tasks seen so far
        nonlocal mem
        mem = [dict({q: v[:MEM // k] for q, v in m.items() if q != "task"}, task=m["task"]) for m in mem] + [R.memory(tk, "random", MEM // k, net)]
    add_memory(R.order[0], 1); own[R.order[0]] = R.predictor(net, "rps", mem, None, R.order[0])
    for pos, tk in enumerate(R.order[1:], start=2):
        M = {q: torch.cat([m[q] for m in mem]) for q in ("x1", "x2", "y")}; x1, x2, y = R.tr[tk][:3]
        net, fixed, log = RP.session(net, fixed, pos - 1, x1, x2, y, M, E2 * (len(y) // BS), R.seed * 10 + pos, BS)
        plog += [dict(task=tk, pos=pos, **r) for r in log]; add_memory(tk, pos); own[tk] = R.predictor(net, "rps", mem, None, tk)
    final = {tk: R.predictor(net, "rps", mem, None, tk) for tk in R.order}
    return own, final, plog, int(fixed.sum())


if __name__ == "__main__":
    r = int(sys.argv[1]); order, seed = ORDERS[r % 6], 31 + r // 6; R = Run(order, seed); t0 = time.time()
    net1, om1, fi1 = R.task1(); rows = {tk: R.tasks[tk]["test"][["unit", "nonadd", "y"]].assign(run=r, task=tk, pos=order.index(tk) + 1) for tk in order}
    for tk in order:
        rows[tk]["scratch"] = R.scratch(tk)
    own, final, plog, nmod = rps(R, net1)
    for tk in order:
        rows[tk]["rps__own"], rows[tk]["rps__final"] = own[tk], final[tk]
    pd.concat(rows.values()).to_csv(os.path.join(OUT, f"run_{r}_rps.csv.gz"), index=False)
    pd.DataFrame(plog).assign(run=r).to_csv(os.path.join(OUT, f"run_{r}_rps_paths.csv"), index=False)
    print("DONE", r, "modules used", nmod, "sec", round(time.time() - t0), flush=True)
