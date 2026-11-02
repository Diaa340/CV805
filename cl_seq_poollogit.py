"""DER++ holding stored logits on a separate exception pool (derpp_poollogit, as cl_metrics_eap / vlm_seq_eap) on the
six-task CLOVE-scene fusion net. Imports cl_seq and cl_seq_eap unchanged: same data, task-1 model, DER++ buffer (random,
split equally over past tasks, 32 replayed + 32 logit-matched samples per step), 10 epochs per task.
  derpp_poollogit     DER++ on the 1000-cell buffer plus 0.5 MSE to stored logits on 32 cells per step from a separate
                      pool of past-task training exception cells (1000 distinct cells split equally over past tasks, as
                      derpp_eappool_0.6; extra storage). A task's pool logits are the model's right after that task, as
                      DER++ stores buffer logits.
  derpp_poollogit_sm  storage-matched: buffer 500 + pool 500, each split equally over past tasks (total 1000).
Pool draws use their own torch generator. Task-1 model, test cells and scratch are deterministic per run (the
make_results_fusion merge asserts it), so the DER++ baseline is the one in run_<r>.csv.gz.
Saves runs/clove_cl_eap/run_<r>_plog.csv.gz and run_<r>_plog_buffer.csv.
Usage: python cl_seq_poollogit.py <run index 0-11>"""
import copy, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import cl_seq as CS, cl_seq_eap as CSE
from cl_seq import Run, ORDERS, MEM, E2, dev
OUT = CSE.OUT; RB = CSE.RB
ARMS = {"derpp_poollogit": MEM, "derpp_poollogit_sm": MEM // 2}   # arm: size of buffer and of pool


def derpp_poollogit(R, net1, mb, arm, blog):
    own, mem, pl = {}, [], []; net = copy.deepcopy(net1); g = torch.Generator(device=dev); g.manual_seed(R.seed * 100 + 50)

    def add_memory(tk, k):
        nonlocal mem, pl
        mem = [dict({q: v[:mb // k] for q, v in m.items() if q != "task"}, task=m["task"]) for m in mem] + [CSE.memory_na(R, tk, mb // k, net)]
        p = CSE.exc_pool(R, tk, mb // k)
        with torch.no_grad():
            p["logit"] = net(p["x1"], p["x2"])
        pl = [{q: v[:mb // k] for q, v in m.items()} for m in pl] + [{q: v for q, v in p.items() if q != "n_cells"}]
    add_memory(R.order[0], 1); own[R.order[0]] = R.predictor(net, "derpp", mem, None, R.order[0])
    for pos, tk in enumerate(R.order[1:], start=2):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        M = {q: torch.cat([m[q] for m in mem]) for q in ("x1", "x2", "y", "logit", "na")}
        X = {q: torch.cat([m[q] for m in pl]) for q in ("x1", "x2", "y", "logit")}
        blog.append(dict(arm=arm, task=tk, pos=pos, buffer=len(M["y"]), buf_exc=int((M["na"] == 1).sum()), pool=len(X["y"])))
        net.train()
        for a, b, y in R.batches(tk, E2):
            loss = F.cross_entropy(net(a, b), y)
            j = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + F.cross_entropy(net(M["x1"][j], M["x2"][j]), M["y"][j])
            j2 = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + 0.5 * F.mse_loss(net(M["x1"][j2], M["x2"][j2]), M["logit"][j2])
            ja = torch.randint(0, len(X["y"]), (RB,), device=dev, generator=g); loss = loss + 0.5 * F.mse_loss(net(X["x1"][ja], X["x2"][ja]), X["logit"][ja])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); add_memory(tk, pos); own[tk] = R.predictor(net, "derpp", mem, None, tk)
    return own, {tk: R.predictor(net, "derpp", mem, None, tk) for tk in R.order}


if __name__ == "__main__":
    r = int(sys.argv[1]); order, seed = ORDERS[r % 6], 31 + r // 6; R = Run(order, seed); t0 = time.time()
    net1, om1, fi1 = R.task1(); rows = {tk: R.tasks[tk]["test"][["unit", "nonadd", "y"]].assign(run=r, task=tk, pos=order.index(tk) + 1) for tk in order}
    for tk in order:
        rows[tk]["scratch"] = R.scratch(tk)
    blog = []
    for arm, mb in ARMS.items():
        own, final = derpp_poollogit(R, net1, mb, arm, blog)
        for tk in order:
            rows[tk][f"{arm}__own"], rows[tk][f"{arm}__final"] = own[tk], final[tk]
        print(r, arm, f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(rows.values()).to_csv(os.path.join(OUT, f"run_{r}_plog.csv.gz"), index=False)
    pd.DataFrame(blog).assign(run=r).to_csv(os.path.join(OUT, f"run_{r}_plog_buffer.csv"), index=False); print("DONE", r)
