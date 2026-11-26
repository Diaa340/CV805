"""Storage-matched DER++ exception anchors on a separate pool (derpp_eappool_sm) on the six-task CLOVE-scene fusion net:
derpp_eappool_0.6 of cl_seq_eap.py (Brier penalty mu/2 mean (y - p_yes)^2, mu = 0.6, RB = 32 pool draws per step, own
generator) with the DER++ buffer and the exception pool each MEM // 2 = 500 cells, each split equally over past tasks
(total 1000, as derpp_poollogit_sm in cl_seq_poollogit.py). Imports cl_seq / cl_seq_eap unchanged: buffer = memory_na
prefix, pool = exc_pool prefix. Task-1 model, test cells and scratch are deterministic per run (merge asserts it).
Saves runs/clove_cl_eap/run_<r>_eapsm.csv.gz and run_<r>_eapsm_buffer.csv.
Usage: python cl_seq_eappool_sm.py <run index 0-11>"""
import copy, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import cl_seq as CS, cl_seq_eap as CSE
from cl_seq import Run, ORDERS, MEM, E2, dev
OUT = CSE.OUT; RB, MU = CSE.RB, CSE.MU; ARM = "derpp_eappool_sm"


def derpp_eappool_sm(R, net1, mb, blog):
    own, mem, pl = {}, [], []; net = copy.deepcopy(net1); g = torch.Generator(device=dev); g.manual_seed(R.seed * 100 + 50)

    def add_memory(tk, k):
        nonlocal mem, pl
        mem = [dict({q: v[:mb // k] for q, v in m.items() if q != "task"}, task=m["task"]) for m in mem] + [CSE.memory_na(R, tk, mb // k, net)]
        pl = [{q: v[:mb // k] for q, v in m.items() if q != "n_cells"} for m in pl] + [CSE.exc_pool(R, tk, mb // k)]
    add_memory(R.order[0], 1); own[R.order[0]] = R.predictor(net, "derpp", mem, None, R.order[0])
    for pos, tk in enumerate(R.order[1:], start=2):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        M = {q: torch.cat([m[q] for m in mem]) for q in ("x1", "x2", "y", "logit", "na")}
        X = {q: torch.cat([m[q] for m in pl]) for q in ("x1", "x2", "y")}
        blog.append(dict(arm=ARM, task=tk, pos=pos, buffer=len(M["y"]), buf_exc=int((M["na"] == 1).sum()), pool=len(X["y"]),
                         pool_yes=float(X["y"].float().mean())))
        net.train()
        for a, b, y in R.batches(tk, E2):
            loss = F.cross_entropy(net(a, b), y)
            j = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + F.cross_entropy(net(M["x1"][j], M["x2"][j]), M["y"][j])
            j2 = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + 0.5 * F.mse_loss(net(M["x1"][j2], M["x2"][j2]), M["logit"][j2])
            ja = torch.randint(0, len(X["y"]), (RB,), device=dev, generator=g); lo = net(X["x1"][ja], X["x2"][ja])
            loss = loss + MU / 2 * ((X["y"][ja].float() - F.softmax(lo, 1)[:, 1]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); add_memory(tk, pos); own[tk] = R.predictor(net, "derpp", mem, None, tk)
    return own, {tk: R.predictor(net, "derpp", mem, None, tk) for tk in R.order}


if __name__ == "__main__":
    r = int(sys.argv[1]); order, seed = ORDERS[r % 6], 31 + r // 6; R = Run(order, seed); t0 = time.time()
    net1, om1, fi1 = R.task1(); rows = {tk: R.tasks[tk]["test"][["unit", "nonadd", "y"]].assign(run=r, task=tk, pos=order.index(tk) + 1) for tk in order}
    for tk in order:
        rows[tk]["scratch"] = R.scratch(tk)
    blog = []; own, final = derpp_eappool_sm(R, net1, MEM // 2, blog)
    for tk in order:
        rows[tk][f"{ARM}__own"], rows[tk][f"{ARM}__final"] = own[tk], final[tk]
    print(r, ARM, f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(rows.values()).to_csv(os.path.join(OUT, f"run_{r}_eapsm.csv.gz"), index=False)
    pd.DataFrame(blog).assign(run=r).to_csv(os.path.join(OUT, f"run_{r}_eapsm_buffer.csv"), index=False); print("DONE", r)
