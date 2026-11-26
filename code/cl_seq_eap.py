"""DER++ with exception anchors (derpp_eap_0.6, as in vlm_seq_eap.py) on the six-task CLOVE-scene fusion net (CLIP
features, cl_seq.py). Imports cl_seq unchanged: same data, Net, task-1 model, 50,000 training draws per task, memory 1000
split equally over past tasks (random), 32 replayed + 32 logit-matched samples per step, 10 epochs per task.
  derpp_eap_0.6   DER++ plus (mu/2) mean (y - p_yes)^2, mu = 0.6, on 32 cells per step drawn from the exception cells
                  inside the DER++ buffer (no extra storage). Anchor draws use their own torch generator.
  derpp_eapce_0.6 the same with the binary cross-entropy (mu/2) CE(logits, y) in place of the squared difference, which
                  does not saturate on confidently wrong exceptions (eapce_0.6 of experiment_guide_exception_anchor.md)
  eap_0.6         finetune plus the same Brier penalty on 32 cells per step from a separate pool of past-task training
                  exceptions: 1000 distinct exception cells split equally over the past tasks (as vlm_seq_eap.exc_pool)
  derpp_eappool_0.6  DER++ plus the penalty on that separate pool (extra storage): tests whether the ~60 exception cells
                  inside the DER++ buffer are too few
  baselines       finetune, derpp, derpp_ruleexc through cl_seq.Run.method, rerun on the rebuilt CLIP features so the new
                  arm is paired with baselines from the same features.
The buffer is the one cl_seq builds for derpp (same rng, same indices); only the exception flag is carried along.
Saves runs/clove_cl_eap/run_<r>.csv.gz (columns as cl_seq) and run_<r>_buffer.csv (exception and yes share of the buffer).
Usage: python cl_seq_eap.py <run index 0-11> [--ce | --pool]   (--ce: only derpp_eapce_0.6 -> run_<r>_ce.csv.gz;
       --pool: only eap_0.6 and derpp_eappool_0.6 -> run_<r>_pool.csv.gz)"""
import copy, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import cl_seq as CS
from cl_seq import Run, ORDERS, MEM, POOL, E2, dev, t
OUT = os.path.join(CS.ROOT, "runs", "clove_cl_eap"); os.makedirs(OUT, exist_ok=True)
MU, RB = 0.6, 32


def memory_na(R, tk, m, net):
    """cl_seq's random memory for task tk, plus the exception flag of each stored sample."""
    idx = np.random.default_rng(R.seed * 100 + R.order.index(tk)).choice(POOL, m, replace=False)   # what Run.memory draws
    out = R.memory(tk, "random", m, net); assert torch.equal(out["y"], R.tr[tk][2][t(idx)])
    out["na"] = t(R.tr[tk][3][idx]); return out


def exc_pool(R, tk, m):
    """Up to m distinct training exception cells of task tk (one draw per cell), own generator."""
    x1, x2, y, na, cell = R.tr[tk]; rng = np.random.default_rng(R.seed * 100 + 50 + R.order.index(tk))
    _, first = np.unique(cell, return_index=True); ex = first[na[first] == 1]; idx = t(rng.choice(ex, min(m, len(ex)), replace=False))
    return dict(x1=x1[idx], x2=x2[idx], y=y[idx], n_cells=len(ex))


def derpp_eap(R, net1, mu, blog, ce=False, replay=True, pool=False):
    """replay: DER++ buffer losses (else finetune); pool: anchors from a separate exception pool (else from the buffer)."""
    own, mem, pl = {}, [], []; net = copy.deepcopy(net1); g = torch.Generator(device=dev); g.manual_seed(R.seed * 100 + 50)
    def add_memory(tk, k):
        nonlocal mem, pl
        mem = [dict({q: v[:MEM // k] for q, v in m.items() if q != "task"}, task=m["task"]) for m in mem] + [memory_na(R, tk, MEM // k, net)]
        if pool:
            pl = [{q: v[:MEM // k] for q, v in m.items() if q != "n_cells"} for m in pl] + [exc_pool(R, tk, MEM // k)]
    add_memory(R.order[0], 1); own[R.order[0]] = R.predictor(net, "derpp", mem, None, R.order[0])
    for pos, tk in enumerate(R.order[1:], start=2):
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        M = {q: torch.cat([m[q] for m in mem]) for q in ("x1", "x2", "y", "logit", "na")}; E = M["na"] == 1
        X = {q: torch.cat([m[q] for m in pl]) for q in ("x1", "x2", "y")} if pool else {q: M[q][E] for q in ("x1", "x2", "y")}
        blog.append(dict(arm=("derpp_" if replay else "") + ("eappool" if pool else "eap"), task=tk, pos=pos, buffer=len(M["y"]), exc=int(E.sum()), anchors=len(X["y"]),
                         anchor_yes=float(X["y"].float().mean()) if len(X["y"]) else np.nan,
                         buf_yes=float(M["y"].float().mean())))
        net.train()
        for a, b, y in R.batches(tk, E2):
            loss = F.cross_entropy(net(a, b), y)
            if replay:
                j = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + F.cross_entropy(net(M["x1"][j], M["x2"][j]), M["y"][j])
                j2 = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + 0.5 * F.mse_loss(net(M["x1"][j2], M["x2"][j2]), M["logit"][j2])
            if mu > 0 and len(X["y"]):
                ja = torch.randint(0, len(X["y"]), (RB,), device=dev, generator=g); lo = net(X["x1"][ja], X["x2"][ja])
                loss = loss + mu / 2 * (F.cross_entropy(lo, X["y"][ja]) if ce else ((X["y"][ja].float() - F.softmax(lo, 1)[:, 1]) ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); add_memory(tk, pos); own[tk] = R.predictor(net, "derpp", mem, None, tk)
    return own, {tk: R.predictor(net, "derpp", mem, None, tk) for tk in R.order}


if __name__ == "__main__":
    r = int(sys.argv[1]); order, seed = ORDERS[r % 6], 31 + r // 6; R = Run(order, seed); t0 = time.time()
    net1, om1, fi1 = R.task1(); rows = {tk: R.tasks[tk]["test"][["unit", "nonadd", "y"]].assign(run=r, task=tk, pos=order.index(tk) + 1) for tk in order}
    for tk in order:
        rows[tk]["scratch"] = R.scratch(tk)
    blog = []
    names = ("derpp_eapce_0.6",) if "--ce" in sys.argv else ("eap_0.6", "derpp_eappool_0.6") if "--pool" in sys.argv else ("finetune", "derpp", "derpp_ruleexc", "derpp_eap_0.6")
    for name in names:
        own, final = derpp_eap(R, net1, MU, blog, ce="eapce" in name, replay=name.startswith("derpp"), pool=name in ("eap_0.6", "derpp_eappool_0.6")) \
            if "eap" in name else R.method(net1, om1, fi1, name, 0.0)
        for tk in order:
            rows[tk][f"{name}__own"], rows[tk][f"{name}__final"] = own[tk], final[tk]
        print(r, name, f"{time.time() - t0:.0f}s", flush=True)
    sfx = "_ce" if "--ce" in sys.argv else "_pool" if "--pool" in sys.argv else ""
    pd.concat(rows.values()).to_csv(os.path.join(OUT, f"run_{r}{sfx}.csv.gz"), index=False)
    pd.DataFrame(blog).assign(run=r).to_csv(os.path.join(OUT, f"run_{r}{sfx}_buffer.csv"), index=False); print("DONE", r)
