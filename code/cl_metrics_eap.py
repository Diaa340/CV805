"""DER++ with a separate 1000-cell exception-anchor pool (derpp_eappool_0.6, as cl_seq_eap.py) on the two-task
constructed benchmarks of cl_metrics.py (CLEVR, GQA). Imports cl_metrics unchanged and replaces only its per-unit loop:
same data, splits, task-1 model (train_task1_si, same draws), buffers (cl_exceptions.buffers), DER++ losses and 10
task-2 epochs.
  derpp_eappool_0.6  DER++ (random buffer 1000) plus (mu/2) mean (y - p_yes)^2, mu = 0.6, on 32 cells per step drawn from
                     a separate pool of up to 1000 distinct task-1 training exception cells (one draw per cell; extra
                     storage). Anchor draws use their own torch generator.
  baselines          finetune, derpp_random_1000, derpp_ruleexc_1000 (probe) through cl_exceptions.task2, rerun on the
                     rebuilt data so the new arm is paired with baselines from the same data; scratch for forward transfer.
Saves runs/clm_eap/task1_<bench>_<shard>.csv.gz / task2_... (columns as cl_metrics, plus y) and pool_<bench>_<shard>.csv.
Usage: python cl_metrics_eap.py <tables|clevr|gqa> <shard> <n_shards> [--mus 2,5,10,20]
  --mus: only derpp_eappool_<mu> for each listed mu (no baselines, no scratch) -> task1_<bench>_<shard>_mu.csv.gz etc.
  --arms: only the listed pool arms (no baselines, no scratch) -> task1_<bench>_<shard>_arm.csv.gz etc.:
     ce<mu>  derpp_eappoolce_<mu>: (mu/2) CE(logits, y) on the pool instead of the Brier term (does not saturate)
     logit   derpp_poollogit: DER++'s logit term on the pool only, 0.5 MSE(f(x), f_task1(x)) (no label term)
     replay  derpp_poolreplay: the pool replayed like the DER++ buffer, CE + 0.5 MSE on logits (probe-style replay, extra storage)
  --full (with --arms): baselines, derpp_eappool_0.6, the listed arms and scratch in one process -> ..._full.csv.gz.
     Needed on GQA: bench_gqa recomputes CLIP features on the GPU in every run and they are not bit-reproducible, so
     separate runs train slightly different task-1 models and cannot be paired cell by cell (CLEVR loads saved features).
  --margins: task-1 model and scratch only -> ..._marg.csv.gz with m_before / m_scratch (margin = logit of the true answer
     minus the other, task-1 model on task-1 test cells / task-2-only model on task-2 test cells) for difficulty matching.
     Valid on CLEVR and tables (saved features, so the same task-1 model as the other runs; merged with a check on before).
  Every run also writes m_before, and m_scratch where scratch is trained."""
import copy, os, sys, time, types
import numpy as np, pandas as pd, torch, torch.nn.functional as F


def _unused(*a, **k):
    raise RuntimeError("tables-benchmark helper called; not available in this rebuild")


# kinds_kill loads CIFAR-100 features at import and tables_step1 imports it; both serve only the synthetic-tables benchmark,
# which this script does not run. Stand-ins (raising if called) let cl_metrics import without those features; once
# data/feats/cifar2v.npz exists the real modules load and the tables benchmark runs too.
for _m, _names in (("kinds_kill", ("balanced_table", "F1", "F2", "ITEMS", "linear_check", "perm")), ("tables_step1", ("POOL2", "table_pairs"))):
    if _m not in sys.modules and not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feats", "cifar2v.npz")):
        sys.modules[_m] = types.SimpleNamespace(**{n: _unused for n in _names})
import cl_metrics as CM
from cl_exceptions import buffers, task2, E2, BS
from route_exps import Net
dev, t = CM.dev, CM.t; OUT = os.path.join(CM.ROOT, "runs", "clm_eap"); os.makedirs(OUT, exist_ok=True)
MU, RB, NPOOL = 0.6, 32, 1000
BASE = [("finetune", "finetune", None), ("derpp_random_1000", "derpp", ("random", 1000)), ("derpp_ruleexc_1000", "derpp", ("ruleexc", 1000))]
PLOG = []; MUS = [float(x) for x in sys.argv[sys.argv.index("--mus") + 1].split(",")] if "--mus" in sys.argv else None
FULL = "--full" in sys.argv; MARG = "--margins" in sys.argv; ARMS = sys.argv[sys.argv.index("--arms") + 1].split(",") if "--arms" in sys.argv else None
ARM = {**{a: (f"derpp_eappoolce_{a[2:]}", "ce", float(a[2:])) for a in (ARMS or []) if a.startswith("ce")},
       "logit": ("derpp_poollogit", "logit", 1.0), "replay": ("derpp_poolreplay", "replay", 1.0)}


def exc_pool(tr1, nonadd_tr, m, seed):
    """Up to m distinct task-1 training exception cells (one draw per cell, over all training draws), own generator."""
    npy = lambda v: v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v)   # tables passes cell indices as tensors
    cell = npy(tr1[3]).astype(np.int64) * 100000 + npy(tr1[4]); _, first = np.unique(cell, return_index=True)
    ex = first[npy(nonadd_tr)[first] == 1]; rng = np.random.default_rng(seed * 100 + 50)
    return rng.choice(ex, min(m, len(ex)), replace=False), len(ex)


def derpp_pool(net1, tr1, tr2, buf, pidx, seed, mu=MU, mode="brier"):
    """cl_exceptions.task2(kind="derpp") plus a term on the pool: brier / ce = (mu/2) Brier or CE against the label;
    logit = 0.5 MSE to the task-1 logits; replay = CE + 0.5 MSE as on the DER++ buffer (separate draws of 32)."""
    net = copy.deepcopy(net1); old = net1; opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); n = len(tr2[2])
    bx1, bx2, by = tr1[0][t(buf)], tr1[1][t(buf)], tr1[2][t(buf)]; ax1, ax2, ay = tr1[0][t(pidx)], tr1[1][t(pidx)], tr1[2][t(pidx)]
    with torch.no_grad():
        blog, alog = old(bx1, bx2), old(ax1, ax2)
    g = torch.Generator(device=dev); g.manual_seed(seed * 100 + 50)
    for _ in range(E2 * (n // BS)):
        i = torch.randint(0, n, (BS,), device=dev); loss = F.cross_entropy(net(tr2[0][i], tr2[1][i]), tr2[2][i])
        j = torch.randint(0, len(by), (32,), device=dev); loss = loss + F.cross_entropy(net(bx1[j], bx2[j]), by[j])
        j2 = torch.randint(0, len(by), (32,), device=dev); loss = loss + 0.5 * F.mse_loss(net(bx1[j2], bx2[j2]), blog[j2])
        if mu > 0 and len(ay):
            ja = torch.randint(0, len(ay), (RB,), device=dev, generator=g); lo = net(ax1[ja], ax2[ja])
            if mode == "brier":
                loss = loss + mu / 2 * ((ay[ja].float() - F.softmax(lo, 1)[:, 1]) ** 2).mean()
            elif mode == "ce":
                loss = loss + mu / 2 * F.cross_entropy(lo, ay[ja])
            elif mode == "logit":
                loss = loss + 0.5 * F.mse_loss(lo, alog[ja])
            elif mode == "replay":
                ja2 = torch.randint(0, len(ay), (RB,), device=dev, generator=g)
                loss = loss + F.cross_entropy(lo, ay[ja]) + 0.5 * F.mse_loss(net(ax1[ja2], ax2[ja2]), alog[ja2])
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def margin(net, te):
    """Logit of the true answer minus the other answer, per test cell."""
    lo = net(te[0], te[1]); y = te[2]
    return (lo.gather(1, y[:, None]) - lo.gather(1, 1 - y[:, None]))[:, 0].float().cpu().numpy()


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    """cl_metrics.run_unit with the method list replaced (same seeds, same task-1 model and buffers)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, Om = CM.train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, y=te1[2].cpu().numpy(), before=(net1(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); D2 = pd.DataFrame(dict(unit=name, nonadd=type2, y=te2[2].cpu().numpy()))
    pidx, ncell = exc_pool(tr1, nonadd_tr, NPOOL, seed)
    if MARG:
        nets = {}
    elif FULL:
        nets = {key: task2(net1, tr1, tr2, kind, 0.0, B[bk] if bk else None) for key, kind, bk in BASE}
        nets["derpp_eappool_0.6"] = derpp_pool(net1, tr1, tr2, B[("random", 1000)], pidx, seed)
        nets |= {ARM[a][0]: derpp_pool(net1, tr1, tr2, B[("random", 1000)], pidx, seed, ARM[a][2], ARM[a][1]) for a in ARMS}
    elif ARMS:
        nets = {ARM[a][0]: derpp_pool(net1, tr1, tr2, B[("random", 1000)], pidx, seed, ARM[a][2], ARM[a][1]) for a in ARMS}
    elif MUS:
        nets = {f"derpp_eappool_{mu:g}": derpp_pool(net1, tr1, tr2, B[("random", 1000)], pidx, seed, mu) for mu in MUS}
    else:
        nets = {key: task2(net1, tr1, tr2, kind, 0.0, B[bk] if bk else None) for key, kind, bk in BASE}
        nets["derpp_eappool_0.6"] = derpp_pool(net1, tr1, tr2, B[("random", 1000)], pidx, seed)
    for key, net in nets.items():
        with torch.no_grad():
            D1[key] = (net(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy(); D2[key] = (net(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
    with torch.no_grad():
        D1["m_before"] = margin(net1, te1)
    if MARG or FULL or (not MUS and not ARMS):
        sn = CM.scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
        with torch.no_grad():
            D2["scratch"] = (sn(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy(); D2["m_scratch"] = margin(sn, te2)
    na = np.asarray(nonadd_tr); rb, pb = B[("random", 1000)], B[("ruleexc", 1000)]; y1 = tr1[2].cpu().numpy()
    with torch.no_grad():
        pc = float((net1(tr1[0][t(pidx)], tr1[1][t(pidx)]).argmax(1) == tr1[2][t(pidx)]).float().mean()) if len(pidx) else np.nan
    PLOG.append(dict(unit=name, exc_cells=ncell, pool=len(pidx), pool_yes=float(y1[pidx].mean()) if len(pidx) else np.nan, pool_t1_correct=pc,
                     derpp_buf_exc=int(na[rb].sum()), probe_buf_exc=int(na[pb].sum()), sec=round(time.time() - t0)))
    print(name, PLOG[-1], flush=True); return D1, D2


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); CM.run_unit = run_unit
    res = {"tables": CM.bench_tables, "clevr": CM.bench_clevr, "gqa": CM.bench_gqa}[bn]((s, ns)); sfx = "_marg" if MARG else "_full" if FULL else "_arm" if ARMS else "_mu" if MUS else ""
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}{sfx}.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}{sfx}.csv.gz"), index=False)
    pd.DataFrame(PLOG).to_csv(os.path.join(OUT, f"pool_{bn}_{s}{sfx}.csv"), index=False); print("DONE", bn, s)
