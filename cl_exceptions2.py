"""F1 confirmatory run (paper/preregistration_f1f2.md): new seed 24, new CLEVR units; finetune, ER random, DER++ with
random / herding / forgetting-events / rule+exceptions buffers at 250 and 1,000. Derived from:
CL limitations: additive 'rule' vs XOR 'exceptions' (paper/preregistration_clx.md). Per unit: task 1 (30 epochs,
per-epoch evaluation of a 10,000-sample candidate pool for forgetting events), then from the same checkpoint 30
task-2 runs (10 epochs): finetune; EWC (lambda 1, 100, 10000); LwF (0.3, 3); feature distillation L2 (0.3, 3);
ER with buffers {random, herding, low-confidence, forgetting-events, exception-cell, rule+exceptions} x {50, 250, 1000};
DER++ with {random, rule+exceptions} x {250, 1000}. Per task-1 test sample: cell, structure label, confidence,
correct before / after each method (and, for tables, whether each buffer covers the sample's cell).
Benchmarks: tables (constructed tables with d_add > 0, two-view CIFAR) and clevr (40 units; symbolic image + CLIP
question features). Usage: python cl_exceptions.py <tables|clevr> <shard> <n_shards>"""
import copy, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from kinds_kill import balanced_table
from route_exps import Net
from tables_step1 import POOL2, table_pairs
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clx2"); dev = "cuda"
t = lambda a: torch.tensor(np.asarray(a), device=dev); BS, E1, E2, POOL = 256, 30, 10, 10000


def train_task1(net, tr):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); n = len(tr[2]); steps = n // BS
    prev = None; fev = torch.zeros(POOL, device=dev)
    for ep in range(E1):
        for _ in range(steps):
            i = torch.randint(0, n, (BS,), device=dev); opt.zero_grad(); F.cross_entropy(net(tr[0][i], tr[1][i]), tr[2][i]).backward(); opt.step()
        with torch.no_grad():
            ok = net(tr[0][:POOL], tr[1][:POOL]).argmax(1) == tr[2][:POOL]
        if prev is not None:
            fev += (prev & ~ok).float()
        prev = ok
    return fev.cpu().numpy()


def buffers(net, tr, fev, nonadd_tr, rng):
    with torch.no_grad():
        lo = net(tr[0][:POOL], tr[1][:POOL]); conf = (lo[:, 1] - lo[:, 0]).abs().cpu().numpy()
        z = F.normalize(torch.cat([net.e1(tr[0][:POOL]), net.e2(tr[1][:POOL])], 1), dim=1)
    y = tr[2][:POOL].cpu().numpy(); out = {}
    for m in (50, 250, 1000):
        out[("random", m)] = rng.choice(POOL, m, replace=False)
        h = []
        for c in (0, 1):  # iCaRL herding per label
            idx = np.where(y == c)[0]; zc = z[idx]; mu = zc.mean(0); s = torch.zeros_like(mu); chosen = []
            for k in range(min(m // 2, len(idx))):
                d = ((mu - (s + zc) / (k + 1)) ** 2).sum(1); d[chosen] = 1e9; j = int(d.argmin()); chosen.append(j); s += zc[j]
            h += list(idx[chosen])
        out[("herding", m)] = np.array(h)
        out[("lowconf", m)] = np.argsort(conf)[:m]
        out[("forget", m)] = np.lexsort((conf, -fev))[:m]
        ex = np.where(nonadd_tr[:POOL] == 1)[0]; cells = {}
        for j in rng.permutation(ex):
            cells.setdefault(int(tr[3][j]) * 100000 + int(tr[4][j]), []).append(j)
        pick, lists = [], [v for v in cells.values()]
        while len(pick) < m and any(lists):
            for v in lists:
                if v and len(pick) < m:
                    pick.append(v.pop())
        rest = np.setdiff1d(np.arange(POOL), pick); pick += list(rng.choice(rest, m - len(pick), replace=False))
        out[("exception", m)] = np.array(pick)
        half = np.array(pick[: m // 2]); rest = np.setdiff1d(np.arange(POOL), half)   # rule + exceptions: half exception-cell, half random
        out[("ruleexc", m)] = np.concatenate([half, rng.choice(rest, m - len(half), replace=False)])
    return out


def task2(net1, tr1, tr2, kind, lam=0.0, buf=None, fisher=None):
    net = copy.deepcopy(net1); old = net1; opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); n = len(tr2[2])
    theta0 = {k: p.detach().clone() for k, p in old.named_parameters()}
    if buf is not None:
        bx1, bx2, by = tr1[0][t(buf)], tr1[1][t(buf)], tr1[2][t(buf)]
        with torch.no_grad():
            blog = old(bx1, bx2)
    for _ in range(E2 * (n // BS)):
        i = torch.randint(0, n, (BS,), device=dev); a, b = tr2[0][i], tr2[1][i]; out = net(a, b); loss = F.cross_entropy(out, tr2[2][i])
        if kind == "lwf":
            with torch.no_grad():
                lo = old(a, b)
            loss = loss + lam * F.kl_div(F.log_softmax(out / 2, 1), F.log_softmax(lo / 2, 1), log_target=True, reduction="batchmean") * 4
        elif kind == "l2":
            with torch.no_grad():
                o1, o2 = old.e1(a), old.e2(b)
            loss = loss + lam * ((net.e1(a) - o1).pow(2).sum(1).mean() + (net.e2(b) - o2).pow(2).sum(1).mean()) / o1.shape[1]
        elif kind == "ewc":
            loss = loss + lam * sum((fisher[k] * (p - theta0[k]) ** 2).sum() for k, p in net.named_parameters())
        elif kind in ("er", "derpp"):
            j = torch.randint(0, len(by), (32,), device=dev); loss = loss + F.cross_entropy(net(bx1[j], bx2[j]), by[j])
            if kind == "derpp":
                j2 = torch.randint(0, len(by), (32,), device=dev); loss = loss + 0.5 * F.mse_loss(net(bx1[j2], bx2[j2]), blog[j2])
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def fisher_diag(net, tr, n=2000):
    net = copy.deepcopy(net); Fi = {k: torch.zeros_like(p) for k, p in net.named_parameters()}
    for p in net.parameters():
        p.requires_grad_(True)
    for s in range(0, n, 32):
        net.zero_grad(); F.cross_entropy(net(tr[0][s:s + 32], tr[1][s:s + 32]), tr[2][s:s + 32]).backward()
        for k, p in net.named_parameters():
            Fi[k] += p.grad.detach() ** 2 * 32 / n
    return Fi


def run_unit(name, tr1, te, te_meta, tr2, te2, nonadd_tr, seed, cover_fn=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev = train_task1(net1, tr1); net1.eval()
    with torch.no_grad():
        lo = net1(te[0], te[1]); before = (lo.argmax(1) == te[2]).cpu().numpy(); conf = (lo[:, 1] - lo[:, 0]).abs().cpu().numpy()
    B = buffers(net1, tr1, fev, nonadd_tr, rng)
    runs = [("finetune", "finetune", 0.0, None)] + [(f"er_random_{m}", "er", 0.0, ("random", m)) for m in (250, 1000)] \
        + [(f"derpp_{s}_{m}", "derpp", 0.0, (s, m)) for s in ("random", "herding", "forget", "ruleexc") for m in (250, 1000)]
    Fi = None
    D = pd.DataFrame(dict(unit=name, **te_meta, y=te[2].cpu().numpy(), conf=conf, before=before)); summ = []
    for key, kind, lam, bk in runs:
        net = task2(net1, tr1, tr2, kind, lam, B[bk] if bk else None, Fi)
        with torch.no_grad():
            D[f"after__{key}"] = (net(te[0], te[1]).argmax(1) == te[2]).cpu().numpy()
            summ.append(dict(unit=name, method=key, task2_acc=(net(te2[0], te2[1]).argmax(1) == te2[2]).float().mean().item()))
    if cover_fn is not None:
        for (s, m), idx in B.items():
            D[f"cover__{s}_{m}"] = cover_fn(idx)
    print(name, f"({time.time() - t0:.0f}s)", flush=True)
    return D, pd.DataFrame(summ)


def bench_tables(shard):
    out, summ = [], []
    for setname, Ff in (("selected", "orig_F.npz"), ("fresh", "fresh.npz")):
        S = np.load(os.path.join(ROOT, "runs", "tables", f"{setname}.npz")); Fs = np.load(os.path.join(ROOT, "runs", "tables", Ff))["F"]
        for k in range(len(S["T"])):
            if S["d_add"][k] == 0 or (k + (0 if setname == "selected" else 1000)) % shard[1] != shard[0]:
                continue
            T, Fn = S["T"][k].astype(int), Fs[k].astype(int); rng = np.random.default_rng(24)
            tr1, te = table_pairs(T, 50000, 0, rng), table_pairs(T, 5000, 1, rng)
            T2 = balanced_table(np.random.default_rng(1024)); r2 = np.random.default_rng(2024); tr2, te2 = table_pairs(T2, 50000, 0, r2, POOL2), table_pairs(T2, 5000, 1, r2, POOL2)
            ci, cj = te[3].cpu().numpy(), te[4].cpu().numpy(); NA = (T != Fn).astype(int); ri, rj = tr1[3].cpu().numpy(), tr1[4].cpu().numpy()
            cover = lambda idx: np.isin(ci * 10 + cj, np.unique(ri[idx] * 10 + rj[idx])).astype(int)
            D, Sm = run_unit(f"{setname}_{k}", tr1, te[:3], dict(r=ci, c=cj, nonadd=NA[ci, cj]), tr2[:3], te2[:3], NA[ri, rj], 24, cover)
            out.append(D); summ.append(Sm)
    return out, summ


def bench_clevr(shard):
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep")
    allU = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); TS = np.load(os.path.join(OUT, "train_structure.npz")); ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz"))
    itr, ite = TS["train_imgs"], ST["clevr_test_imgs"]; Xi = t(Z["Ximg"]).float(); out, summ = [], []
    for n, u in enumerate(TS["units"]):
        if n % shard[1] != shard[0]:
            continue
        k = allU.index(u); p = next(allU[(k + s) % len(allU)] for s in range(len(allU) // 2, len(allU)) if fam[allU[(k + s) % len(allU)]] != fam[u])
        rng = np.random.default_rng(24)
        def cells(unit, imgs, nsamp):
            T = Z[f"T__{unit}"]; Xq = t(Z[f"Cq__{unit}"]).float(); q, ii = rng.integers(0, 10, nsamp), rng.integers(0, len(imgs), nsamp)
            return Xi[t(imgs[ii])], Xq[t(q)], t(T[q, imgs[ii]]).long(), t(q), t(ii)
        tr1, tr2 = cells(u, itr, 50000), cells(p, itr, 50000)
        def full(unit, imgs):
            T = Z[f"T__{unit}"]; Xq = t(Z[f"Cq__{unit}"]).float(); q, ii = np.meshgrid(np.arange(10), np.arange(len(imgs)), indexing="ij"); q, ii = q.ravel(), ii.ravel()
            return (Xi[t(imgs[ii])], Xq[t(q)], t(T[q, imgs[ii]]).long()), q, ii
        te, q, ii = full(u, ite); te2, _, _ = full(p, ite)
        NAtr = TS[f"nonadd__{u}"]; NAte = ST[f"nonadd__clevr__{u}"]
        D, Sm = run_unit(u, tr1, te, dict(r=q, c=ii, nonadd=NAte[q, ii]), tr2[:3], te2, NAtr[tr1[3].cpu().numpy(), tr1[4].cpu().numpy()], 24)
        out.append(D); summ.append(Sm)
    return out, summ


if __name__ == "__main__":
    b, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    out, summ = (bench_tables if b == "tables" else bench_clevr)((s, ns))
    pd.concat(out).to_csv(os.path.join(OUT, f"samples_{b}_{s}.csv.gz"), index=False); pd.concat(summ).to_csv(os.path.join(OUT, f"summary_{b}_{s}.csv"), index=False)
    print("DONE", b, s)
