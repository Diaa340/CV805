"""CL metrics by sample type on four benchmarks (paper/preregistration_clm.md). Per unit: task 1 (30 epochs) once, with
SI path-integral importance tracked; from that checkpoint task 2 (10 epochs) under
  regularisation: EWC (1, 100, 10000), SI (0.1, 1, 10, 100), LwF (0.3, 3), feature distillation (0.3, 3);
  memory: ER, A-GEM, GEM, DER++ (random buffer 1000), iCaRL (herding buffer 1000, distillation 0.1 / 1,
          nearest-mean classifier), DER++ rule+exceptions-1000;
  architectural: DER-expansion (Yan et al. 2021: frozen old encoders + new encoders, joint head, random buffer 1000),
          multi-head (shared encoders, one head per task, task identity given at test);
plus finetune and a task-2-only model (10 epochs from scratch) for forward transfer. Saves per-sample correctness on task-1 test samples (before / after each method) and on task-2 test
samples (after each method, and scratch), each with its cell type (rule / exception).
Usage: python cl_metrics.py <tables|clevr|gqa|pope> <shard> <n_shards>"""
import copy, os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from cl_exceptions import buffers, fisher_diag, task2, train_task1, E1, E2, BS
from kinds_kill import balanced_table
from route_exps import Net
from step1_build import nearest_ferrers
from tables_step1 import POOL2, table_pairs
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clm"); dev = "cuda"; t = lambda a: torch.tensor(np.asarray(a), device=dev)
ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz")); SM = None  # loaded by the GQA / POPE benchmarks
RUNS = [("finetune", "finetune", 0.0, None)] + [(f"ewc_{l:g}", "ewc", l, None) for l in (1, 100, 10000)] + [(f"lwf_{l:g}", "lwf", l, None) for l in (0.3, 3)] \
    + [(f"l2_{l:g}", "l2", l, None) for l in (0.3, 3)] + [("er_random_1000", "er", 0.0, ("random", 1000)), ("derpp_random_1000", "derpp", 0.0, ("random", 1000)),
                                                         ("derpp_ruleexc_1000", "derpp", 0.0, ("ruleexc", 1000))] \
    + [(f"si_{l:g}", "si", l, None) for l in (0.1, 1, 10, 100)] + [("agem_random_1000", "agem", 0.0, ("random", 1000)), ("gem_random_1000", "gem", 0.0, ("random", 1000)),
       ("icarl_0.1", "icarl", 0.1, ("herding", 1000)), ("icarl_1", "icarl", 1.0, ("herding", 1000)), ("der_expand_1000", "expand", 0.0, ("random", 1000)), ("multihead", "multihead", 0.0, None)]
XI = 0.1  # SI damping


def train_task1_si(net, tr):
    """Identical to cl_exceptions.train_task1 (same updates, same random draws) plus SI importance (Zenke et al. 2017)."""
    from cl_exceptions import POOL
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); n = len(tr[2]); steps = n // BS
    prev = None; fev = torch.zeros(POOL, device=dev); start = {k: p.detach().clone() for k, p in net.named_parameters()}
    last = {k: v.clone() for k, v in start.items()}; W = {k: torch.zeros_like(v) for k, v in start.items()}
    for ep in range(E1):
        for _ in range(steps):
            i = torch.randint(0, n, (BS,), device=dev); opt.zero_grad(); F.cross_entropy(net(tr[0][i], tr[1][i]), tr[2][i]).backward()
            g = {k: p.grad.detach().clone() for k, p in net.named_parameters()}; opt.step()
            for k, p in net.named_parameters():
                W[k] -= g[k] * (p.detach() - last[k]); last[k] = p.detach().clone()
        with torch.no_grad():
            ok = net(tr[0][:POOL], tr[1][:POOL]).argmax(1) == tr[2][:POOL]
        if prev is not None:
            fev += (prev & ~ok).float()
        prev = ok
    om = {k: (W[k] / ((last[k] - start[k]) ** 2 + XI)).clamp(min=0) for k in W}
    return fev.cpu().numpy(), om


def hfeat(net, a, b):  # fused hidden layer of the head, L2-normalised (features for iCaRL herding and nearest-mean)
    return F.normalize(net.head[:2](torch.cat([net.e1(a), net.e2(b)], 1)), dim=1)


def herd(z, y, m):
    out = []
    for c in (0, 1):
        idx = torch.where(y == c)[0]; zc = z[idx]; mu = zc.mean(0); s = torch.zeros_like(mu); chosen = []
        for k in range(min(m, len(idx))):
            d = ((mu - (s + zc) / (k + 1)) ** 2).sum(1); d[chosen] = 1e9; j = int(d.argmin()); chosen.append(j); s += zc[j]
        out.append(idx[chosen])
    return torch.cat(out)


class Expand(nn.Module):
    """DER-expansion (Yan et al. 2021): frozen task-1 encoders + new encoders; the head is widened, starting as the
    task-1 head (zero weights on the new features)."""
    def __init__(self, old, d1, d2):
        super().__init__(); new = Net(d1, d2, 2).to(dev); self.o1, self.o2, self.e1, self.e2 = copy.deepcopy(old.e1), copy.deepcopy(old.e2), new.e1, new.e2
        for p in list(self.o1.parameters()) + list(self.o2.parameters()):
            p.requires_grad_(False)
        L0 = old.head[0]; emb = L0.in_features // 2; L = nn.Linear(4 * emb, L0.out_features).to(dev)
        with torch.no_grad():
            L.weight.zero_(); L.weight[:, :2 * emb] = L0.weight; L.bias.copy_(L0.bias)
        self.head = nn.Sequential(L, nn.ReLU(), copy.deepcopy(old.head[2]))

    def forward(self, a, b):
        return self.head(torch.cat([self.o1(a), self.o2(b), self.e1(a), self.e2(b)], 1))


def task2x(net1, tr1, tr2, kind, lam, buf, Fi, Om):
    """Returns (predict on task 1, predict on task 2[, extra predictions]). Older methods come from cl_exceptions.task2."""
    if kind not in ("si", "agem", "gem", "icarl", "expand", "multihead"):
        net = task2(net1, tr1, tr2, kind, lam, buf, Fi); f = lambda a, b: net(a, b).argmax(1); return f, f, None
    if kind == "si":
        net = task2(net1, tr1, tr2, "ewc", lam, None, Om); f = lambda a, b: net(a, b).argmax(1); return f, f, None
    old = net1; n = len(tr2[2]); d1, d2 = tr1[0].shape[1], tr1[1].shape[1]
    if buf is not None:
        bx1, bx2, by = tr1[0][t(buf)], tr1[1][t(buf)], tr1[2][t(buf)]
    if kind == "expand":
        net = Expand(old, d1, d2); params = [p for p in net.parameters() if p.requires_grad]
    elif kind == "multihead":
        net = copy.deepcopy(old); net.head = Net(d1, d2, 2).to(dev).head; params = list(net.parameters())
    else:
        net = copy.deepcopy(old); params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    for _ in range(E2 * (n // BS)):
        i = torch.randint(0, n, (BS,), device=dev); a, b, y = tr2[0][i], tr2[1][i], tr2[2][i]
        if kind in ("agem", "gem"):   # project the task-2 gradient so the loss on the memory does not increase
            g = torch.autograd.grad(F.cross_entropy(net(a, b), y), params)
            j = torch.randint(0, len(by), (256,), device=dev) if kind == "agem" else torch.arange(len(by), device=dev)
            gr = torch.autograd.grad(F.cross_entropy(net(bx1[j], bx2[j]), by[j]), params)
            dot = sum((x * z).sum() for x, z in zip(g, gr))
            if dot < 0:
                nn2 = sum((z * z).sum() for z in gr); g = [x - dot / nn2 * z for x, z in zip(g, gr)]
            opt.zero_grad()
            for p, x in zip(params, g):
                p.grad = x.clone()
            opt.step(); continue
        if kind == "icarl":   # classification + distillation from the task-1 model, on new data and exemplars
            j = torch.randint(0, len(by), (32,), device=dev); a, b, y = torch.cat([a, bx1[j]]), torch.cat([b, bx2[j]]), torch.cat([y, by[j]])
            out = net(a, b)
            with torch.no_grad():
                lo = old(a, b)
            loss = F.cross_entropy(out, y) + lam * F.kl_div(F.log_softmax(out / 2, 1), F.log_softmax(lo / 2, 1), log_target=True, reduction="batchmean") * 4
        else:                 # expand: ER on the widened model; multihead: plain training of the new head + shared encoders
            loss = F.cross_entropy(net(a, b), y)
            if kind == "expand":
                j = torch.randint(0, len(by), (32,), device=dev); loss = loss + F.cross_entropy(net(bx1[j], bx2[j]), by[j])
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    if kind == "multihead":
        h1 = old.head
        f1 = lambda a, b: h1(torch.cat([net.e1(a), net.e2(b)], 1)).argmax(1); f2 = lambda a, b: net(a, b).argmax(1); return f1, f2, None
    if kind == "icarl":       # nearest-mean-of-exemplars: 250 task-1 exemplars per label (herding prefix) + 250 task-2
        from cl_exceptions import POOL
        h = np.asarray(buf); keep = np.concatenate([h[:250], h[len(h) // 2:len(h) // 2 + 250]])
        with torch.no_grad():
            z1 = hfeat(net, tr1[0][t(keep)], tr1[1][t(keep)]); y1 = tr1[2][t(keep)]
            z2a = hfeat(net, tr2[0][:POOL], tr2[1][:POOL]); k2 = herd(z2a, tr2[2][:POOL], 250); z2, y2 = z2a[k2], tr2[2][:POOL][k2]
            Z, Y = torch.cat([z1, z2]), torch.cat([y1, y2]); mu = F.normalize(torch.stack([Z[Y == c].mean(0) for c in (0, 1)]), dim=1)
        f = lambda a, b: torch.cdist(hfeat(net, a, b), mu).argmin(1); g = lambda a, b: net(a, b).argmax(1); return f, f, g
    f = lambda a, b: net(a, b).argmax(1); return f, f, None


def scratch(tr2, d1, d2, seed):
    torch.manual_seed(seed + 500); net = Net(d1, d2, 2).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5); n = len(tr2[2])
    for _ in range(E2 * (n // BS)):
        i = torch.randint(0, n, (BS,), device=dev); opt.zero_grad(); torch.nn.functional.cross_entropy(net(tr2[0][i], tr2[1][i]), tr2[2][i]).backward(); opt.step()
    return net.eval()


def run_unit(name, tr1, te1, type1, tr2, te2, type2, nonadd_tr, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); t0 = time.time()
    net1 = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); fev, Om = train_task1_si(net1, tr1); net1.eval()
    with torch.no_grad():
        D1 = pd.DataFrame(dict(unit=name, nonadd=type1, before=(net1(te1[0], te1[1]).argmax(1) == te1[2]).cpu().numpy()))
    B = buffers(net1, tr1, fev, nonadd_tr, rng); Fi = fisher_diag(net1, tr1); D2 = pd.DataFrame(dict(unit=name, nonadd=type2))
    for key, kind, lam, bk in RUNS:
        f1, f2, g = task2x(net1, tr1, tr2, kind, lam, B[bk] if bk else None, Fi, Om)
        with torch.no_grad():
            D1[key] = (f1(te1[0], te1[1]) == te1[2]).cpu().numpy(); D2[key] = (f2(te2[0], te2[1]) == te2[2]).cpu().numpy()
            if g is not None:
                D1[key + "_head"] = (g(te1[0], te1[1]) == te1[2]).cpu().numpy(); D2[key + "_head"] = (g(te2[0], te2[1]) == te2[2]).cpu().numpy()
    sn = scratch(tr2, tr1[0].shape[1], tr1[1].shape[1], seed)
    with torch.no_grad():
        D2["scratch"] = (sn(te2[0], te2[1]).argmax(1) == te2[2]).cpu().numpy()
    print(name, f"({time.time() - t0:.0f}s)", flush=True); return D1, D2


def bench_tables(shard):
    out = []
    for setname, Ff in (("selected", "orig_F.npz"), ("fresh", "fresh.npz")):
        S = np.load(os.path.join(ROOT, "runs", "tables", f"{setname}.npz")); Fs = np.load(os.path.join(ROOT, "runs", "tables", Ff))["F"]
        for k in range(len(S["T"])):
            if S["d_add"][k] == 0 or (k + (0 if setname == "selected" else 1000)) % shard[1] != shard[0]:
                continue
            T, Fn = S["T"][k].astype(int), Fs[k].astype(int); rng = np.random.default_rng(23)
            tr1, te1 = table_pairs(T, 50000, 0, rng), table_pairs(T, 5000, 1, rng)
            T2 = balanced_table(np.random.default_rng(1023)); F2 = nearest_ferrers(T2.astype(np.int8))[0].astype(int); r2 = np.random.default_rng(2023)
            tr2, te2 = table_pairs(T2, 50000, 0, r2, POOL2), table_pairs(T2, 5000, 1, r2, POOL2)
            NA, NA2 = (T != Fn).astype(int), (T2 != F2).astype(int)
            out.append(run_unit(f"{setname}_{k}", tr1, te1[:3], NA[te1[3].cpu().numpy(), te1[4].cpu().numpy()], tr2[:3], te2[:3],
                                NA2[te2[3].cpu().numpy(), te2[4].cpu().numpy()], NA[tr1[3].cpu().numpy(), tr1[4].cpu().numpy()], 23))
    return out


def bench_clevr(shard):
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep")
    allU = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); TS = np.load(os.path.join(ROOT, "runs", "clx", "train_structure.npz"))
    itr, ite = TS["train_imgs"], ST["clevr_test_imgs"]; Xi = t(Z["Ximg"]).float(); out = []
    for n, u in enumerate(TS["units"]):
        if n % shard[1] != shard[0]:
            continue
        k = allU.index(u); p = next(allU[(k + s) % len(allU)] for s in range(len(allU) // 2, len(allU)) if fam[allU[(k + s) % len(allU)]] != fam[u]); rng = np.random.default_rng(23)
        def cells(unit, imgs, nsamp):
            T = Z[f"T__{unit}"]; Xq = t(Z[f"Cq__{unit}"]).float(); q, ii = rng.integers(0, 10, nsamp), rng.integers(0, len(imgs), nsamp)
            return Xi[t(imgs[ii])], Xq[t(q)], t(T[q, imgs[ii]]).long(), q, ii
        def full(unit):
            T = Z[f"T__{unit}"]; Xq = t(Z[f"Cq__{unit}"]).float(); q, ii = np.meshgrid(np.arange(10), np.arange(len(ite)), indexing="ij"); q, ii = q.ravel(), ii.ravel()
            return (Xi[t(ite[ii])], Xq[t(q)], t(T[q, ite[ii]]).long()), ST[f"nonadd__clevr__{unit}"][q, ii]
        tr1, tr2 = cells(u, itr, 50000), cells(p, itr, 50000); (te1, ty1), (te2, ty2) = full(u), full(p)
        out.append(run_unit(u, tr1, te1, ty1, tr2[:3], te2, ty2, TS[f"nonadd__{u}"][tr1[3], tr1[4]], 23))
    return out


def clip_feats(paths, texts):
    sys.path.insert(0, ROOT); from extract_features import clip_images, clip_texts
    nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6)
    return (t(nz(clip_images(np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)) for p in paths])).astype(np.float32))).float(),
            t(nz(clip_texts(list(texts)).astype(np.float32))).float())


def load_sm():
    global SM
    SM = SM if SM is not None else np.load(os.path.join(OUT, "structure.npz"))


def bench_gqa(shard):
    load_sm()
    G = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep")
    U = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); IMD = os.path.join(ROOT, "data", "raw", "gqa", "images"); feats = {}; out = []
    perm = np.random.default_rng(92).permutation(200); itr, ite = np.sort(perm[:150]), np.sort(perm[150:])
    def F(u):
        if u not in feats:
            feats[u] = clip_feats([os.path.join(IMD, f"{m}.jpg") for m in G[f"imgs__{u}"]], G[f"text__{u}"])
        return feats[u]
    for n, u in enumerate(U):
        if n % shard[1] != shard[0]:
            continue
        p = next(U[(n + s) % len(U)] for s in range(len(U) // 2, len(U)) if fam[U[(n + s) % len(U)]] != fam[u])
        for seed in (23, 24, 25):
            rng = np.random.default_rng(seed)
            def cells(unit, nsamp):
                T = G[f"T__{unit}"]; Xi, Xq = F(unit); q, ii = rng.integers(0, 10, nsamp), rng.integers(0, len(itr), nsamp)
                return Xi[t(itr[ii])], Xq[t(q)], t(T[q, itr[ii]]).long(), q, ii
            def full(unit):
                T = G[f"T__{unit}"]; Xi, Xq = F(unit); q, ii = np.meshgrid(np.arange(10), np.arange(len(ite)), indexing="ij"); q, ii = q.ravel(), ii.ravel()
                return (Xi[t(ite[ii])], Xq[t(q)], t(T[q, ite[ii]]).long()), ST[f"nonadd__gqa__{unit}"][q, ii]
            tr1, tr2 = cells(u, 50000), cells(p, 50000); (te1, ty1), (te2, ty2) = full(u), full(p)
            out.append(run_unit(f"{u}_s{seed}", tr1, te1, ty1, tr2[:3], te2, ty2, SM[f"nonadd__gqa_train__{u}"][tr1[3], tr1[4]], seed))
    return out


def bench_pope(shard):
    load_sm()
    P = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); C, Tf = P["cells"], P["T"].astype(int); Fc = np.load(os.path.join(ROOT, "runs", "pope", "clip.npz"))
    nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6); Xi, Xq = t(nz(Fc["img"])).float(), t(nz(Fc["q"])).float()
    trset = set(np.random.default_rng(71).permutation(500)[:375]); te_m = np.array([k not in trset for k in C[:, 1]]); trc, tec = C[~te_m], C[te_m]
    asked = np.zeros(Tf.shape, bool); asked[C[:, 0], C[:, 1]] = True
    rest = np.array([(o, k, Tf[o, k]) for o in range(80) for k in range(500) if not asked[o, k]]); rtr = rest[[k in trset for k in rest[:, 1]]]; rte = rest[[k not in trset for k in rest[:, 1]]]
    N1te, N1tr = ST["nonadd__pope__all"], SM["nonadd__pope_t1_train"]; N2te = SM["nonadd__pope_t2_test"]; out = []
    for seed in (23, 24, 25, 26, 27):
        if seed % shard[1] != shard[0]:
            continue
        rng = np.random.default_rng(seed); a = trc[rng.integers(0, len(trc), 50000)]
        bal = lambda R, n: np.concatenate([R[R[:, 2] == 1][rng.integers(0, (R[:, 2] == 1).sum(), n // 2)], R[R[:, 2] == 0][rng.integers(0, (R[:, 2] == 0).sum(), n // 2)]])
        b, b2 = bal(rtr, 50000), bal(rte, 5000)
        tr1 = (Xi[t(a[:, 1])], Xq[t(a[:, 0])], t(a[:, 2]).long(), a[:, 0], a[:, 1]); tr2 = (Xi[t(b[:, 1])], Xq[t(b[:, 0])], t(b[:, 2]).long())
        te1 = (Xi[t(tec[:, 1])], Xq[t(tec[:, 0])], t(tec[:, 2]).long()); te2 = (Xi[t(b2[:, 1])], Xq[t(b2[:, 0])], t(b2[:, 2]).long())
        out.append(run_unit(f"pope_s{seed}", tr1, te1, N1te[tec[:, 0], tec[:, 1]], tr2, te2, N2te[b2[:, 0], b2[:, 1]], N1tr[a[:, 0], a[:, 1]], seed))
    return out


if __name__ == "__main__":
    bn, s, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    res = {"tables": bench_tables, "clevr": bench_clevr, "gqa": bench_gqa, "pope": bench_pope}[bn]((s, ns))
    pd.concat([r[0] for r in res]).to_csv(os.path.join(OUT, f"task1_{bn}_{s}.csv.gz"), index=False)
    pd.concat([r[1] for r in res]).to_csv(os.path.join(OUT, f"task2_{bn}_{s}.csv.gz"), index=False); print("DONE", bn, s)
