"""Six-task continual learning on CLOVE-scene (paper/preregistration_clove_cl.md). One run = one CLOVE task order and
seed. Task 1 (30 epochs) is trained once (with SI importance and the Fisher tracked); from that checkpoint every method
learns tasks 2-6 (10 epochs each):
  finetune; EWC (online, 1 / 100 / 10000); SI (0.1 / 1 / 10 / 100); LwF (0.3 / 3); feature distillation (0.3 / 3);
  ER, A-GEM, GEM, DER++ (memory 1000 split equally over past tasks, random); DER++ rule+exceptions; iCaRL (herding,
  distillation 0.1 / 1, nearest-mean); DER-expansion (new encoders per task, old frozen, ER memory); multi-head.
plus a model trained on each task alone (10 epochs) for forward transfer. For every test sample of every task we save
whether it is correct right after its own task (own) and after task 6 (final).
Usage: python cl_seq.py <run index 0-11>   (runs = 6 orders x 2 seeds)"""
import copy, os, pickle, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from route_exps import Net
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clove_cl"); os.makedirs(OUT, exist_ok=True); dev = "cuda"
t = lambda a: torch.tensor(np.asarray(a), device=dev); BS, E1, E2, NTR, POOL, MEM, XI = 256, 30, 10, 50000, 10000, 1000, 0.1
ORDERS = [list("abcdef")] + pickle.load(open(os.path.join(ROOT, "data", "raw", "clove", "scene_perm.pkl"), "rb"))["sampled_perm"]
GRID = [("finetune", "finetune", 0.0)] + [(f"ewc_{l:g}", "ewc", l) for l in (1, 100, 10000)] + [(f"si_{l:g}", "si", l) for l in (0.1, 1, 10, 100)] \
    + [(f"lwf_{l:g}", "lwf", l) for l in (0.3, 3)] + [(f"l2_{l:g}", "l2", l) for l in (0.3, 3)] + [(m, m, 0.0) for m in ("er", "agem", "gem", "derpp", "derpp_ruleexc")] \
    + [(f"icarl_{l:g}", "icarl", l) for l in (0.1, 1)] + [("der_expand", "expand", 0.0), ("multihead", "multihead", 0.0)]


def data():
    U = np.load(os.path.join(ROOT, "runs", "clove", "units.npz")); C = np.load(os.path.join(ROOT, "runs", "clove", "clip.npz")); S = np.load(os.path.join(ROOT, "runs", "clove", "structure.npz"))
    K = pd.read_csv(os.path.join(ROOT, "runs", "clove", "units.csv")).query("keep")
    nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6); XI_, XQ = t(nz(C["img"].astype(np.float32))).float(), t(nz(C["txt"].astype(np.float32))).float()
    row = {m: k for k, m in enumerate(C["ids"])}; qoff = {u: 10 * k for k, u in enumerate(C["units"])}; tasks = {}
    for tk, g in K.groupby("task"):
        cells = {"train": [], "test": []}
        for u in g.unit:
            T, im = U[f"T__{u}"], U[f"imgs__{u}"]
            for sp, sl in (("train", range(150)), ("test", range(150, 200))):
                N = S[f"nonadd__{sp}__{u}"]
                for q in range(10):
                    for j, c in enumerate(sl):
                        cells[sp].append((row[str(im[c])], qoff[u] + q, T[q, c], N[q, j], u))
        tasks[tk] = {sp: pd.DataFrame(v, columns=["img", "q", "y", "nonadd", "unit"]) for sp, v in cells.items()}
    return tasks, XI_, XQ


class Expand(nn.Module):
    """DER-expansion (Yan et al. 2021) over several tasks: frozen encoders from earlier tasks plus one new pair."""
    def __init__(self, base, d1, d2):
        super().__init__(); self.old = nn.ModuleList(); self.e1, self.e2, self.head = base.e1, base.e2, base.head

    def grow(self, d1, d2):
        frozen = nn.ModuleList([copy.deepcopy(self.e1), copy.deepcopy(self.e2)])
        for p in frozen.parameters():
            p.requires_grad_(False)
        self.old.append(frozen); new = Net(d1, d2, 2).to(dev); self.e1, self.e2 = new.e1, new.e2
        L0 = self.head[0]; L = nn.Linear(L0.in_features + 2 * new.e1[-1].out_features, L0.out_features).to(dev)
        with torch.no_grad():
            L.weight.zero_(); L.weight[:, :L0.in_features] = L0.weight; L.bias.copy_(L0.bias)
        self.head = nn.Sequential(L, nn.ReLU(), self.head[2])

    def forward(self, a, b):
        return self.head(torch.cat([z for o in self.old for z in (o[0](a), o[1](b))] + [self.e1(a), self.e2(b)], 1))


def hfeat(net, a, b):
    return F.normalize(net.head[:2](torch.cat([net.e1(a), net.e2(b)], 1)), dim=1)


def herd(z, y, m):
    out = []
    for c in (0, 1):
        idx = torch.where(y == c)[0]; zc = z[idx]; mu = zc.mean(0); s = torch.zeros_like(mu); chosen = []
        for k in range(min(m, len(idx))):
            d = ((mu - (s + zc) / (k + 1)) ** 2).sum(1); d[chosen] = 1e9; j = int(d.argmin()); chosen.append(j); s += zc[j]
        out.append(idx[chosen])
    return torch.cat(out)


def interleave(a, b):
    """a0 b0 a1 b1 ...: any prefix keeps the same mix when the memory is cut down for a new task."""
    out = [x for pair in zip(a, b) for x in pair]; n = min(len(a), len(b)); return np.array(out + list(a[n:]) + list(b[n:]), int)


def gem_project(g, G):
    """GEM (Lopez-Paz & Ranzato 2017): smallest change to g so that no past-task gradient (rows of G) has a negative dot."""
    p = (G @ g).cpu().double().numpy()
    if (p >= 0).all():
        return g
    M = (G @ G.T).cpu().double().numpy(); v = np.zeros(len(p)); lr = 1.0 / (np.linalg.eigvalsh(M).max() + 1e-12)
    for _ in range(500):   # dual: min_v 1/2 v'Mv + p'v, v >= 0 (projected gradient)
        v = np.maximum(0, v - lr * (M @ v + p))
    return g + G.T @ torch.tensor(v, device=dev, dtype=g.dtype)


class Run:
    def __init__(self, order, seed):
        self.tasks, self.XI, self.XQ = data(); self.order, self.seed = order, seed; self.d1, self.d2 = self.XI.shape[1], self.XQ.shape[1]
        torch.manual_seed(seed); rng = np.random.default_rng(seed); self.tr, self.te = {}, {}
        for tk in order:   # 50,000 training draws per task from its training cells; all test cells
            D = self.tasks[tk]["train"]; i = rng.integers(0, len(D), NTR); d = D.iloc[i]
            self.tr[tk] = (self.XI[t(d.img.values)], self.XQ[t(d.q.values)], t(d.y.values).long(), d.nonadd.values, d.img.values * 10000 + d.q.values)
            E = self.tasks[tk]["test"]; self.te[tk] = (self.XI[t(E.img.values)], self.XQ[t(E.q.values)], t(E.y.values).long())

    def batches(self, tk, epochs):
        n = len(self.tr[tk][2])
        for _ in range(epochs * (n // BS)):
            i = torch.randint(0, n, (BS,), device=dev); yield self.tr[tk][0][i], self.tr[tk][1][i], self.tr[tk][2][i]

    def memory(self, tk, kind, m, net):
        """m samples of task tk from its first POOL draws: random, rule+exceptions (half from exception cells, one per cell
        in turn, half random) or herding (iCaRL)."""
        rng = np.random.default_rng(self.seed * 100 + self.order.index(tk)); x1, x2, y, na, cell = self.tr[tk]
        if kind == "herding":
            with torch.no_grad():
                h = herd(hfeat(net, x1[:POOL], x2[:POOL]), y[:POOL], m // 2).cpu().numpy()
            idx = interleave(h[:len(h) // 2], h[len(h) // 2:])   # herding prefixes stay valid when the memory shrinks
        elif kind in ("classbal", "ruleexc_bal"):   # class-balance control (preregistration_strat.md)
            yy = y[:POOL].cpu().numpy(); pick = np.array([], int)
            if kind == "ruleexc_bal":
                ex = np.where(na[:POOL] == 1)[0]; byc = {}
                for j in rng.permutation(ex):
                    byc.setdefault(cell[j], []).append(j)
                lists, pk = list(byc.values()), []
                while len(pk) < m // 2 and any(lists):
                    for v in lists:
                        if v and len(pk) < m // 2:
                            pk.append(v.pop())
                pick = np.array(pk, int)
            rest = np.setdiff1d(np.arange(POOL), pick); n = m - len(pick); want = int(np.clip(m // 2 - yy[pick].sum(), 0, n))
            ry, rn = rest[yy[rest] == 1], rest[yy[rest] == 0]; ky = min(want, len(ry)); kn = min(n - ky, len(rn)); ky = n - kn
            fill = np.concatenate([rng.choice(ry, ky, replace=False), rng.choice(rn, kn, replace=False)]); fy, fn = fill[yy[fill] == 1], fill[yy[fill] == 0]
            if kind == "classbal":
                idx = interleave(fy, fn)
            else:   # exception half and its balancing half interleaved; the balancing half itself interleaves yes / no
                idx = interleave(pick, interleave(rng.permutation(fy), rng.permutation(fn)))
        elif kind == "ruleexc":
            ex = np.where(na[:POOL] == 1)[0]; byc = {}
            for j in rng.permutation(ex):
                byc.setdefault(cell[j], []).append(j)
            lists, pick = list(byc.values()), []
            while len(pick) < m // 2 and any(lists):
                for v in lists:
                    if v and len(pick) < m // 2:
                        pick.append(v.pop())
            rest = np.setdiff1d(np.arange(POOL), pick); idx = interleave(np.array(pick, int), rng.choice(rest, m - len(pick), replace=False))
        else:
            idx = rng.choice(POOL, m, replace=False)
        idx = t(idx)
        with torch.no_grad():
            lo = net(x1[idx], x2[idx])
        return dict(x1=x1[idx], x2=x2[idx], y=y[idx], logit=lo, task=tk)

    def task1(self):
        torch.manual_seed(self.seed); net = Net(self.d1, self.d2, 2).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        start = {k: p.detach().clone() for k, p in net.named_parameters()}; last = {k: v.clone() for k, v in start.items()}; W = {k: torch.zeros_like(v) for k, v in start.items()}
        for a, b, y in self.batches(self.order[0], E1):
            opt.zero_grad(); F.cross_entropy(net(a, b), y).backward(); g = {k: p.grad.detach().clone() for k, p in net.named_parameters()}; opt.step()
            for k, p in net.named_parameters():
                W[k] -= g[k] * (p.detach() - last[k]); last[k] = p.detach().clone()
        net.eval(); om = {k: (W[k] / ((last[k] - start[k]) ** 2 + XI)).clamp(min=0) for k in W}
        return net, om, self.fisher(net, self.order[0])

    def fisher(self, net, tk, n=2000):
        net = copy.deepcopy(net); Fi = {k: torch.zeros_like(p) for k, p in net.named_parameters()}; x1, x2, y = self.tr[tk][:3]
        for s in range(0, n, 32):
            net.zero_grad(); F.cross_entropy(net(x1[s:s + 32], x2[s:s + 32]), y[s:s + 32]).backward()
            for k, p in net.named_parameters():
                Fi[k] += p.grad.detach() ** 2 * 32 / n
        return Fi

    def method(self, net1, om1, fi1, kind, lam):
        """Learns tasks 2..6 from the shared task-1 model. Returns {task: predict function} right after each task and at the end."""
        own, mem = {}, []; net = copy.deepcopy(net1); heads = {self.order[0]: net.head}
        if kind == "expand":
            net = Expand(net, self.d1, self.d2)
        anchor = {k: p.detach().clone() for k, p in net1.named_parameters()}; Om = {k: v.clone() for k, v in om1.items()}; Fi = {k: v.clone() for k, v in fi1.items()}
        mk = {"derpp_ruleexc": "ruleexc", "icarl": "herding", "derpp_classbal": "classbal", "derpp_ruleexc_bal": "ruleexc_bal"}.get(kind, "random")
        def add_memory(tk, k):   # memory is split equally over the tasks seen so far
            nonlocal mem
            if kind in ("er", "agem", "gem", "derpp", "derpp_ruleexc", "derpp_classbal", "derpp_ruleexc_bal", "icarl", "expand"):
                mem = [dict({q: v[:MEM // k] for q, v in m.items() if q != "task"}, task=m["task"]) for m in mem] + [self.memory(tk, mk, MEM // k, net)]
        add_memory(self.order[0], 1); own[self.order[0]] = self.predictor(net, kind, mem, heads, self.order[0])
        for pos, tk in enumerate(self.order[1:], start=2):
            old = copy.deepcopy(net).eval()
            if kind == "expand":
                net.grow(self.d1, self.d2)
            if kind == "multihead":
                net.head = Net(self.d1, self.d2, 2).to(dev).head; heads[tk] = net.head
            params = [p for p in net.parameters() if p.requires_grad]; opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
            if kind == "si":
                start = {k: p.detach().clone() for k, p in net.named_parameters()}; last = {k: v.clone() for k, v in start.items()}; W = {k: torch.zeros_like(v) for k, v in start.items()}
            if mem:
                M = {q: torch.cat([m[q] for m in mem]) for q in ("x1", "x2", "y", "logit") if kind != "expand" or q != "logit"}
            net.train()
            for a, b, y in self.batches(tk, E2):
                if kind == "icarl":
                    j = torch.randint(0, len(M["y"]), (32,), device=dev); a, b, y = torch.cat([a, M["x1"][j]]), torch.cat([b, M["x2"][j]]), torch.cat([y, M["y"][j]])
                out = net(a, b); loss = F.cross_entropy(out, y)
                if kind in ("lwf", "icarl"):
                    with torch.no_grad():
                        lo = old(a, b)
                    loss = loss + lam * F.kl_div(F.log_softmax(out / 2, 1), F.log_softmax(lo / 2, 1), log_target=True, reduction="batchmean") * 4
                elif kind == "l2":
                    with torch.no_grad():
                        o1, o2 = old.e1(a), old.e2(b)
                    loss = loss + lam * ((net.e1(a) - o1).pow(2).sum(1).mean() + (net.e2(b) - o2).pow(2).sum(1).mean()) / o1.shape[1]
                elif kind in ("ewc", "si"):
                    P = Fi if kind == "ewc" else Om; loss = loss + lam * sum((P[k] * (p - anchor[k]) ** 2).sum() for k, p in net.named_parameters())
                elif kind in ("er", "derpp", "derpp_ruleexc", "derpp_classbal", "derpp_ruleexc_bal", "expand"):
                    j = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + F.cross_entropy(net(M["x1"][j], M["x2"][j]), M["y"][j])
                    if kind != "er" and kind != "expand":
                        j2 = torch.randint(0, len(M["y"]), (32,), device=dev); loss = loss + 0.5 * F.mse_loss(net(M["x1"][j2], M["x2"][j2]), M["logit"][j2])
                if kind in ("agem", "gem"):
                    g = torch.cat([x.flatten() for x in torch.autograd.grad(loss, params)])
                    if kind == "agem":
                        j = torch.randint(0, len(M["y"]), (256,), device=dev)
                        r = torch.cat([x.flatten() for x in torch.autograd.grad(F.cross_entropy(net(M["x1"][j], M["x2"][j]), M["y"][j]), params)])
                        if g @ r < 0:
                            g = g - (g @ r) / (r @ r) * r
                    else:
                        G = torch.stack([torch.cat([x.flatten() for x in torch.autograd.grad(F.cross_entropy(net(m["x1"], m["x2"]), m["y"]), params)]) for m in mem]); g = gem_project(g, G)
                    opt.zero_grad(); k = 0
                    for p in params:
                        p.grad = g[k:k + p.numel()].view_as(p).clone(); k += p.numel()
                    opt.step()
                else:
                    opt.zero_grad(); loss.backward()
                    if kind == "si":
                        gg = {k: p.grad.detach().clone() for k, p in net.named_parameters()}
                    opt.step()
                    if kind == "si":
                        for k, p in net.named_parameters():
                            W[k] -= gg[k] * (p.detach() - last[k]); last[k] = p.detach().clone()
            net.eval()
            if kind == "si":
                for k in Om:
                    Om[k] += (W[k] / ((last[k] - start[k]) ** 2 + XI)).clamp(min=0)
            if kind == "ewc":
                fi = self.fisher(net, tk); Fi = {k: Fi[k] + fi[k] for k in Fi}
            anchor = {k: p.detach().clone() for k, p in net.named_parameters()}
            add_memory(tk, pos); own[tk] = self.predictor(net, kind, mem, heads, tk)
        final = {tk: self.predictor(net, kind, mem, heads, tk) for tk in self.order}
        return own, final

    def predictor(self, net, kind, mem, heads, tk):
        """Evaluates now (the model changes later) and returns per-sample correctness for every task's test set."""
        x1, x2, y = self.te[tk]
        with torch.no_grad():
            if kind == "multihead":
                h = heads[tk]; p = h(torch.cat([net.e1(x1), net.e2(x2)], 1)).argmax(1)
            elif kind == "icarl" and mem:
                Z = torch.cat([hfeat(net, m["x1"], m["x2"]) for m in mem]); Y = torch.cat([m["y"] for m in mem])
                mu = F.normalize(torch.stack([Z[Y == c].mean(0) for c in (0, 1)]), dim=1); p = torch.cdist(hfeat(net, x1, x2), mu).argmin(1)
            else:
                p = net(x1, x2).argmax(1)
        return (p == y).cpu().numpy()

    def scratch(self, tk):
        torch.manual_seed(self.seed + 500); net = Net(self.d1, self.d2, 2).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        for a, b, y in self.batches(tk, E2):
            opt.zero_grad(); F.cross_entropy(net(a, b), y).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return (net(self.te[tk][0], self.te[tk][1]).argmax(1) == self.te[tk][2]).cpu().numpy()


GRID_BAL = [(m, m, 0.0) for m in ("derpp", "derpp_ruleexc", "derpp_classbal", "derpp_ruleexc_bal")]   # class-balance control

if __name__ == "__main__":
    BAL = "--bal" in sys.argv; GRID = GRID_BAL if BAL else GRID
    r = int(sys.argv[1]); order, seed = ORDERS[r % 6], 31 + r // 6; R = Run(order, seed); t0 = time.time()
    net1, om1, fi1 = R.task1(); rows = {tk: R.tasks[tk]["test"][["unit", "nonadd", "y"]].assign(run=r, task=tk, pos=order.index(tk) + 1) for tk in order}
    for tk in order:
        rows[tk]["scratch"] = R.scratch(tk)
    for name, kind, lam in GRID:
        own, final = R.method(net1, om1, fi1, kind, lam)
        for tk in order:
            rows[tk][f"{name}__own"], rows[tk][f"{name}__final"] = own[tk], final[tk]
        print(r, name, f"{time.time() - t0:.0f}s", flush=True)
    pd.concat(rows.values()).to_csv(os.path.join(OUT, f"run_{'bal_' if BAL else ''}{r}.csv.gz"), index=False); print("DONE", r)
