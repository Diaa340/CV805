"""F2 confirmatory rerun (paper/preregistration_f1f2.md) with new seeds (tables / CLEVR 24, GQA 26-28, POPE 28-32).
Derived from: Per-sample unrepairability test (paper/preregistration_unrepair.md). For each benchmark unit: train the two-branch
model on task 1 (30 epochs), drift it on task 2 (10 epochs, whole model), then on the frozen drifted encoders retrain a
LINEAR head (cheap, additive) and an MLP head (reference) on 20,000 task-1 training samples. Per test sample record the
cell's structure label (non-additive), local synergy-minus-redundancy, the task-1 model's confidence, and correctness
before drift / after drift / after each repair.
Benchmarks: tables (125 constructed tables, two-view CIFAR, seed 23), clevr (103 units; symbolic image + CLIP question
features; task 2 = a unit of another family), gqa (24 units; CLIP; task 2 = a unit of another family; seeds 23-25),
pope (CLIP; task 2 = COCO cells not asked by POPE, balanced; seeds 23-27). Usage: python unrepair.py <bench> [shard n]"""
import copy, os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from kinds_kill import balanced_table, linear_check
from route_exps import Net, train
from tables_step1 import POOL2, mlp_head, table_pairs
from unrepair_structure import local_synred
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "unrepair2"); dev = "cuda"
ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz")); t = lambda a: torch.tensor(np.asarray(a), device=dev)


def run(tr1, te, tr2, extra, seed):
    """tr1 / tr2: (x1, x2, y) training tensors of task 1 / task 2; te: (x1, x2, y) task-1 test tensors; extra: per-test-sample dict."""
    torch.manual_seed(seed); net = Net(tr1[0].shape[1], tr1[1].shape[1], 2).to(dev); train(net, tr1, 30, 1e-3)
    with torch.no_grad():
        lo = net(te[0], te[1]); before = (lo.argmax(1) == te[2]).cpu().numpy(); conf = (lo[:, 1] - lo[:, 0]).abs().cpu().numpy()
    net2 = copy.deepcopy(net); train(net2, tr2, 10, 1e-3)
    with torch.no_grad():
        after = (net2(te[0], te[1]).argmax(1) == te[2]).cpu().numpy()
        ztr = torch.cat([net2.e1(tr1[0][:20000]), net2.e2(tr1[1][:20000])], 1); zte = torch.cat([net2.e1(te[0]), net2.e2(te[1])], 1)
    lin = torch.nn.Linear(ztr.shape[1], 2).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(ztr), tr1[2][:20000]).backward(); opt.step()
    with torch.no_grad():
        rec_lin = (lin(zte).argmax(1) == te[2]).cpu().numpy()
    h = torch.nn.Sequential(torch.nn.Linear(ztr.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(dev); opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    for _ in range(2000):
        i = torch.randint(0, len(ztr), (512,), device=dev); opt.zero_grad(); torch.nn.functional.cross_entropy(h(ztr[i]), tr1[2][:20000][i]).backward(); opt.step()
    with torch.no_grad():
        rec_mlp = (h(zte).argmax(1) == te[2]).cpu().numpy()
    return pd.DataFrame(dict(**extra, y=te[2].cpu().numpy(), conf=conf, before=before, after=after, rec_lin=rec_lin, rec_mlp=rec_mlp))


def bench_tables(shard):
    from kinds_kill import F1, F2  # noqa (features loaded by kinds_kill)
    out = []
    for setname, Ffile in (("selected", "orig_F.npz"), ("fresh", "fresh.npz")):
        S = np.load(os.path.join(ROOT, "runs", "tables", f"{setname}.npz")); Fs = np.load(os.path.join(ROOT, "runs", "tables", Ffile))["F"]
        for k in range(len(S["T"])):
            if (k + (0 if setname == "selected" else 1000)) % shard[1] != shard[0]:
                continue
            T, Fn = S["T"][k].astype(int), Fs[k].astype(int); rng = np.random.default_rng(24)
            tr1, te = table_pairs(T, 50000, 0, rng), table_pairs(T, 5000, 1, rng)
            T2 = balanced_table(np.random.default_rng(1024)); tr2 = table_pairs(T2, 50000, 0, np.random.default_rng(2024), POOL2)
            ci, cj = te[3].cpu().numpy(), te[4].cpu().numpy(); L = local_synred(T, np.ones(T.shape))
            d = run(tr1[:3], te[:3], tr2[:3], dict(unit=f"{setname}_{k}", nonadd=(T != Fn)[ci, cj].astype(int), synred=L[ci, cj]), 24)
            out.append(d); print(setname, k, len(d), flush=True)
    return pd.concat(out)


def bench_clevr(shard):
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep")
    U = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); perm = np.random.default_rng(14).permutation(1000); itr, ite = perm[:700], ST["clevr_test_imgs"]
    Xi = t(Z["Ximg"]).float(); out = []
    for n, u in enumerate(U):
        if n % shard[1] != shard[0]:
            continue
        p = next(U[(n + s) % len(U)] for s in range(len(U) // 2, len(U)) if fam[U[(n + s) % len(U)]] != fam[u])
        def cells(unit, imgs, nsamp, rng):
            T = Z[f"T__{unit}"]; Xq = t(Z[f"Cq__{unit}"]).float(); q, i = rng.integers(0, 10, nsamp), imgs[rng.integers(0, len(imgs), nsamp)]
            return Xi[t(i)], Xq[t(q)], t(T[q, i]).long()
        rng = np.random.default_rng(24); tr1, tr2 = cells(u, itr, 50000, rng), cells(p, itr, 50000, rng)
        T = Z[f"T__{u}"]; Xq = t(Z[f"Cq__{u}"]).float(); q, i = np.meshgrid(np.arange(10), np.arange(len(ite)), indexing="ij"); q, i = q.ravel(), i.ravel()
        te = (Xi[t(ite[i])], Xq[t(q)], t(T[q, ite[i]]).long()); N, L = ST[f"nonadd__clevr__{u}"], ST[f"synred__clevr__{u}"]
        out.append(run(tr1, te, tr2, dict(unit=u, nonadd=N[q, i], synred=L[q, i]), 24)); print(u, flush=True)
    return pd.concat(out)


def clip_feats(paths, texts):
    sys.path.insert(0, ROOT); from extract_features import clip_images, clip_texts
    nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6)
    return (t(nz(clip_images(np.stack([np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC)) for p in paths])).astype(np.float32))).float(),
            t(nz(clip_texts(list(texts)).astype(np.float32))).float())


def bench_gqa(shard):
    G = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep")
    U = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); IMD = os.path.join(ROOT, "data", "raw", "gqa", "images"); feats = {}
    def F(u):
        if u not in feats:
            feats[u] = clip_feats([os.path.join(IMD, f"{m}.jpg") for m in G[f"imgs__{u}"]], G[f"text__{u}"])
        return feats[u]
    out = []
    for n, u in enumerate(U):
        if n % shard[1] != shard[0]:
            continue
        p = next(U[(n + s) % len(U)] for s in range(len(U) // 2, len(U)) if fam[U[(n + s) % len(U)]] != fam[u])
        perm = np.random.default_rng(92).permutation(200); itr, ite = perm[:150], np.sort(perm[150:])
        for seed in (26, 27, 28):
            rng = np.random.default_rng(seed)
            def cells(unit, imgs, nsamp):
                T = G[f"T__{unit}"]; Xi, Xq = F(unit); q, i = rng.integers(0, 10, nsamp), imgs[rng.integers(0, len(imgs), nsamp)]
                return Xi[t(i)], Xq[t(q)], t(T[q, i]).long()
            tr1, tr2 = cells(u, itr, 50000), cells(p, np.arange(200), 50000)
            T = G[f"T__{u}"]; Xi, Xq = F(u); q, i = np.meshgrid(np.arange(10), np.arange(len(ite)), indexing="ij"); q, i = q.ravel(), i.ravel()
            te = (Xi[t(ite[i])], Xq[t(q)], t(T[q, ite[i]]).long())
            out.append(run(tr1, te, tr2, dict(unit=f"{u}_s{seed}", nonadd=ST[f"nonadd__gqa__{u}"][q, i], synred=ST[f"synred__gqa__{u}"][q, i]), seed)); print(u, seed, flush=True)
    return pd.concat(out)


def bench_pope(shard):
    P = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); C, Tfull = P["cells"], P["T"]; Fc = np.load(os.path.join(ROOT, "runs", "pope", "clip.npz"))
    nz = lambda f: (f - f.mean(0)) / (f.std(0) + 1e-6); Xi, Xq = t(nz(Fc["img"])).float(), t(nz(Fc["q"])).float()
    tr_im = set(np.random.default_rng(71).permutation(500)[:375]); te_m = np.array([k not in tr_im for k in C[:, 1]]); trc, tec = C[~te_m], C[te_m]
    asked = {(o, k) for o, k, _ in C}; rest = np.array([(o, k, Tfull[o, k]) for o in range(80) for k in range(500) if (o, k) not in asked])
    N, L = ST["nonadd__pope__all"], ST["synred__pope__all"]; out = []
    for seed in (28, 29, 30, 31, 32):
        if seed % shard[1] != shard[0]:
            continue
        rng = np.random.default_rng(seed); a = trc[rng.integers(0, len(trc), 50000)]
        pos, neg = rest[rest[:, 2] == 1], rest[rest[:, 2] == 0]; b = np.concatenate([pos[rng.integers(0, len(pos), 25000)], neg[rng.integers(0, len(neg), 25000)]])
        tr1 = (Xi[t(a[:, 1])], Xq[t(a[:, 0])], t(a[:, 2]).long()); tr2 = (Xi[t(b[:, 1])], Xq[t(b[:, 0])], t(b[:, 2]).long())
        te = (Xi[t(tec[:, 1])], Xq[t(tec[:, 0])], t(tec[:, 2]).long())
        out.append(run(tr1, te, tr2, dict(unit=f"pope_s{seed}", nonadd=N[tec[:, 0], tec[:, 1]], synred=L[tec[:, 0], tec[:, 1]]), seed)); print("pope", seed, flush=True)
    return pd.concat(out)


if __name__ == "__main__":
    b = sys.argv[1]; shard = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    D = {"tables": bench_tables, "clevr": bench_clevr, "gqa": bench_gqa, "pope": bench_pope}[b](shard)
    D.to_csv(os.path.join(OUT, f"samples_{b}_{shard[0]}.csv.gz"), index=False); print("DONE", b, shard, len(D))
