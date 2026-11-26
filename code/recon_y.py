"""Rebuild the answer (y) of every saved test sample of the two-task CL-metrics study (runs/clm), which did not save it.
Tables: replays the same random draws (table_pairs with the same seeds); CLEVR / GQA: the fixed test grids. Every
rebuilt row is checked against the saved exception label. Writes runs/clm/y_<bench>.npz (task1 / task2 arrays in file
row order). Outcome-free."""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clm")


def load(b, task):
    return [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, f"task{task}_{b}_*.csv.gz")))], sorted(glob.glob(os.path.join(OUT, f"task{task}_{b}_*.csv.gz")))


def tables():
    from kinds_kill import balanced_table
    from step1_build import nearest_ferrers
    from tables_step1 import POOL2, table_pairs
    T2 = balanced_table(np.random.default_rng(1023)); r2 = np.random.default_rng(2023); table_pairs(T2, 50000, 0, r2, POOL2); te2 = table_pairs(T2, 5000, 1, r2, POOL2)
    y2 = te2[2].cpu().numpy(); F2 = nearest_ferrers(T2.astype(np.int8))[0].astype(int); na2 = (T2 != F2).astype(int)[te2[3].cpu().numpy(), te2[4].cpu().numpy()]
    cache = {}
    for setname, Ff in (("selected", "orig_F.npz"), ("fresh", "fresh.npz")):
        S = np.load(os.path.join(ROOT, "runs", "tables", f"{setname}.npz")); Fs = np.load(os.path.join(ROOT, "runs", "tables", Ff))["F"]
        for k in range(len(S["T"])):
            if S["d_add"][k] == 0:
                continue
            T, Fn = S["T"][k].astype(int), Fs[k].astype(int); rng = np.random.default_rng(23); table_pairs(T, 50000, 0, rng); te1 = table_pairs(T, 5000, 1, rng)
            cache[f"{setname}_{k}"] = (te1[2].cpu().numpy(), (T != Fn).astype(int)[te1[3].cpu().numpy(), te1[4].cpu().numpy()])
    return lambda u: cache[u], lambda u: (y2, na2)


def grids(b):
    ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz"))
    if b == "clevr":
        Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep")
        allU = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); ite = ST["clevr_test_imgs"]; Tget = lambda u: Z[f"T__{u}"]; base = lambda un: un
        partner = lambda u: next(allU[(allU.index(u) + s) % len(allU)] for s in range(len(allU) // 2, len(allU)) if fam[allU[(allU.index(u) + s) % len(allU)]] != fam[u])
        lab = lambda u: ST[f"nonadd__clevr__{u}"]; cols = lambda T: ite
    else:
        G = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz")); FU = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep")
        U = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family)); ite = np.sort(np.random.default_rng(92).permutation(200)[150:]); Tget = lambda u: G[f"T__{u}"]
        base = lambda un: un.rsplit("_s", 1)[0]; partner = lambda u: next(U[(U.index(u) + s) % len(U)] for s in range(len(U) // 2, len(U)) if fam[U[(U.index(u) + s) % len(U)]] != fam[u])
        lab = lambda u: ST[f"nonadd__gqa__{u}"]; cols = lambda T: ite
    def grid(u):
        T = Tget(u); q, ii = np.meshgrid(np.arange(10), np.arange(len(ite)), indexing="ij"); q, ii = q.ravel(), ii.ravel(); return T[q, ite[ii]], lab(u)[q, ii]
    return (lambda un: grid(base(un))), (lambda un: grid(partner(base(un))))


for b in ("tables", "clevr", "gqa"):
    f1, f2 = tables() if b == "tables" else grids(b); out = {}
    for task, f in ((1, f1), (2, f2)):
        Ds, names = load(b, task); ys = []
        for D in Ds:
            for un, x in D.groupby("unit", sort=False):
                y, na = f(un); assert len(y) == len(x) and np.array_equal(na, x.nonadd.values), (b, task, un); ys.append(y)
        out[f"task{task}"] = np.concatenate(ys); out[f"task{task}_files"] = np.array(names)
    np.savez(os.path.join(OUT, f"y_{b}.npz"), **out); print(b, "rebuilt and verified:", len(out["task1"]), len(out["task2"]), flush=True)
