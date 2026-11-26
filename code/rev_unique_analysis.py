"""Revision R3 analysis (paper/preregistration_revision.md): non-uniqueness of exceptions, and whether the key results
survive using only cells whose status is the same in every optimal rule found (runs/rev/unique.npz).
Run with .venv_cu128 on GPU 2 (rebuilding the constructed tasks' test cells loads their features on the GPU).
1. Ambiguity: share of cells whose status differs across optimal rules; share of the saved exception labels that are
   stable exceptions.
2. Continual learning (two-task study, saved results): same-answer gaps dBWT_s / dFWT_s recomputed with stable labels
   (ambiguous cells dropped), for the pre-specified results: CLEVR backward (ER, A-GEM, GEM, DER++) and forward (DER++,
   feature distillation) on constructed tasks, CLEVR and GQA.
3. VLM cells (Study C sub-tables): AUC for the cells EMAP flips, with stable labels only (descriptive).
Writes paper/results_rev_unique.txt"""
import glob, json, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); lines = []; NB = 1000; rng = np.random.default_rng(0)
def say(s=""): print(s, flush=True); lines.append(s)
UQ = dict(np.load(os.path.join(ROOT, "runs", "rev", "unique.npz"))); UC = pd.read_csv(os.path.join(ROOT, "runs", "rev", "unique.csv"))

say("1. How unique are the exceptions? (random tie-breaking among optimal rules; ambiguity is a lower bound)")
for b, x in UC.groupby("bench"):
    say(f"  {b:7s} {len(x):3d} tables | ambiguous cells {x.ambiguous_share.mean():.3f} (median {x.ambiguous_share.median():.3f}) | "
        f"stable exceptions {x.stable_exc_share.mean():.3f} of cells vs {x.min_errors.div(x.cells).mean():.3f} exception share | "
        f"accepted solutions {x.accepted.mean():.1f}/{x.solves.mean():.0f} | proven optimal {x.proven.div(x.solves).mean():.2f}")


def label(key, rows, cols):
    """Stable label per cell: 1 = exception in every optimal rule found, 0 = in none, NaN = ambiguous."""
    E, A = UQ[f"exc__{key}"][rows, cols].astype(float), UQ[f"amb__{key}"][rows, cols].astype(bool); E[A] = np.nan; return E


def cells(b):
    """(table key, row index, column index) for every saved test sample of the two-task study, in file order."""
    Y = np.load(os.path.join(ROOT, "runs", "clm", f"y_{b}.npz")); out = {}
    for task in (1, 2):
        Ds = [pd.read_csv(f) for f in Y[f"task{task}_files"]]; keys, rr, cc = [], [], []
        if b == "tables":
            from tables_step1 import POOL2, table_pairs
            from rev_unique import tables as _t
            T2 = dict((k, T) for k, T, _, _ in _t() if k == "tables__task2")["tables__task2"]
            r2 = np.random.default_rng(2023); table_pairs(T2, 50000, 0, r2, POOL2); te2 = table_pairs(T2, 5000, 1, r2, POOL2)
            for D in Ds:
                for un, x in D.groupby("unit", sort=False):
                    if task == 1:
                        setname, k = un.rsplit("_", 1); S = np.load(os.path.join(ROOT, "runs", "tables", f"{setname}.npz")); T = S["T"][int(k)].astype(int)
                        rng1 = np.random.default_rng(23); table_pairs(T, 50000, 0, rng1); te = table_pairs(T, 5000, 1, rng1); key = f"tables__{un}"
                    else:
                        te, key = te2, "tables__task2"
                    keys += [key] * len(x); rr += list(te[3].cpu().numpy()); cc += list(te[4].cpu().numpy())
        else:
            ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz"))
            if b == "clevr":
                FU = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).query("keep"); n = len(ST["clevr_test_imgs"]); base = lambda un: un
            else:
                FU = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep"); n = 50; base = lambda un: un.rsplit("_s", 1)[0]
            U = FU.unit.tolist(); fam = dict(zip(FU.unit, FU.family))
            partner = lambda u: next(U[(U.index(u) + s) % len(U)] for s in range(len(U) // 2, len(U)) if fam[U[(U.index(u) + s) % len(U)]] != fam[u])
            q, ii = np.meshgrid(np.arange(10), np.arange(n), indexing="ij"); q, ii = q.ravel(), ii.ravel()
            for D in Ds:
                for un, x in D.groupby("unit", sort=False):
                    u = base(un); u = u if task == 1 else partner(u); keys += [f"{b}__{u}"] * len(x); rr += list(q); cc += list(ii)
        out[task] = (np.array(keys), np.array(rr), np.array(cc))
    return out


G2 = [("l2", ("0.3", "3")), ("lwf", ("0.3", "3")), ("ewc", ("1", "100", "10000")), ("si", ("0.1", "1", "10", "100")), ("icarl", ("0.1", "1"))]


def strat_gap(D, after, ref, w=None):
    """Pooled same-answer gap: sum_y pi_y [(after-ref | E, y) - (after-ref | R, y)], pi from exceptions; unit weights w."""
    w = np.ones(len(D)) if w is None else w
    ch = (D[after].astype(float) - D[ref].astype(float)).values; g = D.lab.values; y = D.y.values; out = 0.0; pis = []
    tot_e = (w * (g == 1)).sum()
    for yy in (0, 1):
        me, mr = (g == 1) & (y == yy), (g == 0) & (y == yy)
        if (w * me).sum() > 0 and (w * mr).sum() > 0:
            pi = (w * me).sum() / tot_e; pis.append(pi); out += pi * ((w * ch * me).sum() / (w * me).sum() - (w * ch * mr).sum() / (w * mr).sum())
    return out / sum(pis) if pis else np.nan


def boot_gap(D, after, ref):
    cl = D.cluster.values; u, inv = np.unique(cl, return_inverse=True)
    return np.nanpercentile([strat_gap(D, after, ref, np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)) for _ in range(NB)], [2.5, 97.5])


say("\n2. Continual learning: same-answer gaps with the saved labels vs with stable labels only (points; [95% CI])")
PRE = {"backward": [("clevr", m) for m in ("er_random_1000", "agem_random_1000", "gem_random_1000", "derpp_random_1000")],
       "forward": [(b, m) for b in ("tables", "clevr", "gqa") for m in ("derpp_random_1000", "l2")]}
res = {}
for b in ("tables", "clevr", "gqa"):
    Y = np.load(os.path.join(ROOT, "runs", "clm", f"y_{b}.npz")); C = cells(b)
    D1 = pd.concat([pd.read_csv(f) for f in Y["task1_files"]]).reset_index(drop=True); D2 = pd.concat([pd.read_csv(f) for f in Y["task2_files"]]).reset_index(drop=True)
    D1["y"], D2["y"] = Y["task1"], Y["task2"]
    a2, a1 = D2.groupby("unit").mean(numeric_only=True), D1.groupby("unit").mean(numeric_only=True)
    for fam, lams in G2:
        pick = {u: (max(ok, key=lambda l: a1.loc[u, f"{fam}_{l}"]) if (ok := [l for l in lams if a2.loc[u, f"{fam}_{l}"] >= a2.loc[u, "finetune"] - 0.02]) else lams[0]) for u in a2.index}
        for D in (D1, D2):
            D[fam] = 0.0
            for l in lams:
                m = D.unit.map(pick) == l; D.loc[m, fam] = D.loc[m, f"{fam}_{l}"].astype(float)
    for task, D in ((1, D1), (2, D2)):
        k, r, c = C[task]; st = np.full(len(D), np.nan)
        for kk in np.unique(k):
            i = np.where(k == kk)[0]; st[i] = label(kk, r[i], c[i])
        D["stable"] = st
        D["cluster"] = D.unit.str.rsplit("_s", n=1).str[0] if b == "gqa" else D.unit
        say(f"  {b} task {task}: saved exceptions that are stable exceptions {np.nanmean(D.stable[D.nonadd == 1] == 1):.3f}; "
            f"saved rule cells that are stable rule cells {np.nanmean(D.stable[D.nonadd == 0] == 0):.3f}; ambiguous {D.stable.isna().mean():.3f}")
    for kind, items in PRE.items():
        for bb, m in items:
            if bb != b:
                continue
            D, ref = (D1, "before") if kind == "backward" else (D2, "scratch")
            o = D.assign(lab=D.nonadd); s = D[D.stable.notna()].assign(lab=lambda x: x.stable.astype(int))
            res[(kind, b, m)] = [(strat_gap(X, m, ref), boot_gap(X, m, ref)) for X in (o, s)]
for (kind, b, m), ((g0, c0), (g1, c1)) in res.items():
    say(f"  {kind:8s} {b:6s} {m:18s} saved {100 * g0:+6.1f} [{100 * c0[0]:+5.1f},{100 * c0[1]:+5.1f}] | stable {100 * g1:+6.1f} [{100 * c1[0]:+5.1f},{100 * c1[1]:+5.1f}]")
okk = lambda t: t[0] <= -0.03 and t[1][1] < 0
nb = sum(okk(res[("backward", "clevr", m)][1]) for m in ("er_random_1000", "agem_random_1000", "gem_random_1000", "derpp_random_1000"))
nf = sum(okk(res[("forward", b, "derpp_random_1000")][1]) for b in ("tables", "clevr", "gqa"))
say(f"\nU1 (backward, CLEVR, stable labels): {nb}/4 replay methods keep gap <= -3 points with upper bound < 0 -> {'PASS' if nb >= 3 else 'FAIL'}")
say(f"U2 (forward, DER++, stable labels): {nf}/3 benchmarks keep gap <= -3 points with upper bound < 0 -> {'PASS' if nf >= 2 else 'FAIL'}")

say("\n3. VLM cells (Study C sub-tables): AUC for cells whose answer EMAP flips")
G = np.load(os.path.join(ROOT, "runs", "vlm3", "grids.npz")); SU = pd.read_csv(os.path.join(ROOT, "runs", "vlm3", "selected2.csv"))
for mname in ("idefics", "internvl", "llava", "qwen"):
    S = np.load(os.path.join(ROOT, "runs", "vlm3", f"scores_{mname}.npz")); fl, lab, labs, mis, conf = [], [], [], [], []
    for cid, key, cols in zip(SU.cid, SU.key, SU.cols):
        cols = json.loads(cols); L = S[f"L__{key}"][:, cols]; E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean()
        fl += list(((L > 0) != (E > 0)).ravel()); st = label(f"vlm__{cid}", slice(None), slice(None)).ravel(); labs += list(st)
        lab += list(UQ[f"exc__vlm__{cid}"].ravel() | UQ[f"amb__vlm__{cid}"].ravel()); mis += list(G[f"misfit__{key}"][:, cols].ravel()); conf += list(-np.abs(L).ravel())
    fl, labs, lab, mis, conf = map(np.array, (fl, labs, lab, mis, conf)); k = ~np.isnan(labs)
    say(f"  {mname:9s} stable exception label {roc_auc_score(fl[k], labs[k]):.3f} (on {k.mean():.2f} of cells) | exception in any optimal rule {roc_auc_score(fl, lab):.3f} | "
        f"graded misfit {roc_auc_score(fl, mis):.3f} | low confidence {roc_auc_score(fl, conf):.3f}")
open(os.path.join(ROOT, "paper", "results_rev_unique.txt"), "w").write("\n".join(lines) + "\n")
