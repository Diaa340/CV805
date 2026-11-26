"""Exception-anchored continual learning on CLOVE-scene (experiment_guide_exception_anchor.md). Imports vlm_seq unchanged:
same cells, model, LoRA, test sample, 300 steps x 8 per task, lr 1e-4, task-1 model shared by all arms, memory 1000 split
equally over past tasks with 4 replayed examples per step. Arms:
  eap_0.6          finetune + (lam/2) mean (y - sigmoid(score))^2 on 4 cells per step from a 1000-cell pool of past-task
                   TRAINING exceptions (budget split as in vlm_seq.add)
  derpp_eap_0.6    DER++ with its usual random buffer, plus the same penalty on 4 exception cells per step drawn from the
                   exceptions inside that buffer (no extra storage)
  eapbal_0.6       eap_0.6 with an answer-balanced draw: 2 "yes" and 2 "no" anchor cells per step from the same pool
                   (eap_0.6 drifted toward "yes": it protected yes-exceptions and lost no-cells)
  derpp_poollogit  DER++ with its usual random buffer, plus DER++'s logit term on the eap pool: 0.5 MSE(score, stored score)
                   on 4 pool cells per step (anchor generator); each task's pool cells get their score right after that task,
                   as DER++ does for its buffer. No label term on the pool (fusion-net follow-up; extra storage)
  derpp_poollogit_sm  storage-matched derpp_poollogit: buffer and pool each get half the memory (500 + 500 = 1000 cells,
                   split over past tasks as usual), so total storage equals DER++'s
  ewcexc_10000     online EWC whose Fisher is computed on past-task training exceptions only (up to 200 per task)
  ewcrand_10000    control: the same with the Fisher on 200 random training cells of each task (vlm_seq's EWC takes the
                   first 200 rows, which are one unit's first two questions), so ewcexc vs ewcrand isolates exceptions
  baselines        finetune, ewc_10000, derpp: rerun so the new arms are paired with baselines from the same rebuild
Anchor cells are drawn from their own generator, so new-task batches (and DER++ replay draws) match the baselines.
Test cells never enter a pool: pools come from the training split (images 0-149), test cells from images 150-199.
Saves runs/clove_vlm_eap/order_<o>_seed_<s>[_<arms>].csv.gz (columns as vlm_seq, plus seed and <arm>__ownscore, the
score right after the cell's own task, for difficulty bins) and a pool log.
Usage: python vlm_seq_eap.py <order index> [--seed 31] [--arms a,b] [--smoke]"""
import argparse, os, time, zipfile
import numpy as np, pandas as pd, torch, torch.nn.functional as F
_zip = zipfile.ZipFile   # vlm_forget opens the CLEVR zip on import; only its CLEVR helpers use it, so allow it to be absent
zipfile.ZipFile = lambda f, *a, **k: None if str(f).endswith("CLEVR_v1.0.zip") and not os.path.exists(f) else _zip(f, *a, **k)
import vlm_forget   # noqa: F401  (imported here, under the patch; vlm_seq.Learner imports it later)
zipfile.ZipFile = _zip
import vlm_seq as VS
from vlm_seq import Learner, memory, dev
OUT = os.path.join(VS.ROOT, "runs", "clove_vlm_eap"); os.makedirs(OUT, exist_ok=True)
ARMS = {"finetune": ("finetune", 0.0, 0.0), "ewc_10000": ("ewc", 1e4, 0.0), "derpp": ("derpp", 0.0, 0.0),
        "eap_0.6": ("eap", 0.0, 0.6), "derpp_eap_0.6": ("derpp_eap", 0.0, 0.6), "ewcexc_10000": ("ewcexc", 1e4, 0.0),
        "ewcrand_10000": ("ewcrand", 1e4, 0.0), "eapbal_0.6": ("eapbal", 0.0, 0.6),
        "derpp_poollogit": ("derpp_poollogit", 0.0, 1.0), "derpp_poollogit_sm": ("derpp_poollogit", 0.0, 1.0)}
HALF = ("derpp_poollogit_sm",)   # arms whose buffer and pool share one MEM budget
REPLAY = ("derpp", "derpp_eap", "derpp_poollogit")


def exc_pool(D, m, rng):
    """Up to m training exception cells of one task."""
    ix = D.index[D.nonadd == 1]; return D.loc[rng.choice(ix, min(m, len(ix)), replace=False)].copy()


def fisher_exc(L, D, rng, n=200, exc_only=True):
    """vlm_seq.Learner.fisher on (up to) n random training exceptions of the task (exc_only=False: n random training
    cells of any kind; vlm_seq's own EWC uses D.iloc[:200], i.e. one unit's first two questions)."""
    E = D[D.nonadd == 1] if exc_only else D; E = E.loc[rng.permutation(E.index)[:n]].reset_index(drop=True); n = len(E) - len(E) % VS.BSZ
    return L.fisher(E.iloc[:n], n)


def train(L, D, rng, arng, kind, lam, mu, mem=None, fi=None, anchor=None, pool=None, log=""):
    """vlm_seq.Learner.train for finetune / ewc / derpp, plus the anchor penalty (weight mu) for eap and derpp_eap.
    rng is consumed exactly as in vlm_seq; the anchor cells come from arng."""
    idx = rng.integers(0, len(D), (VS.STEPS, VS.BSZ))
    L.v.model.train(); opt = torch.optim.AdamW(list(L.P.values()), lr=VS.LR, weight_decay=0.0); t0 = time.time()
    exc = mem[mem.nonadd == 1].reset_index(drop=True) if kind == "derpp_eap" else pool
    for s in range(VS.STEPS):
        d = D.iloc[idx[s]]; sc = L.score(d.img.values, d.text.values)
        loss = F.binary_cross_entropy_with_logits(sc, torch.tensor(d.y.values, device=dev).float())
        if kind in ("ewc", "ewcexc", "ewcrand"):
            loss = loss + lam * sum((fi[k] * (p - anchor[k]) ** 2).sum() for k, p in L.P.items())
        if kind in REPLAY:
            j = rng.integers(0, len(mem), VS.RB); m = mem.iloc[j]; ms = L.score(m.img.values, m.text.values)
            loss = loss + F.binary_cross_entropy_with_logits(ms, torch.tensor(m.y.values, device=dev).float())
            loss = loss + 0.5 * F.mse_loss(ms, torch.tensor(m.logit.values, device=dev).float())
        if mu > 0 and exc is not None and len(exc):
            if kind == "eapbal":   # answer-balanced: half of the anchor cells from each answer (the pool is ~85% yes)
                a = pd.concat([g.iloc[arng.integers(0, len(g), VS.RB // 2)] for _, g in exc.groupby("y")])
            else:
                a = exc.iloc[arng.integers(0, len(exc), VS.RB)]
            if kind == "derpp_poollogit":
                loss = loss + 0.5 * F.mse_loss(L.score(a.img.values, a.text.values), torch.tensor(a.logit.values, device=dev).float())
            else:
                p = torch.sigmoid(L.score(a.img.values, a.text.values))
                loss = loss + mu / 2 * ((torch.tensor(a.y.values, device=dev).float() - p) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            print(f"  {log} step {s}/{VS.STEPS} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
    L.v.model.eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("order", type=int); ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    a = ap.parse_args(); arms = a.arms.split(","); order, seed = VS.ORDERS[a.order], a.seed
    tasks = VS.cells(); r51 = np.random.default_rng(51); te = {}
    # Test exception labels from the archived runs (../cl_raw_bundle/clove_vlm): the exact-rule optimum is not unique and
    # the rebuilt MILP breaks ties differently in 21 of 43 units (same exception count in all 43). With these labels the
    # test sample is cell-for-cell the archived one. Training labels are the rebuilt optimum.
    arch = set(pd.read_csv(os.path.join(VS.ROOT, "runs", "clove", "test_exceptions_archived.csv")).cell)
    for tk in order:   # test sample exactly as vlm_seq.main
        E = tasks[tk]["test"]; E["nonadd"] = E.cell.isin(arch).astype(int); rule = E[E.nonadd == 0]; te[tk] = pd.concat([E[E.nonadd == 1], rule.loc[r51.choice(rule.index, VS.NRULE, replace=False)]]).reset_index(drop=True)
        tasks[tk]["train"] = tasks[tk]["train"].reset_index(drop=True)
        assert not set(tasks[tk]["train"].img) & set(te[tk].img), "train and test images overlap"
    if a.smoke:
        VS.STEPS = 6; te = {tk: d.iloc[:40] for tk, d in te.items()}
    tag = f"order_{a.order}_seed_{seed}{'_smoke' if a.smoke else ''}" + (f"_{'-'.join(arms)}" if arms != list(ARMS) else "")   # one file per arm subset
    L = Learner(a.model); rows = {tk: te[tk][["unit", "nonadd", "y", "cell"]].assign(order=a.order, seed=seed, task=tk, pos=order.index(tk) + 1) for tk in order}
    t0 = time.time(); plog = []
    for tk in order:
        rows[tk]["zeroshot"] = (L.scores(te[tk]) > 0) == te[tk].y.values
    L.load(L.init); L.train(tasks[order[0]]["train"], np.random.default_rng(seed), log=f"task1 {order[0]}"); s1 = L.state()
    sc1 = L.scores(te[order[0]]); own1 = (sc1 > 0) == te[order[0]].y.values; fi1 = L.fisher(tasks[order[0]]["train"]); print(f"task 1 done ({time.time() - t0:.0f}s)", flush=True)
    for name in arms:
        kind, lam, mu = ARMS[name]; L.load(s1); rng = np.random.default_rng(seed + 1); arng = np.random.default_rng(seed + 7)
        rows[order[0]][f"{name}__own"] = own1; rows[order[0]][f"{name}__ownscore"] = sc1; mems, pools = [], []; anchor = L.state()
        fi = fisher_exc(L, tasks[order[0]]["train"], np.random.default_rng(seed * 100 + 99), exc_only=kind == "ewcexc") if kind in ("ewcexc", "ewcrand") else {k: v.clone() for k, v in fi1.items()}

        def add(tk, k):
            nonlocal mems, pools
            D = tasks[tk]["train"]; mb = VS.MEM // 2 if name in HALF else VS.MEM
            if kind in REPLAY:   # same buffer and seeds as vlm_seq's DER++
                M = memory(D, "derpp", mb // k, np.random.default_rng(seed * 100 + order.index(tk))); M["logit"] = L.scores(M); mems = [x.iloc[:mb // k] for x in mems] + [M]
            if kind in ("eap", "eapbal", "derpp_poollogit"):
                P = exc_pool(D, mb // k, np.random.default_rng(seed * 100 + 50 + order.index(tk)))
                if kind == "derpp_poollogit":
                    P = P.assign(logit=L.scores(P))
                pools = [x.iloc[:mb // k] for x in pools] + [P]
        add(order[0], 1)
        for pos, tk in enumerate(order[1:], start=2):
            D = tasks[tk]["train"]; mem = pd.concat(mems).reset_index(drop=True) if mems else None; pool = pd.concat(pools).reset_index(drop=True) if pools else None
            src = mem[mem.nonadd == 1] if kind == "derpp_eap" else pool
            if src is not None:
                plog.append(dict(arm=name, task=tk, pos=pos, buffer=len(mem) if mem is not None else 0, anchor_cells=len(src),
                                 anchor_yes=round(src.y.mean(), 3) if len(src) else None, anchor_test_overlap=len(set(src.img) & set().union(*[set(te[t].img) for t in order]))))
            train(L, D, rng, arng, kind, lam, mu, mem, fi, anchor, pool, log=f"{name} {tk}")
            if kind == "ewc":
                f2 = L.fisher(D); fi = {k: fi[k] + f2[k] for k in fi}
            if kind in ("ewcexc", "ewcrand"):
                f2 = fisher_exc(L, D, np.random.default_rng(seed * 100 + 99 + pos), exc_only=kind == "ewcexc"); fi = {k: fi[k] + f2[k] for k in fi}
            anchor = L.state(); add(tk, pos); sc = L.scores(te[tk]); rows[tk][f"{name}__own"] = (sc > 0) == te[tk].y.values; rows[tk][f"{name}__ownscore"] = sc
        for tk in order:
            rows[tk][f"{name}__final"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(rows.values()).to_csv(os.path.join(OUT, f"{tag}.csv.gz"), index=False); pd.DataFrame(plog).to_csv(os.path.join(OUT, f"{tag}_pools.csv"), index=False)
    print("DONE", a.order, seed, flush=True)


if __name__ == "__main__":
    main()
