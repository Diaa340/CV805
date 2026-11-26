"""Exception curricula for a VLM (paper/preregistration_curric.md, step 2). Nonce-word CLEVR (vlm_seq_nonce.py): for each
of the six tasks, a fresh LoRA on Qwen2.5-VL-3B-Instruct is trained on that task alone (as the task-only model of the
nonce study: lr 1e-4, batch 8) under five conditions:
  uniform_300, misfit_300, loss_300, uniform_600, misfit_600
misfit: sampling weight 1 + 4 q, q = percentile (within the task) of the cell's graded misfit in its unit's training
answer table (natural cells; model-free). loss: per step, 16 uniform candidates scored without gradients, 8 kept with
probability proportional to their loss. Test cells: same construction as the nonce study (500 exception + 500 rule cells per task,
seed 51, drawn in task order 0-5; not the identical sample). All conditions share them. Saves per test cell: unit, answer, exception label, correct under each condition, and zero-shot.
Usage: python vlm_curric.py <task indices, e.g. 0,1,2> [--smoke]"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import vlm_seq as VS
import vlm_seq_nonce as VN
import vlm_seq_clevr as VC
from unrepair_structure import local_synred
OUT = os.path.join(VS.ROOT, "runs", "vlm_curric"); os.makedirs(OUT, exist_ok=True); dev = "cuda"; BSZ = 8
CONDS = [("uniform_300", "uniform", 300), ("misfit_300", "misfit", 300), ("loss_300", "loss", 300), ("uniform_600", "uniform", 600), ("misfit_600", "misfit", 600)]


def misfit_q(D):
    """Percentile within the task of each training cell's graded misfit in its unit's (partial) training table."""
    m = np.zeros(len(D))
    for u, x in D.groupby("unit"):
        q = x.cell.str.rsplit("_", n=1).str[1].astype(int).values; c = x.cell.str.split("_", n=1).str[0].astype(int).values
        qi, ci = np.unique(q, return_inverse=True)[1], np.unique(c, return_inverse=True)[1]
        T = -np.ones((qi.max() + 1, ci.max() + 1), int); T[qi, ci] = x.y.values; M = local_synred(T, (T >= 0).astype(float)); m[x.index] = M[qi, ci]
    return pd.Series(m).rank(pct=True).values


def train(L, D, kind, steps, q, seed):
    L.load(L.init); L.v.model.train(); opt = torch.optim.AdamW(list(L.P.values()), lr=VS.LR, weight_decay=0.0); rng = np.random.default_rng(seed)
    w = (1 + 4 * q) / (1 + 4 * q).sum()
    for s in range(steps):
        if kind == "misfit":
            i = rng.choice(len(D), BSZ, p=w)
        elif kind == "loss":
            cand = rng.integers(0, len(D), 2 * BSZ); d = D.iloc[cand]
            with torch.no_grad():
                L.v.model.eval(); l = F.binary_cross_entropy_with_logits(L.score(d.img.values, d.text.values), torch.tensor(d.y.values, device=dev).float(), reduction="none")
                L.v.model.train()
            p = (l + 1e-6).cpu().numpy(); i = cand[rng.choice(2 * BSZ, BSZ, replace=False, p=p / p.sum())]
        else:
            i = rng.integers(0, len(D), BSZ)
        d = D.iloc[i]; loss = F.binary_cross_entropy_with_logits(L.score(d.img.values, d.text.values), torch.tensor(d.y.values, device=dev).float())
        opt.zero_grad(); loss.backward(); opt.step()
    L.v.model.eval()


def main():
    tasks = [str(t) for t in sys.argv[1].split(",")]; smoke = "--smoke" in sys.argv; T = VN.nonce_tasks(); r51 = np.random.default_rng(51); te = {}
    for tk in sorted(T):   # 500 exception + 500 rule test cells per task, one generator, task order 0..5
        E = T[tk]["test"]; ex, ru = E[E.nonadd == 1], E[E.nonadd == 0]
        te[tk] = pd.concat([ex.loc[r51.choice(ex.index, min(500, len(ex)), replace=False)], ru.loc[r51.choice(ru.index, min(500, len(ru)), replace=False)]]).reset_index(drop=True)
    L = VC.ClevrLearner("Qwen/Qwen2.5-VL-3B-Instruct"); L.names = ["default"]; t0 = time.time()
    for tk in tasks:
        D = T[tk]["train"].reset_index(drop=True); q = misfit_q(D); X = te[tk].iloc[:40] if smoke else te[tk]
        R = X[["unit", "nonadd", "y", "cell"]].assign(task=tk); L.load(L.init); R["zeroshot"] = (L.scores(X) > 0) == X.y.values
        for name, kind, steps in CONDS:
            train(L, D, kind, 4 if smoke else steps, q, 31 + int(tk)); R[name] = (L.scores(X) > 0) == X.y.values
            print(tk, name, f"({time.time() - t0:.0f}s)", flush=True)
        R.to_csv(os.path.join(OUT, f"task_{tk}{'_smoke' if smoke else ''}.csv.gz"), index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
