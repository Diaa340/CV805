"""Six-task CLOVE-scene continual learning with a VLM (paper/preregistration_clove_vlm.md).
Qwen2.5-VL-3B-Instruct with LoRA (r 16) on the language model; images resized to 336 x 336; prompt "<question> Answer
with yes or no."; the score is the log-odds of yes vs no at the next token, trained with a logistic loss.
Each task: 300 steps of 8 examples (lr 1e-4) from its training cells. Task 1 is trained once; from it every method
learns tasks 2-6: finetune; EWC (100, 10000; online); LwF (1; teacher = model after the previous task, its scores on
the examples the task will train on, computed before the task); ER, DER++ (alpha 0.5 on stored scores), DER++ rule+exceptions
(memory 1000 split equally over past tasks; 4 replayed examples per step). Plus a LoRA trained on each task alone.
Test cells per task: all exception cells and 600 rule cells (seed 51). Saves per test cell: correct right after its
task (own), after task 6 (final), and for the task-only LoRA (scratch).
Usage: python vlm_seq.py <order index> [--smoke]"""
import argparse, os, pickle, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clove_vlm"); os.makedirs(OUT, exist_ok=True); dev = "cuda"
ORDERS = [list("abcdef")] + pickle.load(open(os.path.join(ROOT, "data", "raw", "clove", "scene_perm.pkl"), "rb"))["sampled_perm"]
STEPS, BSZ, RB, LR, MEM, NRULE = 300, 8, 4, 1e-4, 1000, 600
METHODS = [("finetune", "finetune", 0.0), ("ewc_100", "ewc", 100.0), ("ewc_10000", "ewc", 1e4), ("lwf_1", "lwf", 1.0), ("er", "er", 0.0),
           ("derpp", "derpp", 0.0), ("derpp_ruleexc", "derpp_ruleexc", 0.0)]


def cells():
    U = np.load(os.path.join(ROOT, "runs", "clove", "units.npz")); S = np.load(os.path.join(ROOT, "runs", "clove", "structure.npz"))
    K = pd.read_csv(os.path.join(ROOT, "runs", "clove", "units.csv")).query("keep"); tasks = {}
    for tk, g in K.groupby("task"):
        out = {"train": [], "test": []}
        for u in g.unit:
            T, im, tx = U[f"T__{u}"], U[f"imgs__{u}"], U[f"text__{u}"]
            for sp, sl in (("train", range(150)), ("test", range(150, 200))):
                N = S[f"nonadd__{sp}__{u}"]
                for q in range(10):
                    for j, c in enumerate(sl):
                        out[sp].append((str(im[c]), f"{tx[q]} Answer with yes or no.", int(T[q, c]), int(N[q, j]), u, f"{im[c]}_{u}_{q}"))
        tasks[tk] = {sp: pd.DataFrame(v, columns=["img", "text", "y", "nonadd", "unit", "cell"]) for sp, v in out.items()}
    return tasks


class Learner:
    def __init__(self, path):
        from peft import LoraConfig, get_peft_model
        from vlm_forget import VLM
        self.v = VLM(path); cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        self.v.model = get_peft_model(self.v.model, cfg); self.P = {n: p for n, p in self.v.model.named_parameters() if "lora_" in n}; self.cache = {}
        self.init = self.state()

    def state(self):
        return {n: p.detach().clone() for n, p in self.P.items()}

    def load(self, st):
        with torch.no_grad():
            for n, p in self.P.items():
                p.copy_(st[n])

    def img(self, m):
        if m not in self.cache:
            self.cache[m] = Image.open(os.path.join(ROOT, "data", "raw", "clove", "images", f"{m}.jpg")).convert("RGB").resize((336, 336), Image.BICUBIC)
        return self.cache[m]

    def score(self, imgs, texts):
        v = self.v; v.tok.padding_side = "left"
        inp = v.proc(text=[v.prompt(t) for t in texts], images=[self.img(m) for m in imgs], padding=True, return_tensors="pt").to(dev)
        lo = v.model(**inp).logits[:, -1].float().log_softmax(-1)
        return torch.logsumexp(lo[:, v.YES], 1) - torch.logsumexp(lo[:, v.NO], 1)

    @torch.no_grad()
    def scores(self, D, bs=32):
        self.v.model.eval(); return np.concatenate([self.score(D.img.values[b:b + bs], D.text.values[b:b + bs]).cpu().numpy() for b in range(0, len(D), bs)])

    def fisher(self, D, n=200):
        self.v.model.eval(); Fi = {k: torch.zeros_like(p) for k, p in self.P.items()}
        for b in range(0, n, BSZ):
            d = D.iloc[b:b + BSZ]; self.v.model.zero_grad()
            F.binary_cross_entropy_with_logits(self.score(d.img.values, d.text.values), torch.tensor(d.y.values, device=dev).float()).backward()
            for k, p in self.P.items():
                Fi[k] += p.grad.detach() ** 2 * BSZ / n
        return Fi

    def train(self, D, rng, kind="finetune", lam=0.0, mem=None, fi=None, anchor=None, log=""):
        """New-task examples are drawn up front; for LwF the teacher (current model) scores only those examples."""
        idx = rng.integers(0, len(D), (STEPS, BSZ)); teacher = None
        if kind == "lwf":
            u = np.unique(idx); ts = dict(zip(u, self.scores(D.iloc[u]))); teacher = np.vectorize(ts.get)(idx)
        self.v.model.train(); opt = torch.optim.AdamW(list(self.P.values()), lr=LR, weight_decay=0.0); t0 = time.time()
        for s in range(STEPS):
            i = idx[s]; d = D.iloc[i]; sc = self.score(d.img.values, d.text.values)
            loss = F.binary_cross_entropy_with_logits(sc, torch.tensor(d.y.values, device=dev).float())
            if kind == "lwf":
                tq = torch.tensor(teacher[s], device=dev).float(); loss = loss + lam * F.binary_cross_entropy_with_logits(sc / 2, torch.sigmoid(tq / 2)) * 4
            if kind == "ewc":
                loss = loss + lam * sum((fi[k] * (p - anchor[k]) ** 2).sum() for k, p in self.P.items())
            if kind in ("er", "derpp", "derpp_ruleexc"):
                j = rng.integers(0, len(mem), RB); m = mem.iloc[j]; ms = self.score(m.img.values, m.text.values)
                loss = loss + F.binary_cross_entropy_with_logits(ms, torch.tensor(m.y.values, device=dev).float())
                if kind != "er":
                    loss = loss + 0.5 * F.mse_loss(ms, torch.tensor(m.logit.values, device=dev).float())
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 100 == 0:
                print(f"  {log} step {s}/{STEPS} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        self.v.model.eval()


def memory(D, kind, m, rng):
    if kind == "derpp_ruleexc":   # half from exception cells (one per cell in turn), half random; interleaved so prefixes keep the mix
        ex = D[D.nonadd == 1]; byc = {c: list(rng.permutation(ex.index[ex.cell == c])) for c in ex.cell.unique()}; pick = []
        while len(pick) < m // 2 and any(byc.values()):
            for c in list(byc):
                if byc[c] and len(pick) < m // 2:
                    pick.append(byc[c].pop())
        rest = rng.choice(np.setdiff1d(D.index, pick), m - len(pick), replace=False)
        idx = [x for p in zip(pick, rest) for x in p] + list(pick[len(rest):]) + list(rest[len(pick):])
    else:
        idx = list(rng.choice(D.index, m, replace=False))
    return D.loc[idx].copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("order", type=int); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    a = ap.parse_args(); global STEPS
    order, seed = ORDERS[a.order], 31; tasks = cells(); r51 = np.random.default_rng(51); te = {}
    for tk in order:
        E = tasks[tk]["test"]; rule = E[E.nonadd == 0]; te[tk] = pd.concat([E[E.nonadd == 1], rule.loc[r51.choice(rule.index, NRULE, replace=False)]]).reset_index(drop=True)
        tasks[tk]["train"] = tasks[tk]["train"].reset_index(drop=True)
    if a.smoke:
        STEPS = 6; te = {tk: d.iloc[:40] for tk, d in te.items()}
    L = Learner(a.model); rows = {tk: te[tk][["unit", "nonadd", "y", "cell"]].assign(order=a.order, task=tk, pos=order.index(tk) + 1) for tk in order}; t0 = time.time()
    for tk in order:
        rows[tk]["zeroshot"] = (L.scores(te[tk]) > 0) == te[tk].y.values
    for tk in order:   # task-only LoRA (forward transfer reference)
        L.load(L.init); L.train(tasks[tk]["train"], np.random.default_rng(seed + 500), log=f"scratch {tk}"); rows[tk]["scratch"] = (L.scores(te[tk]) > 0) == te[tk].y.values
    L.load(L.init); L.train(tasks[order[0]]["train"], np.random.default_rng(seed), log=f"task1 {order[0]}"); s1 = L.state()
    own1 = (L.scores(te[order[0]]) > 0) == te[order[0]].y.values; fi1 = L.fisher(tasks[order[0]]["train"]); print(f"task 1 done ({time.time() - t0:.0f}s)", flush=True)
    for name, kind, lam in METHODS:
        L.load(s1); rng = np.random.default_rng(seed + 1); rows[order[0]][f"{name}__own"] = own1; mems = []; fi = {k: v.clone() for k, v in fi1.items()}; anchor = L.state()
        def add(tk, k):
            nonlocal mems
            if kind in ("er", "derpp", "derpp_ruleexc"):
                M = memory(tasks[tk]["train"], kind, MEM // k, np.random.default_rng(seed * 100 + order.index(tk))); M["logit"] = L.scores(M); mems = [x.iloc[:MEM // k] for x in mems] + [M]
        add(order[0], 1)
        for pos, tk in enumerate(order[1:], start=2):
            D = tasks[tk]["train"]
            L.train(D, rng, kind, lam, pd.concat(mems).reset_index(drop=True) if mems else None, fi, anchor, log=f"{name} {tk}")
            if kind == "ewc":
                f2 = L.fisher(D); fi = {k: fi[k] + f2[k] for k in fi}
            anchor = L.state(); add(tk, pos); rows[tk][f"{name}__own"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        for tk in order:
            rows[tk][f"{name}__final"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(rows.values()).to_csv(os.path.join(OUT, f"order_{a.order}{'_smoke' if a.smoke else ''}.csv.gz"), index=False)
    print("DONE", a.order, flush=True)


if __name__ == "__main__":
    main()
