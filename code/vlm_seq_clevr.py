"""Six-task CLEVR continual learning with a VLM (paper/preregistration_guide.md). Same learner as vlm_seq.py (Qwen2.5-VL-
3B-Instruct, LoRA r 16, 336 x 336 images, yes / no log-odds, 300 steps x 8 per task, lr 1e-4).
Tasks: the 40 CLEVR units of the CL studies, grouped by question family into six tasks (families sorted, each added to
the task with the fewest units). Cells: only natural cells (the question is well-posed on the image and its natural
answer equals the table's), as in the earlier VLM test; training cells on the 700 training images, test cells on the
300 test images (per task 500 exception and 500 rule cells, seed 51; fewer if not available). Labels: the exact rule /
exception labels already computed for these cells.
Methods from the shared task-1 model: finetune; online EWC (100, 10000); LwF (1); ER; DER++; class-balanced ER
(memory 50% yes); LoRA-expansion (a new LoRA per task, earlier LoRAs frozen and kept active, with ER memory; the
architectural analogue of DER-expansion). Memory 1000 split over past tasks; 4 replayed examples per step. Plus a task-
only LoRA per task. Saves per test cell: answer, type, correct after its task (own), after task 6 (final), task-only.
Usage: python vlm_seq_clevr.py <order 0|1> [--smoke]"""
import argparse, io, os, time, zipfile
import numpy as np, pandas as pd, torch
from PIL import Image
import vlm_seq as VS
ROOT = VS.ROOT; OUT = os.path.join(ROOT, "runs", "clevr_vlm"); os.makedirs(OUT, exist_ok=True); dev = "cuda"
METHODS = [("finetune", "finetune", 0.0), ("ewc_100", "ewc", 100.0), ("ewc_10000", "ewc", 1e4), ("lwf_1", "lwf", 1.0), ("er", "er", 0.0),
           ("derpp", "derpp", 0.0), ("er_classbal", "er_classbal", 0.0), ("lora_expand", "lora_expand", 0.0)]


def tasks_and_cells():
    from vlm_test import units
    U, imgs, S = units(); TS = np.load(os.path.join(ROOT, "runs", "clx", "train_structure.npz")); ST = np.load(os.path.join(ROOT, "runs", "unrepair", "structure.npz"))
    fam = {u: int(u.split("_")[0]) for u in TS["units"]}; groups = [[] for _ in range(6)]
    for f in sorted(set(fam.values())):
        min(groups, key=len).extend([u for u in TS["units"] if fam[u] == f])
    itr, ite = TS["train_imgs"], ST["clevr_test_imgs"]; tasks = {}
    for k, us in enumerate(groups):
        out = {"train": [], "test": []}
        for u in us:
            d = U[u]
            for sp, cols, N in (("train", itr, TS[f"nonadd__{u}"]), ("test", ite, ST[f"nonadd__clevr__{u}"])):
                for q in range(10):
                    for j, c in enumerate(cols):
                        if d["native"][q, c] == d["T"][q, c]:
                            out[sp].append((S[imgs[c]]["image_filename"], f"{d['text'][q]} Answer with yes or no.", int(d["T"][q, c]), int(N[q, j]), u, f"{c}_{u}_{q}"))
        tasks[str(k)] = {sp: pd.DataFrame(v, columns=["img", "text", "y", "nonadd", "unit", "cell"]).reset_index(drop=True) for sp, v in out.items()}
    return tasks


class ClevrLearner(VS.Learner):
    def __init__(self, path):
        super().__init__(path); self.zip = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip"))

    def img(self, m):
        if m not in self.cache:
            self.cache[m] = Image.open(io.BytesIO(self.zip.read(f"CLEVR_v1.0/images/val/{m}"))).convert("RGB").resize((336, 336), Image.BICUBIC)
        return self.cache[m]

    def expand(self, name):
        """Add a new LoRA, keep all earlier ones active but frozen; train only the new one."""
        from peft import LoraConfig
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        self.v.model.add_adapter(name, cfg); self.names.append(name); self.v.model.base_model.set_adapter(self.names)
        for n, p in self.v.model.named_parameters():
            p.requires_grad_(f".{name}." in n)
        self.P = {n: p for n, p in self.v.model.named_parameters() if f".{name}." in n}

    def reset_expansion(self):
        for nm in self.names[1:]:
            self.v.model.delete_adapter(nm)
        self.names = ["default"]; self.v.model.base_model.set_adapter("default")
        for n, p in self.v.model.named_parameters():
            p.requires_grad_(".default." in n and "lora_" in n)
        self.P = {n: p for n, p in self.v.model.named_parameters() if "lora_" in n and ".default." in n}


def classbal_memory(D, m, rng):
    y1, y0 = D.index[D.y == 1], D.index[D.y == 0]; k1 = min(m // 2, len(y1)); k0 = m - k1
    a, b = rng.choice(y1, k1, replace=False), rng.choice(y0, k0, replace=False); return D.loc[[x for p in zip(a, b) for x in p] + list(b[len(a):])].copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("order", type=int); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); tasks = tasks_and_cells(); order = [str(x) for x in np.random.default_rng(60 + a.order).permutation(6)]; seed = 31
    r51 = np.random.default_rng(51); te = {}
    for tk in order:
        E = tasks[tk]["test"]; ex, ru = E[E.nonadd == 1], E[E.nonadd == 0]
        te[tk] = pd.concat([ex.loc[r51.choice(ex.index, min(500, len(ex)), replace=False)], ru.loc[r51.choice(ru.index, min(500, len(ru)), replace=False)]]).reset_index(drop=True)
    if a.smoke:
        VS.STEPS = 6; te = {tk: d.sample(40, random_state=0).reset_index(drop=True) for tk, d in te.items()}
    for tk in order:
        print(tk, "train", len(tasks[tk]["train"]), "test", len(te[tk]), "exceptions %.2f" % te[tk].nonadd.mean(), "yes %.2f" % te[tk].y.mean(), flush=True)
    L = ClevrLearner(a.model); L.names = ["default"]; rows = {tk: te[tk][["unit", "nonadd", "y", "cell"]].assign(order=a.order, task=tk, pos=order.index(tk) + 1) for tk in order}; t0 = time.time()
    path = os.path.join(OUT, f"order_{a.order}{'_smoke' if a.smoke else ''}.csv.gz"); done = set()
    if a.resume:   # reuse saved columns (zero-shot, task-only, finished methods); rows are in the same order (same sampling)
        P = pd.read_csv(path)
        for tk in order:
            x = P[P.task.astype(str) == tk].reset_index(drop=True); assert (x.cell.values == rows[tk].cell.values).all(), tk
            for c in x.columns:
                if c not in rows[tk].columns:
                    rows[tk][c] = x[c].values
        done = {c[:-7] for c in P.columns if c.endswith("__final")}; print("resuming; finished:", sorted(done), flush=True)
    else:
        for tk in order:
            rows[tk]["zeroshot"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        for tk in order:
            L.load(L.init); L.train(tasks[tk]["train"], np.random.default_rng(seed + 500), log=f"scratch {tk}"); rows[tk]["scratch"] = (L.scores(te[tk]) > 0) == te[tk].y.values
    L.load(L.init); L.train(tasks[order[0]]["train"], np.random.default_rng(seed), log=f"task1 {order[0]}"); s1 = L.state()
    own1 = (L.scores(te[order[0]]) > 0) == te[order[0]].y.values; fi1 = L.fisher(tasks[order[0]]["train"]); print(f"task 1 done ({time.time() - t0:.0f}s)", flush=True)
    if a.resume:
        print(f"retrained task-1 model agrees with the saved one on {np.mean(own1 == rows[order[0]]['finetune__own'].values):.4f} of task-1 test cells "
              f"(accuracy now {own1.mean():.4f}, saved {rows[order[0]]['finetune__own'].mean():.4f})", flush=True)
    for name, kind, lam in METHODS:
        if name in done:
            continue
        L.load(s1); rng = np.random.default_rng(seed + 1); rows[order[0]][f"{name}__own"] = own1; mems = []; fi = {k: v.clone() for k, v in fi1.items()}; anchor = L.state()
        base = {"er_classbal": "er", "lora_expand": "er"}.get(kind, kind)
        def add(tk, k):
            nonlocal mems
            if base in ("er", "derpp"):
                r = np.random.default_rng(seed * 100 + order.index(tk)); D = tasks[tk]["train"]
                M = classbal_memory(D, MEMK(k), r) if kind == "er_classbal" else VS.memory(D, "random", MEMK(k), r); M["logit"] = L.scores(M)
                mems = [x.iloc[:MEMK(k)] for x in mems] + [M]
        add(order[0], 1)
        for pos, tk in enumerate(order[1:], start=2):
            if kind == "lora_expand":
                L.expand(f"t{pos}")
            L.train(tasks[tk]["train"], rng, base, lam, pd.concat(mems).reset_index(drop=True) if mems else None, fi, anchor, log=f"{name} {tk}")
            if kind == "ewc":
                f2 = L.fisher(tasks[tk]["train"]); fi = {k: fi[k] + f2[k] for k in fi}
            anchor = L.state(); add(tk, pos); rows[tk][f"{name}__own"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        for tk in order:
            rows[tk][f"{name}__final"] = (L.scores(te[tk]) > 0) == te[tk].y.values
        if kind == "lora_expand":
            L.reset_expansion()
        print(f"{name} done ({time.time() - t0:.0f}s)", flush=True)
        pd.concat(rows.values()).to_csv(path, index=False)
    print("DONE", a.order, flush=True)


MEMK = lambda k: VS.MEM // k

if __name__ == "__main__":
    main()
