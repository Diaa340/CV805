"""EMAP (Hessel & Lee 2020) on three settings (paper/preregistration_emap.md).
EMAP of a model f on a set of n items {(a_i, b_i)}: f_hat(a_i, b_i) = mean_j f(a_i, b_j) + mean_j f(a_j, b_i) - mean_jk f(a_j, b_k),
applied to the logits (log-odds for the VLM); accuracy of argmax f_hat vs the true labels.
 --part vlm:    Qwen2.5-VL-7B on the 24 CLEVR units whose questions are well-posed on every image (complete 10 x 100 tables,
                100 held-out images); compared with the VLM's own accuracy and the best additive fit of the same sub-table.
 --part bench:  frozen-feature full models (as in additive_estimator.py) on the 7 benchmarks; EMAP on 2,000 test items.
 --part tables: full models on the 63 label tables (as in the label-table test); EMAP on 2,000 test pairs.
Output: runs/emap_<part>.csv"""
import argparse, io, os, time, zipfile
import numpy as np, pandas as pd, torch, torch.nn.functional as F
dev = "cuda"; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def emap_diag(ga, gb, head, chunk=256):
    """ga, gb: per-item codes of modality a and b (n x d); head(ga_i, gb_j) -> logits. Returns f(i,i) and EMAP f_hat(i,i)."""
    n = len(ga); row = None; col = None; tot = 0
    for s in range(0, n, chunk):
        A = ga[s:s + chunk]  # f(a_s.., b_all): rows
        L = torch.stack([head(A[k:k + 1].expand(n, -1), gb) for k in range(len(A))])  # (c, n, C)
        r = L.mean(1); row = r if row is None else torch.cat([row, r]); tot = tot + L.sum((0, 1))
        cs = L.sum(0); col = cs if col is None else col + cs
    col = col / n; grand = tot / (n * n)
    diag = torch.stack([head(ga[i:i + 1], gb[i:i + 1])[0] for i in range(n)])
    return diag, row + col - grand


def part_vlm(out):
    from vlm_test import units
    from step2_train import additive_pred
    from transformers import AutoModelForImageTextToText, AutoProcessor
    U, imgs, S = units(); fam = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).set_index("unit")
    keep = [u for u in U if fam.lifted[u] <= 0.0]
    ite = np.random.default_rng(14).permutation(len(imgs))[700:800]  # 100 held-out images (step-2 test split)
    path = "Qwen/Qwen2.5-VL-7B-Instruct"; proc = AutoProcessor.from_pretrained(path); proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval(); tok = proc.tokenizer
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws}); YES, NO = ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"])
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip"))
    IM = {k: Image.open(io.BytesIO(z.read(f"CLEVR_v1.0/images/val/{S[imgs[k]]['image_filename']}"))).convert("RGB") for k in ite}
    for u in keep:
        T = U[u]["T"][:, ite]; nat = U[u]["native"][:, ite]; assert (nat == T).all(), u
        cells = [(q, j) for q in range(T.shape[0]) for j in range(len(ite))]; lo = np.zeros(len(cells)); t0 = time.time()
        for b in range(0, len(cells), 16):
            B = cells[b:b + 16]
            msgs = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{U[u]['text'][q]} Answer with yes or no."}]}],
                                             add_generation_prompt=True, tokenize=False) for q, _ in B]
            inp = proc(text=msgs, images=[IM[ite[j]] for _, j in B], padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                l = model(**inp).logits[:, -1].float().log_softmax(-1)
            lo[b:b + len(B)] = (torch.logsumexp(l[:, YES], 1) - torch.logsumexp(l[:, NO], 1)).cpu().numpy()
        Lg = lo.reshape(T.shape); Eh = Lg.mean(1, keepdims=True) + Lg.mean(0, keepdims=True) - Lg.mean()
        Tt = np.where(T == 1, 1.0, -1.0); Et = Tt.mean(1, keepdims=True) + Tt.mean(0, keepdims=True) - Tt.mean()
        A = additive_pred(T)
        r = dict(unit=u, final=fam.final[u], acc_vlm=float(((Lg > 0) == T).mean()), acc_emap_vlm=float(((Eh > 0) == T).mean()),
                 acc_emap_truth=float(((Et > 0) == T).mean()), ceiling=float((A == T).mean()), yes_share=float(T.mean()), sec=time.time() - t0)
        pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); print(r, flush=True)


def part_bench(seeds, out, n=2000):
    from additive_estimator import fit
    from route_exps import Net
    for name in ("avmnist", "cifar2v", "clevr", "cremad", "food101", "n24news", "vqa"):
        Z = np.load(os.path.join(ROOT, "data", "feats", f"{name}.npz"), allow_pickle=True); s = Z["split"]
        f1, f2 = Z["f1"].astype(np.float32), Z["f2"].astype(np.float32); y = pd.factorize(Z["y"])[0]
        nz = lambda f: (f - f[s == 0].mean(0)) / (f[s == 0].std(0) + 1e-6)
        X1, X2, Y = torch.tensor(nz(f1), device=dev), torch.tensor(nz(f2), device=dev), torch.tensor(y, device=dev); C = int(y.max()) + 1
        tr = tuple(v[torch.tensor(s == 0, device=dev)] for v in (X1, X2, Y)); te = tuple(v[torch.tensor(s == 1, device=dev)] for v in (X1, X2, Y))
        for seed in seeds:
            torch.manual_seed(seed); net = fit(Net(X1.shape[1], X2.shape[1], C).to(dev), tr).eval()
            idx = torch.tensor(np.random.default_rng(seed).choice(len(te[2]), min(n, len(te[2])), replace=False), device=dev)
            with torch.no_grad():
                ga, gb = net.e1(te[0][idx]), net.e2(te[1][idx])
            d, e = emap_diag(ga, gb, lambda a, b: net.head(torch.cat([a, b], 1)))
            yy = te[2][idx]; r = dict(data=name, seed=seed, n=len(idx), acc_full=(d.argmax(1) == yy).float().mean().item(), acc_emap=(e.argmax(1) == yy).float().mean().item())
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); print(r, flush=True)


def part_tables(seeds, out, n=2000):
    from route_exps import Net, train
    from tables_train import table_pairs
    S = np.load(os.path.join(ROOT, "runs", "tables", "selected.npz"))
    for k in range(len(S["T"])):
        T = S["T"][k].astype(int)
        for seed in seeds:  # same data and model as the label-table test (tables_train.py)
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            tr, te = table_pairs(T, 50000, 0, rng), table_pairs(T, 20000, 1, rng)
            net = Net(512, 512, 2).to(dev); train(net, tr[:3], 30, 1e-3); net.eval()
            idx = torch.arange(n, device=dev)
            with torch.no_grad():
                ga, gb = net.e1(te[0][idx]), net.e2(te[1][idx])
            d, e = emap_diag(ga, gb, lambda a, b: net.head(torch.cat([a, b], 1)))
            yy = te[2][idx]; p = yy.float().mean().item()
            r = dict(table=k, seed=seed, acc_full=(d.argmax(1) == yy).float().mean().item(), acc_emap=(e.argmax(1) == yy).float().mean().item(), acc_maj=max(p, 1 - p))
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False); print(r, flush=True)


if __name__ == "__main__":
    from PIL import Image  # noqa: E402  (used by part_vlm)
    globals()["Image"] = Image
    ap = argparse.ArgumentParser(); ap.add_argument("--part", required=True); ap.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25])
    ap.add_argument("--out", default=None); a = ap.parse_args(); out = os.path.join(ROOT, "runs", a.out or f"emap_{a.part}.csv")
    {"vlm": lambda: part_vlm(out), "bench": lambda: part_bench(a.seeds, out), "tables": lambda: part_tables(a.seeds, out)}[a.part]()
