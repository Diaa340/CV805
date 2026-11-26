"""VLM scale test (paper/preregistration_vlm.md): are a large VLM's errors concentrated on the non-additive cells of
CLEVR's exact answer tables? Units = step-2 CLEVR units (10 questions x 1,000 val images, set-lifted tables). A cell is
non-additive if the best additive fit [u_q + w_img > 0] of the full table gets it wrong. The VLM is only asked cells
where the native CLEVR question is well-posed (native answer = set-lifted answer). Answer = argmax of the next-token
probability of Yes vs No. Output: runs/clevr2/vlm_<model>.csv"""
import argparse, io, json, os, time, zipfile
import numpy as np, pandas as pd, torch
from PIL import Image
from clevr_exec import execute_total, load, safe
from step2_train import additive_pred
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2"); dev = "cuda"


def units():
    """Rebuild each step-2 unit's programs and question texts (same deterministic selection as step2_build.py)."""
    Q, S = load("val"); imgs = np.sort(np.random.default_rng(11).choice(len(S), 1000, replace=False)); fams = {}
    for q in Q:
        if q["answer"] in ("yes", "no"):
            fams.setdefault(q["question_family_index"], {})[json.dumps(q["program"])] = q["question"]
    U = pd.read_csv(os.path.join(OUT, "families.csv")); U = U[U.keep]; Z = np.load(os.path.join(OUT, "data.npz")); out = {}
    for u in U.unit:
        f, g = map(int, u.split("_")); progs = sorted(fams[f])
        pick = np.random.default_rng(12 + f).choice(len(progs), min(60, len(progs)), replace=False)[g * 10:(g + 1) * 10]
        P = [json.loads(progs[p]) for p in pick]; txt = [fams[f][progs[p]] for p in pick]
        T = np.array([[int(execute_total(p, S[k])[0] == "yes") for k in imgs] for p in P])
        assert (T == Z[f"T__{u}"]).all(), u
        nat = np.array([[{"yes": 1, "no": 0, None: -1}[safe(p, S[k])] for k in imgs] for p in P])
        out[u] = dict(T=T, native=nat, text=txt)
    return out, imgs, S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--per_unit", type=int, default=250); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--limit_units", type=int, default=0); ap.add_argument("--out", default=None); args = ap.parse_args()
    U, imgs, S = units(); rng = np.random.default_rng(31); rows = []
    for u, d in U.items():
        A = additive_pred(d["T"]); valid = np.argwhere((d["native"] >= 0) & (d["native"] == d["T"]))
        pick = valid[rng.choice(len(valid), min(args.per_unit, len(valid)), replace=False)] if len(valid) else []
        for q, i in pick:
            rows.append(dict(unit=u, q=int(q), img=int(i), image_index=int(imgs[i]), text=d["text"][q], truth=int(d["T"][q, i]),
                             nonadd=int(A[q, i] != d["T"][q, i]), add_pred=int(A[q, i])))
    D = pd.DataFrame(rows)
    if args.limit_units:
        D = D[D.unit.isin(sorted(D.unit.unique())[:args.limit_units])]
    print(f"{len(D)} cells from {D.unit.nunique()} units; non-additive share {D.nonadd.mean():.3f}", flush=True)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model); proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer; ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    YES, NO = ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"])
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip")); cache = {}
    def image(k):
        if k not in cache:
            cache[k] = Image.open(io.BytesIO(z.read(f"CLEVR_v1.0/images/val/{S[k]['image_filename']}"))).convert("RGB")
        return cache[k]
    D = D.sort_values("image_index").reset_index(drop=True); lp = np.zeros(len(D)); t0 = time.time()
    for b in range(0, len(D), args.bs):
        B = D.iloc[b:b + args.bs]
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{t} Answer with yes or no."}]}] for t in B.text]
        texts = [proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        inp = proc(text=texts, images=[image(k) for k in B.image_index], padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            lo = model(**inp).logits[:, -1].float().log_softmax(-1)
        lp[b:b + len(B)] = (torch.logsumexp(lo[:, YES], 1) - torch.logsumexp(lo[:, NO], 1)).cpu().numpy()
        if b % (args.bs * 50) == 0:
            print(f"  {b + len(B)}/{len(D)} ({time.time() - t0:.0f}s)", flush=True)
    D["logodds_yes"] = lp; D["pred"] = (lp > 0).astype(int); D["err"] = (D.pred != D.truth).astype(int)
    name = args.out or args.model.split("/")[-1]
    D.drop(columns=["text"]).to_csv(os.path.join(OUT, f"vlm_{name}.csv"), index=False)
    print(f"done: accuracy {1 - D.err.mean():.3f}; error additive cells {D[D.nonadd == 0].err.mean():.3f}, non-additive {D[D.nonadd == 1].err.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
