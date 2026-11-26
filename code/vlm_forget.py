"""Do big models forget fusion first? (paper/preregistration_vlm_forget.md)
Qwen2.5-VL-3B-Instruct + LoRA on the language model, fine-tuned on an unrelated task (synthetic OCR) with checkpoints at
several drift levels; CLEVR cells (exact, model-free fusion labels from step 2) evaluated at each level; then two repair
arms at the largest drift that retrain only a small part on a little CLEVR train data: the vision-language projector
(merger) or a LoRA on the last 4 language layers. Writes runs/clevr2/forget_<tag>.csv (per cell log-odds at each stage)."""
import argparse, copy, glob, io, json, os, random, time, zipfile
import numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageFont
from clevr_exec import load
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "clevr2"); dev = "cuda"
FONTS = sorted(glob.glob("/usr/share/fonts/truetype/dejavu/*.ttf") + glob.glob("/usr/share/fonts/truetype/liberation/*.ttf")
               + glob.glob("/usr/share/fonts/truetype/ubuntu/*.ttf") + glob.glob("/usr/share/fonts/truetype/lato/*.ttf"))
ZIP = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip"))


def ocr_example(rng):
    words = ["".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"), rng.integers(3, 9)))
             for _ in range(rng.integers(1, 4))]
    bg = tuple(int(c) for c in rng.integers(0, 256, 3)); fg = tuple(255 - c if abs(255 - 2 * c) > 60 else (c + 128) % 256 for c in bg)
    im = Image.new("RGB", (448, 448), bg); d = ImageDraw.Draw(im); y = int(rng.integers(20, 120))
    for w in words:
        f = ImageFont.truetype(FONTS[int(rng.integers(len(FONTS)))], int(rng.integers(28, 60)))
        d.text((int(rng.integers(10, 120)), y), w, fill=fg, font=f); y += int(rng.integers(70, 110))
    return im, "What text is written in the image? Reply with the text only.", " ".join(words)


def clevr_train_examples(n, seed):
    Q, S = load("train"); rng = np.random.default_rng(seed); yn = [q for q in Q if q["answer"] in ("yes", "no")]
    out = []
    for k in rng.choice(len(yn), n, replace=False):
        q = yn[k]; im = Image.open(io.BytesIO(ZIP.read(f"CLEVR_v1.0/images/train/{q['image_filename']}"))).convert("RGB")
        out.append((im, f"{q['question']} Answer with yes or no.", q["answer"].capitalize()))
    return out


class VLM:
    def __init__(self, path):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(path); self.tok = self.proc.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).to(dev)
        ids = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})
        self.YES, self.NO = ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"])

    def prompt(self, text):
        return self.proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
                                             add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def logodds(self, images, texts, bs=16):
        self.model.eval(); self.tok.padding_side = "left"; out = []
        for b in range(0, len(texts), bs):
            inp = self.proc(text=[self.prompt(t) for t in texts[b:b + bs]], images=images[b:b + bs], padding=True, return_tensors="pt").to(dev)
            lo = self.model(**inp).logits[:, -1].float().log_softmax(-1)
            out.append((torch.logsumexp(lo[:, self.YES], 1) - torch.logsumexp(lo[:, self.NO], 1)).cpu().numpy())
        return np.concatenate(out)

    def train(self, examples, steps, lr, bs=8, params=None, log=""):
        """Supervised fine-tuning on (image, prompt, answer); loss on answer tokens only. params: trainable parameters."""
        self.model.train(); self.tok.padding_side = "right"
        opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr, weight_decay=0.0); rng = random.Random(0); t0 = time.time()
        for s in range(steps):
            B = rng.sample(examples, bs) if callable(getattr(examples, "__len__", None)) else [examples() for _ in range(bs)]
            prompts = [self.prompt(p) for _, p, _ in B]; full = [pp + a + "<|im_end|>" for pp, (_, _, a) in zip(prompts, B)]
            ims = [im for im, _, _ in B]
            inp = self.proc(text=full, images=ims, padding=True, return_tensors="pt").to(dev)
            plen = [len(self.proc(text=[pp], images=[im], return_tensors="pt")["input_ids"][0]) for pp, im in zip(prompts, ims)]
            lab = inp["input_ids"].clone(); lab[inp["attention_mask"] == 0] = -100
            for i, n in enumerate(plen):
                lab[i, :n] = -100
            loss = self.model(**inp, labels=lab).loss; opt.zero_grad(); loss.backward(); opt.step()
            if s % 100 == 0:
                print(f"  {log} step {s}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)


def eval_cells(vlm, D, S):
    ims = [Image.open(io.BytesIO(ZIP.read(f"CLEVR_v1.0/images/val/{S[k]['image_filename']}"))).convert("RGB") for k in D.image_index]
    return vlm.logodds(ims, [f"{t} Answer with yes or no." for t in D.text])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--levels", type=int, nargs="+", default=[250, 1000, 3000, 8000]); ap.add_argument("--ocr_lr", type=float, default=2e-4)
    ap.add_argument("--repair_steps", type=int, default=250); ap.add_argument("--n_add", type=int, default=7630)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--tag", default="3b"); args = ap.parse_args()
    from peft import LoraConfig, get_peft_model
    from vlm_test import units
    U, imgs, S = units(); C = pd.read_csv(os.path.join(OUT, "vlm_Qwen2.5-VL-7B-Instruct.csv"))
    C["text"] = [U[u]["text"][q] for u, q in zip(C.unit, C.q)]
    rng = np.random.default_rng(41); add = C[C.nonadd == 0]; D = pd.concat([C[C.nonadd == 1], add.iloc[rng.choice(len(add), min(args.n_add, len(add)), replace=False)]])
    if args.smoke:
        D = D.iloc[rng.choice(len(D), 64, replace=False)]; args.levels = [20]; args.repair_steps = 10
    D = D.sort_values("image_index").reset_index(drop=True)[["unit", "q", "img", "image_index", "truth", "nonadd", "text"]]
    vlm = VLM(args.model); D["lo_base"] = eval_cells(vlm, D, S); print(f"base accuracy {((D.lo_base > 0) == D.truth).mean():.3f}", flush=True)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    vlm.model = get_peft_model(vlm.model, cfg); vlm.model.print_trainable_parameters()
    orng = np.random.default_rng(7); done = 0
    for lev in args.levels:
        vlm.train(lambda: ocr_example(orng), lev - done, args.ocr_lr, params=list(vlm.model.parameters()), log=f"OCR->{lev}"); done = lev
        D[f"lo_ocr{lev}"] = eval_cells(vlm, D, S); print(f"after {lev} OCR steps accuracy {((D[f'lo_ocr{lev}'] > 0) == D.truth).mean():.3f}", flush=True)
        D.drop(columns=["text"]).to_csv(os.path.join(OUT, f"forget_{args.tag}.csv"), index=False)
    drifted = {k: v.detach().clone() for k, v in vlm.model.state_dict().items()}
    rep = clevr_train_examples(2000, 5)
    for arm in ("projector", "last4"):
        vlm.model.load_state_dict(drifted)
        for p in vlm.model.parameters():
            p.requires_grad_(False)
        if arm == "projector":
            ps = [p for n, p in vlm.model.named_parameters() if ".merger." in n]
        else:
            import re
            lid = lambda n: int(re.search(r"language_model\.layers\.(\d+)\.", n).group(1)) if re.search(r"language_model\.layers\.(\d+)\.", n) else -1
            nl = 1 + max(lid(n) for n, _ in vlm.model.named_parameters())
            ps = [p for n, p in vlm.model.named_parameters() if "lora_" in n and lid(n) >= nl - 4]
        for p in ps:
            p.requires_grad_(True)
        print(f"repair {arm}: {sum(p.numel() for p in ps)} trainable parameters", flush=True)
        vlm.train(rep, args.repair_steps, 1e-4, params=ps, log=f"repair {arm}")
        D[f"lo_rep_{arm}"] = eval_cells(vlm, D, S); print(f"repair {arm} accuracy {((D[f'lo_rep_{arm}'] > 0) == D.truth).mean():.3f}", flush=True)
        D.drop(columns=["text"]).to_csv(os.path.join(OUT, f"forget_{args.tag}.csv"), index=False)
    print("FORGET DONE", flush=True)


if __name__ == "__main__":
    main()
