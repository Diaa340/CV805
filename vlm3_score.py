"""Zero-shot yes / no log-odds of a VLM on every cell of the fixed grids (paper/preregistration_vlm3.md).
Models (7-8B, three families): qwen = Qwen2.5-VL-7B-Instruct; internvl = InternVL3.5-8B (HF port); llava =
LLaVA-OneVision-1.5-8B-Instruct (publisher code, pinned revision; run with .venv_llava15, transformers 4.53, which the
publisher's code targets). Same input for all: image scaled to a longest side of
448 pixels (InternVL: one 448 tile, no multi-tile cropping); prompt "<question> Answer with yes or no."; score = log-odds
of yes vs no at the next token (first token of Yes / yes / " Yes" / " yes" vs the same for no).
Writes runs/vlm3/scores_<model>.npz (L__<unit>: 10 x 100); resumes finished units.
Usage: python vlm3_score.py <qwen|internvl|llava> [--limit N]"""
import io, os, sys, time, zipfile
import numpy as np, torch
from PIL import Image
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "vlm3"); dev = "cuda"
MODELS = {"qwen": ("Qwen/Qwen2.5-VL-7B-Instruct", False, None), "internvl": ("OpenGVLab/InternVL3_5-8B-HF", False, None),
          "llava": ("lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", True, "e137910552d616eb0f7305147ca054311a90c542"),
          "idefics": ("HuggingFaceM4/Idefics3-8B-Llama3", False, None)}   # added for preregistration_vlm3_conf.md
import transformers
DT = {"dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype": torch.bfloat16}   # llava runs in .venv_llava15 (transformers 4.53)
ZIP = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip")); CACHE = {}


def image(ref):
    if ref not in CACHE:
        src, name = ref.split(":", 1)
        im = (Image.open(io.BytesIO(ZIP.read(f"CLEVR_v1.0/images/val/{name}"))) if src == "clevr" else
              Image.open(os.path.join(ROOT, "data", "raw", {"gqa": "gqa", "clove": "clove", "pope": "pope"}[src], "images", name))).convert("RGB")
        im.thumbnail((448, 448), Image.BICUBIC); CACHE[ref] = im
    return CACHE[ref]


def load(name):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    path, remote, rev = MODELS[name]
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=remote, revision=rev); kw = {}
    if remote:   # the publisher's config predates transformers 5 and lacks pad_token_id; set it from the tokenizer
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, revision=rev)
        for c in (cfg, cfg.text_config):
            if getattr(c, "pad_token_id", None) is None:
                setattr(c, "pad_token_id", proc.tokenizer.pad_token_id)
        kw["config"] = cfg
    M = (AutoModelForCausalLM if remote else AutoModelForImageTextToText).from_pretrained(path, trust_remote_code=remote, revision=rev, **kw, **DT).to(dev).eval()
    proc.extra = {"internvl": {"crop_to_patches": False}, "idefics": {"do_image_splitting": False}}.get(name, {})   # one image tile for InternVL and Idefics3
    tok = proc.tokenizer; tok.padding_side = "left"
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})
    YES, NO = ids(["Yes", "yes", " Yes", " yes"]), ids(["No", "no", " No", " no"]); assert not set(YES) & set(NO)
    return proc, M, YES, NO


@torch.no_grad()
def score(proc, M, YES, NO, texts, imgs, bs=16):
    out = []
    for b in range(0, len(texts), bs):
        msgs = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}], add_generation_prompt=True, tokenize=False)
                for t in texts[b:b + bs]]
        inp = proc(text=msgs, images=[image(r) for r in imgs[b:b + bs]], padding=True, return_tensors="pt", **proc.extra).to(dev)
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        lo = M(**inp).logits[:, -1].float().log_softmax(-1)
        out.append((torch.logsumexp(lo[:, YES], 1) - torch.logsumexp(lo[:, NO], 1)).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    name = sys.argv[1]; limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    G = dict(np.load(os.path.join(OUT, "grids.npz"))); keys = sorted(k[3:] for k in G if k.startswith("T__"))[:limit]
    path = os.path.join(OUT, f"scores_{name}{'_smoke' if limit else ''}.npz"); done = dict(np.load(path)) if os.path.exists(path) else {}
    proc, M, YES, NO = load(name); print(name, "yes ids", YES, "no ids", NO, flush=True)
    for k in keys:
        if f"L__{k}" in done:
            continue
        t0 = time.time(); T, tx, im = G[f"T__{k}"], G[f"text__{k}"], G[f"img__{k}"]
        q, j = np.meshgrid(np.arange(T.shape[0]), np.arange(T.shape[1]), indexing="ij")
        order = np.argsort(j.ravel(), kind="stable")   # group cells by image
        L = np.zeros(T.size); L[order] = score(proc, M, YES, NO, list(tx[q.ravel()[order]]), list(im[j.ravel()[order]]))
        done[f"L__{k}"] = L.reshape(T.shape).astype(np.float32); np.savez(path, **done)
        print(k, f"acc {((L.reshape(T.shape) > 0) == T).mean():.3f}", f"({time.time() - t0:.0f}s)", flush=True)
    print("DONE", name, flush=True)
