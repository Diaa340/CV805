"""Option-level scores for the additive-recombination test (paper/preregistration_mm.md), same model and processor
settings as Xiu et al.'s qwen25_small_embed.py (Qwen2.5-VL-7B, min/max pixels, forced choice over the valid letters):
  full : image + [hint] + question + lettered options      (their prompt)
  txt  : the same text, no image
  img  : image + lettered options only (no hint, no question) -> keeps the option CONTENTS (pitfall 1)
Usage (in .venv_xiu): python mm_scores.py <mmbench|mmstar>; writes results/mm_scores_<ds>.jsonl"""
import base64, io, json, sys
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
ds = sys.argv[1]; torch.manual_seed(42); name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
proc = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28); tok = AutoTokenizer.from_pretrained(name)
LET = [tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"]; TAIL = "\nAnswer with the option's letter from the given choices directly."


@torch.no_grad()
def score(image, text):
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    msg = [{"role": "user", "content": content}]; t = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ims, vids = process_vision_info(msg) if image is not None else (None, None)
    inp = proc(text=[t], images=ims, videos=vids, padding=True, return_tensors="pt").to("cuda")
    return model(**inp).logits[0, -1].float().log_softmax(-1)[LET].cpu().tolist()   # log-probs of A..D (absolute)


out = open(f"results/mm_scores_{ds}.jsonl", "w")
for split in ("train", "val"):
    for it in json.load(open(f"data/{ds}/{ds}_{split}.json")):
        img = Image.open(io.BytesIO(base64.b64decode(it["image"]))).convert("RGB"); k = it["num_options"]
        full_text = [c["value"] for c in it["conversations"] if c["from"] == "human"][0].replace("\n<image>", "")
        opts = "".join(f"\n{L}. {t}" for L, t in it["options"]).lstrip("\n")
        r = dict(id=it["id"], split=split, k=k, category=it["category"], answer="ABCD".index(it["conversations"][1]["value"]),
                 full=score(img, full_text)[:k], txt=score(None, full_text)[:k], img=score(img, "Options:\n" + opts + TAIL)[:k])
        out.write(json.dumps(r) + "\n"); out.flush()
print("MM SCORES DONE", ds)
