"""Pitfall-3 variant of Xiu et al.'s Stage 2 (paper/preregistration_pope.md): instead of replacing a modality's token
embeddings with Gaussian noise, replace the modality with a RANDOM REAL sample from the same split (another image for
the language-only prediction; another question for the vision-only prediction). Same model, processor settings
(min/max pixels), prompt, forced-choice A/B probabilities and 0.3 confidence rule as their qwen25_small_embed.py.
Features and vl_prob are copied from their Stage-2 output, so only v_prob / l_prob change.
Usage (in .venv_xiu): python real_replace.py <train|val>"""
import json, os, random, sys
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
mode = sys.argv[1]; torch.manual_seed(42); random.seed(42)
name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
proc = AutoProcessor.from_pretrained(name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28); tok = AutoTokenizer.from_pretrained(name)
AB = [tok.encode("A", add_special_tokens=False)[0], tok.encode("B", add_special_tokens=False)[0]]
IMG = "/home/diaaabuhani/mm_pid/data/raw/pope/images"
data = json.load(open(f"results/embeds/pope/{mode}/qwen25_7.json"))
src = json.load(open(f"data/pope/pope_{mode}.json"))
qtext = lambda it: [c["value"] for c in it["conversations"] if c["from"] == "human"][0].replace("\n<image>", "")


@torch.no_grad()
def probs(image_file, question):
    msg = [{"role": "user", "content": [{"type": "image", "image": Image.open(os.path.join(IMG, image_file)).convert("RGB")}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True); ims, vids = process_vision_info(msg)
    inp = proc(text=[text], images=ims, videos=vids, padding=True, return_tensors="pt").to("cuda")
    last = model(**inp).logits[0, -1].float(); absolute = torch.softmax(last, -1)[AB]; fc = torch.softmax(last[AB], -1)
    if absolute.sum() < 0.3:
        fc = torch.full((2,), 0.5)
    return fc.cpu().tolist()


n = len(src)
for k, (it, out) in enumerate(zip(src, data)):
    j = random.choice([x for x in range(n) if x != k]); m = random.choice([x for x in range(n) if x != k])
    g = [c for c in out["conversations"] if c["from"] == "gpt"][0]
    g["l_prob"] = probs(src[j]["image"], qtext(it))       # language-only: question k with a random other image
    g["v_prob"] = probs(it["image"], qtext(src[m]))        # vision-only: image k with a random other question
    if k % 250 == 0:
        print(k, n, g["vl_prob"], g["l_prob"], g["v_prob"], flush=True)
os.makedirs(f"results/embeds_real/pope/{mode}", exist_ok=True)
json.dump(data, open(f"results/embeds_real/pope/{mode}/qwen25_7.json", "w"))
print("REAL DONE", mode, flush=True)
