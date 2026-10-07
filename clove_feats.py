"""CLIP features for CLOVE units (images resized 224 bicubic, as for GQA; questions as text). Writes runs/clove/clip.npz"""
import os, zipfile, io
import numpy as np
from PIL import Image
from extract_features import clip_images, clip_texts
ROOT = os.path.dirname(os.path.abspath(__file__)); U = np.load(os.path.join(ROOT, "runs", "clove", "units.npz"))
ids = sorted({str(m) for k in U.files if k.startswith("imgs__") for m in U[k]}); Z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "gqa", "images.zip"))
os.makedirs(os.path.join(ROOT, "data", "raw", "clove", "images"), exist_ok=True); arr = []
for m in ids:
    b = Z.read(f"images/{m}.jpg"); open(os.path.join(ROOT, "data", "raw", "clove", "images", f"{m}.jpg"), "wb").write(b)
    arr.append(np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BICUBIC)))
units = sorted(k[3:] for k in U.files if k.startswith("T__")); texts = np.concatenate([U[f"text__{u}"] for u in units])
np.savez(os.path.join(ROOT, "runs", "clove", "clip.npz"), ids=np.array(ids), img=clip_images(np.stack(arr)), units=np.array(units), txt=clip_texts(list(texts)))
print(len(ids), "images,", len(texts), "questions")
