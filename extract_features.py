"""Frozen-encoder features for the multi-dataset forgetting study. One file per dataset:
data/feats/<name>.npz with f1, f2 (fp16 features of modality 1 / 2), y (int labels), split (0 train / 1 test),
plus optional extras (e.g. CREMA-D human interaction types).

Encoders (frozen): CLIP ViT-B/32 (openai/clip-vit-base-patch32) projected 512-d image / text embeddings;
wav2vec2-base (facebook/wav2vec2-base) last hidden layer, mean-pooled over time (768-d) for speech audio.

  cifar2v : CIFAR-100, modality 1 = left half, modality 2 = right half (each resized to 224x224). 100 classes.
  avmnist : MNIST image + Free Spoken Digit Dataset recording of the same digit (random pairing; test images only
            get test recordings). 10 classes.
  cremad  : CREMA-D audio (wav2vec2 on the 16 kHz track) + face (CLIP, mean over the three 1-fps frames).
            6 classes; stratified 70/30 split (seed 0), as in cremad_cl.py; human interaction types attached.
  food101 : UPMC Food-101 image + text (CLIP).          [after download]
  n24news : N24News image + text (CLIP).                 [after download]
Usage: python extract_features.py cifar2v avmnist cremad
"""
import io
import os
import pickle
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "feats")
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
_clip = _w2v = None


def clip():
    global _clip
    if _clip is None:
        from transformers import CLIPModel, CLIPTokenizer
        _clip = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval().half(),
                 CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32"))
    return _clip


@torch.no_grad()
def clip_images(u8, bs=512):
    """u8: uint8 array (N, H, W, 3) or (N, H, W). Resized to 224x224 (no crop), CLIP-normalised."""
    model = clip()[0]; out = []
    for i in range(0, len(u8), bs):
        x = torch.tensor(u8[i:i + bs], device=dev)
        if x.dim() == 3:
            x = x[..., None].expand(-1, -1, -1, 3)
        x = x.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        e = model.get_image_features(pixel_values=((x - CLIP_MEAN) / CLIP_STD).half())
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        out.append(e.float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


@torch.no_grad()
def clip_texts(texts, bs=512):
    model, tok = clip(); out = []
    for i in range(0, len(texts), bs):
        t = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
        e = model.get_text_features(**t)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        out.append(e.float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


@torch.no_grad()
def w2v(waves):
    """waves: list of 1-D float32 arrays at 16 kHz -> mean-pooled last hidden state (768-d)."""
    global _w2v
    if _w2v is None:
        from transformers import Wav2Vec2Model
        _w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(dev).eval()
    out = []
    for w in waves:
        w = (w - w.mean()) / (w.std() + 1e-7)  # wav2vec2 feature-extractor normalisation
        h = _w2v(torch.tensor(w, device=dev)[None]).last_hidden_state
        out.append(h.mean(1)[0].float().cpu())
    return torch.stack(out).numpy().astype(np.float16)


def save(name, **arrays):
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{name}.npz"), **arrays)
    print(f"saved {name}: " + ", ".join(f"{k}{v.shape}" for k, v in arrays.items()), flush=True)


# ------------------------------------------------------------------------------------------ datasets
def cifar2v():
    base = os.path.expanduser("~/cifar100_pid/data/cifar-100-python")
    X, y, sp = [], [], []
    for s, fn in ((0, "train"), (1, "test")):
        d = pickle.load(open(os.path.join(base, fn), "rb"), encoding="latin1")
        X.append(d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)); y.append(np.array(d["fine_labels"]))
        sp.append(np.full(len(d["fine_labels"]), s))
    X = np.concatenate(X)
    save("cifar2v", f1=clip_images(np.ascontiguousarray(X[:, :, :16])), f2=clip_images(np.ascontiguousarray(X[:, :, 16:])),
         y=np.concatenate(y), split=np.concatenate(sp))


def _decode_audio(b, sr_in):
    import soundfile as sf
    import librosa
    w, sr = sf.read(io.BytesIO(b), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    return librosa.resample(w, orig_sr=sr, target_sr=16000) if sr != 16000 else w


def avmnist():
    from PIL import Image
    raw = os.path.join(ROOT, "data", "raw")
    imgs, ys, sp = [], [], []
    for s, fn in ((0, "mnist_train.parquet"), (1, "mnist_test.parquet")):
        df = pd.read_parquet(os.path.join(raw, fn))
        imgs += [np.array(Image.open(io.BytesIO(r["bytes"]))) for r in df["image"]]; ys.append(df["label"].values)
        sp.append(np.full(len(df), s))
    imgs = np.stack(imgs); y = np.concatenate(ys); split = np.concatenate(sp)
    aud = {}
    for s, fn in ((0, "fsdd_train.parquet"), (1, "fsdd_test.parquet")):
        df = pd.read_parquet(os.path.join(raw, fn))
        lab_col = "label" if "label" in df.columns else [c for c in df.columns if c != "audio"][0]
        waves = [_decode_audio(a["bytes"], None) for a in df["audio"]]
        aud[s] = (w2v(waves), df[lab_col].values.astype(int))
        print(f"  fsdd split {s}: {len(waves)} recordings", flush=True)
    rng = np.random.default_rng(0)
    f2 = np.zeros((len(y), 768), np.float16); src = np.zeros(len(y), int)
    for i in range(len(y)):
        feats, labs = aud[split[i]]
        j = rng.choice(np.where(labs == y[i])[0]); f2[i] = feats[j]; src[i] = j
    save("avmnist", f1=clip_images(imgs), f2=f2, y=y, split=split, audio_src=src)


def _cremad_wave(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def cremad():
    meta = pd.read_csv(os.path.join(ROOT, "data", "cremad_meta.csv"))
    paths = {}
    for dp, _, fs in os.walk(os.path.join(ROOT, "data", "CREMA-D")):
        for f in fs:
            if f.endswith(".flv"):
                paths[f[:-4]] = os.path.join(dp, f)
    with Pool(48) as p:
        waves = p.map(_cremad_wave, [paths[n] for n in meta.name], chunksize=16)
    fa = w2v(waves)
    V = np.load(os.path.join(ROOT, "data", "cremad_frames.npy"), mmap_mode="r")  # N,3,256,256,3
    fv = np.mean([clip_images(np.ascontiguousarray(V[:, k])).astype(np.float32) for k in range(V.shape[1])], 0).astype(np.float16)
    y = meta.label.values
    rng = np.random.default_rng(0); split = np.zeros(len(y), int)  # same stratified 70/30 split as cremad_cl.py
    for c in range(6):
        ids = np.where(y == c)[0]; split[rng.choice(ids, int(0.3 * len(ids)), replace=False)] = 1
    H = pd.read_csv(os.path.join(ROOT, "data", "cremad_human_pid.csv"), index_col=0).loc[meta.name]
    save("cremad", f1=fa, f2=fv, y=y, split=split, human_type=H.htype.values.astype(str),
         human_atoms=H[["i_a", "i_v", "i_av", "red", "u_a", "u_v", "syn"]].values.astype(np.float32))


def _load224(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC))


def food101():
    """UPMC Food-101: image + web-page title (CLIP image / CLIP text). Official train/test split. 101 classes."""
    import html
    base = os.path.join(ROOT, "data", "raw", "UPMC-Food-101")
    rows = []
    for s, sp in ((0, "train"), (1, "test")):
        for line in open(os.path.join(base, "texts", f"{sp}_titles.csv"), encoding="utf-8", errors="replace"):
            line = line.rstrip("\n")
            if not line:
                continue
            fname, rest = line.split(",", 1); title, label = rest.rsplit(",", 1)
            path = os.path.join(base, "images", sp, label, fname)
            if os.path.exists(path):
                rows.append((path, html.unescape(title).strip(), label, s))
    classes = sorted({r[2] for r in rows}); cid = {c: i for i, c in enumerate(classes)}
    print(f"  {len(rows)} image-title pairs, {len(classes)} classes", flush=True)
    f_img = []
    with Pool(48) as p:
        for i in range(0, len(rows), 4096):
            f_img.append(clip_images(np.stack(p.map(_load224, [r[0] for r in rows[i:i + 4096]], chunksize=32))))
    save("food101", f1=np.concatenate(f_img), f2=clip_texts([r[1] for r in rows]),
         y=np.array([cid[r[2]] for r in rows]), split=np.array([r[3] for r in rows]))


def n24news():
    """N24News: news image + headline (CLIP image / CLIP text). train+dev -> split 0, test -> split 1. 24 sections."""
    import json
    base = os.path.join(ROOT, "data", "raw", "N24News")
    rows = []
    for s, sp in ((0, "train"), (0, "dev"), (1, "test")):
        for r in json.load(open(os.path.join(base, "news", f"nytimes_{sp}.json"))):
            path = os.path.join(base, "imgs", f"{r['image_id']}.jpg")
            if os.path.exists(path):
                rows.append((path, r["headline"], r["section"], s))
    classes = sorted({r[2] for r in rows}); cid = {c: i for i, c in enumerate(classes)}
    print(f"  {len(rows)} image-headline pairs, {len(classes)} sections", flush=True)
    f_img = []
    with Pool(48) as p:
        for i in range(0, len(rows), 4096):
            f_img.append(clip_images(np.stack(p.map(_load224, [r[0] for r in rows[i:i + 4096]], chunksize=32))))
    save("n24news", f1=np.concatenate(f_img), f2=clip_texts([r[1] for r in rows]),
         y=np.array([cid[r[2]] for r in rows]), split=np.array([r[3] for r in rows]))


def _img_bytes224(b):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BICUBIC))


def vqa():
    """VQA v2 validation subset (merve/vqav2-small): image + question (CLIP). Answers kept if among the top-100.
    Own 70/30 split by image (images never shared across splits)."""
    import pyarrow.parquet as pq
    base = os.path.join(ROOT, "data", "raw", "vqa")
    tabs = [pq.read_table(os.path.join(base, f)).to_pandas() for f in sorted(os.listdir(base)) if f.endswith(".parquet")]
    df = pd.concat(tabs, ignore_index=True)
    top = df.multiple_choice_answer.value_counts().index[:100]; df = df[df.multiple_choice_answer.isin(top)].reset_index(drop=True)
    keys = df.image.apply(lambda d: d["path"] or str(hash(d["bytes"][:2048])))
    uk = keys.unique(); rng = np.random.default_rng(0); test_imgs = set(rng.choice(uk, int(0.3 * len(uk)), replace=False))
    print(f"  {len(df)} QA pairs (top-100 answers), {len(uk)} images", flush=True)
    f_img = []
    with Pool(48) as p:
        for i in range(0, len(df), 4096):
            f_img.append(clip_images(np.stack(p.map(_img_bytes224, [d["bytes"] for d in df.image[i:i + 4096]], chunksize=32))))
    cid = {a: i for i, a in enumerate(top)}
    save("vqa", f1=np.concatenate(f_img), f2=clip_texts(df.question.tolist()), y=df.multiple_choice_answer.map(cid).values,
         split=keys.isin(test_imgs).astype(int).values, answers=np.array(list(top)))


def clevr(max_train_q=150000):
    """CLEVR v1.0: image + question (CLIP). Question family = last program function; train questions subsampled to
    max_train_q, all val questions used as the test split. Answers: all 28."""
    import json
    import zipfile
    base = os.path.join(ROOT, "data", "raw", "clevr")
    z = zipfile.ZipFile(os.path.join(base, "CLEVR_v1.0.zip"))
    rows = []
    for s, sp in ((0, "train"), (1, "val")):
        qs = json.loads(z.read(f"CLEVR_v1.0/questions/CLEVR_{sp}_questions.json"))["questions"]
        if s == 0:
            qs = [qs[i] for i in np.random.default_rng(0).choice(len(qs), min(max_train_q, len(qs)), replace=False)]
        for q in qs:
            rows.append((f"CLEVR_v1.0/images/{sp}/{q['image_filename']}", q["question"], q["answer"], q["program"][-1]["function"], s))
    df = pd.DataFrame(rows, columns=["img", "question", "answer", "family", "split"])
    answers = sorted(df.answer.unique()); cid = {a: i for i, a in enumerate(answers)}
    print(f"  {len(df)} questions, {df.img.nunique()} images, {len(answers)} answers", flush=True)
    uimg = df.img.unique(); idx = {k: i for i, k in enumerate(uimg)}
    from PIL import Image
    feats = []
    for i in range(0, len(uimg), 2048):
        arr = np.stack([np.asarray(Image.open(z.open(k)).convert("RGB").resize((224, 224), Image.BICUBIC)) for k in uimg[i:i + 2048]])
        feats.append(clip_images(arr))
    fimg = np.concatenate(feats)
    save("clevr", f1=fimg[df.img.map(idx).values], f2=clip_texts(df.question.tolist()), y=df.answer.map(cid).values,
         split=df.split.values, family=df.family.values.astype(str), answers=np.array(answers))


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"== {name}", flush=True)
        globals()[name]()
