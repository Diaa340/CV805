"""Preprocess CREMA-D (7442 .flv clips: MP3 audio + VP6 video) into GPU-friendly arrays, following OGM-GE
(GeWu-Lab/OGM-GE_CVPR2022, dataset/CramedDataset.py + data/CREMAD/video_preprocessing.py):
  audio : decode to mono 22.05 kHz, tile to 3 s, clip to [-1, 1], log|STFT| (n_fft=512, hop=353) -> 257 x 188 (fp16)
          (OGM-GE reads AudioWAV; here the audio track of the same recording inside the .flv is used)
  visual: frames at 1 fps (t = 0, 1, 2 s -> 3 frames, last one repeated if the clip is shorter), resized to
          256 x 256 (uint8); train-time RandomResizedCrop(224) / test-time Resize(224) are done on the GPU.
  split : OGM-GE train.csv (6698) / test.csv (744).
Output: data/cremad_{audio.npy, frames.npy, meta.csv}"""
import os
import subprocess
from multiprocessing import Pool

import librosa
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "CREMA-D")
CLASSES = {"NEU": 0, "HAP": 1, "SAD": 2, "FEA": 3, "DIS": 4, "ANG": 5}
SR, SIZE, NF = 22050, 256, 3


def audio_spec(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    s = np.frombuffer(raw, np.float32)
    # OGM-GE tiles exactly 3x, which leaves clips shorter than 1 s under 3 s; tile as often as needed instead
    s = np.tile(s, int(np.ceil(SR * 3 / max(len(s), 1))))[:SR * 3].copy()
    assert len(s) == SR * 3, f"empty audio track in {path}"
    np.clip(s, -1, 1, out=s)
    spec = np.log(np.abs(librosa.stft(s, n_fft=512, hop_length=353)) + 1e-7)
    return spec.astype(np.float16)


def frames(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps=1,scale={SIZE}:{SIZE}", "-frames:v", str(NF),
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, SIZE, SIZE, 3)
    if len(f) == 0:  # e.g. 1076_MTI_NEU_XX.flv is truncated (0.066 s) in the source; dropped like OGM-GE's loader would
        raise ValueError("no frame at 1 fps")
    if len(f) < NF:
        f = np.concatenate([f, np.repeat(f[-1:], NF - len(f), 0)])
    return f[:NF]


def work(item):
    name, path = item
    try:
        return name, audio_spec(path), frames(path), None
    except Exception as e:  # noqa: BLE001
        return name, None, None, repr(e)


def main():
    ref = os.path.join(ROOT, "ref_ogmge", "data", "CREMAD")
    split = {}
    for sp in ("train", "test"):
        for n, lab in pd.read_csv(os.path.join(ref, f"{sp}.csv"), header=None).values:
            split[n] = (sp, CLASSES[lab])
    paths = {}
    for dp, _, fs in os.walk(SRC):
        for f in fs:
            if f.endswith(".flv"):
                paths[f[:-4]] = os.path.join(dp, f)
    names = sorted(split)
    missing = [n for n in names if n not in paths]
    print(f"{len(names)} clips in OGM-GE split, {len(paths)} .flv found, {len(missing)} missing")
    items = [(n, paths[n]) for n in names if n in paths]
    with Pool(48) as pool:
        out = pool.map(work, items, chunksize=8)
    bad = [(n, e) for n, a, v, e in out if e]
    good = [(n, a, v) for n, a, v, e in out if not e]
    print(f"decoded {len(good)}, failed {len(bad)}: {bad[:3]}")
    A = np.stack([a for _, a, _ in good]); V = np.stack([v for _, _, v in good])
    meta = pd.DataFrame(dict(name=[n for n, _, _ in good], split=[split[n][0] for n, _, _ in good],
                             label=[split[n][1] for n, _, _ in good]))
    np.save(os.path.join(ROOT, "data", "cremad_audio.npy"), A)
    np.save(os.path.join(ROOT, "data", "cremad_frames.npy"), V)
    meta.to_csv(os.path.join(ROOT, "data", "cremad_meta.csv"), index=False)
    print("audio", A.shape, A.dtype, "frames", V.shape, V.dtype)
    print(meta.groupby(["split", "label"]).size().unstack())


if __name__ == "__main__":
    main()
