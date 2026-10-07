#!/usr/bin/env bash
# GQA chain (preregistration_gqa.md): wait for images, extract the needed ones, run training and VLM parts, analyse. GPU 2.
cd "$(dirname "$0")"; export CUDA_VISIBLE_DEVICES=2
until [ -f data/raw/gqa/done.txt ]; do sleep 60; done
.venv/bin/python - <<'PY'
import numpy as np, zipfile, os
Z = np.load("runs/gqa/units.npz"); need = sorted({m for k in Z.files if k.startswith("imgs__") for m in Z[k]})
z = zipfile.ZipFile("data/raw/gqa/images.zip"); os.makedirs("data/raw/gqa/images", exist_ok=True); names = set(z.namelist())
for m in need:
    n = f"images/{m}.jpg"
    if n in names and not os.path.exists(f"data/raw/gqa/{n}"): z.extract(n, "data/raw/gqa")
print(len(need), "images extracted")
PY
.venv/bin/python -u gqa_train.py > runs/gqa/train.log 2>&1 &
.venv/bin/python -u gqa_vlm.py > runs/gqa/vlm.log 2>&1
wait
.venv/bin/python analyze_gqa.py > runs/gqa/analysis.log 2>&1; echo "GQA DONE" >> runs/gqa/analysis.log
