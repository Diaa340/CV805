"""POPE exact table (paper/preregistration_pope.md). Outcome-free. Truth = POPE's own ground-truth object lists
(segmentations + captions, POPE/segmentation/coco_ground_truth_segmentation.json) for the 500 POPE images x 80 COCO
categories. Checks agreement with all 9,000 POPE labels; computes structure (additive-fit distance, XOR density) and
exact MMI synergy on the full table and on POPE's own sampled cells. Writes data/raw/pope/pope_table.npz"""
import json, os
import numpy as np, torch
from pid_utils import mmi
from step2_build import additive_fit, density
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = os.path.join(ROOT, "data", "raw", "pope")
A = json.load(open(os.path.join(ROOT, "data", "raw", "coco", "annotations", "instances_val2014.json")))
names = [c["name"] for c in sorted(A["categories"], key=lambda c: c["id"])]
GT = {g["image"]: set(g["objects"]) for g in map(json.loads, open("/home/diaaabuhani/ext/POPE/segmentation/coco_ground_truth_segmentation.json"))}
ims = open(os.path.join(P, "images.txt")).read().split()
T = np.array([[int(o in GT[f]) for f in ims] for o in names], np.int8)
cells, split, ok = [], [], 0
for s in ("random", "popular", "adversarial"):
    for l in open(f"/home/diaaabuhani/ext/POPE/output/coco/coco_pope_{s}.json"):
        q = json.loads(l); o = q["text"].replace("Is there a ", "").replace("Is there an ", "").replace(" in the image?", "")
        c = (names.index(o), ims.index(q["image"]), int(q["label"] == "yes")); cells.append(c); split.append(s); ok += int(T[c[0], c[1]] == c[2])
print(f"table {T.shape}, yes share {T.mean():.3f}; agreement with POPE labels {ok}/{len(cells)}")
C = np.array(cells); Tp = np.full(T.shape, -1, np.int8); Tp[C[:, 0], C[:, 1]] = C[:, 2]
for nm, X in (("full table", T), ("POPE cells only", Tp)):
    d, n4 = density(X, np.random.default_rng(13))
    print(f"  {nm}: cells {int((X >= 0).sum())}, yes {X[X >= 0].mean():.3f}, additive-fit distance {additive_fit(X):.3f}, XOR density {d:.4f} (n4 {n4}), exact MMI synergy {mmi(X):.3f}")
np.savez(os.path.join(P, "pope_table.npz"), T=T, T_pope=Tp, objects=np.array(names), images=np.array(ims), cells=C, split=np.array(split))
