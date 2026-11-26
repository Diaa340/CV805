"""Fixed question x image grids for the multi-VLM structure test (paper/preregistration_vlm3.md). Outcome-free.
Every unit is a complete 10-question x 100-image answer table on which every question is well-posed on every image, so
the structural score and EMAP are computed on the identical, fully labelled cells (no mismatched pairs):
  clevr: the 24 CLEVR units whose questions are well-posed on every image (as in the EMAP study), 100 held-out images;
  gqa:   the 24 GQA existence units, 100 of their 200 images (seed 93);
  clove: the 43 CLOVE-scene existence units, their 50 test images + 50 of their training images (seed 94);
  pope:  8 units of 10 COCO objects (objects shuffled, seed 95) x 100 POPE images (seed 96 + unit), POPE's question.
Per unit: exact additive rule (MILP, 600 s) -> exception labels and structural score s = exception share (= minimum
error of any additive rule); graded misfit per cell; MMI synergy. BROJA is added by vlm3_pid.py (.venv_dit).
Writes runs/vlm3/grids.npz and runs/vlm3/units.csv"""
import os
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "runs", "vlm3"); os.makedirs(OUT, exist_ok=True)


def job(a):
    from exact_additive import exact_partial_fit
    from unrepair_structure import local_synred
    from pid_utils import mmi
    key, T = a; P, b, up, m = exact_partial_fit(T, np.ones(T.shape), 600)
    return key, (P != T).astype(np.int8), local_synred(T, np.ones(T.shape)), b, up, mmi(T)


def build():
    G = {}
    from vlm_test import units
    U, imgs, S = units(); fam = pd.read_csv(os.path.join(ROOT, "runs", "clevr2", "families.csv")).set_index("unit")
    ite = np.random.default_rng(14).permutation(len(imgs))[700:800]
    for u in [u for u in U if fam.lifted[u] <= 0.0]:
        assert (U[u]["native"][:, ite] == U[u]["T"][:, ite]).all()
        G[f"clevr__{u}"] = dict(T=U[u]["T"][:, ite], text=[f"{t} Answer with yes or no." for t in U[u]["text"]], img=[f"clevr:{S[imgs[k]]['image_filename']}" for k in ite])
    Q = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz")); K = pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep")
    cols = np.sort(np.random.default_rng(93).choice(200, 100, replace=False))
    for u in K.unit:
        G[f"gqa__{u}"] = dict(T=Q[f"T__{u}"][:, cols], text=[f"{t} Answer with yes or no." for t in Q[f"text__{u}"]], img=[f"gqa:{m}.jpg" for m in Q[f"imgs__{u}"][cols]])
    C = np.load(os.path.join(ROOT, "runs", "clove", "units.npz")); K = pd.read_csv(os.path.join(ROOT, "runs", "clove", "units.csv")).query("keep")
    cols = np.concatenate([np.sort(np.random.default_rng(94).choice(150, 50, replace=False)), np.arange(150, 200)])
    for u in K.unit:
        G[f"clove__{u}"] = dict(T=C[f"T__{u}"][:, cols], text=[f"{t} Answer with yes or no." for t in C[f"text__{u}"]], img=[f"clove:{m}.jpg" for m in C[f"imgs__{u}"][cols]])
    P = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); objs = np.random.default_rng(95).permutation(80)
    for g in range(8):
        o = objs[10 * g:10 * g + 10]; k = np.sort(np.random.default_rng(96 + g).choice(500, 100, replace=False))
        G[f"pope__g{g}"] = dict(T=P["T"][np.ix_(o, k)].astype(int), text=[f"Is there a {P['objects'][x]} in the image? Answer with yes or no." for x in o],
                                img=[f"pope:{P['images'][x]}" for x in k])
    return G


if __name__ == "__main__":
    G = build(); rows = []; out = {}
    with Pool(12) as p:
        for key, N, M, b, up, mm in p.imap_unordered(job, [(k, d["T"].astype(int)) for k, d in G.items()]):
            d = G[key]; out[f"T__{key}"] = d["T"].astype(np.int8); out[f"exc__{key}"] = N; out[f"misfit__{key}"] = M
            out[f"text__{key}"] = np.array(d["text"]); out[f"img__{key}"] = np.array(d["img"])
            rows.append(dict(key=key, bench=key.split("__")[0], s=float(N.mean()), ceiling=b, bound=up, exact=bool(abs(up - b) < 1e-6), mmi=mm, yes=float(d["T"].mean())))
            print(key, round(N.mean(), 3), round(b, 4), round(up, 4), flush=True)
    np.savez(os.path.join(OUT, "grids.npz"), **out); pd.DataFrame(rows).sort_values("key").to_csv(os.path.join(OUT, "units.csv"), index=False)
