"""Step 2 (paper/preregistration_step2.md): exact answer tables for CLEVR yes/no question families under set-lifted
(existential) program semantics (clevr_exec.execute_total). Outcome-free.
For each family: up to 60 distinct programs (questions) x 400 fixed val images, answers by executing the program on the
scene graph (-1 = ill-posed). Structure (XOR density, additive-fit distance), exact-table MMI synergy, clustered joint
distribution for BROJA (computed separately with dit), symbolic and CLIP features.
Writes runs/clevr2/families.npz (+ per-family arrays) and runs/clevr2/clusters.npz"""
import json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from clevr_exec import execute_total, load
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "clevr2"); dev = "cuda"
NQ, NI, GQ = 60, 1000, 10
COL = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]; SHP = ["cube", "sphere", "cylinder"]
MAT = ["rubber", "metal"]; SIZ = ["small", "large"]


def img_feat(scene):
    """Symbolic image code: objects sorted by x; per slot one-hot color / shape / material / size + 3-d coords (10 slots)."""
    v = np.zeros((10, 18), np.float32)
    for s, o in enumerate(sorted(scene["objects"], key=lambda o: o["3d_coords"][0])[:10]):
        v[s, COL.index(o["color"])] = 1; v[s, 8 + SHP.index(o["shape"])] = 1; v[s, 11 + MAT.index(o["material"])] = 1
        v[s, 13 + SIZ.index(o["size"])] = 1; v[s, 15:18] = np.array(o["3d_coords"]) / 3
    h = np.zeros(8 * 3 * 2 * 2, np.float32)  # counts of every attribute combination
    for o in scene["objects"]:
        h[((COL.index(o["color"]) * 3 + SHP.index(o["shape"])) * 2 + MAT.index(o["material"])) * 2 + SIZ.index(o["size"])] += 1
    single = np.r_[h.reshape(8, 3, 2, 2).sum((1, 2, 3)), h.reshape(8, 3, 2, 2).sum((0, 2, 3)), h.reshape(8, 3, 2, 2).sum((0, 1, 3)),
                   h.reshape(8, 3, 2, 2).sum((0, 1, 2))]
    return np.r_[v.ravel(), h, single, len(scene["objects"])]


def q_feat(progs):
    """Symbolic question code: for each program step, one-hot of its value input (programs of one family share structure)."""
    vocab = sorted({v for p in progs for f in p for v in f["value_inputs"]}); L = max(len(p) for p in progs)
    X = np.zeros((len(progs), L, len(vocab) + 1), np.float32)
    for n, p in enumerate(progs):
        for s, f in enumerate(p):
            X[n, s, vocab.index(f["value_inputs"][0]) if f["value_inputs"] else len(vocab)] = 1
    return X.reshape(len(progs), -1)


def density(T, rng, n=3000000):
    q, i = rng.integers(0, T.shape[0], (n, 2)), rng.integers(0, T.shape[1], (n, 2))
    k = (q[:, 0] != q[:, 1]) & (i[:, 0] != i[:, 1]); q, i = q[k], i[k]
    a, b, c, d = T[q[:, 0], i[:, 0]], T[q[:, 1], i[:, 1]], T[q[:, 0], i[:, 1]], T[q[:, 1], i[:, 0]]
    v = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    return float(((a == b) & (c == d) & (a != c))[v].mean()), int(v.sum())


def additive_fit(T, steps=3000):
    """Best threshold-additive fit [u_q + w_i > 0] by logistic regression on valid cells; returns its error (d_add upper bound)."""
    q, i = np.where(T >= 0); y = torch.tensor(T[q, i], device=dev)
    u = torch.zeros(T.shape[0], device=dev, requires_grad=True); w = torch.zeros(T.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True); opt = torch.optim.Adam([u, w, b], lr=0.1)
    qt, it = torch.tensor(q, device=dev), torch.tensor(i, device=dev)
    for _ in range(steps):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(u[qt] + w[it] + b, y.float()).backward(); opt.step()
    with torch.no_grad():
        return float(((u[qt] + w[it] + b > 0).long() != y).float().mean())


def mmi(T):
    H = lambda p: -sum(x * np.log2(x) for x in (p, 1 - p) if 0 < x < 1)
    m = T >= 0; p = T[m].mean(); pq = m.sum(1) / m.sum(); pi = m.sum(0) / m.sum()
    rq = np.array([T[r][m[r]].mean() if m[r].any() else 0 for r in range(T.shape[0])]); ri = np.array([T[:, c][m[:, c]].mean() if m[:, c].any() else 0 for c in range(T.shape[1])])
    I12 = H(p); Iq = H(p) - sum(pq[r] * H(rq[r]) for r in range(len(rq))); Ii = H(p) - sum(pi[c] * H(ri[c]) for c in range(len(ri)))
    return I12 - max(Iq, Ii)


def kmeans(X, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(k, n_init=10, random_state=seed).fit_predict(X)


def main():
    Q, S = load("val"); rng = np.random.default_rng(11); imgs = np.sort(rng.choice(len(S), NI, replace=False))
    fams = {}
    for q in Q:
        if q["answer"] in ("yes", "no"):
            fams.setdefault(q["question_family_index"], {})[json.dumps(q["program"])] = q["question"]
    Ximg = np.stack([img_feat(S[k]) for k in imgs]); rows = []; store = {}
    for f in sorted(fams):
        progs = sorted(fams[f]); r2 = np.random.default_rng(12 + f)
        pick = r2.choice(len(progs), min(NQ, len(progs)), replace=False); P = [json.loads(progs[p]) for p in pick]
        txt = [fams[f][progs[p]] for p in pick]
        E = [[execute_total(p, S[k]) for k in imgs] for p in P]
        T = np.array([[int(a == "yes") for a, _ in r] for r in E], np.int8); LIFT = np.array([[l for _, l in r] for r in E])
        # units: disjoint groups of GQ questions within the family
        for g in range(len(P) // GQ):
            sl = slice(g * GQ, (g + 1) * GQ); Tg = T[sl]; yes = Tg.mean(); keep = 0.10 <= yes <= 0.90
            row = dict(unit=f"{f}_{g}", family=f, group=g, final=P[0][-1]["function"], lifted=float(LIFT[sl].mean()), yes=yes, keep=keep)
            if keep:
                dn, n4 = density(Tg, np.random.default_rng(13)); keep = n4 >= 20000; row["keep"] = keep
                if keep:
                    row.update(dens=dn, dens_n=n4, d_add_hat=additive_fit(Tg), syn_mmi_exact=mmi(Tg))
                    store[f"{f}_{g}"] = dict(T=Tg, Xq=q_feat(P)[sl], text=np.array(txt[sl]))
            rows.append(row); print(row, flush=True)
    import pandas as pd
    D = pd.DataFrame(rows); D.to_csv(os.path.join(OUT, "families.csv"), index=False)
    print(f"kept {D.keep.sum()} of {len(D)} units from {D[D.keep].family.nunique()} families")
    # CLIP features: images and question texts
    sys.path.insert(0, ROOT); from extract_features import clip_images, clip_texts
    import zipfile
    from PIL import Image
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "clevr", "CLEVR_v1.0.zip"))
    arr = np.stack([np.asarray(Image.open(z.open(f"CLEVR_v1.0/images/val/{S[k]['image_filename']}")).convert("RGB").resize((224, 224), Image.BICUBIC)) for k in imgs])
    Cimg = clip_images(arr).astype(np.float32)
    # clustered joint distributions for BROJA (k = 10 per modality; symbolic and CLIP features), as in Liang et al.
    ci_sym, ci_clip = kmeans(Ximg, 10, 0), kmeans(Cimg, 10, 0); clus = {}
    for f, d in store.items():
        d["Cq"] = clip_texts(list(d["text"])).astype(np.float32)
        for nm, cq, ci in (("sym", kmeans(d["Xq"], 10, 0), ci_sym), ("clip", kmeans(d["Cq"], 10, 0), ci_clip)):
            J = np.zeros((10, 10, 2)); T = d["T"]
            for a in range(T.shape[0]):
                for b in range(T.shape[1]):
                    if T[a, b] >= 0:
                        J[cq[a], ci[b], T[a, b]] += 1
            clus[f"{f}_{nm}"] = J / J.sum()
    np.savez(os.path.join(OUT, "clusters.npz"), **clus)
    np.savez(os.path.join(OUT, "data.npz"), imgs=imgs, Ximg=Ximg, Cimg=Cimg,
             **{f"{k}__{f}": v for f, d in store.items() for k, v in d.items() if k != "text"})


if __name__ == "__main__":
    main()
