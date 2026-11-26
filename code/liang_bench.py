"""Liang et al.'s PID estimate (their default pipeline: normalize + PCA(5) + KMeans(30) per modality, CVX solver) on the
test split of each frozen-feature benchmark. Run with .venv_pid. Writes runs/liang_bench.csv"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job(name):
    from liang_pid import pid_bits
    Z = np.load(os.path.join(ROOT, "data", "feats", f"{name}.npz"), allow_pickle=True); s = Z["split"] == 1
    cl = lambda X: KMeans(n_clusters=30, random_state=0, n_init=10).fit(PCA(n_components=5).fit_transform(normalize(np.nan_to_num(X)))).labels_
    c1, c2 = cl(Z["f1"][s].astype(np.float32)), cl(Z["f2"][s].astype(np.float32)); y = pd.factorize(Z["y"][s])[0]
    P = np.zeros((30, 30, y.max() + 1)); np.add.at(P, (c1, c2, y), 1); P /= P.sum()
    r = pid_bits(P); H = -(lambda p: (p[p > 0] * np.log2(p[p > 0])).sum())(np.bincount(y) / len(y))
    print(name, r, flush=True)
    return dict(data=name, H_y=H, **{k: float(v) for k, v in r.items()})


if __name__ == "__main__":
    names = ["avmnist", "cifar2v", "clevr", "cremad", "food101", "n24news", "vqa"]
    with Pool(7) as p:
        rows = p.map(job, names)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "runs", "liang_bench.csv"), index=False)
