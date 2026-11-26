"""Revision R5 (paper/preregistration_revision.md): how does the exception share s(T) depend on the table's size?
Sources (complete tables): the 40 CLEVR continual-learning units (10 questions x 1,000 images), the 24 GQA units
(10 x 200) and the 43 CLOVE units (10 x 200). For each unit, column subsets of n images (CLEVR n in 25, 50, 100, 200,
400, 1000; GQA / CLOVE n in 25, 50, 100, 200), 5 random draws per n (1 draw when n is the full width), with all 10
questions and with 5 random questions. Exact rule per sub-table (MILP, 120 s; the proven lower bound is kept).
Reported (descriptive): mean s by n and rows; the share of variance in s between units vs between draws of the same
unit at n = 50 and n = 100 (intraclass correlation); Spearman correlation of unit rankings across n.
Writes runs/rev/stability.csv and paper/results_rev_stability.txt"""
import os, zlib
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "runs", "rev"); os.makedirs(OUT, exist_ok=True)


def sources():
    Z = np.load(os.path.join(ROOT, "runs", "clevr2", "data.npz")); TS = np.load(os.path.join(ROOT, "runs", "clx", "train_structure.npz"))
    for u in TS["units"]:
        yield "clevr", u, Z[f"T__{u}"].astype(int), (25, 50, 100, 200, 400, 1000)
    Q = np.load(os.path.join(ROOT, "runs", "gqa", "units.npz"))
    for u in pd.read_csv(os.path.join(ROOT, "runs", "gqa", "units.csv")).query("keep").unit:
        yield "gqa", u, Q[f"T__{u}"].astype(int), (25, 50, 100, 200)
    C = np.load(os.path.join(ROOT, "runs", "clove", "units.npz"))
    for u in pd.read_csv(os.path.join(ROOT, "runs", "clove", "units.csv")).query("keep").unit:
        yield "clove", u, C[f"T__{u}"].astype(int), (25, 50, 100, 200)


def job(a):
    from exact_additive import exact_partial_fit
    b, u, n, rows, d, T = a; P, best, up, m = exact_partial_fit(T, np.ones(T.shape), 120)
    return dict(bench=b, unit=u, n=n, rows=rows, draw=d, s=float((P != T).mean()), s_low=float(1 - up), yes=float(T.mean()))


if __name__ == "__main__":
    jobs = []
    for b, u, T, ns in sources():
        for rows in (10, 5):
            for n in ns:
                for d in range(1 if n >= T.shape[1] else 5):
                    r = np.random.default_rng(zlib.crc32(f'{u}|{n}|{rows}|{d}'.encode())); cols = np.sort(r.choice(T.shape[1], n, replace=False)); qs = np.sort(r.choice(10, rows, replace=False))
                    jobs.append((b, u, n, rows, d, T[np.ix_(qs, cols)]))
    print(len(jobs), "sub-tables", flush=True)
    with Pool(12) as p:
        R = pd.DataFrame(p.map(job, jobs, chunksize=4))
    R.to_csv(os.path.join(OUT, "stability.csv"), index=False); lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("Mean exception share s by number of images n (10 questions / 5 questions):")
    for b, x in R.groupby("bench"):
        say(f"  {b:6s} " + " | ".join(f"n={n}: {g[g.rows == 10].s.mean():.3f} / {g[g.rows == 5].s.mean():.3f}" for n, g in x.groupby("n")))
    say("\nBetween-unit share of variance in s (intraclass correlation over draws), 10 questions:")
    for b, x in R[R.rows == 10].groupby("bench"):
        parts = []
        for n in (50, 100):
            g = x[x.n == n]; wv = g.groupby("unit").s.var().mean(); bv = g.groupby("unit").s.mean().var(); parts.append(f"n={n}: {bv / (bv + wv):.2f}")
        say(f"  {b:6s} " + ", ".join(parts))
    say("\nSpearman correlation of unit rankings (mean s per unit) between n = 50 and the largest n, 10 questions:")
    for b, x in R[R.rows == 10].groupby("bench"):
        m = x.groupby(["unit", "n"]).s.mean().unstack(); say(f"  {b:6s} {m[50].corr(m[m.columns.max()], method='spearman'):+.2f}")
    open(os.path.join(ROOT, "paper", "results_rev_stability.txt"), "w").write("\n".join(lines) + "\n")
