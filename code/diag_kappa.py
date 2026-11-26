"""POST-HOC DIAGNOSTICS for v4 H2 (not pre-registered; cannot change the v4 verdicts).
Part A: does kappa_c respond to KNOWN co-drift? Artificial linear drifts of the task-1 encoder outputs, z_m' = z_m + D_m z_m.
  D1 is random; D2 = gamma * D2_comp + (1 - gamma) * D2_rand, where D2_comp minimises the change of the fusion layer's
  first-layer input on REAL pairs when both drift (true compensation) and D2_rand is random with the same norm.
  Measured per gamma: N (ground truth), kappa_c as implemented (beta with replacement features), and an on-pair variant
  kappa_c^pair (margin distortion on the true pairs).
Part B: is there anything to predict? Spread of N across units vs bootstrap standard error within units.
Writes paper/results_diag_kappa.txt"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from route_exps import Net, load_unit, train
from v3_drift import model_types
from v4_measure import margin, profile

dev = "cuda"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = []


def say(s=""):
    print(s, flush=True); lines.append(s)


class A_:
    pass


def part_a(data, seed=23, gammas=(0.0, 0.25, 0.5, 0.75, 1.0), sizes=(0.3, 0.6)):
    a = A_(); a.data = data; a.snr = [4.5, 3.0]; a.shift = "12"; a.class_split = 0; a.order = "normal"
    U = load_unit(a, seed); c1 = U["c1"]; torch.manual_seed(seed)
    net = Net(U["tr1"][0].shape[1], U["tr1"][1].shape[1], U["C"]).to(dev); train(net, U["tr1"], 30, 1e-3, w=U["w1"])
    exact = data in ("synth", "fv_cremad") or data.startswith("bridge_")
    types = U["types"] if exact else model_types(net, U["te1"], c1, seed)
    x1, x2, y = U["te1"]; t1, t2, _ = U["tr1"]
    with torch.no_grad():
        z1, z2, o1, o2 = net.e1(x1), net.e2(x2), net.e1(t1[:20000]), net.e2(t2[:20000])
        W0 = net.head[0].weight; e = z1.shape[1]; Am, Bm = W0[:, :e], W0[:, e:]
        base_ok = (margin(net.head, z1, z2, y, c1) > 0).cpu().numpy()
    keep = base_ok & (types == "S"); say(f"  [{data}] task-1 S samples correct: {keep.sum()}")
    g = torch.Generator(device=dev).manual_seed(77)
    perms = [(torch.randperm(len(y), generator=g, device=dev), torch.randperm(len(y), generator=g, device=dev)) for _ in range(3)]
    prof0 = profile(net.head, z1, z2, y, c1, perms)
    for size in sizes:
        G = torch.randn(e, e, generator=g, device=dev)
        D1 = size * G / torch.linalg.matrix_norm(G, 2)
        with torch.no_grad():
            # D2_comp: least squares so that B D2 z2 ≈ -A D1 z1 on real train pairs (ridge)
            Y = -(o1 @ D1.T) @ Am.T                        # (n, h) target change from modality 2
            Bp = torch.linalg.pinv(Bm)                     # (e, h)
            Tz = Y @ Bp.T                                  # desired D2 z2 values (n, e)
            X = o2
            D2c = torch.linalg.solve(X.T @ X + 1e-2 * len(X) * torch.eye(e, device=dev), X.T @ Tz).T
            R = torch.randn(e, e, generator=g, device=dev); D2r = R / torch.linalg.matrix_norm(R, 2) * torch.linalg.matrix_norm(D2c, 2)
        for gm in gammas:
            D2 = gm * D2c + (1 - gm) * D2r
            M1 = lambda v, D=D1: v + v @ D.T; M2 = lambda v, D=D2: v + v @ D.T
            with torch.no_grad():
                n1, n2 = M1(z1), M2(z2)
                S1 = (margin(net.head, n1, z2, y, c1) > 0).cpu().numpy()[keep]; S2 = (margin(net.head, z1, n2, y, c1) > 0).cpu().numpy()[keep]
                S12 = (margin(net.head, n1, n2, y, c1) > 0).cpu().numpy()[keep]
                N = S12.mean() - (S1 & S2).mean()
                b0 = prof0["beta"].abs().mean()
                kj = (profile(net.head, z1, z2, y, c1, perms, (M1, M2))["beta"] - prof0["beta"]).abs().mean() / b0
                k1 = (profile(net.head, z1, z2, y, c1, perms, (M1, None))["beta"] - prof0["beta"]).abs().mean() / b0
                k2 = (profile(net.head, z1, z2, y, c1, perms, (None, M2))["beta"] - prof0["beta"]).abs().mean() / b0
                s0 = margin(net.head, z1, z2, y, c1); d = lambda s: (s - s0).abs().mean() / s0.abs().mean()
                pj, p1, p2 = d(margin(net.head, n1, n2, y, c1)), d(margin(net.head, n1, z2, y, c1)), d(margin(net.head, z1, n2, y, c1))
            say(f"    size {size:.1f} gamma {gm:.2f}: P(S1) {S1.mean():.2f} P(S2) {S2.mean():.2f} r12 {S12.mean():.2f} N {N:+.3f} | "
                f"kappa_c (implemented) {(0.5 * (k1 + k2) - kj).item():+.3f} | kappa_c^pair {(0.5 * (p1 + p2) - pj).item():+.3f}")


def part_b():
    say("\nPART B: spread of N across units vs within-unit bootstrap SE (needs-both group, runs/v4)")
    rows = []
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "v4", "v4_*.npz"))):
        z = np.load(f, allow_pickle=True); data = str(z["data"]); regime = os.path.basename(f)[:-4].rsplit("_", 1)[1]
        k = z["before"].astype(bool) & (z["types"] == "S")
        if k.sum() < 30:
            continue
        s1, s2, s12 = z["S1"][k], z["S2"][k], z["S12"][k]; n = len(s1)
        N = s12.mean() - (s1 & s2).mean()
        boot = [(lambda i: s12[i].mean() - (s1[i] & s2[i]).mean())(rng.integers(0, n, n)) for _ in range(200)]
        grp = "synth+bridge" if data == "synth" or data.startswith("bridge_") else data
        rows.append(dict(group=grp, regime=regime, N=N, se=np.std(boot), kappa_c=float(z["kappa_c"]), n=n))
    D = pd.DataFrame(rows)
    for (gname, reg), x in D.groupby(["group", "regime"]):
        sd, se = x.N.std(), np.sqrt((x.se ** 2).mean())
        rel = max(0.0, 1 - se ** 2 / sd ** 2) if sd > 0 else 0.0
        say(f"  {gname:13s} {reg:7s} units={len(x):2d} mean N {x.N.mean():+.3f} | SD across units {sd:.3f} | within-unit SE {se:.3f} | "
            f"reliability {rel:.2f} | Spearman(kappa_c, N) {spearmanr(x.kappa_c, x.N)[0]:+.2f}")
    sd, se = D.N.std(), np.sqrt((D.se ** 2).mean())
    say(f"  POOLED units={len(D)} SD across units {sd:.3f} | within-unit SE {se:.3f} | reliability {max(0, 1 - se ** 2 / sd ** 2):.2f} | "
        f"between-group share of N variance {D.groupby('group').N.transform('mean').var() / D.N.var():.2f}")


if __name__ == "__main__":
    say("POST-HOC DIAGNOSTICS (not pre-registered)\nPART A: kappa_c under KNOWN co-drift")
    for data in sys.argv[1:] or ["fv_cremad", "synth"]:
        part_a(data)
    part_b()
    open(os.path.join(ROOT, "paper", "results_diag_kappa.txt"), "w").write("\n".join(lines) + "\n")
