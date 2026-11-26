"""Mismatched-drift control + independent task 2 (paper/preregistration_kinds_2x2.md). Output (appended): runs/kinds_2x2.csv"""
import argparse, copy, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from kinds_kill import F1, F2, ITEMS, balanced_table, linear_check, pairs, perm
from route_exps import Net, train
dev = "cuda"; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLS = {"A": perm[10:20], "B": perm[20:30]}; LEVELS = (1, 3, 10); ILEVELS = (1, 3, 10, 30, 100); SPE = 50000 // 256


def pool_items(pool):
    idx = np.concatenate([ITEMS[0][c] for c in pool]); lab = np.concatenate([[k] * len(ITEMS[0][c]) for k, c in enumerate(pool)])
    return torch.tensor(idx, device=dev), torch.tensor(lab, device=dev)


def run_task2(net1, mode, pool, seed, levels=LEVELS, bs=256, tau=0.1):
    """Continuing task-2 run; returns {level: (e1, e2)} snapshots. mode: coupled (InfoNCE) or independent (own CE heads)."""
    net = copy.deepcopy(net1); e1, e2 = net.e1, net.e2
    for p in list(e1.parameters()) + list(e2.parameters()):
        p.requires_grad_(True)
    idx, lab = pool_items(pool); params = list(e1.parameters()) + list(e2.parameters())
    if mode == "independent":
        h1, h2 = torch.nn.Linear(128, 10).to(dev), torch.nn.Linear(128, 10).to(dev); params += list(h1.parameters()) + list(h2.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5); g = torch.Generator(device=dev).manual_seed(seed)
    snaps, done = {}, 0
    for lev in levels:
        for _ in range((lev - done) * SPE):
            j = torch.randint(0, len(idx), (bs,), device=dev, generator=g); i = idx[j]
            if mode == "coupled":
                a, b = F.normalize(e1(F1[i]), dim=1), F.normalize(e2(F2[i]), dim=1); lo = a @ b.T / tau; t = torch.arange(bs, device=dev)
                loss = (F.cross_entropy(lo, t) + F.cross_entropy(lo.T, t)) / 2
            else:
                loss = F.cross_entropy(h1(e1(F1[i])), lab[j]) + F.cross_entropy(h2(e2(F2[i])), lab[j])
            opt.zero_grad(); loss.backward(); opt.step()
        done = lev; snaps[lev] = (copy.deepcopy(e1).eval(), copy.deepcopy(e2).eval())
    return snaps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[28, 29, 30, 31, 32])
    ap.add_argument("--out", default="kinds_2x2"); args = ap.parse_args(); out = os.path.join(ROOT, "runs", f"{args.out}.csv")
    for seed in args.seeds:
        rng = np.random.default_rng(seed); T = balanced_table(rng); attr = rng.permutation([0] * 5 + [1] * 5)
        for kind in ("comparative", "conjunctive", "xor"):
            t0 = time.time(); torch.manual_seed(seed)
            tr, te = pairs(kind, 50000, 0, rng, T, attr), pairs(kind, 20000, 1, rng, T, attr)
            net1 = Net(512, 512, 2).to(dev); train(net1, tr, 30, 1e-3)
            for p in net1.parameters():
                p.requires_grad_(False)
            x1, x2, y = te
            with torch.no_grad():
                o1, o2 = net1.e1(x1), net1.e2(x2); ok0 = net1.head(torch.cat([o1, o2], 1)).argmax(1) == y
                ztr = torch.cat([net1.e1(tr[0][:20000]), net1.e2(tr[1][:20000])], 1)
            lin = linear_check(ztr, tr[2][:20000], torch.cat([o1, o2], 1), y); acc0 = ok0.float().mean().item()
            S = {"A": run_task2(net1, "coupled", POOLS["A"], 5000 + seed), "B": run_task2(net1, "coupled", POOLS["B"], 6000 + seed),
                 "I": run_task2(net1, "independent", POOLS["A"], 7000 + seed, ILEVELS)}
            for lev in ILEVELS:
                with torch.no_grad():
                    z = {k: (S[k][lev][0](x1), S[k][lev][1](x2)) for k in S if lev in S[k]}
                    h = lambda a, b: net1.head(torch.cat([a, b], 1)).argmax(1) == y
                    cosd = lambda a, b: 1 - F.cosine_similarity(a, b, 1).mean().item()
                    coord = F.cosine_similarity((z["A"][1] - o2).flatten(), (z["B"][1] - o2).flatten(), 0).item() if "A" in z else np.nan
                    conds = [("independent", z["I"])] + ([("matched", z["A"]), ("mismatched_AB", (z["A"][0], z["B"][1])),
                                                          ("mismatched_BA", (z["B"][0], z["A"][1]))] if "A" in z else [])
                    for cond, (n1, n2) in conds:
                        s1, s2, s12 = h(n1, o2), h(o1, n2), h(n1, n2)
                        acc = lambda s: s.float().mean().item(); k = lambda s: (acc(s) - 0.5) / (acc0 - 0.5)
                        r = dict(seed=seed, kind=kind, level=lev, cond=cond, acc0=acc0, lin_acc=lin, k1=k(s1), k2=k(s2), k12=k(s12),
                                 E=k(s12) - k(s1) * k(s2), P1=acc(s1[ok0]), P2=acc(s2[ok0]), r12=acc(s12[ok0]), P1n2=acc((s1 & s2)[ok0]),
                                 drift1=cosd(n1, o1), drift2=cosd(n2, o2), coordAB_e2=coord)
                        r["C"] = r["P1n2"] - r["P1"] * r["P2"]; r["N"] = r["r12"] - r["P1n2"]
                        pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                        print(f"s{seed} {kind:11s} L{lev:2d} {cond:13s} k1 {r['k1']:.2f} k2 {r['k2']:.2f} k12 {r['k12']:.2f} E {r['E']:+.3f} "
                              f"C {r['C']:+.3f} N {r['N']:+.3f} drift {r['drift1']:.2f}/{r['drift2']:.2f}", flush=True)
            print(f"  ({time.time() - t0:.0f}s) acc0 {acc0:.3f} lin {lin:.3f}", flush=True)


if __name__ == "__main__":
    main()
