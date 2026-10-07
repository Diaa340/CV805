"""Kinds of cross-modal knowledge v2 (paper/preregistration_kinds_v2.md): mismatched-drift control with a mismatch
check, independent task 2 (damage-matched), bias-only threshold refit, label split. Output: runs/<out>.csv"""
import argparse, os, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from kinds_kill import F1, F2, ITEMS, balanced_table, linear_check, pairs, perm
from kinds_2x2 import run_task2
from route_exps import Net, train
dev = "cuda"; ROOT = os.path.dirname(os.path.abspath(__file__))
POOLS = {"A": perm[10:20], "B": perm[20:30]}; CL = (1, 3, 10); IL = (1, 3, 10, 30, 100)
RET = torch.tensor(np.concatenate([ITEMS[1][c] for c in POOLS["A"]]), device=dev)  # 1,000 held-out pool-A test items


@torch.no_grad()
def retrieval(e1, e2):
    a, b = F.normalize(e1(F1[RET]), dim=1), F.normalize(e2(F2[RET]), dim=1)
    return ((a @ b.T).argmax(1) == torch.arange(len(RET), device=dev)).float().mean().item()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[28, 29, 30, 31, 32])
    ap.add_argument("--out", default="kinds_v2"); args = ap.parse_args(); out = os.path.join(ROOT, "runs", f"{args.out}.csv")
    for seed in args.seeds:
        rng = np.random.default_rng(seed); T = balanced_table(rng); attr = rng.permutation([0] * 5 + [1] * 5)
        for kind in ("comparative", "conjunctive", "xor"):
            t0 = time.time(); torch.manual_seed(seed)
            tr, te = pairs(kind, 50000, 0, rng, T, attr), pairs(kind, 20000, 1, rng, T, attr)
            net1 = Net(512, 512, 2).to(dev); train(net1, tr, 30, 1e-3)
            for p in net1.parameters():
                p.requires_grad_(False)
            x1, x2, y = te; c1, c2, cy = tr[0][:500], tr[1][:500], tr[2][:500]  # 500 labelled task-1 pairs for the refit
            mg = lambda a, b: (lambda l: l[:, 1] - l[:, 0])(net1.head(torch.cat([a, b], 1)))
            with torch.no_grad():
                o1, o2, q1, q2 = net1.e1(x1), net1.e2(x2), net1.e1(c1), net1.e2(c2)
                ok0 = mg(o1, o2) > 0
                ok0 = ok0 == y.bool()
                ztr = torch.cat([net1.e1(tr[0][:20000]), net1.e2(tr[1][:20000])], 1)
            lin = linear_check(ztr, tr[2][:20000], torch.cat([o1, o2], 1), y); acc0 = ok0.float().mean().item()
            S = {"A": run_task2(net1, "coupled", POOLS["A"], 5000 + seed), "B1": run_task2(net1, "coupled", POOLS["A"], 6000 + seed),
                 "B2": run_task2(net1, "coupled", POOLS["B"], 8000 + seed), "I": run_task2(net1, "independent", POOLS["A"], 7000 + seed, IL)}
            for lev in IL:
                conds = [("independent", S["I"][lev], S["I"][lev])]
                if lev in CL:
                    conds += [("matched", S["A"][lev], S["A"][lev])]
                    for b in ("B1", "B2"):
                        conds += [(f"mis{b}_AB", S["A"][lev], S[b][lev]), (f"mis{b}_BA", S[b][lev], S["A"][lev])]
                for cond, (e1a, _), (_, e2b) in conds:
                    with torch.no_grad():
                        n1, n2, p1, p2 = e1a(x1), e2b(x2), e1a(c1), e2b(c2)
                        M = {"1": (mg(n1, o2), mg(p1, q2)), "2": (mg(o1, n2), mg(q1, p2)), "12": (mg(n1, n2), mg(p1, p2))}
                    yb = y.bool(); cyb = cy.bool(); acc = lambda s: s.float().mean().item(); k = lambda s: (acc(s) - 0.5) / (acc0 - 0.5)
                    surv = {m: (v[0] > 0) == yb for m, v in M.items()}

                    def refit_sup(mt, mc):  # one bias number, chosen to maximise accuracy on 500 labelled task-1 pairs
                        cand = torch.cat([-mc, torch.tensor([0.0, 1e6, -1e6], device=dev)])
                        best = cand[((mc[None] + cand[:, None] > 0) == cyb[None]).float().mean(1).argmax()]
                        return (mt + best > 0) == yb
                    sup = {m: refit_sup(*v) for m, v in M.items()}
                    free = (M["12"][0] - M["12"][0].median() > 0) == yb
                    r = dict(seed=seed, kind=kind, level=lev, cond=cond, acc0=acc0, lin_acc=lin,
                             k1=k(surv["1"]), k2=k(surv["2"]), k12=k(surv["12"]), k12_sup=k(sup["12"]), k12_free=k(free),
                             k1_sup=k(sup["1"]), k2_sup=k(sup["2"]), retrieval=retrieval(e1a, e2b),
                             drift1=1 - F.cosine_similarity(n1, o1, 1).mean().item(), drift2=1 - F.cosine_similarity(n2, o2, 1).mean().item())
                    for lab, m in (("all", ok0), ("y1", ok0 & yb), ("y0", ok0 & ~yb)):
                        s1, s2, s12 = surv["1"][m], surv["2"][m], surv["12"][m]
                        P1, P2, R, b = acc(s1), acc(s2), acc(s12), acc(s1 & s2)
                        r.update({f"P1_{lab}": P1, f"P2_{lab}": P2, f"r12_{lab}": R, f"C_{lab}": b - P1 * P2, f"N_{lab}": R - b, f"n_{lab}": int(m.sum())})
                    pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"s{seed} {kind:11s} done ({time.time() - t0:.0f}s) acc0 {acc0:.3f} lin {lin:.3f}", flush=True)


if __name__ == "__main__":
    main()
