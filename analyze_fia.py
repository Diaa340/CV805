"""Pre-registered analysis of FIA predictions P-a, P-b, P-c (paper/preregistration_v4_fia.md), plus Pareto fronts and
the ablation / baseline tables. Writes paper/results_fia.txt and paper/figs/fia_pareto_<data>.png."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "paper", "figs"); os.makedirs(FIG, exist_ok=True)
lines = []


def say(s=""):
    print(s); lines.append(s)


D = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(ROOT, "runs", "fia_*_fia.csv"))], ignore_index=True)
D["buffer"] = D["buffer"].fillna("")
D["unit"] = D.data + "|" + D.order.astype(str) + "|" + D.seed.astype(str)
NAT = ("cifar2v", "cremad", "food101", "n24news", "avmnist")
say(f"FIA: {D.unit.nunique()} units, {len(D)} runs")


def matched(u, meth, col):
    """Value of `col` for `meth` at the best-S-retention lambda among lambdas with task-2 acc >= fine-tuning - 0.02."""
    ft = u[u.method == "finetune"].task2_acc.iloc[0]
    g = u[(u.method == meth) & (u.task2_acc >= ft - 0.02)]
    if not len(g) or g["ret_S"].isna().all():
        return np.nan, np.nan
    r = g.loc[g["ret_S"].idxmax()]
    return r[col], r.lam


def paired(units, a, b, col="ret_S"):
    out = []
    for _, u in units.groupby("unit"):
        va, _ = matched(u, a, col); vb, _ = matched(u, b, col)
        out.append(va - vb)
    return np.array(out, float)


say("\n=== P-a (P1 bridge): FIA - L2 in S retention at matched plasticity (need mean >= 0.05) ===")
B = D[D.data.str.startswith("bridge_")]
d = paired(B, "fia", "l2"); pa = np.nanmean(d) >= 0.05
say(f"  per unit: {np.round(d, 3)} | mean {np.nanmean(d):+.3f} -> {'PASS' if pa else 'FAIL'}")
say("\n=== P-b (natural): FIA - L2 gain on S exceeds gain on R by >= 0.03; FIA vs FIA-no-beta |dS| < 0.02 ===")
Nt = D[D.data.isin(NAT)]
gS, gR = paired(Nt, "fia", "l2", "ret_S"), paired(Nt, "fia", "l2", "ret_R")
dB = paired(Nt, "fia", "fia_nobeta", "ret_S")
pb = (np.nanmean(gS) - np.nanmean(gR) >= 0.03) and abs(np.nanmean(dB)) < 0.02
say(f"  gain S {np.nanmean(gS):+.3f}, gain R {np.nanmean(gR):+.3f}, difference {np.nanmean(gS) - np.nanmean(gR):+.3f}; "
    f"FIA - FIA_nobeta on S {np.nanmean(dB):+.3f} -> {'PASS' if pb else 'FAIL'}")
say("\n=== P-c (face-voice): at equal lambda, FIA-joint task-2 acc >= FIA-full + 0.01 with S retention within +-0.02 ===")
Fv = D[D.data == "fv_cremad"]
x = Fv[Fv.method.isin(["fia", "fia_joint"])].pivot_table(index=["unit", "lam"], columns="method", values=["task2_acc", "ret_S"])
dt = (x[("task2_acc", "fia_joint")] - x[("task2_acc", "fia")]).mean(); ds = (x[("ret_S", "fia_joint")] - x[("ret_S", "fia")]).mean()
pc = dt >= 0.01 and abs(ds) <= 0.02
say(f"  task-2 acc difference {dt:+.3f}, S retention difference {ds:+.3f} -> {'PASS' if pc else 'FAIL'}")
say("\n=== VERDICT: " + " | ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in (("P-a", pa), ("P-b", pb), ("P-c", pc))) + " ===")

say("\n=== [reported] methods at matched plasticity: mean S retention / R retention / task-2 acc / avg acc / BWT, by dataset group ===")
grp = lambda d_: "bridge" if d_.startswith("bridge_") else ("natural" if d_ in NAT else d_)
D["group"] = D.data.map(grp)
METH = ["finetune", "ewc", "lwf", "l2", "jac", "fia", "fia_joint", "fia_swap", "fia_nobeta", "fia_mixed"]
for gname, G in D.groupby("group"):
    say(f"  -- {gname} ({G.unit.nunique()} units)")
    for meth in METH:
        vals = []
        for _, u in G.groupby("unit"):
            if meth == "finetune":
                r = u[u.method == "finetune"].iloc[0]
            else:
                ft = u[u.method == "finetune"].task2_acc.iloc[0]; g = u[(u.method == meth) & (u.task2_acc >= ft - 0.02)]
                if not len(g) or g.ret_S.isna().all():
                    continue
                r = g.loc[g.ret_S.idxmax()]
            vals.append([r.ret_S, r.get("ret_R", np.nan), r.task2_acc, r.avg_acc, r.bwt])
        if vals:
            v = np.nanmean(np.array(vals, float), 0)
            say(f"     {meth:11s} S {v[0]:.3f} R {v[1]:.3f} task2 {v[2]:.3f} avg {v[3]:.3f} BWT {v[4]:+.3f} (units {len(vals)})")
    for meth, bk in (("replay", "random"), ("derpp", "random"), ("fia+buf", "random"), ("fia+buf", "needsboth"), ("fia+buf", "beta"), ("joint", ""), ("oracle_realign", "")):
        g = G[(G.method == meth) & (G.buffer == bk)]
        if len(g):
            say(f"     {meth + ('[' + bk + ']' if bk else ''):22s} S {g.ret_S.mean():.3f} R {g.get('ret_R', pd.Series(dtype=float)).mean():.3f} "
                f"task2 {g.task2_acc.mean():.3f} avg {g.avg_acc.mean():.3f} BWT {g.bwt.mean():+.3f} sec/step {g.sec_per_step.mean():.4f}")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for meth in ("lwf", "l2", "jac", "fia", "fia_joint", "ewc"):
        pts = G[G.method == meth].groupby("lam")[["task2_acc", "ret_S"]].mean()
        ax.plot(pts.task2_acc, pts.ret_S, marker="o", label=meth)
    ft = G[G.method == "finetune"][["task2_acc", "ret_S"]].mean(); ax.scatter([ft.task2_acc], [ft.ret_S], c="k", marker="x", label="finetune")
    ax.set_xlabel("task-2 accuracy"); ax.set_ylabel("needs-both (S) retention"); ax.set_title(f"Pareto: {gname}"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"fia_pareto_{gname}.png"), dpi=140); plt.close()
say("\n=== [reported] retention by interaction-share quintile (share of task-1-correct samples kept), matched plasticity, natural ===")
for meth in ("finetune", "l2", "lwf", "fia"):
    q = []
    for _, u in Nt.groupby("unit"):
        if meth == "finetune":
            r = u[u.method == "finetune"].iloc[0]
        else:
            ft = u[u.method == "finetune"].task2_acc.iloc[0]; g = u[(u.method == meth) & (u.task2_acc >= ft - 0.02)]
            if not len(g) or g.ret_S.isna().all():
                continue
            r = g.loc[g.ret_S.idxmax()]
        q.append([r[f"keep_q{i}"] for i in range(1, 6)])
    say(f"  {meth:9s} " + " ".join(f"Q{i + 1}={v:.3f}" for i, v in enumerate(np.nanmean(np.array(q, float), 0))))
open(os.path.join(ROOT, "paper", "results_fia.txt"), "w").write("\n".join(lines) + "\n")
