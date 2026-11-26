"""Main results table (paper/table_main.tex): standard CL metrics next to exception-aware ones, per benchmark and method.
Matched plasticity as in the pre-registered analyses. Same-answer gaps from figures/strat_metrics.csv (and
clevr_vlm_metrics.csv when present). Two-task: ACC = mean final accuracy of the two tasks; BWT = task-1 change;
E-ACC = mean final exception accuracy of the two tasks. Six-task: ACC = mean final accuracy over tasks; BWT = mean over
tasks 1-5 of (final - right after the task); E-ACC = final exception accuracy. VLM test sets oversample exceptions, so
ACC / BWT there weight each task's rule and exception accuracy by the task's natural exception share."""
import glob, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); S = pd.read_csv(os.path.join(ROOT, "figures", "strat_metrics.csv"))
G = [("ewc", ("1", "100", "10000")), ("si", ("0.1", "1", "10", "100")), ("lwf", ("0.3", "3")), ("l2", ("0.3", "3")), ("icarl", ("0.1", "1"))]
TWO = {"er": "er_random_1000", "agem": "agem_random_1000", "gem": "gem_random_1000", "derpp": "derpp_random_1000", "der_expand": "der_expand_1000", "probe": "derpp_ruleexc_1000"}
SEQ = {"probe": "derpp_ruleexc"}
res = {}
for b in ("tables", "clevr", "gqa"):
    Y = np.load(os.path.join(ROOT, "runs", "clm", f"y_{b}.npz"))
    D1 = pd.concat([pd.read_csv(f) for f in Y["task1_files"]]).reset_index(drop=True); D2 = pd.concat([pd.read_csv(f) for f in Y["task2_files"]]).reset_index(drop=True)
    a2 = D2.groupby("unit").mean(numeric_only=True); a1 = D1.groupby("unit").mean(numeric_only=True)
    for fam, lams in G:
        pick = {u: (max(ok, key=lambda l: a1.loc[u, f"{fam}_{l}"]) if (ok := [l for l in lams if a2.loc[u, f"{fam}_{l}"] >= a2.loc[u, "finetune"] - 0.02]) else lams[0]) for u in a2.index}
        for D in (D1, D2):
            D[fam] = 0.0
            for l in lams:
                m = D.unit.map(pick) == l; D.loc[m, fam] = D.loc[m, f"{fam}_{l}"].astype(float)
    for m in ["finetune", "ewc", "si", "lwf", "l2", "er", "agem", "gem", "derpp", "icarl", "der_expand", "multihead", "probe"]:
        c = TWO.get(m, m); s = S[(S.bench == b) & (S.method == c)].iloc[0]
        res[(b, m)] = dict(ACC=0.5 * (D1[c].mean() + D2[c].mean()), BWT=D1[c].mean() - D1.before.mean(),
                           EACC=0.5 * (D1[D1.nonadd == 1][c].mean() + D2[D2.nonadd == 1][c].mean()), dB=s.dBWT_s, dF=s.dFWT_s)
C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", "clove_cl", "run_[0-9]*.csv.gz")))])
for fam, lams in G:
    late = C[C.pos >= 2].groupby("run").mean(numeric_only=True); allr = C.groupby("run").mean(numeric_only=True); pick = {}
    for r in late.index:
        ok = [l for l in lams if late.loc[r, f"{fam}_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]; pick[r] = max(ok, key=lambda l: allr.loc[r, f"{fam}_{l}__final"]) if ok else lams[0]
    for s_ in ("own", "final"):
        C[f"{fam}__{s_}"] = 0.0
        for l in lams:
            mm = C.run.map(pick) == l; C.loc[mm, f"{fam}__{s_}"] = C.loc[mm, f"{fam}_{l}__{s_}"]
for m in ["finetune", "ewc", "si", "lwf", "l2", "er", "agem", "gem", "derpp", "icarl", "der_expand", "multihead", "probe"]:
    c = SEQ.get(m, m); s = S[(S.bench == "clove") & (S.method == c)].iloc[0]; e5 = C[C.pos <= 5]
    res[("clove", m)] = dict(ACC=C[f"{c}__final"].mean(), BWT=e5[f"{c}__final"].mean() - e5[f"{c}__own"].mean(), EACC=C[C.nonadd == 1][f"{c}__final"].mean(), dB=s.dBWT_s, dF=s.dFWT_s)


def vlm(path_glob, bench_key, strat_df, names, natural_share):
    V = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(path_glob))])
    late = V[V.pos >= 2].groupby("order").mean(numeric_only=True); allr = V.groupby("order").mean(numeric_only=True); pick = {}
    for r in late.index:
        ok = [l for l in ("100", "10000") if late.loc[r, f"ewc_{l}__own"] >= late.loc[r, "finetune__own"] - 0.02]; pick[r] = max(ok, key=lambda l: allr.loc[r, f"ewc_{l}__final"]) if ok else "100"
    for s_ in ("own", "final"):
        V[f"ewc__{s_}"] = 0.0
        for l in ("100", "10000"):
            mm = V.order.map(pick) == l; V.loc[mm, f"ewc__{s_}"] = V.loc[mm, f"ewc_{l}__{s_}"]
    def wacc(X, col):   # natural-share weighted accuracy, averaged over tasks
        out = []
        for tk, x in X.groupby("task"):
            p = natural_share[tk]; out.append(p * x[x.nonadd == 1][col].mean() + (1 - p) * x[x.nonadd == 0][col].mean())
        return np.mean(out)
    for m, c in names.items():
        s = strat_df[strat_df.method == c].iloc[0]; e5 = V[V.pos <= 5]
        res[(bench_key, m)] = dict(ACC=wacc(V, f"{c}__final"), BWT=wacc(e5, f"{c}__final") - wacc(e5, f"{c}__own"), EACC=V[V.nonadd == 1][f"{c}__final"].mean(),
                                   dB=s.dBWT_s, dF=s.dFWT_s)
    res[(bench_key, "zeroshot")] = dict(ACC=wacc(V, "zeroshot"), BWT=np.nan, EACC=V[V.nonadd == 1].zeroshot.mean(), dB=np.nan, dF=np.nan)


import pickle
from vlm_seq import cells
TK = cells(); share = {tk: TK[tk]["test"].nonadd.mean() for tk in TK}
vlm(os.path.join(ROOT, "runs", "clove_vlm", "order_[0-9].csv.gz"), "clove_vlm", S[S.bench == "clove_vlm"],
    {"finetune": "finetune", "ewc": "ewc", "lwf": "lwf_1", "er": "er", "derpp": "derpp", "probe": "derpp_ruleexc"}, share)
if glob.glob(os.path.join(ROOT, "runs", "clevr_vlm", "order_[0-9].csv.gz")) and os.path.exists(os.path.join(ROOT, "figures", "clevr_vlm_metrics.csv")):
    from vlm_seq_clevr import tasks_and_cells
    CT = tasks_and_cells(); sh = {int(tk): CT[tk]["test"].nonadd.mean() for tk in CT}
    CV = pd.read_csv(os.path.join(ROOT, "figures", "clevr_vlm_metrics.csv")).rename(columns={"dFWT_s": "dFWT_s"}); CV["dBWT_s"] = CV.dBWT_s
    vlm(os.path.join(ROOT, "runs", "clevr_vlm", "order_[0-9].csv.gz"), "clevr_vlm", CV, {"finetune": "finetune", "ewc": "ewc", "lwf": "lwf_1", "er": "er", "derpp": "derpp",
        "er_classbal": "er_classbal", "lora_expand": "lora_expand"}, sh)
pickle.dump(res, open(os.path.join(ROOT, "runs", "table_main.pkl"), "wb")); print("computed", len(res), "cells")
