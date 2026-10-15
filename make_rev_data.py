"""Exports the data behind every table and figure of the revised paper into paper/revision/data/ (CSV), from the saved
results. Run: .venv/bin/python make_rev_data.py"""
import glob, os, pickle, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "paper", "revision", "data"); os.makedirs(OUT, exist_ok=True)
P = lambda *a: os.path.join(ROOT, *a)
def save(df, name): df.to_csv(os.path.join(OUT, name), index=False); print(f"{name}: {len(df)} rows")

# 01 controlled tables (Sec. 4.1; supplement Tab. sup_controlled)
S = np.load(P("runs", "tables", "selected.npz"))
M = pd.DataFrame(dict(table=range(len(S["T"])), xor_density=S["dens"], exception_share=S["d_add"], syn_mmi=S["syn_mmi"]))
M = M.merge(pd.read_csv(P("runs", "tables", "pid.csv"))[["table", "syn_broja", "syn_ccs"]], on="table")
R = pd.read_csv(P("runs", "tables", "train.csv")).groupby("table", as_index=False).mean(numeric_only=True)
A = M.merge(R[["table", "acc_full", "acc_lin", "acc_maj"]], on="table").assign(set="selected")
F = np.load(P("runs", "tables", "fresh.npz"))
MF = pd.DataFrame(dict(table=range(len(F["T"])), xor_density=F["dens"], exception_share=F["d_add"]))
MF = MF.merge(pd.read_csv(P("runs", "tables", "pid_fresh.csv"))[["table", "syn_broja", "syn_ccs"]], on="table")
RF = pd.concat([pd.read_csv(f) for f in glob.glob(P("runs", "tables", "step1_fresh_s*.csv"))]).groupby("table", as_index=False).mean(numeric_only=True)
B = MF.merge(RF[["table", "acc_before", "acc_lin", "acc_maj"]].rename(columns={"acc_before": "acc_full"}), on="table").assign(set="fresh_replication")
C = pd.concat([A, B]); C["learned"] = C.acc_full - C.acc_maj >= 0.10; C["readout_gap"] = (C.acc_full - C.acc_lin) / (C.acc_full - C.acc_maj)
save(C[["set", "table", "exception_share", "xor_density", "syn_broja", "syn_ccs", "syn_mmi", "acc_full", "acc_lin", "acc_maj", "readout_gap", "learned"]].round(5),
     "01_controlled_tables.csv")

# 02-04 VLM per-table data
ren = {"key": "table", "s": "exception_share", "s_low": "exception_share_lower_bound", "pid": "syn_broja", "acc": "accuracy", "acc_emap": "accuracy_emap", "delta": "emap_drop"}
save(pd.read_csv(P("paper", "figs", "vlm3_units.csv")).rename(columns=ren | {"mmi": "syn_mmi", "acc_exc": "acc_exceptions", "acc_rule": "acc_rule_cells"}).round(5),
     "02_vlm_studyA_full_tables.csv")
save(pd.read_csv(P("paper", "figs", "vlm3_decor_units.csv")).drop(columns=["acc_exc", "acc_rule"], errors="ignore").rename(columns=ren | {"mmi": "syn_mmi"}).round(5),
     "03_vlm_studyB_subtables.csv")
save(pd.read_csv(P("paper", "figs", "rev_null_units.csv")).rename(columns=ren | {"yes": "yes_share", "delta_null": "emap_drop_null", "acc_null": "accuracy_null",
     "excess": "emap_drop_minus_null", "slack": "slack_above_floor"}).round(5), "04_vlm_studyC_subtables_with_null.csv")

# 05 VLM partial correlations (Studies B, C, null; Tab. null, Fig. 1 VLM panel, supplement)
rows = [("B", "Qwen2.5-VL-7B", "rho(drop, s | PID, acc)", 0.325, 0.102, 0.501), ("B", "Qwen2.5-VL-7B", "rho(drop, PID | s, acc)", 0.188, np.nan, np.nan),
        ("B", "InternVL3.5-8B", "rho(drop, s | PID, acc)", 0.368, 0.169, 0.537), ("B", "InternVL3.5-8B", "rho(drop, PID | s, acc)", 0.437, np.nan, np.nan),
        ("B", "LLaVA-OV-1.5-8B", "rho(drop, s | PID, acc)", 0.317, 0.091, 0.503), ("B", "LLaVA-OV-1.5-8B", "rho(drop, PID | s, acc)", 0.427, np.nan, np.nan),
        ("C", "Idefics3-8B", "rho(drop, s | PID, acc, yes)", 0.416, 0.212, 0.559), ("C", "Idefics3-8B", "rho(drop, PID | s, acc, yes)", 0.065, -0.167, 0.285),
        ("C", "InternVL3.5-8B", "rho(drop, s | PID, acc, yes)", 0.418, 0.197, 0.584), ("C", "InternVL3.5-8B", "rho(drop, PID | s, acc, yes)", 0.048, -0.157, 0.261),
        ("C", "LLaVA-OV-1.5-8B", "rho(drop, s | PID, acc, yes)", 0.433, 0.228, 0.584), ("C", "LLaVA-OV-1.5-8B", "rho(drop, PID | s, acc, yes)", 0.239, 0.022, 0.417),
        ("C", "Qwen2.5-VL-7B", "rho(drop, s | PID, acc, yes)", 0.260, 0.059, 0.440), ("C", "Qwen2.5-VL-7B", "rho(drop, PID | s, acc, yes)", 0.027, -0.181, 0.252)]
names = {"Idefics3-8B (confirmatory model of Study C)": "Idefics3-8B", "InternVL3.5-8B": "InternVL3.5-8B", "LLaVA-OV-1.5-8B": "LLaVA-OV-1.5-8B", "Qwen2.5-VL-7B": "Qwen2.5-VL-7B"}
what = {"rho_obs": "rho(drop, s | PID, acc, yes) [re-bootstrapped]", "rho_null": "rho(null drop, s | PID, acc, yes)", "rho_excess": "rho(drop - null drop, s | PID, acc, yes)",
        "rho_adj": "rho(drop, s | PID, acc, yes, null drop)", "rho_slack": "rho(drop - (s - error), s | PID, acc, yes)", "rho_yes": "rho(drop, yes | s, PID, acc)"}
cur = None
for line in open(P("paper", "results_rev_null.txt")):
    m = re.match(r"--- (.*?) ---", line)
    if m:
        cur = names[m.group(1)]
    m = re.match(r"\s+(rho_\w+)\s+([+-][\d.]+) \[([+-][\d.]+),([+-][\d.]+)\]", line)
    if m and cur:
        rows.append(("C + null", cur, what[m.group(1)], float(m.group(2)), float(m.group(3)), float(m.group(4))))
save(pd.DataFrame(rows, columns=["study", "model", "quantity", "rho", "ci_low", "ci_high"]), "05_vlm_partial_correlations.csv")

# 06 cell-level AUC (Sec. 4.2 "Which cells"; supplement)
auc = [("full tables (solver's labels)", "Qwen2.5-VL-7B", 0.675, np.nan, 0.866, 0.735), ("full tables (solver's labels)", "InternVL3.5-8B", 0.623, np.nan, 0.783, 0.672),
       ("full tables (solver's labels)", "LLaVA-OV-1.5-8B", 0.695, np.nan, 0.880, 0.639)]
for line in open(P("paper", "results_rev_unique.txt")):
    m = re.match(r"\s+(idefics|internvl|llava|qwen)\s+stable exception label ([\d.]+).*any optimal rule ([\d.]+) \| graded misfit ([\d.]+) \| low confidence ([\d.]+)", line)
    if m:
        auc.append(("Study C sub-tables", {"idefics": "Idefics3-8B", "internvl": "InternVL3.5-8B", "llava": "LLaVA-OV-1.5-8B", "qwen": "Qwen2.5-VL-7B"}[m.group(1)],
                    float(m.group(3)), float(m.group(2)), float(m.group(4)), float(m.group(5))))
save(pd.DataFrame(auc, columns=["tables", "model", "auc_exception_label", "auc_stable_exception_label", "auc_graded_misfit", "auc_low_confidence"]), "06_vlm_cell_auc.csv")

# 07 continual-learning main table (Tab. main), points
res = pickle.load(open(P("runs", "table_main.pkl"), "rb"))
save(pd.DataFrame([dict(benchmark=b, method=m, **{k: 100 * v for k, v in d.items()}) for (b, m), d in res.items()]).rename(
     columns={"ACC": "ACC", "BWT": "BWT", "EACC": "E_ACC", "dB": "delta_B_same_answer", "dF": "delta_F_same_answer"}).round(2), "07_cl_main_table_points.csv")

# 08 same-answer gaps with intervals (Fig. gaps), fractions
save(pd.read_csv(P("paper", "figs", "strat_metrics.csv")).round(5), "08_cl_same_answer_gaps.csv")

# 09 difficulty-matched gaps (Tab. difficulty; supplement), points
rows, bench = [], None
for line in open(P("paper", "results_rev_difficulty.txt")):
    m = re.match(r"=== (\w+):", line)
    if m:
        bench = m.group(1)
    m = re.match(r"\s+([A-Za-z+\-. ()]+?)\s+\|(.*)$", line)
    if m and bench and "method" not in m.group(1):
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) == 6:
            vals = []
            for c in cells:
                n = re.findall(r"[+-]?\d+\.\d+", c); vals += [float(n[0]), float(n[1]), float(n[2]), float(n[3]) if len(n) > 3 else np.nan]
            cols = [f"{d}_{k}_{q}" for d in ("backward", "forward") for k in ("same_answer", "difficulty_matched", "vs_hardest_rule_quarter") for q in ("gap", "ci_low", "ci_high", "coverage")]
            rows.append(dict(benchmark=bench, method=m.group(1).strip(), **dict(zip(cols, vals))))
D9 = pd.DataFrame(rows); D9 = D9.drop(columns=[c for c in D9.columns if c.endswith("coverage") and "difficulty_matched" not in c]); save(D9, "09_cl_difficulty_matched_points.csv")

# 10 uniqueness of exceptions (supplement Tabs. unique, unique_cl)
save(pd.read_csv(P("runs", "rev", "unique.csv")).round(5), "10a_exception_uniqueness_per_table.csv")
rows = []
for line in open(P("paper", "results_rev_unique.txt")):
    m = re.match(r"\s+(backward|forward)\s+(\w+)\s+(\w+)\s+saved\s+([+-]?[\d.]+) \[\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)\] \| stable\s+([+-]?[\w.]+) \[\s*([+-]?[\w.]+),\s*([+-]?[\w.]+)\]", line)
    if m:
        rows.append(dict(direction=m.group(1), benchmark=m.group(2), method=m.group(3), saved_gap=float(m.group(4)), saved_ci_low=float(m.group(5)), saved_ci_high=float(m.group(6)),
                         stable_gap=float(m.group(7)), stable_ci_low=float(m.group(8)), stable_ci_high=float(m.group(9))))
save(pd.DataFrame(rows), "10b_cl_gaps_saved_vs_stable_labels_points.csv")

# 11 buffer control (supplement Tab. der_buffers)
save(pd.read_csv(P("paper", "figs", "strat_buffers.csv")).round(5), "11_cl_buffer_control.csv")

# 12 VLM continual learning on CLOVE (Tab. vlm_cl), points
V = pd.read_csv(P("paper", "figs", "clove_vlm_metrics.csv"))
acc = {m: res[("clove_vlm", k)] for m, k in (("finetune", "finetune"), ("ewc", "ewc"), ("lwf_1", "lwf"), ("er", "er"), ("derpp", "derpp"), ("derpp_ruleexc", "probe"))}
V["ACC"] = V.method.map(lambda m: 100 * acc[m]["ACC"]); V["E_ACC"] = V.method.map(lambda m: 100 * acc[m]["EACC"]); V["BWT_overall"] = V.method.map(lambda m: 100 * acc[m]["BWT"])
z = res[("clove_vlm", "zeroshot")]; V = pd.concat([pd.DataFrame([dict(method="zeroshot", ACC=100 * z["ACC"], E_ACC=100 * z["EACC"])]), V])
save(V.round(4), "12_vlm_cl_clove.csv")

# 13 size dependence of s (supplement Tab. sup_stability)
St = pd.read_csv(P("runs", "rev", "stability.csv")); save(St.round(5), "13a_stability_subtables.csv")
save(St.groupby(["bench", "rows", "n"], as_index=False).agg(mean_s=("s", "mean"), sd_s=("s", "std"), subtables=("s", "size")).round(5), "13b_stability_summary.csv")

# 14 decorrelated selections (supplement Fig. decor)
for nm, f in (("B", "selected.csv"), ("C", "selected2.csv")):
    x = pd.read_csv(P("runs", "vlm3", f))[["cid", "key", "bench", "s", "s_low", "exact", "broja", "mmi", "yes", "cols"]].assign(study=nm)
    save(x.rename(columns={"cid": "subtable", "key": "source_table", "s": "exception_share", "broja": "syn_broja", "mmi": "syn_mmi", "yes": "yes_share", "cols": "image_columns"}).round(5),
         f"14{'a' if nm == 'B' else 'b'}_decorrelated_selection_study{nm}.csv")
