"""Renders paper/table_main.tex from runs/table_main.pkl (and a standalone PDF / PNG preview)."""
import os, pickle, subprocess
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); res = pickle.load(open(os.path.join(ROOT, "runs", "table_main.pkl"), "rb"))
S = pd.read_csv(os.path.join(ROOT, "figures", "strat_metrics.csv")); CV = os.path.join(ROOT, "figures", "clevr_vlm_metrics.csv")
TWO = {"er": "er_random_1000", "agem": "agem_random_1000", "gem": "gem_random_1000", "derpp": "derpp_random_1000", "der_expand": "der_expand_1000", "probe": "derpp_ruleexc_1000"}
def sig(bench, m, key):   # 95% interval entirely below zero?
    if bench == "clevr_vlm":
        if not os.path.exists(CV): return False
        r = pd.read_csv(CV).set_index("method"); c = {"lwf": "lwf_1", "probe": None}.get(m, m)
        return c in r.index and (r.loc[c, "hi"] < 0 if key == "dB" else r.loc[c, "fhi"] < 0)
    c = TWO.get(m, m) if bench in ("tables", "clevr", "gqa") else {"probe": "derpp_ruleexc", "lwf": "lwf_1" if bench == "clove_vlm" else "lwf"}.get(m, m)
    x = S[(S.bench == bench) & (S.method == c)]
    return len(x) and (x.iloc[0][f"{'dBWT' if key == 'dB' else 'dFWT'}_s_hi"] < 0)
SMALL = [("Reference", []), ("Regularisation (parameter)", [("ewc", "EWC", "Fisher"), ("si", "SI", "Path integral")]),
         ("Regularisation (distillation)", [("lwf", "LwF", "Logit distill."), ("l2", "Feature distill.", "Feature distill.")]),
         ("Memory / replay", [("er", "ER", "Buffer"), ("agem", "A-GEM", "Buffer + grad. proj."), ("gem", "GEM", "Buffer + grad. QP"), ("derpp", "DER++", "Buffer + logits"),
                              ("icarl", "iCaRL", "Herding + NME")]),
         ("Architectural", [("der_expand", "DER (expansion)", "Frozen encoders + buffer"), ("multihead", "Multi-head", "Task ID")])]
SMALL[0] = ("Reference", [("finetune", "Finetune", "None")])
VLM = [("Reference", [("zeroshot", "Zero-shot", "None"), ("finetune", "Seq-LoRA", "None")]), ("Regularisation", [("ewc", "EWC", "Fisher"), ("lwf", "LwF", "Logit distill.")]),
       ("Memory / replay", [("er", "ER", "Buffer"), ("er_classbal", "ER, class-balanced", "Balanced buffer"), ("derpp", "DER++", "Buffer + logits")]),
       ("Architectural", [("lora_expand", "LoRA expansion", "Frozen LoRAs + buffer")])]
COLS = [("tables", "Constructed"), ("clevr", "CLEVR"), ("gqa", "GQA"), ("clove", "CLOVE-scene")]; MET = ["ACC", "BWT", "EACC", "dB", "dF"]
RANKED = ["ACC", "BWT", "EACC"]; COLR = ["red!75!black", "blue!70!black", "green!45!black"]


def block(sections, colmap, probe_bench):
    names = [m for _, rows in sections for m, _, _ in rows if m != "zeroshot"]; out = []
    rank = {}
    for b in colmap.values():
        for k in RANKED:
            v = {m: res[(b, m)][k] for m in names if (b, m) in res}
            for i, m in enumerate(sorted(v, key=lambda m: -v[m])[:3]):
                rank[(b, m, k)] = i
    def cell(b, m, k):
        if b is None or (b, m) not in res or np.isnan(res[(b, m)][k]): return "\\textemdash" if b is None or (b, m) not in res else "\\textemdash"
        v = 100 * res[(b, m)][k]; s = f"{v:.1f}"
        if k in RANKED and (b, m, k) in rank: return f"\\textcolor{{{COLR[rank[(b, m, k)]]}}}{{\\textbf{{{s}}}}}"
        if k in ("dB", "dF") and m not in ("zeroshot", "probe") and sig(b, m, k): return f"\\textbf{{{s}}}"
        return s
    for title, rows in sections:
        out.append(f"\\rowcolor{{gray!12}}\\multicolumn{{{2 + 5 * len(COLS)}}}{{l}}{{\\textbf{{{title}}}}}\\\\")
        for m, nm, sigl in rows:
            out.append(f"{nm} & {sigl} & " + " & ".join(cell(colmap.get(b), m, k) for b, _ in COLS for k in MET) + "\\\\")
    out.append("\\midrule\\rowcolor{gray!12}" + f"\\multicolumn{{{2 + 5 * len(COLS)}}}{{l}}{{\\textbf{{Diagnostic probe (not a proposed method)}}}}\\\\")
    out.append("\\textit{DER++ rule+exc.} & \\textit{Exception cells in buffer} & " + " & ".join(
        ("\\textcolor{gray}{" + f"{100 * res[(colmap[b], 'probe')][k]:.1f}" + "}") if colmap.get(b) and (colmap[b], "probe") in res else "\\textemdash" for b, _ in COLS for k in MET) + "\\\\")
    return out


head = ["\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.2pt}",
        "\\caption{\\textbf{Standard vs.\\ exception-aware continual-learning metrics.} ACC, BWT and exception accuracy (E-ACC) in \\%; best, second and third per column (within each backbone) in "
        "\\textcolor{red!75!black}{red}, \\textcolor{blue!70!black}{blue}, \\textcolor{green!45!black}{green}. $\\Delta$B and $\\Delta$F: same-answer gap (exceptions minus rule) in backward / forward "
        "transfer; \\textbf{bold}: 95\\% interval below zero (exceptions lost, or learned, worse than the rule with the same answer). Gaps are only meaningful for methods that keep the rule "
        "(finetune and weak regularisers show small or positive gaps because exceptions start near chance). Constructed, CLEVR, GQA: two tasks; CLOVE-scene: six tasks.}",
        "\\label{tab:main}\\resizebox{\\textwidth}{!}{\\begin{tabular}{ll" + "|rrrrr" * len(COLS) + "}\\toprule",
        "\\textbf{Method} & \\textbf{Extra signal} & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{t}}}}}" for _, t in COLS) + "\\\\",
        " & & " + " & ".join(["ACC$\\uparrow$ & BWT$\\uparrow$ & E-ACC$\\uparrow$ & $\\Delta$B$\\uparrow$ & $\\Delta$F$\\uparrow$"] * len(COLS)) + "\\\\\\midrule",
        f"\\multicolumn{{{2 + 5 * len(COLS)}}}{{c}}{{\\textit{{Two-branch fusion network on frozen CLIP features}}}}\\\\\\midrule"]
body = block(SMALL, {"tables": "tables", "clevr": "clevr", "gqa": "gqa", "clove": "clove"}, None)
mid = ["\\midrule", f"\\multicolumn{{{2 + 5 * len(COLS)}}}{{c}}{{\\textit{{Qwen2.5-VL-3B-Instruct + LoRA (CLOVE-scene: six tasks; nonce-word CLEVR: pre-registered gate not met, see text)}}}}\\\\\\midrule"]
vb = block(VLM, {"clevr": "clevr_vlm" if ("clevr_vlm", "finetune") in res else None, "clove": "clove_vlm"}, None)
tail = ["\\bottomrule\\end{tabular}}\\end{table*}"]
tex = "\n".join(head + body + mid + vb + tail); open(os.path.join(ROOT, "paper", "table_main.tex"), "w").write(tex + "\n")
doc = "\\documentclass{article}\\usepackage[landscape,margin=0.4in]{geometry}\\usepackage{booktabs,graphicx}\\usepackage[table]{xcolor}\\pagestyle{empty}\\begin{document}" + tex + "\\end{document}"
os.makedirs(os.path.join(ROOT, "paper", "table_preview"), exist_ok=True); open(os.path.join(ROOT, "paper", "table_preview", "table_main.tex"), "w").write(doc)
subprocess.run(["pdflatex", "-interaction=nonstopmode", "table_main.tex"], cwd=os.path.join(ROOT, "paper", "table_preview"), capture_output=True)
subprocess.run(["pdftoppm", "-png", "-r", "170", "-singlefile", "table_main.pdf", "table_main"], cwd=os.path.join(ROOT, "paper", "table_preview")); print("rendered")
