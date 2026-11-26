"""Figure: partial correlation of the EMAP drop with structure and with PID synergy (accuracy and answer balance held
fixed), per model, on the confirmatory sub-tables (paper/results_vlm3_conf.txt). Writes figures/fig_partial.{pdf,png}"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"]})
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); STRUCT, PID, INK, MUTED = "#F1515E", "#B8C1C9", "#1F2A33", "#5E6B76"
rows = [("Idefics3-8B\n(confirmatory)", (0.416, 0.212, 0.559), (0.065, -0.167, 0.285)), ("InternVL3.5-8B", (0.418, 0.197, 0.584), (0.048, -0.157, 0.261)),
        ("LLaVA-OV-1.5-8B", (0.433, 0.228, 0.584), (0.239, 0.022, 0.417)), ("Qwen2.5-VL-7B", (0.260, 0.059, 0.440), (0.027, -0.181, 0.252))]
fig, ax = plt.subplots(figsize=(4.6, 3.1)); h = 0.34
for i, (nm, s, p) in enumerate(rows):
    y = len(rows) - 1 - i
    for dy, (v, lo, hi), col, lab in ((h / 2, s, STRUCT, "exception share"), (-h / 2, p, PID, "PID synergy")):
        ax.barh(y + dy, v, height=h, color=col, label=lab if i == 0 else None)
        ax.errorbar(v, y + dy, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
ax.axvline(0, color=MUTED, lw=0.8); ax.axvline(0.3, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.set_ylim(-0.6, len(rows) - 0.1); ax.text(0.31, len(rows) - 0.12, "pre-registered threshold", fontsize=7, color=MUTED, va="top")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
ax.set_xlim(-0.25, 0.65); ax.set_xlabel("partial rank correlation with EMAP drop", fontsize=8.5); ax.tick_params(axis="x", labelsize=8)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0); ax.legend(fontsize=8, frameon=False, loc="lower right")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "figures", f"fig_partial.{ext}"), dpi=300, facecolor="white")
