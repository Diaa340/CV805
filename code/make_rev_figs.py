"""Revision figures (figures/revision/): fig_gaps.pdf (same-answer gaps without the constructed-task forward panel)
and fig_vlm_panel.pdf (Fig. 1 VLM panel with the accuracy-matched null scorer). Data: figures/strat_metrics.csv,
paper/results_vlm3_conf.txt values, paper/results_rev_null.txt values (copied below, with their sources)."""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "figures", "revision"); os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#1f1f1f", "#6b6b68", "#d9d9d6"; RED, BLUE, GREY = "#D5484F", "#3F6FB0", "#A3A6AB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.size": 0, "pdf.fonttype": 42})

# ---------------------------------------------------------------- Figure: same-answer gaps (replaces fig_gaps.pdf)
S = pd.read_csv(os.path.join(ROOT, "figures", "strat_metrics.csv"))
ROWS = [("Finetune", "finetune", "finetune"), ("EWC", "ewc", "ewc"), ("SI", "si", "si"), ("LwF", "lwf", "lwf"), ("Feature distill.", "l2", "l2"),
        ("ER", "er_random_1000", "er"), ("A-GEM", "agem_random_1000", "agem"), ("GEM", "gem_random_1000", "gem"), ("DER++", "derpp_random_1000", "derpp"),
        ("iCaRL", "icarl", "icarl"), ("DER (expansion)", "der_expand_1000", "der_expand"), ("Multi-head", "multihead", "multihead")]
GROUP_GAP = {4: 0.5, 9: 0.5}   # extra space after Feature distill. (regularisation) and iCaRL (memory)
ypos, y = [], 0.0
for i in range(len(ROWS)):
    ypos.append(y); y += 1 + GROUP_GAP.get(i, 0)
ypos = -np.array(ypos)
PANELS_B = [("tables", "Constructed"), ("clevr", "CLEVR"), ("gqa", "GQA"), ("clove", "CLOVE-scene")]
PANELS_F = [("clevr", "CLEVR"), ("gqa", "GQA"), ("clove", "CLOVE-scene")]
fig = plt.figure(figsize=(7.0, 2.75)); gs = fig.add_gridspec(1, 8, width_ratios=[1, 1, 1, 1, 0.35, 1, 1, 1], wspace=0.12)
LIM = (-48, 48)


def panel(ax, bench, col, title, first):
    x = S[S.bench == bench].set_index("method")
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for t in (-25, 25):
        ax.axvline(t, color=GRID, lw=0.5, zorder=0)
    for (lab, two, seq), yy in zip(ROWS, ypos):
        m = two if bench in ("tables", "clevr", "gqa") else seq
        v, lo, hi = 100 * x.loc[m, col], 100 * x.loc[m, f"{col}_lo"], 100 * x.loc[m, f"{col}_hi"]
        sig = hi < 0
        ax.plot([lo, hi], [yy, yy], color=RED if sig else GREY, lw=1.0, solid_capstyle="round", zorder=2)
        ax.scatter([v], [yy], s=16, zorder=3, facecolor=RED if sig else "white", edgecolor=RED if sig else MUTED, linewidth=0.8)
    ax.set_xlim(*LIM); ax.set_xticks([-25, 0, 25]); ax.set_ylim(ypos.min() - 0.8, 0.8); ax.set_title(title, fontsize=7.5, color=INK, pad=3)
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in ROWS] if first else [])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


for j, (b, t) in enumerate(PANELS_B):
    panel(fig.add_subplot(gs[0, j]), b, "dBWT_s", t, j == 0)
for j, (b, t) in enumerate(PANELS_F):
    panel(fig.add_subplot(gs[0, 5 + j]), b, "dFWT_s", t, False)
fig.subplots_adjust(left=0.145, right=0.985, top=0.86, bottom=0.25)
axs = fig.axes; xb = (axs[0].get_position().x0 + axs[3].get_position().x1) / 2; xf = (axs[4].get_position().x0 + axs[6].get_position().x1) / 2
fig.text(xb, 0.955, "Old exceptions: forgotten more than the rule?", ha="center", fontsize=8, color=INK, weight="bold")
fig.text(xf, 0.955, "New exceptions: learned worse?", ha="center", fontsize=8, color=INK, weight="bold")
fig.text(xb, 0.13, "ΔB, backward-transfer gap (points)", ha="center", color=INK)
fig.text(xf, 0.13, "ΔF, forward-transfer gap (points)", ha="center", color=INK)
fig.text(xf, 0.02, "Constructed tasks omitted: their second table\nhas no stable exceptions.", ha="center", color=MUTED, fontsize=6.3)
fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc=RED, mec=RED, ms=4, label="95% interval below zero"),
                    plt.Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=4, label="interval includes zero")],
           loc="lower center", bbox_to_anchor=(xb, -0.01), ncol=2, frameon=False, fontsize=6.5)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_gaps.{ext}"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- Figure: Fig. 1 VLM panel with the null scorer
# observed and PID: paper/results_vlm3_conf.txt (Study C); null: paper/results_rev_null.txt (rho_null)
V = [("Idefics3", (0.416, 0.212, 0.559), (0.374, 0.166, 0.533), (0.065, -0.167, 0.285)),
     ("InternVL3.5", (0.418, 0.197, 0.584), (0.499, 0.289, 0.646), (0.048, -0.157, 0.261)),
     ("LLaVA-OV-1.5", (0.433, 0.228, 0.584), (0.396, 0.172, 0.581), (0.239, 0.022, 0.417)),
     ("Qwen2.5-VL", (0.260, 0.059, 0.440), (0.301, 0.084, 0.476), (0.027, -0.181, 0.252))]
fig, ax = plt.subplots(figsize=(2.5, 2.2)); h = 0.24
for i, (nm, *vals) in enumerate(V):
    for j, ((v, lo, hi), c, hatch) in enumerate(zip(vals, (RED, BLUE, GREY), (None, "////", None))):
        yy = -i - (j - 1) * (h + 0.03)
        ax.barh(yy, v, height=h, color=c if hatch is None else "white", edgecolor=c, hatch=hatch, linewidth=0.8, zorder=2)
        ax.plot([lo, hi], [yy, yy], color=INK, lw=0.7, zorder=3)
ax.axvline(0, color=MUTED, lw=0.6); ax.axvline(0.3, color=MUTED, lw=0.6, ls=(0, (3, 2)))
ax.set_yticks([-i for i in range(len(V))]); ax.set_yticklabels([v[0] for v in V]); ax.set_xlim(-0.25, 0.7); ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
ax.set_xlabel("partial ρ with EMAP drop"); ax.grid(axis="x", color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=RED, ec=RED, label="exception share"),
                   plt.Rectangle((0, 0), 1, 1, fc="white", ec=BLUE, hatch="////", label="null: accuracy alone"),
                   plt.Rectangle((0, 0), 1, 1, fc=GREY, ec=GREY, label="PID synergy")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, frameon=False, fontsize=6.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_vlm_panel.{ext}"), dpi=250)
print("written", os.listdir(OUT))
