"""Figure: where VLMs' answers depend on interactions (figures/revision/fig_flips.pdf). Data: runs/rev/flip_cells.csv.gz
(per cell of the 99 full tables, 4 VLMs: graded misfit, exception label, |log-odds|, whether the answer flips under
EMAP, and the accuracy-matched null scorer's flip probability over 20 draws). Also writes paper/revision/data/15_*.csv"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "figures", "revision"); DOUT = os.path.join(ROOT, "paper", "revision", "data")
INK, MUTED, GRID = "#1f1f1f", "#6b6b68", "#d9d9d6"
MOD = {"qwen": ("Qwen2.5-VL-7B", "#2a78d6"), "internvl": ("InternVL3.5-8B", "#eb6834"), "llava": ("LLaVA-OV-1.5-8B", "#1baf7a"), "idefics": ("Idefics3-8B", "#eda100")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42})
D = pd.read_csv(os.path.join(ROOT, "runs", "rev", "flip_cells.csv.gz")); D["dec"] = pd.qcut(D.misfit.rank(method="first"), 10, labels=False)
rng = np.random.default_rng(0); tabs = np.array(sorted(D.table.unique())); ti = {t: i for i, t in enumerate(tabs)}


def curve(x, col):
    """Flip rate by misfit decile with a 95% bootstrap over tables."""
    g = x.groupby([x.table.map(ti), "dec"])[col].agg(["sum", "count"]).unstack(fill_value=0); S, N = g["sum"].reindex(range(len(tabs)), fill_value=0).values, g["count"].reindex(range(len(tabs)), fill_value=0).values
    est = S.sum(0) / N.sum(0); B = []
    for _ in range(1000):
        w = np.bincount(rng.integers(0, len(tabs), len(tabs)), minlength=len(tabs)); B.append((w @ S) / np.maximum(w @ N, 1))
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0); return est, lo, hi


fig = plt.figure(figsize=(7.0, 2.7)); gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.25], wspace=0.75)
# (a) flip rate by misfit decile
ax = fig.add_subplot(gs[0, 0]); x = np.arange(1, 11); rows = []
nulls = []
for m, (nm, c) in MOD.items():
    e, lo, hi = curve(D[D.model == m], "flip"); en, _, _ = curve(D[D.model == m], "flip_null"); nulls.append(en)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0); ax.plot(x, e, color=c, lw=1.6, marker="o", ms=2.8, label=nm)
    rows += [dict(model=nm, misfit_decile=int(d), flip_rate=e[d - 1], ci_low=lo[d - 1], ci_high=hi[d - 1], null_flip_rate=en[d - 1]) for d in x]
nulls = np.array(nulls); ax.plot(x, nulls.mean(0), color=MUTED, lw=1.2, ls=(0, (3, 2)), label="accuracy-matched null", zorder=1)
ax.fill_between(x, nulls.min(0), nulls.max(0), color=MUTED, alpha=0.10, lw=0)
ax.set_xticks([1, 4, 7, 10]); ax.set_xlabel("graded misfit (decile)"); ax.set_ylabel("answers flipped when interactions\nare removed (share of cells)")
ax.set_ylim(0, 0.9); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=1.6); ax.set_title("(a) flips concentrate at high misfit", fontsize=7.5, loc="left", color=INK)
# (b) AUC for predicting flips
ax = fig.add_subplot(gs[0, 1]); feats = [("graded misfit", "misfit", 1), ("model confidence", "conf", -1), ("exception label", "exc", 1)]; auc_rows = []
w = 0.24
for i, (m, (nm, c)) in enumerate(MOD.items()):
    x_ = D[D.model == m]
    for j, (fl, col, sg) in enumerate(feats):
        a = roc_auc_score(x_.flip, sg * x_[col]); auc_rows.append(dict(model=nm, predictor=fl, auc=a))
        ax.barh(-i + (1 - j) * (w + 0.02), a - 0.5, left=0.5, height=w, color=["#3a3a38", "#9a9a96", "#d0d0cc"][j], label=fl if i == 0 else None)
    an = roc_auc_score(x_.flip_null > 0.5, x_.misfit); auc_rows.append(dict(model=nm, predictor="graded misfit, null scorer's flips", auc=an))
    ax.plot([an], [-i + (w + 0.02)], marker="|", color=MUTED, ms=9, mew=1.4, label="misfit, null scorer" if i == 0 else None)
ax.set_yticks([-i for i in range(4)]); ax.set_yticklabels([v[0] for v in MOD.values()], color=INK); ax.set_xlim(0.5, 1.0); ax.set_xticks([0.5, 0.75, 1.0])
ax.set_xlabel("AUC for predicting which answers flip"); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, handlelength=1.2, columnspacing=0.8); ax.set_title("(b) misfit predicts the flips", fontsize=7.5, loc="left", color=INK)
# (c) one table: misfit as shade, flipped cells outlined, exceptions dotted (LLaVA-OV-1.5, a CLEVR table)
G = np.load(os.path.join(ROOT, "runs", "vlm3", "grids.npz")); key = "clevr__85_1"
if f"T__{key}" not in G.files:
    key = sorted(k[3:] for k in G.files if k.startswith("T__clevr"))[0]
S = np.load(os.path.join(ROOT, "runs", "vlm3", "scores_llava.npz")); T, L, Mf, X = G[f"T__{key}"].astype(int), S[f"L__{key}"], G[f"misfit__{key}"], G[f"exc__{key}"]
E = L.mean(1, keepdims=True) + L.mean(0, keepdims=True) - L.mean(); F = (L > 0) != (E > 0)
ro = np.argsort(-T.mean(1)); co = np.argsort(-T.mean(0), kind="stable"); sel = co[np.linspace(0, len(co) - 1, 40).astype(int)]
ax = fig.add_subplot(gs[0, 2]); Mv = Mf[np.ix_(ro, sel)]
im = ax.imshow(Mv, cmap="Blues", vmin=np.percentile(Mf, 5), vmax=np.percentile(Mf, 99), aspect="auto", interpolation="nearest")
for i in range(Mv.shape[0]):
    for j in range(Mv.shape[1]):
        r, c_ = ro[i], sel[j]
        if F[r, c_]:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#c0392b", lw=1.0))
        if X[r, c_]:
            ax.plot(j, i, marker="o", ms=1.8, color=INK)
        if T[r, c_]:
            ax.plot(j, i, marker="s", ms=1.0, color="white", alpha=0.0)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("40 of 100 images (sorted by yes-rate)"); ax.set_ylabel("10 questions")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label("graded misfit", fontsize=6.5, labelpad=2); cb.ax.tick_params(labelsize=6)
ax.set_title("(c) one CLEVR table, LLaVA-OV-1.5", fontsize=7.5, loc="left", color=INK)
ax.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec="#c0392b", lw=1.0, label="answer flips"), plt.Line2D([], [], marker="o", ls="", ms=2.5, color=INK, label="exception")],
          frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.subplots_adjust(left=0.085, right=0.955, top=0.89, bottom=0.3)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"fig_flips.{ext}"), dpi=220)
pd.DataFrame(rows).round(4).to_csv(os.path.join(DOUT, "15a_flip_rate_by_misfit_decile.csv"), index=False)
pd.DataFrame(auc_rows).round(4).to_csv(os.path.join(DOUT, "15b_flip_auc.csv"), index=False); print("example table:", key)
