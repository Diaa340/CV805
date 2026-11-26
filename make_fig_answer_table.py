"""Figure: answer table, best additive rule and XOR square on real POPE data (COCO images and object annotations).
Exact rule from exact_additive.exact_partial_fit (proven optimal: 23 of 24 cells; one exception).
Writes paper/figs/fig_answer_table.png and .pdf"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from exact_additive import exact_partial_fit
ROOT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"]})
YES, EXC, NO = "#1DBDE6", "#F1515E", "#EDF0F2"            # sky blue, coral red, light cool grey
INK, YES_TXT, NO_TXT, MUTED = "#1F2A33", "#06303D", "#7A8691", "#5E6B76"
EXC_TXT = "white"
P = np.load(os.path.join(ROOT, "data", "raw", "pope", "pope_table.npz")); O = list(P["objects"]); IM = list(P["images"])
OBJ = ["car", "chair", "dining table", "person"]
COLS = dict(A="COCO_val2014_000000120792.jpg", B="COCO_val2014_000000006033.jpg", C="COCO_val2014_000000018150.jpg",
            D="COCO_val2014_000000121959.jpg", E="COCO_val2014_000000178078.jpg", F="COCO_val2014_000000209290.jpg")
T = P["T"].astype(int)[np.ix_([O.index(o) for o in OBJ], [IM.index(c) for c in COLS.values()])]
R, best, bound, _ = exact_partial_fit(T, np.ones(T.shape), 60); assert abs(best - bound) < 1e-9 and (R != T).sum() == 1


def thumb(name, w=4.0, h=3.0):
    im = Image.open(os.path.join(ROOT, "data", "raw", "pope", "images", name)).convert("RGB"); W, H = im.size
    s = min(W / w, H / h); cw, ch = int(w * s), int(h * s); x0, y0 = (W - cw) // 2, (H - ch) // 2
    return np.asarray(im.crop((x0, y0, x0 + cw, y0 + ch)).resize((int(300 * w / h), 300), Image.LANCZOS))


TH = {k: thumb(v) for k, v in COLS.items()}
CW, CH, IH, GAP = 1.0, 0.82, 0.75, 0.06   # cell width / height, image height, gap


def grid(ax, ox, rows, cols, val, kind, ylab=True):
    """Draws one panel with its left edge at x = ox. kind(i, j) -> 'yes' | 'no' | 'exc'."""
    for c, j in enumerate(cols):
        ax.imshow(TH[list(COLS)[j]], extent=(ox + c * CW + GAP / 2, ox + (c + 1) * CW - GAP / 2, 0.1, 0.1 + IH), zorder=2)
        ax.text(ox + (c + .5) * CW, 0.1 + IH + 0.1, list(COLS)[j], ha="center", va="bottom", fontsize=13, color=MUTED)
    for r, i in enumerate(rows):
        y = -(r + 1) * CH
        ax.text(ox - 0.12, y + CH / 2, f"{OBJ[i]}?", ha="right", va="center", fontsize=14, color=INK)
        for c, j in enumerate(cols):
            k = kind(i, j); fill = {"yes": YES, "no": NO, "exc": EXC}[k]
            ax.add_patch(Rectangle((ox + c * CW + GAP / 2, y + GAP / 2), CW - GAP, CH - GAP, facecolor=fill, edgecolor="none", zorder=1))
            ax.text(ox + (c + .5) * CW, y + CH / 2, "yes" if val[i, j] else "no", ha="center", va="center", fontsize=13.5,
                    color={"yes": YES_TXT, "no": NO_TXT, "exc": EXC_TXT}[k], fontweight="bold" if k == "exc" else "normal", zorder=3)
    if ylab:
        ax.text(ox - 0.12, 0.1 + IH / 2, "Is there a …", ha="right", va="center", fontsize=13, color=MUTED, style="italic")


def title(ax, x, letter, text):
    ax.text(x, 0.1 + IH + 0.62, letter, fontsize=17, fontweight="bold", color=INK, va="bottom")
    ax.text(x + 0.38, 0.1 + IH + 0.62, text, fontsize=15, color=INK, va="bottom")


OA, OB, OC = 0.0, 8.35, 16.5          # left edges of the three panels' grids
fig = plt.figure(figsize=(18, 5.6)); ax = fig.add_axes([0, 0.1, 1, 0.9]); ax.set_aspect("equal"); ax.axis("off")
# (a) answer table
grid(ax, OA, [0, 1, 2, 3], range(6), T, lambda i, j: "yes" if T[i, j] else "no"); title(ax, OA - 2.05, "a", "Answer table")
# (b) best additive rule: rows / columns sorted by the rule's scores (number of predicted yes), staircase boundary
rr = sorted(range(4), key=lambda i: -R[i].sum()); cc = sorted(range(6), key=lambda j: (-R[:, j].sum(), j))
grid(ax, OB, rr, cc, T, lambda i, j: "exc" if R[i, j] != T[i, j] else ("yes" if T[i, j] else "no")); title(ax, OB - 2.05, "b", "Best additive rule")
k = [int(R[i].sum()) for i in rr]; xs, ys = [OB + k[0] * CW], [0]   # staircase: yes region at the top left
for r in range(4):
    if k[r] == 0:
        break
    xs += [OB + k[r] * CW]; ys += [-(r + 1) * CH]
    if r + 1 < 4:
        xs += [OB + k[r + 1] * CW]; ys += [-(r + 1) * CH]
ax.plot(xs, ys, color=INK, lw=3, solid_capstyle="projecting", zorder=4)
ax.text(OB + 3 * CW, -4 * CH - 0.12, "image score: high → low", ha="center", va="top", fontsize=12, color=MUTED)
ax.text(OB + 6 * CW + 0.1, -2 * CH, "question score: high → low", rotation=-90, ha="left", va="center", fontsize=12, color=MUTED)
# (c) XOR square: person / car x B / E
xi, xj = [OBJ.index("person"), OBJ.index("car")], [list(COLS).index("B"), list(COLS).index("E")]
Y0 = -CH                                      # 2 x 2 grid with the usual cell size, centred on the four rows of the other tables
for c, j in enumerate(xj):
    nm = list(COLS)[j]
    ax.imshow(TH[nm], extent=(OC + c * CW + GAP / 2, OC + (c + 1) * CW - GAP / 2, Y0 + 0.1, Y0 + 0.1 + IH), zorder=2)
    ax.text(OC + (c + .5) * CW, Y0 + 0.1 + IH + 0.1, nm, ha="center", va="bottom", fontsize=13, color=MUTED)
for r, i in enumerate(xi):
    y = Y0 - (r + 1) * CH; ax.text(OC - 0.12, y + CH / 2, f"{OBJ[i]}?", ha="right", va="center", fontsize=14, color=INK)
    for c, j in enumerate(xj):
        v = T[i, j]
        ax.add_patch(Rectangle((OC + c * CW + GAP / 2, y + GAP / 2), CW - GAP, CH - GAP, facecolor=YES if v else NO, edgecolor="none", zorder=1))
        ax.text(OC + (c + .5) * CW, y + CH / 2, "yes" if v else "no", ha="center", va="center", fontsize=13.5, color=YES_TXT if v else NO_TXT, zorder=3)
title(ax, OC - 1.05, "c", "XOR square")
ax.text(OC + CW, Y0 - 2 * CH - 0.15, "Yes on one diagonal, no on the other:\nno additive rule fits all four.", ha="center", va="top", fontsize=12,
        color=MUTED, linespacing=1.35)
ax.set_xlim(OA - 2.15, OC + 2 * CW + 0.9); ax.set_ylim(-4 * CH - 0.75, 0.1 + IH + 1.05)
# legend
L = fig.add_axes([0.24, 0.0, 0.62, 0.08]); L.axis("off"); L.set_xlim(0, 10); L.set_ylim(0, 1)
for x, col, lab in ((0.0, YES, "yes"), (1.2, NO, "no"), (2.4, EXC, "exception: answer is yes, the rule says no")):
    L.add_patch(Rectangle((x, .28), .36, .48, facecolor=col)); L.text(x + .5, .52, lab, va="center", fontsize=13, color=INK)
L.plot([6.55, 7.15], [.52, .52], color=INK, lw=3); L.text(7.3, .52, "best additive rule", va="center", fontsize=13, color=INK)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "paper", "figs", f"fig_answer_table.{ext}"), dpi=300, bbox_inches="tight", facecolor="white")
print("exception at", [(OBJ[i], list(COLS)[j]) for i, j in np.argwhere(R != T)])
