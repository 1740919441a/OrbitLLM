from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#223B63"
BLUE = "#69A7DB"
BLUE_L = "#EAF4FB"
GREEN = "#72A956"
GREEN_L = "#EEF7E9"
ORANGE = "#EA9860"
ORANGE_L = "#FFF0E5"
PURPLE = "#A77DB1"
PURPLE_L = "#F3ECF5"
RED = "#D94D4D"
RED_L = "#FDECEC"
GRAY = "#6F7882"
LIGHT = "#F8FAFB"
BORDER = "#C3CCD5"


def rounded(ax, x, y, w, h, fc="white", ec=BORDER, lw=1.5, r=0.018, z=1):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.007,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=2.0, z=6):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=15, linewidth=lw, color=color, zorder=z)
    ax.add_patch(p)
    return p


fig = plt.figure(figsize=(16, 7.2), dpi=190, facecolor="white")
ax = fig.add_axes([0.018, 0.03, 0.964, 0.94])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 0.945, "INSIDE ORBITLLM", ha="center", va="center",
        fontsize=22, weight="bold", color=NAVY)
ax.text(0.5, 0.905, "Measured costs, feasible choice, guarded actions",
        ha="center", va="center", fontsize=11, color=GRAY)

panels = [(0.035, 0.105, 0.285, 0.745),
          (0.357, 0.105, 0.285, 0.745),
          (0.679, 0.105, 0.285, 0.745)]
for x, y, w, h in panels:
    rounded(ax, x, y, w, h, fc="#FCFDFE", ec=BORDER, lw=1.7, r=0.024, z=0)

titles = [(0.055, "1", "LLM COST PROFILE", BLUE),
          (0.377, "2", "FEASIBILITY + TVD", PURPLE),
          (0.699, "3", "GUARDED ACTIONS", GREEN)]
for x, n, title, color in titles:
    ax.add_patch(Circle((x + 0.018, 0.805), 0.022, facecolor=color, edgecolor="none", zorder=5))
    ax.text(x + 0.018, 0.805, n, ha="center", va="center", fontsize=11,
            color="white", weight="bold", zorder=6)
    ax.text(x + 0.052, 0.805, title, ha="left", va="center", fontsize=14,
            color=NAVY, weight="bold")

# ---------------------------------------------------------------------------
# Panel 1: measured cost surface, no numerical table.
# ---------------------------------------------------------------------------
ax.text(0.064, 0.735, "ENERGY", fontsize=9.5, color=GRAY, weight="bold")
ax.add_patch(Rectangle((0.066, 0.64), 0.075, 0.07,
                       facecolor=BLUE_L, edgecolor=BLUE, linewidth=1.4))
ax.text(0.1035, 0.675, "Prefill", ha="center", va="center", fontsize=9.5,
        color=NAVY, weight="bold")
for i in range(7):
    ax.add_patch(Rectangle((0.158 + i * 0.019, 0.64), 0.014, 0.052,
                           facecolor=ORANGE_L, edgecolor=ORANGE, linewidth=1.0))
ax.text(0.221, 0.615, "Decode", ha="center", va="top", fontsize=9.5,
        color="#9C5B31", weight="bold")
arrow(ax, 0.145, 0.675, 0.155, 0.675, color=GRAY, lw=1.2)

ax.text(0.064, 0.545, "MEMORY", fontsize=9.5, color=GRAY, weight="bold")
ax.plot([0.068, 0.286], [0.385, 0.385], color="#A7AFB7", lw=1.0)
ax.plot([0.068, 0.068], [0.385, 0.535], color="#A7AFB7", lw=1.0)
xs = [0.075, 0.11, 0.145, 0.18, 0.215, 0.25, 0.278]
ys = [0.405, 0.414, 0.428, 0.448, 0.475, 0.51, 0.545]
ax.plot(xs, ys, color=NAVY, lw=2.5, marker="o", markersize=3.5)
ax.plot([0.068, 0.286], [0.515, 0.515], color=RED, lw=1.5, ls="--")
ax.text(0.278, 0.525, "limit", fontsize=8, color=RED, ha="right")
ax.text(0.17, 0.355, "context", fontsize=8.5, color=GRAY, ha="center")

# Compact profile-table icon.
rounded(ax, 0.09, 0.19, 0.17, 0.105, fc=BLUE_L, ec=BLUE, lw=1.4, r=0.012)
for i in range(1, 4):
    ax.plot([0.09 + i * 0.0425, 0.09 + i * 0.0425], [0.19, 0.295], color=BLUE, lw=0.8)
for i in range(1, 3):
    ax.plot([0.09, 0.26], [0.19 + i * 0.035, 0.19 + i * 0.035], color=BLUE, lw=0.8)
ax.text(0.175, 0.155, "measured profile", fontsize=9, color=NAVY,
        ha="center", weight="bold")

# ---------------------------------------------------------------------------
# Panel 2: hard feasibility filtering precedes TVD ranking.
# ---------------------------------------------------------------------------
# Value decay cue.
ax.plot([0.39, 0.465], [0.69, 0.69], color="#A8AFB6", lw=1.0)
ax.plot([0.39, 0.39], [0.69, 0.755], color="#A8AFB6", lw=1.0)
ax.plot([0.395, 0.41, 0.43, 0.452, 0.46], [0.748, 0.735, 0.715, 0.697, 0.693],
        color=PURPLE, lw=2.2)
ax.text(0.428, 0.665, "Value", ha="center", fontsize=8.5, color=GRAY)

# Battery icon.
ax.add_patch(Rectangle((0.49, 0.704), 0.055, 0.035, facecolor="white",
                       edgecolor=ORANGE, linewidth=1.5))
ax.add_patch(Rectangle((0.545, 0.714), 0.007, 0.015, facecolor=ORANGE, edgecolor=ORANGE))
ax.add_patch(Rectangle((0.495, 0.709), 0.032, 0.025, facecolor=ORANGE_L, edgecolor="none"))
ax.text(0.52, 0.665, "Energy", ha="center", fontsize=8.5, color=GRAY)

# Bandwidth waves.
for rad in [0.018, 0.032, 0.046]:
    theta = [i / 30 * 1.2 - 0.6 for i in range(31)]
    xx = [0.58 + rad * __import__('math').cos(t) for t in theta]
    yy = [0.72 + rad * __import__('math').sin(t) for t in theta]
    ax.plot(xx, yy, color=BLUE, lw=1.5)
ax.add_patch(Circle((0.58, 0.72), 0.005, facecolor=BLUE, edgecolor="none"))
ax.text(0.58, 0.665, "Bandwidth", ha="center", fontsize=8.5, color=GRAY)

# Memory chip icon.
ax.add_patch(Rectangle((0.405, 0.545), 0.055, 0.055, facecolor=GREEN_L,
                       edgecolor=GREEN, linewidth=1.4))
ax.add_patch(Rectangle((0.417, 0.557), 0.031, 0.031, facecolor="white",
                       edgecolor=GREEN, linewidth=1.0))
ax.text(0.4325, 0.52, "Memory", ha="center", fontsize=8.5, color=GRAY)

# Hard feasibility gate.
gate = Polygon([[0.445, 0.505], [0.475, 0.54], [0.505, 0.505],
                [0.475, 0.47]], closed=True, facecolor=GREEN_L,
               edgecolor=GREEN, linewidth=1.6)
ax.add_patch(gate)
ax.text(0.475, 0.505, "feasible?", ha="center", va="center", fontsize=7.2,
        weight="bold", color=GREEN)

# TVD ranking lens.
lens = Polygon([[0.575, 0.51], [0.62, 0.55], [0.62, 0.46], [0.575, 0.42], [0.53, 0.46], [0.53, 0.55]],
               closed=True, facecolor=PURPLE_L, edgecolor=PURPLE, linewidth=1.8)
ax.add_patch(lens)
ax.text(0.575, 0.495, "TVD", ha="center", va="center", fontsize=17,
        weight="bold", color="#60466B")

# Value and bandwidth are scoring inputs. Energy is both a hard feasibility
# input and a scarcity cost; memory is a hard feasibility input only.
arrow(ax, 0.43, 0.69, 0.55, 0.55, color=PURPLE, lw=1.4)
arrow(ax, 0.58, 0.69, 0.603, 0.55, color=BLUE, lw=1.4)
arrow(ax, 0.52, 0.704, 0.475, 0.54, color=ORANGE, lw=1.4)
arrow(ax, 0.432, 0.545, 0.447, 0.515, color=GREEN, lw=1.4)
arrow(ax, 0.505, 0.505, 0.53, 0.505, color=GREEN, lw=1.4)
ax.plot([0.52, 0.548], [0.66, 0.555], color=ORANGE, lw=1.2, ls="--")

# The lens scores only feasible actions; the result then feeds the ranking.
arrow(ax, 0.575, 0.42, 0.575, 0.397, color=PURPLE, lw=1.4)

# Ranked action cards. Only action names and visual bars.
cards = [("ON", ORANGE, ORANGE_L),
         ("PRE", GREEN, GREEN_L),
         ("DOWN", BLUE, BLUE_L)]
for i, (name, color, fc) in enumerate(cards):
    y = 0.325 - i * 0.09
    rounded(ax, 0.39, y, 0.215, 0.065, fc=fc, ec=color, lw=1.5, r=0.012)
    ax.text(0.415, y + 0.0325, name, ha="left", va="center", fontsize=10.5,
            weight="bold", color=color)
    ax.add_patch(Rectangle((0.47, y + 0.023), 0.11, 0.018,
                           facecolor="white", edgecolor=color, linewidth=0.8,
                           hatch="////", alpha=0.55))
# All three candidates remain visually unresolved until the max selector.
ax.plot([0.61, 0.623], [0.19, 0.19], color=PURPLE, lw=1.5)
ax.plot([0.623, 0.623], [0.19, 0.39], color=PURPLE, lw=1.5)
ax.plot([0.61, 0.623], [0.39, 0.39], color=PURPLE, lw=1.5)
arrow(ax, 0.623, 0.29, 0.635, 0.29, color=PURPLE, lw=1.4)
ax.text(0.617, 0.405, "select max", ha="center", fontsize=7.2,
        color=PURPLE, weight="bold")

# ---------------------------------------------------------------------------
# Panel 3: concise execution mechanisms, no data or formulas.
# ---------------------------------------------------------------------------
# ON lane.
rounded(ax, 0.71, 0.615, 0.22, 0.105, fc=ORANGE_L, ec=ORANGE, lw=1.4, r=0.014)
ax.text(0.73, 0.682, "ON", fontsize=11, weight="bold", color=ORANGE)
for i in range(6):
    ax.add_patch(Rectangle((0.77 + i * 0.022, 0.65), 0.015, 0.026,
                           facecolor="white", edgecolor=ORANGE, linewidth=0.9))
# Equal-height tokens represent sequential decode; the curve represents KV growth.
kv_x = [0.777, 0.799, 0.821, 0.843, 0.865, 0.887]
kv_y = [0.674, 0.678, 0.683, 0.689, 0.697, 0.706]
ax.plot(kv_x, kv_y, color=NAVY, lw=1.6, marker="o", markersize=2.3)
ax.plot([0.765, 0.91], [0.7, 0.7], color=RED, lw=1.3, ls="--")
ax.add_patch(Rectangle((0.882, 0.641), 0.006, 0.066, facecolor=RED, edgecolor="none"))
ax.text(0.885, 0.628, "STOP", fontsize=7.2, color=RED, ha="center", weight="bold")

# PRE lane with image-mask-retained visual.
rounded(ax, 0.71, 0.435, 0.22, 0.125, fc=GREEN_L, ec=GREEN, lw=1.4, r=0.014)
ax.text(0.73, 0.525, "PRE", fontsize=11, weight="bold", color=GREEN_DARK if 'GREEN_DARK' in globals() else GREEN)
ax.add_patch(Rectangle((0.765, 0.46), 0.052, 0.052, facecolor="#B9D79A", edgecolor=GREEN, linewidth=0.9))
ax.add_patch(Polygon([[0.765, 0.46], [0.79, 0.485], [0.817, 0.472], [0.817, 0.46]],
                     closed=True, facecolor=BLUE, edgecolor="none"))
arrow(ax, 0.82, 0.486, 0.85, 0.486, color=GREEN, lw=1.3)
ax.add_patch(Rectangle((0.858, 0.465), 0.045, 0.04, facecolor=ORANGE_L, edgecolor=ORANGE, linewidth=1.0))
ax.add_patch(Rectangle((0.869, 0.476), 0.022, 0.018, facecolor="#D9A66B", edgecolor="none"))

# DOWN lane.
rounded(ax, 0.71, 0.265, 0.22, 0.105, fc=BLUE_L, ec=BLUE, lw=1.4, r=0.014)
ax.text(0.73, 0.332, "DOWN", fontsize=11, weight="bold", color=BLUE)
ax.add_patch(Circle((0.79, 0.315), 0.008, facecolor=BLUE, edgecolor="none"))
for rad in [0.025, 0.043, 0.061]:
    theta = [i / 30 * 1.2 - 0.6 for i in range(31)]
    xx = [0.79 + rad * __import__('math').cos(t) for t in theta]
    yy = [0.315 + rad * __import__('math').sin(t) for t in theta]
    ax.plot(xx, yy, color=BLUE, lw=1.3)
arrow(ax, 0.855, 0.315, 0.91, 0.315, color=BLUE, lw=1.5)

# Compact state-feedback symbol; the full feedback path belongs to the architecture figure.
ax.add_patch(FancyArrowPatch((0.852, 0.2), (0.785, 0.2),
                            connectionstyle="arc3,rad=-0.65", arrowstyle="-|>",
                            mutation_scale=12, color=GRAY, linewidth=1.3))
ax.add_patch(FancyArrowPatch((0.785, 0.2), (0.852, 0.2),
                            connectionstyle="arc3,rad=-0.65", arrowstyle="-|>",
                            mutation_scale=12, color=GRAY, linewidth=1.3))

# Between-panel arrows.
arrow(ax, 0.322, 0.48, 0.354, 0.48, color=PURPLE, lw=2.2)
arrow(ax, 0.644, 0.48, 0.676, 0.48, color=GREEN, lw=2.2)

fig.savefig(OUT / "core_highlights_minimal_reference.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "core_highlights_minimal_reference.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(OUT / "core_highlights_minimal_reference.png")
print(OUT / "core_highlights_minimal_reference.pdf")
