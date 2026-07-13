from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


NAVY = "#213B63"
BLUE = "#6EA6D9"
BLUE_LIGHT = "#EAF4FB"
GREEN = "#75A95A"
GREEN_DARK = "#4B853C"
GREEN_LIGHT = "#EEF7E9"
ORANGE = "#E99A62"
ORANGE_LIGHT = "#FFF0E5"
RED = "#D94C4C"
RED_LIGHT = "#FDECEC"
GRAY = "#727A84"
GRAY_LIGHT = "#F4F5F6"
PURPLE = "#A783B4"
PURPLE_LIGHT = "#F3ECF5"


def box(ax, xy, width, height, text="", face="white", edge=NAVY, lw=1.5,
        radius=0.018, fontsize=10, weight="normal", color="#1F1F1F", z=2):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(patch)
    if text:
        ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
                ha="center", va="center", fontsize=fontsize,
                weight=weight, color=color, zorder=z + 1)
    return patch


def arrow(ax, start, end, color=NAVY, lw=1.8, style="-|>", z=4, dashed=False):
    a = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13,
                        linewidth=lw, color=color, zorder=z,
                        linestyle="--" if dashed else "-")
    ax.add_patch(a)
    return a


fig = plt.figure(figsize=(16, 8.8), dpi=180, facecolor="white")
ax = fig.add_axes([0.018, 0.025, 0.964, 0.95])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Header: a light profile-interface ribbon rather than a third flow diagram.
ax.text(0.5, 0.966, "PROFILE-ANCHORED RUNTIME MECHANISMS",
        ha="center", va="center", fontsize=21, weight="bold", color=NAVY)
ax.text(0.5, 0.933,
        "Two internal mechanisms consumed by the same replaceable OrbitLLM profile interface",
        ha="center", va="center", fontsize=10.5, color=GRAY)
box(ax, (0.365, 0.872), 0.27, 0.045,
    "PROFILE INTERFACE   energy | latency | memory | utility",
    face=PURPLE_LIGHT, edge=PURPLE, lw=1.4, fontsize=9.5, weight="bold", color="#60466B")

# Main panels.
left = FancyBboxPatch((0.035, 0.08), 0.445, 0.755,
                      boxstyle="round,pad=0.012,rounding_size=0.026",
                      facecolor="#FCFDFE", edgecolor="#BEC8D2", linewidth=1.7)
right = FancyBboxPatch((0.52, 0.08), 0.445, 0.755,
                       boxstyle="round,pad=0.012,rounding_size=0.026",
                       facecolor="#FCFDFE", edgecolor="#BEC8D2", linewidth=1.7)
ax.add_patch(left)
ax.add_patch(right)

ax.text(0.055, 0.805, "A   Decode-Aware Memory Guard",
        fontsize=15, weight="bold", color=NAVY, va="center")
ax.text(0.54, 0.805, "B   Measured Semantic Pre-Screening",
        fontsize=15, weight="bold", color=NAVY, va="center")

# Panel A: admission check.
box(ax, (0.058, 0.71), 0.118, 0.063,
    "Predicted peak\nmemory", face=BLUE_LIGHT, edge=BLUE,
    fontsize=9.5, weight="bold", color=NAVY)
diamond = Polygon([[0.205, 0.742], [0.236, 0.775], [0.267, 0.742], [0.236, 0.709]],
                  closed=True, facecolor=PURPLE_LIGHT, edgecolor=PURPLE, linewidth=1.5)
ax.add_patch(diamond)
ax.text(0.236, 0.742, "fit?", ha="center", va="center", fontsize=9.5, weight="bold")
box(ax, (0.298, 0.71), 0.148, 0.063,
    "Admit ON\nunder guard", face=GREEN_LIGHT, edge=GREEN,
    fontsize=9.5, weight="bold", color=GREEN_DARK)
arrow(ax, (0.176, 0.742), (0.205, 0.742), color=BLUE)
arrow(ax, (0.267, 0.742), (0.298, 0.742), color=GREEN)
ax.text(0.274, 0.764, "yes", fontsize=8.5, color=GREEN_DARK)
arrow(ax, (0.236, 0.709), (0.236, 0.68), color=RED)
ax.text(0.247, 0.687, "reject / DOWN", fontsize=8.2, color=RED, va="center")

# Panel A: time axis and prefill/decode tokens.
ax.text(0.063, 0.645, "Execution timeline", fontsize=10.5, weight="bold", color=GRAY)
ax.plot([0.075, 0.445], [0.42, 0.42], color="#AAB2BA", lw=1.2)
ax.text(0.448, 0.415, "time", fontsize=8.5, color=GRAY, ha="right", va="top")

prefill_x, prefill_w = 0.078, 0.098
ax.add_patch(Rectangle((prefill_x, 0.43), prefill_w, 0.095,
                       facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.5))
ax.text(prefill_x + prefill_w / 2, 0.48, "PREFILL", ha="center", va="center",
        fontsize=10, weight="bold", color=NAVY)
ax.text(prefill_x + prefill_w / 2, 0.448, "parallel burst", ha="center", va="center",
        fontsize=7.8, color=GRAY)

token_x = 0.188
for i in range(10):
    color = ORANGE_LIGHT if i < 7 else RED_LIGHT
    edge = ORANGE if i < 7 else RED
    ax.add_patch(Rectangle((token_x + i * 0.025, 0.43), 0.019, 0.065,
                           facecolor=color, edgecolor=edge, linewidth=1.0))
    ax.text(token_x + i * 0.025 + 0.0095, 0.462, f"t{i+1}", ha="center", va="center",
            fontsize=6.2, color="#6B4930" if i < 7 else RED)
ax.text(0.306, 0.515, "DECODE: sequential tokens", ha="center", va="bottom",
        fontsize=9.5, weight="bold", color="#9C5A30")

# KV cache growth and guardband.
guard_y = 0.622
ax.plot([0.075, 0.445], [guard_y, guard_y], color=RED, lw=1.7, ls="--")
ax.text(0.442, guard_y + 0.012, "Guardband  (1-gamma) Mmax",
        fontsize=8.5, color=RED, ha="right", va="bottom", weight="bold")
x_curve = [0.079, 0.176, 0.198, 0.223, 0.248, 0.273, 0.298, 0.323, 0.348, 0.373, 0.398]
y_curve = [0.535, 0.545, 0.551, 0.558, 0.566, 0.575, 0.585, 0.596, 0.608, 0.622, 0.635]
ax.plot(x_curve, y_curve, color=NAVY, lw=2.4, marker="o", markersize=3.3)
ax.text(0.095, 0.565, "KV-cache growth", fontsize=8.5, color=NAVY, rotation=5)

# Early-stop gate at crossing.
ax.plot([0.373, 0.373], [0.405, 0.655], color=RED, lw=1.3, ls=":")
box(ax, (0.335, 0.33), 0.112, 0.062,
    "Hard cap /\nearly stop", face=RED_LIGHT, edge=RED,
    fontsize=9.2, weight="bold", color=RED)
arrow(ax, (0.373, 0.405), (0.39, 0.392), color=RED, lw=1.4)
ax.text(0.075, 0.375, "Energy accumulates with generated length",
        fontsize=8.8, color="#9C5A30")
ax.plot([0.078, 0.315], [0.352, 0.352], color=ORANGE, lw=4, solid_capstyle="round")
ax.plot([0.315, 0.373], [0.352, 0.352], color=RED, lw=4, solid_capstyle="round")

box(ax, (0.075, 0.18), 0.165, 0.092,
    "Admission guard\n+ runtime hard cap", face=GREEN_LIGHT,
    edge=GREEN, fontsize=10, weight="bold", color=GREEN_DARK)
box(ax, (0.272, 0.18), 0.165, 0.092,
    "0 OOM events\nin evaluated traces", face=BLUE_LIGHT,
    edge=BLUE, fontsize=10, weight="bold", color=NAVY)
arrow(ax, (0.24, 0.226), (0.272, 0.226), color=GREEN)
ax.text(0.258, 0.145, "Not a universal hardware guarantee",
        fontsize=8.3, color=GRAY, ha="center", style="italic")

# Panel B: stylized raw remote-sensing tile.
raw_x, raw_y, raw_w, raw_h = 0.545, 0.55, 0.13, 0.17
box(ax, (raw_x, raw_y), raw_w, raw_h, face=BLUE_LIGHT, edge=BLUE, lw=1.5)
ax.add_patch(Rectangle((raw_x + 0.01, raw_y + 0.012), raw_w - 0.02, raw_h - 0.024,
                       facecolor="#B9D79A", edgecolor="none", zorder=3))
ax.add_patch(Polygon([[raw_x + 0.01, raw_y + 0.03], [raw_x + 0.05, raw_y + 0.08],
                      [raw_x + 0.075, raw_y + 0.06], [raw_x + 0.12, raw_y + 0.11],
                      [raw_x + 0.12, raw_y + 0.012], [raw_x + 0.01, raw_y + 0.012]],
                     closed=True, facecolor="#7FB7D6", edgecolor="none", zorder=3))
ax.add_patch(Rectangle((raw_x + 0.073, raw_y + 0.095), 0.028, 0.025,
                       facecolor="#D9A66B", edgecolor="#95663D", linewidth=0.7, zorder=4))
ax.add_patch(Rectangle((raw_x + 0.038, raw_y + 0.118), 0.020, 0.017,
                       facecolor="#D9A66B", edgecolor="#95663D", linewidth=0.7, zorder=4))
ax.text(raw_x + raw_w / 2, raw_y - 0.023, "Raw sensor tile", ha="center",
        fontsize=9.2, weight="bold", color=NAVY)

# MobileSAM selector, not a scheduler/funnel duplication.
box(ax, (0.704, 0.575), 0.095, 0.12,
    "MobileSAM\nmask selector", face=GREEN_LIGHT, edge=GREEN,
    fontsize=10, weight="bold", color=GREEN_DARK)
arrow(ax, (raw_x + raw_w, raw_y + 0.09), (0.704, 0.635), color=GREEN)

# Retained and dropped outputs.
box(ax, (0.83, 0.625), 0.105, 0.094,
    "RETAIN\nsemantic regions", face=ORANGE_LIGHT, edge=ORANGE,
    fontsize=9.4, weight="bold", color="#9C5A30")
box(ax, (0.83, 0.505), 0.105, 0.072,
    "DROP\nbackground", face=GRAY_LIGHT, edge="#A7ACB2",
    fontsize=9.2, weight="bold", color=GRAY)
arrow(ax, (0.799, 0.65), (0.83, 0.672), color=ORANGE)
arrow(ax, (0.799, 0.615), (0.83, 0.542), color="#9FA4AA")

# Small measured tradeoff plot.
plot_x0, plot_y0, plot_w, plot_h = 0.555, 0.245, 0.235, 0.205
box(ax, (plot_x0 - 0.012, plot_y0 - 0.03), plot_w + 0.03, plot_h + 0.06,
    face="white", edge="#D0D5DA", lw=1.1)
ax.plot([plot_x0, plot_x0], [plot_y0, plot_y0 + plot_h], color=GRAY, lw=1.0)
ax.plot([plot_x0, plot_x0 + plot_w], [plot_y0, plot_y0], color=GRAY, lw=1.0)
pts = [(0.21, 0.42), (0.32, 0.37), (0.50, 0.53)]
px = [plot_x0 + p[0] * plot_w / 0.6 for p in pts]
py = [plot_y0 + p[1] * plot_h / 0.6 for p in pts]
ax.plot(px, py, color=GREEN_DARK, lw=2.0, marker="o", markersize=6,
        markerfacecolor=GREEN_LIGHT, markeredgewidth=1.4)
for (rho, eta), x, y in zip(pts, px, py):
    ax.text(x + 0.006, y + 0.008, f"({rho:.2f}, {eta:.2f})",
            fontsize=7.5, color=GREEN_DARK)
ax.text(plot_x0 + plot_w / 2, plot_y0 - 0.018, "retained ratio  rho",
        ha="center", va="top", fontsize=8.3, color=GRAY)
ax.text(plot_x0 - 0.012, plot_y0 + plot_h / 2, "utility  eta",
        ha="right", va="center", rotation=90, fontsize=8.3, color=GRAY)
ax.text(plot_x0 + 0.008, plot_y0 + plot_h - 0.012, "Measured cost-utility profile",
        fontsize=9.2, weight="bold", color=NAVY, va="top")

# Measured MobileSAM costs and profile tuple.
box(ax, (0.82, 0.35), 0.125, 0.104,
    "Measured cost\n17.9 J / image\n250 ms / image", face=BLUE_LIGHT,
    edge=BLUE, fontsize=9.2, weight="bold", color=NAVY)
box(ax, (0.82, 0.19), 0.125, 0.105,
    "PRE profile row\n(rho, eta, E, L, M)", face=PURPLE_LIGHT,
    edge=PURPLE, fontsize=9.2, weight="bold", color="#60466B")
arrow(ax, (0.79, 0.35), (0.82, 0.40), color=BLUE)
arrow(ax, (0.882, 0.35), (0.882, 0.295), color=PURPLE)
ax.text(0.75, 0.135, "Semantic compression is a measured tradeoff, not lossless filtering",
        ha="center", va="center", fontsize=8.5, color=GRAY, style="italic")

# Short profile-interface links stay in the header and do not cross panel content.
arrow(ax, (0.43, 0.872), (0.29, 0.838), color=PURPLE, lw=1.2, dashed=True)
arrow(ax, (0.57, 0.872), (0.72, 0.838), color=PURPLE, lw=1.2, dashed=True)
ax.text(0.305, 0.849, "reads memory profile", fontsize=7.7, color=PURPLE, ha="center")
ax.text(0.695, 0.849, "writes PRE profile", fontsize=7.7, color=PURPLE, ha="center")

fig.savefig(OUT / "core_mechanisms_reference.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "core_mechanisms_reference.pdf", bbox_inches="tight", facecolor="white")
plt.close(fig)

print(OUT / "core_mechanisms_reference.png")
print(OUT / "core_mechanisms_reference.pdf")
