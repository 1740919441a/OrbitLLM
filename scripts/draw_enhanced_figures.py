from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, RegularPolygon


FIG_DIR = Path("figures")

COL = {
    "blue": "#1f5a9d",
    "blue_dark": "#123c69",
    "sky": "#dff0ff",
    "green": "#2a8c4a",
    "green_bg": "#e8f6ea",
    "orange": "#c76b00",
    "orange_bg": "#fff2df",
    "red": "#c73a32",
    "red_bg": "#ffe7e4",
    "gray": "#65717d",
    "gray_bg": "#f1f3f5",
    "ink": "#17212b",
    "panel": "#fbfcfe",
    "teal": "#087c7c",
    "teal_bg": "#e7f5f4",
}


def add_shadow(patch, alpha: float = 0.12):
    patch.set_path_effects(
        [
            pe.SimplePatchShadow(offset=(2, -2), shadow_rgbFace=(0, 0, 0), alpha=alpha),
            pe.Normal(),
        ]
    )
    return patch


def round_box(ax, xy, wh, text="", fc="white", ec=COL["blue"], lw=1.8, r=0.02, fs=10, weight="regular", color=COL["ink"], shadow=False):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    if shadow:
        add_shadow(patch)
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight=weight, color=color, linespacing=1.15)
    return patch


def group_box(ax, xy, wh, title, fc, ec):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        alpha=0.95,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + h - 0.035, title, ha="left", va="center", fontsize=10.5, fontweight="bold", color=ec)
    return patch


def arrow(ax, start, end, color=COL["blue_dark"], lw=1.8, rad=0.0, ms=14):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def ortho_arrow(ax, start, end, color=COL["blue_dark"], lw=1.8, mid_x=None, ms=14):
    sx, sy = start
    ex, ey = end
    mx = mid_x if mid_x is not None else (sx + ex) / 2
    ax.plot([sx, mx, mx], [sy, sy, ey], color=color, lw=lw, solid_capstyle="round")
    arr = FancyArrowPatch((mx, ey), (ex, ey), arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, shrinkA=0, shrinkB=2)
    ax.add_patch(arr)
    return arr


def draw_satellite(ax, x, y, s=1.0, color=COL["blue"]):
    ax.add_patch(Rectangle((x - 0.018 * s, y - 0.014 * s), 0.036 * s, 0.028 * s, facecolor="white", edgecolor=color, lw=1.4))
    ax.add_patch(Rectangle((x - 0.064 * s, y - 0.018 * s), 0.038 * s, 0.036 * s, facecolor=COL["sky"], edgecolor=color, lw=1.2))
    ax.add_patch(Rectangle((x + 0.026 * s, y - 0.018 * s), 0.038 * s, 0.036 * s, facecolor=COL["sky"], edgecolor=color, lw=1.2))
    ax.plot([x - 0.026 * s, x - 0.018 * s], [y, y], color=color, lw=1.1)
    ax.plot([x + 0.018 * s, x + 0.026 * s], [y, y], color=color, lw=1.1)
    ax.plot([x, x + 0.025 * s], [y + 0.014 * s, y + 0.04 * s], color=color, lw=1.1)
    ax.add_patch(Arc((x + 0.03 * s, y + 0.045 * s), 0.03 * s, 0.03 * s, theta1=200, theta2=330, edgecolor=color, lw=1.0))


def draw_task(ax, x, y, s=1.0, color=COL["gray"]):
    for i in range(3):
        dx = i * 0.008 * s
        dy = -i * 0.006 * s
        ax.add_patch(Rectangle((x + dx, y + dy), 0.045 * s, 0.055 * s, facecolor="white", edgecolor=color, lw=1.0))
        ax.plot([x + dx + 0.008 * s, x + dx + 0.036 * s], [y + dy + 0.037 * s, y + dy + 0.037 * s], color=color, lw=0.8)
        ax.plot([x + dx + 0.008 * s, x + dx + 0.031 * s], [y + dy + 0.025 * s, y + dy + 0.025 * s], color=color, lw=0.8)


def draw_gpu(ax, x, y, s=1.0, color=COL["orange"]):
    ax.add_patch(Rectangle((x - 0.04 * s, y - 0.028 * s), 0.08 * s, 0.056 * s, facecolor="white", edgecolor=color, lw=1.4))
    ax.add_patch(Rectangle((x - 0.022 * s, y - 0.014 * s), 0.044 * s, 0.028 * s, facecolor=COL["orange_bg"], edgecolor=color, lw=1.1))
    for i in range(5):
        yy = y - 0.022 * s + i * 0.011 * s
        ax.plot([x - 0.048 * s, x - 0.04 * s], [yy, yy], color=color, lw=1.0)
        ax.plot([x + 0.04 * s, x + 0.048 * s], [yy, yy], color=color, lw=1.0)


def draw_station(ax, x, y, s=1.0, color=COL["blue"]):
    ax.add_patch(Polygon([[x - 0.03 * s, y - 0.035 * s], [x + 0.03 * s, y - 0.035 * s], [x, y + 0.01 * s]], closed=True, facecolor="white", edgecolor=color, lw=1.3))
    ax.add_patch(Arc((x, y + 0.012 * s), 0.08 * s, 0.055 * s, theta1=20, theta2=160, edgecolor=color, lw=1.4))
    ax.add_patch(Arc((x, y + 0.012 * s), 0.12 * s, 0.085 * s, theta1=25, theta2=155, edgecolor=color, lw=0.9, alpha=0.75))


def draw_filter(ax, x, y, s=1.0, color=COL["green"]):
    ax.add_patch(Polygon([[x - 0.045 * s, y + 0.025 * s], [x + 0.045 * s, y + 0.025 * s], [x + 0.014 * s, y - 0.008 * s], [x + 0.014 * s, y - 0.038 * s], [x - 0.014 * s, y - 0.038 * s], [x - 0.014 * s, y - 0.008 * s]], closed=True, facecolor="white", edgecolor=color, lw=1.4))
    ax.add_patch(Circle((x + 0.04 * s, y - 0.03 * s), 0.018 * s, facecolor=COL["green_bg"], edgecolor=color, lw=1.1))
    ax.plot([x + 0.053 * s, x + 0.073 * s], [y - 0.043 * s, y - 0.063 * s], color=color, lw=1.4)


def draw_profile_icon(ax, x, y, s=1.0):
    ec = COL["teal"]
    ax.add_patch(Rectangle((x - 0.055 * s, y - 0.035 * s), 0.11 * s, 0.07 * s, facecolor="white", edgecolor=ec, lw=1.3))
    for i in range(1, 3):
        ax.plot([x - 0.055 * s, x + 0.055 * s], [y - 0.035 * s + i * 0.023 * s, y - 0.035 * s + i * 0.023 * s], color=ec, lw=0.8)
    ax.plot([x - 0.018 * s, x - 0.018 * s], [y - 0.035 * s, y + 0.035 * s], color=ec, lw=0.8)
    ax.plot([x + 0.018 * s, x + 0.018 * s], [y - 0.035 * s, y + 0.035 * s], color=ec, lw=0.8)
    ax.add_patch(Rectangle((x + 0.068 * s, y - 0.026 * s), 0.05 * s, 0.052 * s, facecolor=COL["teal_bg"], edgecolor=ec, lw=1.1))
    for i in range(4):
        yy = y - 0.018 * s + i * 0.012 * s
        ax.plot([x + 0.06 * s, x + 0.068 * s], [yy, yy], color=ec, lw=0.8)


def warning(ax, x, y, label):
    tri = RegularPolygon((x, y + 0.018), 3, radius=0.035, orientation=0, facecolor=COL["red_bg"], edgecolor=COL["red"], lw=1.6)
    ax.add_patch(tri)
    ax.text(x, y + 0.017, "!", ha="center", va="center", fontsize=14, fontweight="bold", color=COL["red"])
    ax.text(x + 0.055, y + 0.016, label, ha="left", va="center", fontsize=10, color=COL["red"], fontweight="bold")


def draw_motivation():
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.965, "Why Profile-Anchored Scheduling?", ha="center", va="top", fontsize=22, fontweight="bold", color=COL["ink"])
    round_box(ax, (0.035, 0.54), (0.93, 0.34), fc="#fbfbfc", ec="#d5dbe1", lw=1.2, r=0.018)
    round_box(ax, (0.035, 0.10), (0.93, 0.34), fc="#fbfdff", ec="#c9dff3", lw=1.2, r=0.018)
    ax.text(0.055, 0.845, "Traditional generic offloading", ha="left", va="center", fontsize=14, fontweight="bold", color=COL["gray"])
    ax.text(0.055, 0.405, "OrbitLLM profile-anchored scheduling", ha="left", va="center", fontsize=14, fontweight="bold", color=COL["blue_dark"])

    # Traditional row
    draw_satellite(ax, 0.105, 0.705, 1.1)
    draw_task(ax, 0.155, 0.675, 1.0)
    ax.text(0.145, 0.625, "Task as generic bits\n+ CPU cycles", ha="center", va="top", fontsize=10, color=COL["gray"])
    round_box(ax, (0.31, 0.64), (0.18, 0.13), "Generic scheduler\nLLM costs hidden", fc=COL["gray_bg"], ec=COL["gray"], fs=11, shadow=True)
    round_box(ax, (0.61, 0.63), (0.18, 0.15), "", fc="#fffaf2", ec=COL["orange"], fs=11, shadow=True)
    draw_gpu(ax, 0.655, 0.705, 0.62, COL["orange"])
    ax.text(0.72, 0.705, "Mispriced\nON decision", ha="center", va="center", fontsize=11, color=COL["ink"], linespacing=1.15)
    warning(ax, 0.83, 0.735, "OOM from KV cache")
    warning(ax, 0.83, 0.660, "Energy drained by decode")
    arrow(ax, (0.21, 0.705), (0.31, 0.705), color=COL["gray"])
    arrow(ax, (0.49, 0.705), (0.61, 0.705), color=COL["gray"])
    arrow(ax, (0.79, 0.705), (0.82, 0.705), color=COL["red"], lw=1.6)

    # OrbitLLM row
    draw_satellite(ax, 0.105, 0.265, 1.1)
    draw_task(ax, 0.155, 0.235, 1.0)
    ax.text(0.145, 0.185, "Same task stream\nwith value decay", ha="center", va="top", fontsize=10, color=COL["blue_dark"])
    round_box(ax, (0.285, 0.205), (0.19, 0.13), "", fc=COL["teal_bg"], ec=COL["teal"], fs=12, weight="bold", shadow=True)
    ax.text(0.38, 0.302, "LLM Cost Profile", ha="center", va="center", fontsize=12, fontweight="bold", color=COL["ink"])
    draw_profile_icon(ax, 0.38, 0.245, 0.48)
    ax.text(0.38, 0.172, "KV boundary  +  prefill/decode energy", ha="center", va="center", fontsize=9.5, color=COL["teal"])
    hexagon = RegularPolygon((0.57, 0.27), 6, radius=0.082, orientation=0, facecolor="#e8f1fb", edgecolor=COL["blue_dark"], lw=2.0)
    add_shadow(hexagon, 0.14)
    ax.add_patch(hexagon)
    ax.text(0.57, 0.27, "Heuristic-TVD\nScheduler", ha="center", va="center", fontsize=12, fontweight="bold", color=COL["blue_dark"])

    cards = [
        ((0.72, 0.32), (0.14, 0.07), "ON", "on-board inference", COL["orange_bg"], COL["orange"]),
        ((0.72, 0.225), (0.14, 0.07), "PRE", "semantic screening", COL["green_bg"], COL["green"]),
        ((0.72, 0.13), (0.14, 0.07), "DOWN", "direct downlink", COL["sky"], COL["blue"]),
    ]
    for (xy, wh, head, sub, fc, ec) in cards:
        round_box(ax, xy, wh, f"{head}\n{sub}", fc=fc, ec=ec, fs=9.5, weight="bold" if head == "PRE" else "regular")
    draw_gpu(ax, 0.69, 0.355, 0.45, COL["orange"])
    draw_filter(ax, 0.69, 0.26, 0.45, COL["green"])
    draw_station(ax, 0.69, 0.165, 0.45, COL["blue"])
    round_box(ax, (0.89, 0.205), (0.075, 0.13), "OK\nZero OOM\nhigh value", fc=COL["green_bg"], ec=COL["green"], fs=9.5, weight="bold", shadow=True)
    arrow(ax, (0.21, 0.265), (0.285, 0.265), color=COL["blue_dark"])
    arrow(ax, (0.475, 0.265), (0.49, 0.265), color=COL["teal"])
    arrow(ax, (0.65, 0.302), (0.72, 0.355), color=COL["orange"])
    arrow(ax, (0.65, 0.27), (0.72, 0.26), color=COL["green"])
    arrow(ax, (0.65, 0.238), (0.72, 0.165), color=COL["blue"])
    arrow(ax, (0.86, 0.355), (0.89, 0.29), color=COL["green"], lw=1.5)
    arrow(ax, (0.86, 0.26), (0.89, 0.27), color=COL["green"], lw=1.5)
    arrow(ax, (0.86, 0.165), (0.89, 0.245), color=COL["green"], lw=1.5)

    ax.text(0.5, 0.055, "OrbitLLM makes LLM memory, energy, and value tradeoffs explicit before scheduling.", ha="center", va="center", fontsize=11, color=COL["gray"])
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"motivation_profile_vs_generic.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)


def draw_architecture():
    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.5, 0.965, "OrbitLLM System Architecture", ha="center", va="top", fontsize=22, fontweight="bold", color=COL["ink"])

    group_box(ax, (0.035, 0.17), (0.26, 0.69), "Environment & Inputs", "#f8fbff", COL["blue"])
    group_box(ax, (0.335, 0.17), (0.34, 0.69), "OrbitLLM Controller", "#fbfdff", COL["teal"])
    group_box(ax, (0.715, 0.17), (0.25, 0.69), "Action Space", "#fffdf9", COL["orange"])

    # Inputs
    round_box(ax, (0.075, 0.705), (0.17, 0.09), "", fc="white", ec=COL["blue"], fs=10.5, shadow=True)
    draw_satellite(ax, 0.105, 0.75, 0.65)
    ax.text(0.17, 0.75, "LEO satellite\nsensor stream", ha="center", va="center", fontsize=10.5, color=COL["ink"])
    round_box(ax, (0.075, 0.55), (0.17, 0.095), "", fc="white", ec=COL["blue"], fs=10.2, shadow=True)
    draw_task(ax, 0.095, 0.578, 0.50)
    ax.text(0.17, 0.598, "Task queue", ha="center", va="center", fontsize=10.5, color=COL["ink"])
    ax.text(0.17, 0.568, "bits, value, half-life", ha="center", va="center", fontsize=9.5, color=COL["ink"])
    round_box(ax, (0.075, 0.395), (0.17, 0.095), "", fc="white", ec=COL["blue"], fs=10.2, shadow=True)
    ax.text(0.17, 0.458, "Visibility windows", ha="center", va="center", fontsize=10.5, color=COL["ink"])
    ax.text(0.17, 0.427, "rate + capacity", ha="center", va="center", fontsize=9.5, color=COL["ink"])
    ax.plot([0.105, 0.225], [0.405, 0.405], color=COL["blue"], lw=1.1)
    for tx in [0.11, 0.15, 0.19, 0.215]:
        ax.plot([tx, tx], [0.395, 0.415], color=COL["blue"], lw=1.0)
    round_box(ax, (0.075, 0.24), (0.17, 0.095), "", fc="white", ec=COL["blue"], fs=10.2, shadow=True)
    ax.add_patch(Rectangle((0.095, 0.277), 0.05, 0.021, facecolor=COL["green_bg"], edgecolor=COL["green"], lw=1.0))
    ax.add_patch(Rectangle((0.145, 0.283), 0.006, 0.009, facecolor=COL["green"], edgecolor=COL["green"], lw=0.8))
    ax.add_patch(Rectangle((0.102, 0.282), 0.032, 0.011, facecolor=COL["green"], edgecolor=COL["green"], lw=0.8, alpha=0.65))
    ax.add_patch(Rectangle((0.19, 0.272), 0.04, 0.032, facecolor=COL["sky"], edgecolor=COL["blue"], lw=1.0))
    ax.text(0.165, 0.305, "Resource state", ha="center", va="center", fontsize=10.5, color=COL["ink"])
    ax.text(0.165, 0.266, "battery + memory", ha="center", va="center", fontsize=9.5, color=COL["ink"])

    # Controller
    round_box(ax, (0.375, 0.69), (0.24, 0.115), "", fc=COL["teal_bg"], ec=COL["teal"], fs=12, weight="bold", shadow=True)
    ax.text(0.495, 0.765, "Parameterized LLM\nCost Profile", ha="center", va="center", fontsize=12, fontweight="bold", color=COL["ink"], linespacing=1.05)
    draw_profile_icon(ax, 0.495, 0.713, 0.42)
    ax.text(0.495, 0.655, "prefill energy | decode energy | KV memory", ha="center", va="center", fontsize=9.3, color=COL["teal"])
    scheduler = RegularPolygon((0.505, 0.48), 6, radius=0.105, orientation=0, facecolor="#e8f1fb", edgecolor=COL["blue_dark"], lw=2.1)
    add_shadow(scheduler, 0.16)
    ax.add_patch(scheduler)
    ax.text(0.505, 0.48, "Heuristic-TVD\nScheduler", ha="center", va="center", fontsize=13, fontweight="bold", color=COL["blue_dark"])
    round_box(ax, (0.385, 0.205), (0.24, 0.105), "Residual-state update\nenergy, window bits, value", fc="white", ec=COL["gray"], fs=10.2, shadow=True)

    # Actions
    action_specs = [
        ((0.755, 0.665), (0.18, 0.11), "ON", "full inference", COL["orange_bg"], COL["orange"], draw_gpu),
        ((0.755, 0.445), (0.18, 0.11), "PRE", "semantic filter", COL["green_bg"], COL["green"], draw_filter),
        ((0.755, 0.225), (0.18, 0.11), "DOWN", "raw downlink", COL["sky"], COL["blue"], draw_station),
    ]
    for xy, wh, head, sub, fc, ec, icon in action_specs:
        round_box(ax, xy, wh, "", fc=fc, ec=ec, fs=10.2, weight="bold" if head == "PRE" else "regular", shadow=True)
        icon(ax, xy[0] + 0.035, xy[1] + wh[1] / 2, 0.32, ec)
        ax.text(xy[0] + 0.116, xy[1] + wh[1] * 0.62, head, ha="center", va="center", fontsize=11, color=COL["ink"], fontweight="bold" if head == "PRE" else "regular")
        ax.text(xy[0] + 0.116, xy[1] + wh[1] * 0.38, sub, ha="center", va="center", fontsize=9.8, color=COL["ink"])

    # Flow arrows
    ortho_arrow(ax, (0.245, 0.598), (0.405, 0.50), color=COL["blue_dark"], mid_x=0.31)
    ortho_arrow(ax, (0.245, 0.442), (0.405, 0.47), color=COL["blue_dark"], mid_x=0.31)
    ortho_arrow(ax, (0.245, 0.288), (0.405, 0.445), color=COL["blue_dark"], mid_x=0.31)
    arrow(ax, (0.495, 0.69), (0.505, 0.59), color=COL["teal"], lw=1.9)
    arrow(ax, (0.61, 0.52), (0.755, 0.72), color=COL["orange"], lw=1.9)
    arrow(ax, (0.615, 0.48), (0.755, 0.50), color=COL["green"], lw=1.9)
    arrow(ax, (0.61, 0.44), (0.755, 0.28), color=COL["blue"], lw=1.9)
    ortho_arrow(ax, (0.755, 0.72), (0.625, 0.27), color=COL["gray"], mid_x=0.69, lw=1.3)
    ortho_arrow(ax, (0.755, 0.50), (0.625, 0.27), color=COL["gray"], mid_x=0.69, lw=1.3)
    ortho_arrow(ax, (0.755, 0.28), (0.625, 0.27), color=COL["gray"], mid_x=0.69, lw=1.3)
    arrow(ax, (0.505, 0.31), (0.505, 0.38), color=COL["gray"], lw=1.5)

    ax.text(0.505, 0.105, "Profile rows are replaceable: changing hardware, quantization, or VLM/LLM runtime updates scheduling costs without changing the controller.", ha="center", va="center", fontsize=10.2, color=COL["gray"])

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"architecture_enhanced.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    draw_motivation()
    draw_architecture()
    print("wrote enhanced figures to figures/")


if __name__ == "__main__":
    main()
