from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np
import pandas as pd

from .config import ensure_output_dirs, load_config, resolve_project_path
from .profile import load_profile, profile_table_rows


OK = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#777777",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 220,
            "savefig.dpi": 220,
        }
    )


def _box(ax, xy: tuple[float, float], wh: tuple[float, float], text: str, *, fc: str = "#F7FBFF", ec: str = "#174A7C", fs: int = 8) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _diamond(ax, center: tuple[float, float], wh: tuple[float, float], text: str, *, fc: str = "#EEF7FF", ec: str = "#174A7C", fs: int = 8) -> None:
    cx, cy = center
    w, h = wh
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], label: str = "", *, color: str = "#174A7C", fs: int = 7) -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=color)
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, label, ha="center", va="bottom", fontsize=fs, color=color)


def plot_architecture(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "OrbitLLM Energy-Value Scheduling Framework", ha="center", va="center", fontsize=14, weight="bold")

    _box(ax, (0.04, 0.61), (0.22, 0.16), "LEO satellite\nsensor + task queue", fc="#F3F8FF")
    _box(ax, (0.04, 0.31), (0.22, 0.16), "Task state\nbits, value, half-life,\nmodel scale", fc="#F3F8FF")
    _box(ax, (0.34, 0.69), (0.22, 0.16), "Parameterized LLM\ncost profile\nenergy, latency, memory", fc="#EEF8F5", ec="#08756F")
    _box(ax, (0.62, 0.69), (0.22, 0.16), "Visibility windows\ndownlink rate + capacity", fc="#EEF8F5", ec="#08756F")
    _diamond(ax, (0.50, 0.43), (0.20, 0.22), "Heuristic-TVD\nscheduler", fc="#F4FAFF")
    _box(ax, (0.74, 0.50), (0.20, 0.12), "ON\non-board inference", fc="#FFF7ED", ec="#C76700")
    _box(ax, (0.74, 0.32), (0.20, 0.12), "PRE\nscreen + partial downlink", fc="#F1FAEE", ec="#278A42")
    _box(ax, (0.74, 0.14), (0.20, 0.12), "DOWN\nraw data to ground", fc="#EEF5FF", ec="#2A62A8")
    _box(ax, (0.36, 0.12), (0.28, 0.11), "Update residual energy,\nwindow bits, realized value", fc="#FBFBFB", ec="#555555")

    _arrow(ax, (0.15, 0.61), (0.15, 0.47))
    _arrow(ax, (0.26, 0.39), (0.40, 0.43), "task")
    _arrow(ax, (0.45, 0.69), (0.49, 0.54), "cost")
    _arrow(ax, (0.72, 0.69), (0.58, 0.52), "windows")
    _arrow(ax, (0.60, 0.46), (0.74, 0.56), "choose")
    _arrow(ax, (0.60, 0.43), (0.74, 0.38))
    _arrow(ax, (0.60, 0.40), (0.74, 0.20))
    _arrow(ax, (0.84, 0.32), (0.62, 0.19), "commit", color="#555555")
    _arrow(ax, (0.84, 0.14), (0.62, 0.16), color="#555555")
    _arrow(ax, (0.74, 0.50), (0.62, 0.21), color="#555555")
    fig.tight_layout()
    fig.savefig(fig_dir / "architecture.png", bbox_inches="tight")
    plt.close(fig)


def plot_decision_flow(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.96, "Heuristic-TVD Per-Task Flow", ha="center", va="center", fontsize=12, weight="bold")

    _box(ax, (0.18, 0.82), (0.64, 0.10), "New task arrives\n(d, V, half-life, model)", fc="#FBFBFB", ec="#333333", fs=7)
    _box(ax, (0.18, 0.68), (0.64, 0.09), "Look up LLM profile\nenergy, latency, peak memory", fc="#F3F8FF", fs=7)
    _diamond(ax, (0.50, 0.55), (0.62, 0.16), "Remove infeasible actions\nenergy or memory violation?", fc="#EEF7FF", fs=6)
    _box(ax, (0.14, 0.36), (0.72, 0.10), "Score remaining ON / PRE / DOWN\nby task value density", fc="#EEF8F5", ec="#08756F", fs=7)
    _diamond(ax, (0.50, 0.24), (0.50, 0.11), "argmax TVD", fc="#FFF7ED", ec="#C76700", fs=8)
    _box(ax, (0.07, 0.06), (0.24, 0.10), "ON\ninfer now", fc="#FFF7ED", ec="#C76700", fs=7)
    _box(ax, (0.38, 0.06), (0.24, 0.10), "PRE\nscreen bits", fc="#F1FAEE", ec="#278A42", fs=7)
    _box(ax, (0.69, 0.06), (0.24, 0.10), "DOWN\nqueue data", fc="#EEF5FF", ec="#2A62A8", fs=7)

    _arrow(ax, (0.50, 0.82), (0.50, 0.77))
    _arrow(ax, (0.50, 0.68), (0.50, 0.615))
    _arrow(ax, (0.50, 0.47), (0.50, 0.46))
    _arrow(ax, (0.50, 0.36), (0.50, 0.295))
    _arrow(ax, (0.44, 0.19), (0.19, 0.16))
    _arrow(ax, (0.50, 0.185), (0.50, 0.16))
    _arrow(ax, (0.56, 0.19), (0.81, 0.16))
    fig.tight_layout()
    fig.savefig(fig_dir / "decision_flow.png", bbox_inches="tight")
    plt.close(fig)


def ci95(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return 1.96 * float(series.std(ddof=1)) / np.sqrt(len(series))


def plot_energy(profile: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    colors = {"FP16": OK["red"], "INT8": OK["orange"], "INT4": OK["green"]}
    for quant in ["FP16", "INT8", "INT4"]:
        rows = profile[(profile["quant"] == quant) & (profile["context_len"] == 2048)].sort_values("scale_b")
        ax.plot(rows["scale_b"], rows["prefill_j_per_tok"] * 1000, marker="o", color=colors[quant], label=f"{quant} prefill")
        ax.plot(rows["scale_b"], rows["decode_j_per_tok"] * 1000, marker="s", ls="--", color=colors[quant], label=f"{quant} decode")
    ax.set_xlabel("Model scale (B params)")
    ax.set_ylabel("Energy per token (mJ)")
    ax.set_xticks(sorted(profile["scale_b"].unique()))
    ax.legend(fontsize=6.2, ncol=2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "energy_per_token.png", bbox_inches="tight")
    plt.close(fig)


def plot_mem(profile: pd.DataFrame, fig_dir: Path) -> None:
    contexts = sorted(profile["context_len"].unique())
    scales = sorted(profile["scale_b"].unique())
    data = np.zeros((len(scales), len(contexts)))
    for i, scale in enumerate(scales):
        for j, ctx in enumerate(contexts):
            row = profile[(profile["scale_b"] == scale) & (profile["context_len"] == ctx) & (profile["quant"] == "INT4")]
            data[i, j] = float(row.iloc[0]["peak_mem_gb"])
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts, fontsize=7)
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{s:g}B" for s in scales])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Model scale")
    cs = ax.contour(np.arange(len(contexts)), np.arange(len(scales)), data, levels=[16.0], colors="white", linewidths=1.3, linestyles="--")
    if cs.allsegs and any(len(seg) for level in cs.allsegs for seg in level):
        ax.clabel(cs, fmt="16 GB", fontsize=6)
    for i in range(len(scales)):
        for j in range(len(contexts)):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if data[i, j] < 10 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Peak memory (GB)", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "mem_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_value(timeseries: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    colors = {
        "MILP-Bound": "black",
        "Heuristic-TVD": OK["blue"],
        "Fixed-Threshold": OK["green"],
        "All-On-Board": OK["orange"],
        "All-Downlink": OK["red"],
    }
    main = timeseries.copy()
    max_by_seed = main.groupby("seed")["cum_value"].transform("max")
    main["norm_cum"] = main["cum_value"] / max_by_seed
    grid = np.linspace(0, 24, 121)
    for policy in ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD"]:
        curves = []
        for _, seed_df in main[main["policy"] == policy].groupby("seed"):
            seed_df = seed_df.sort_values("arrival_h")
            curves.append(np.interp(grid, seed_df["arrival_h"], seed_df["norm_cum"], left=0, right=seed_df["norm_cum"].iloc[-1]))
        arr = np.vstack(curves)
        mean = arr.mean(axis=0)
        ax.plot(grid, mean, color=colors[policy], lw=2.0 if policy == "Heuristic-TVD" else 1.4, label=policy)
        if policy == "Heuristic-TVD":
            err = 1.96 * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ax.fill_between(grid, mean - err, mean + err, color=colors[policy], alpha=0.18)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cumulative value (normalized)")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "value_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, fig_dir: Path) -> None:
    order = ["Full", "No prefill/decode split", "No PRE action"]
    grouped = ablation.groupby("variant")["normalized_value"]
    means = [grouped.mean()[x] for x in order]
    errs = [ci95(ablation[ablation["variant"] == x]["normalized_value"]) for x in order]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    labels = ["Full", "No split", "No PRE"]
    bars = ax.bar(labels, means, yerr=errs, capsize=3, color=[OK["blue"], OK["sky"], OK["gray"]], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Cumulative value (normalized)")
    ax.set_ylim(max(0.0, min(means) - 0.08), 1.04)
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "ablation.png", bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(sensitivity: pd.DataFrame, fig_dir: Path) -> None:
    grouped = sensitivity.groupby("energy_factor")
    factors = sorted(grouped.groups)
    values = [grouped.get_group(f)["normalized_value"].mean() for f in factors]
    jpt = [grouped.get_group(f)["j_per_task"].mean() for f in factors]
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    x = np.arange(len(factors))
    ax.bar(x, values, width=0.52, color=OK["blue"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Cumulative value (normalized)", color=OK["blue"])
    ax.set_ylim(0.55, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.1f}x" for f in factors])
    ax.set_xlabel("On-board energy budget")
    ax2 = ax.twinx()
    ax2.plot(x, jpt, marker="o", color=OK["red"], lw=1.6)
    ax2.set_ylabel("Energy per useful task (J)", color=OK["red"])
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "power_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_network_sensitivity(network: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))
    policy_order = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD"]
    colors = {
        "All-Downlink": OK["red"],
        "All-On-Board": OK["orange"],
        "Fixed-Threshold": OK["green"],
        "Heuristic-TVD": OK["blue"],
    }
    rate = network[network["sweep_type"] == "rate"].copy()
    for policy in policy_order:
        rows = rate[rate["policy"] == policy].groupby("sweep_value")["normalized_value"].mean().sort_index()
        axes[0].plot(rows.index, rows.values, marker="o", label=policy, color=colors[policy], lw=1.8 if policy == "Heuristic-TVD" else 1.1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Downlink rate (Mbps)")
    axes[0].set_ylabel("Value (normalized)")
    axes[0].set_ylim(0.45, 1.03)
    axes[0].legend(fontsize=5.8, loc="lower right")

    window = network[(network["sweep_type"] == "window") & (network["policy"] == "Heuristic-TVD")].copy()
    grouped = window.groupby("sweep_value")[["on_pct", "pre_pct", "down_pct"]].mean().sort_index()
    x = np.arange(len(grouped))
    bottom = np.zeros(len(grouped))
    for col, label, color in [("on_pct", "ON", OK["orange"]), ("pre_pct", "PRE", OK["green"]), ("down_pct", "DOWN", OK["sky"])]:
        axes[1].bar(x, grouped[col], bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.3)
        bottom += grouped[col].to_numpy()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{v:g}" for v in grouped.index])
    axes[1].set_xlabel("Visibility window (min)")
    axes[1].set_ylabel("Heuristic action mix (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(fig_dir / "network_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_pre_sensitivity(pre: pd.DataFrame, fig_dir: Path) -> None:
    data = pre[pre["variant"] == "PRE"].copy()
    pivot = data.groupby(["eta_model", "rho"])["normalized_value"].mean().unstack("rho").reindex(["conservative", "nominal", "optimistic"])
    fig, ax = plt.subplots(figsize=(3.35, 2.45))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="YlGnBu", vmin=max(0.65, np.nanmin(pivot.to_numpy()) - 0.02), vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(["conserv.", "nominal", "optim."])
    ax.set_xlabel("PRE retention ratio $\\rho$")
    ax.set_ylabel("Utility retention model")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Value (normalized)", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "pre_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_hardware_sensitivity(hardware: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))
    heur = hardware[hardware["policy"] == "Heuristic-TVD"].copy()
    grouped = heur.groupby(["energy_scale", "profile_label"])
    labels = [label for _, label in sorted(grouped.groups.keys())]
    scales = [scale for scale, _ in sorted(grouped.groups.keys())]
    values = [grouped.get_group((scale, label))["normalized_value"].mean() for scale, label in sorted(grouped.groups.keys())]
    axes[0].plot(scales, values, marker="o", color=OK["blue"], lw=1.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(scales)
    axes[0].set_xticklabels([str(s).rstrip("0").rstrip(".") + "x" for s in scales])
    axes[0].set_xlabel("Energy/latency scale vs. 4090")
    axes[0].set_ylabel("Heuristic value (normalized)")
    axes[0].set_ylim(0.55, 1.03)

    mix = heur.groupby("energy_scale")[["on_pct", "pre_pct", "down_pct"]].mean().sort_index()
    x = np.arange(len(mix))
    bottom = np.zeros(len(mix))
    for col, label, color in [("on_pct", "ON", OK["orange"]), ("pre_pct", "PRE", OK["green"]), ("down_pct", "DOWN", OK["sky"])]:
        axes[1].bar(x, mix[col], bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.3)
        bottom += mix[col].to_numpy()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{v:g}x" for v in mix.index])
    axes[1].set_xlabel("Energy/latency scale")
    axes[1].set_ylabel("Action mix (%)")
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(fig_dir / "hardware_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_tle_realism(tle: pd.DataFrame, contacts: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45))
    order = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD"]
    labels = ["Down", "On", "Thresh", "TVD"]
    models = ["fixed-window", "tle-derived"]
    width = 0.36
    x = np.arange(len(order))
    colors = {"fixed-window": OK["gray"], "tle-derived": OK["blue"]}
    for j, model in enumerate(models):
        vals = [tle[(tle["contact_model"] == model) & (tle["policy"] == p)]["normalized_value"].mean() for p in order]
        axes[0].bar(x + (j - 0.5) * width, vals, width=width, label=model, color=colors[model], edgecolor="black", linewidth=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel("Value (normalized)")
    axes[0].set_ylim(0.45, 1.05)
    axes[0].legend(fontsize=6, loc="lower right")

    if not contacts.empty:
        starts_h = contacts["start_s"] / 3600.0
        durations = contacts["duration_s"] / 60.0
        caps = contacts["capacity_mbit"] / 1000.0
        axes[1].scatter(starts_h, durations, s=np.clip(caps * 10, 10, 80), color=OK["green"], alpha=0.75, edgecolor="black", linewidth=0.3)
        axes[1].set_xlabel("Contact start (h)")
        axes[1].set_ylabel("Duration (min)")
        axes[1].set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(fig_dir / "tle_realism.png", bbox_inches="tight")
    plt.close(fig)


def write_tables(profile: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    lines = ["% Auto-generated by orbitllm.plot\n"]
    lines.append("\\newcommand{\\ProfileRows}{%\n")
    for row in profile_table_rows(profile, "INT4", 2048):
        lines.append(
            f"{row['model']} & {row['quant']} & {row['throughput']:.1f} & "
            f"{row['prefill_mj']:.1f}/{row['decode_mj']:.1f} & {row['peak_mem']:.1f}\\\\\n"
        )
    lines.append("}\n")
    main = metrics[metrics["variant"] == "main"].copy()
    max_by_seed = main.groupby("seed")["total_value"].transform("max")
    main["normalized_value"] = main["total_value"] / max_by_seed
    order = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD", "MILP-Bound"]
    lines.append("\\newcommand{\\ActionMixRows}{%\n")
    for policy in order:
        rows = main[main["policy"] == policy]
        if rows.empty or policy == "MILP-Bound":
            continue
        name = "\\textbf{Heuristic-TVD}" if policy == "Heuristic-TVD" else policy
        lines.append(
            f"{name} & {rows['on_pct'].mean():.1f} & {rows['pre_pct'].mean():.1f} & "
            f"{rows['down_pct'].mean():.1f} & {rows['drop_pct'].mean():.1f}\\\\\n"
        )
    lines.append("}\n")
    lines.append("\\newcommand{\\MainResultRows}{%\n")
    for policy in order:
        rows = main[main["policy"] == policy]
        if rows.empty:
            continue
        val = rows["normalized_value"].mean()
        val_ci = ci95(rows["normalized_value"])
        jpt = rows["j_per_task"].replace([np.inf, -np.inf], np.nan).mean()
        bw = rows["bw_saving"].mean()
        lat = rows["median_latency_min"].mean()
        bw_s = "--" if pd.isna(bw) else f"{bw:.1f}"
        lat_s = "--" if pd.isna(lat) else f"{lat:.1f}"
        jpt_s = "--" if pd.isna(jpt) else f"{jpt:.1f}"
        name = "\\textbf{Heuristic-TVD}" if policy == "Heuristic-TVD" else policy
        lines.append(f"{name} & {val:.2f}$\\pm${val_ci:.2f} & {jpt_s} & {bw_s} & {lat_s}\\\\\n")
    lines.append("}\n")
    out = out_dir / "generated_tables.tex"
    out.write_text("".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures and LaTeX tables.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    fig_dir = resolve_project_path(cfg["figures_dir"])
    out_dir = resolve_project_path(cfg["output_dir"])
    style()
    profile = load_profile(cfg["profile_path"])
    metrics = pd.read_csv(resolve_project_path(cfg["metrics_path"]))
    timeseries = pd.read_csv(resolve_project_path(cfg["timeseries_path"]))
    ablation = pd.read_csv(resolve_project_path(cfg["ablation_path"]))
    sensitivity = pd.read_csv(resolve_project_path(cfg["sensitivity_path"]))
    network_path = resolve_project_path(cfg.get("network_sensitivity_path", "results/network_sensitivity_metrics.csv"))
    pre_path = resolve_project_path(cfg.get("pre_sensitivity_path", "results/pre_sensitivity_metrics.csv"))
    hardware_path = resolve_project_path(cfg.get("hardware_sensitivity_path", "results/hardware_sensitivity_metrics.csv"))
    tle_path = resolve_project_path(cfg.get("tle_realism_path", "results/tle_realism_metrics.csv"))
    tle_contact_path = resolve_project_path(cfg.get("tle_contact_path", "results/tle_contact_trace.csv"))
    plot_architecture(fig_dir)
    plot_decision_flow(fig_dir)
    plot_energy(profile, fig_dir)
    plot_mem(profile, fig_dir)
    plot_value(timeseries, fig_dir)
    plot_ablation(ablation, fig_dir)
    plot_sensitivity(sensitivity, fig_dir)
    if network_path.exists():
        plot_network_sensitivity(pd.read_csv(network_path), fig_dir)
    if pre_path.exists():
        plot_pre_sensitivity(pd.read_csv(pre_path), fig_dir)
    if hardware_path.exists():
        plot_hardware_sensitivity(pd.read_csv(hardware_path), fig_dir)
    if tle_path.exists() and tle_contact_path.exists():
        plot_tle_realism(pd.read_csv(tle_path), pd.read_csv(tle_contact_path), fig_dir)
    write_tables(profile, metrics, out_dir)
    print(f"figures written to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
