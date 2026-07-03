from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def plot_decode_prior() -> None:
    path = RESULTS / "decode_prior_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    metric = "normalized_value"
    grouped = (
        df.groupby(["decode_sigma", "scheduler_info"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["ci95"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for label, sub in grouped.groupby("scheduler_info"):
        sub = sub.sort_values("decode_sigma")
        ax.errorbar(
            sub["decode_sigma"],
            sub["mean"],
            yerr=sub["ci95"],
            marker="o",
            linewidth=1.6,
            capsize=2.5,
            label=label,
        )
    ax.set_xlabel("Decode-length uncertainty sigma")
    ax.set_ylabel("Normalized value")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "decode_prior_sensitivity.png")


def plot_hardware_action_mix() -> None:
    path = RESULTS / "hardware_portability_action_mix.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["policy"].isin(["Heuristic-TVD", "Priority-Aware", "Deadline-Energy"])]
    df["label"] = df["hardware_profile"].str.replace("-measured", "", regex=False) + "\n" + df["policy"]
    cols = ["on_pct", "pre_pct", "down_pct", "drop_pct"]
    colors = ["#3b6ea8", "#6aa84f", "#e5a03c", "#999999"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for col, color in zip(cols, colors):
        ax.bar(x, df[col], bottom=bottom, label=col.replace("_pct", "").upper(), color=color)
        bottom += df[col].to_numpy()
    ax.set_ylabel("Action mix (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=25, ha="right", fontsize=7)
    ax.legend(frameon=False, ncols=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.2)
    _save(fig, "hardware_portability_action_mix.png")


def plot_guardband_heatmaps() -> None:
    path = RESULTS / "guardband_decode_cap_ablation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    for metric, name, label in [
        ("early_stop_rate", "guardband_early_stop_heatmap.png", "Early-stop rate"),
        ("truncation_value_loss_pct", "guardband_value_loss_heatmap.png", "Truncation value loss (%)"),
    ]:
        pivot = df.pivot_table(index="gamma", columns="decode_cap", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(4.3, 2.8))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(float(c))) if str(c).replace(".", "", 1).isdigit() else str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel("Decode cap")
        ax.set_ylabel("Memory guardband gamma")
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, name)


def plot_vlm_pre_eta() -> None:
    path = RESULTS / "vlm_pre_eta_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for selector, sub in df.groupby("selector"):
        sub = sub.sort_values("rho")
        ax.plot(sub["rho"], sub["eta"], marker="o", linewidth=1.8, label=selector)
    ax.set_xlabel("Retained patch fraction rho")
    ax.set_ylabel("Utility retention eta")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "vlm_pre_eta_curve.png")


def main() -> None:
    plot_decode_prior()
    plot_hardware_action_mix()
    plot_guardband_heatmaps()
    plot_vlm_pre_eta()


if __name__ == "__main__":
    main()
