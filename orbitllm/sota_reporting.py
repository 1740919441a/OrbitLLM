from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import resolve_project_path


POLICY_ORDER = [
    "Deadline-Energy",
    "Energy-WSRPT",
    "ActiveInf-inspired",
    "LEO-DRL-inspired",
    "Hierarchical-RA-inspired",
    "DRL-UA-inspired",
    "QoS-Multihop-projected",
    "DelayCost-adapted",
    "Heuristic-TVD",
]

COLORS = {
    "Deadline-Energy": "#4C78A8",
    "Energy-WSRPT": "#72B7B2",
    "ActiveInf-inspired": "#F58518",
    "LEO-DRL-inspired": "#B279A2",
    "Hierarchical-RA-inspired": "#8C6D31",
    "DRL-UA-inspired": "#E17C05",
    "QoS-Multihop-projected": "#59A14F",
    "DelayCost-adapted": "#AF7AA1",
    "Heuristic-TVD": "#2E7D32",
}

DISPLAY_NAMES = {
    "ActiveInf-inspired": "ActiveInf",
    "LEO-DRL-inspired": "LEO-DRL",
    "Hierarchical-RA-inspired": "Hierarchical-RA",
    "DRL-UA-inspired": "DRL-UA",
    "QoS-Multihop-projected": "QoS-Multihop",
    "DelayCost-adapted": "DelayCost",
}


def _display_name(policy: str) -> str:
    return DISPLAY_NAMES.get(policy, policy)

ORIGINS = {
    "Deadline-Energy": "Internal",
    "Energy-WSRPT": "Internal",
    "ActiveInf-inspired": "He'24, IEEE TMC (inspired)",
    "LEO-DRL-inspired": "Tang'23, IEEE Satellite (inspired)",
    "Hierarchical-RA-inspired": "Gao'24, IEEE IoT-J (inspired)",
    "DRL-UA-inspired": "Zhang'24, IEEE TGCN (inspired)",
    "QoS-Multihop-projected": "Zhao'24, IEEE IoT-J (projection)",
    "DelayCost-adapted": "Li'25, FGCS (adapted)",
    "Heuristic-TVD": "This work (2026)",
}

MECHANISM_ROWS = [
    ("Deadline-Energy", "2026", "Internal", "E", "--", "--", "--", "--", "--", "--", "--"),
    ("Energy-WSRPT", "2026", "Internal", "E", "--", "--", "--", "--", "--", "--", "--"),
    ("Lyapunov DPP (source)", "2023", "Applied Sciences", "E", "--", "--", "--", "--", "P", "--", "P"),
    ("ActiveInf-inspired", "2024", "IEEE TMC", "--", "E", "P", "P", "--", "P", "P", "--"),
    ("LEO-DRL-inspired", "2023", "IEEE Satellite", "E", "--", "--", "--", "--", "P", "--", "P"),
    ("Hierarchical-RA-inspired", "2024", "IEEE IoT-J", "E", "--", "--", "--", "--", "P", "--", "E"),
    ("DRL-UA-inspired", "2024", "IEEE TGCN", "E", "--", "--", "--", "--", "P", "--", "P"),
    ("QoS-Multihop-projected", "2024", "IEEE IoT-J", "E", "--", "--", "--", "--", "E", "--", "E"),
    ("DelayCost-adapted", "2025", "FGCS", "E", "--", "--", "--", "--", "P", "--", "--"),
    (r"\textbf{Heuristic-TVD (this work)}", "2026", "This work", "E", "E", "E", "E", "E", "E", "E", "--"),
]


def _summary(df: pd.DataFrame, metrics: list[str], group: str = "policy") -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for label, part in df.groupby(group):
        row: dict[str, float | str | int] = {group: label, "n": len(part)}
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            mean = float(values.mean()) if not values.empty else float("nan")
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = 1.96 * std / np.sqrt(max(len(values), 1))
        rows.append(row)
    order = {name: idx for idx, name in enumerate(POLICY_ORDER)}
    return pd.DataFrame(rows).sort_values(group, key=lambda col: col.map(order).fillna(999))


def _paper_numerical_rows(df: pd.DataFrame, *, renormalize: bool = True) -> pd.DataFrame:
    result = df[df["policy"] != "DPP-LLM"].copy()
    if renormalize and "total_value" in result.columns:
        group_cols = [
            column
            for column in ("seed", "scenario", "robustness_variant", "contact_model")
            if column in result.columns
        ]
        if group_cols:
            result["normalized_value"] = result.groupby(group_cols)["total_value"].transform(
                lambda values: values / max(float(values.max()), 1e-9)
            )
    return result


def _latex_name(policy: str) -> str:
    return r"\textbf{Heuristic-TVD}" if policy == "Heuristic-TVD" else policy


def _write_latex(
    main: pd.DataFrame,
    oracle: pd.DataFrame,
    physical: pd.DataFrame,
    eurosat: pd.DataFrame | None,
    output: Path,
) -> None:
    lines = [r"\newcommand{\SotaMainRows}{"]
    for _, row in main.iterrows():
        lines.append(
            f"{_latex_name(str(row['policy']))} & "
            f"{ORIGINS.get(str(row['policy']), '--')} & "
            f"{row['normalized_value_mean']:.3f}\\,$\\pm$\\,{row['normalized_value_ci95']:.3f} & "
            f"{row['total_value_mean']:.1f}\\,$\\pm$\\,{row['total_value_ci95']:.1f} & "
            f"{row['energy_j_mean'] / 1000.0:.1f} & "
            f"{row['median_latency_min_mean']:.2f} & "
            f"{row['mean_decision_ms_mean'] * 1000.0:.1f}\\\\"
        )
    lines.append("}")
    lines.append(r"\newcommand{\SotaMechanismRows}{")
    for row in MECHANISM_ROWS:
        lines.append(" & ".join(row) + r"\\")
    lines.append("}")
    if eurosat is not None:
        lines.append(r"\newcommand{\SotaEuroSATRows}{")
        for _, row in eurosat.iterrows():
            lines.append(
                f"{_latex_name(str(row['policy']))} & "
                f"{row['normalized_value_mean']:.3f}\\,$\\pm$\\,{row['normalized_value_ci95']:.3f} & "
                f"{row['total_value_mean']:.1f}\\,$\\pm$\\,{row['total_value_ci95']:.1f} & "
                f"{row['energy_j_mean'] / 1000.0:.1f} & "
                f"{row['on_pct_mean']:.1f}/{row['pre_pct_mean']:.1f}/{row['down_pct_mean']:.1f}\\\\"
            )
        lines.append("}")
    lines.append(r"\newcommand{\SotaOracleRows}{")
    for _, row in oracle.iterrows():
        lines.append(
            f"{_latex_name(str(row['policy']))} & "
            f"{row['optimality_gap_pct_mean']:.2f}\\,$\\pm$\\,{row['optimality_gap_pct_ci95']:.2f} & "
            f"{row['mean_decision_ms_mean'] * 1000.0:.1f}\\\\"
        )
    lines.append("}")
    lines.append(r"\newcommand{\SotaPhysicalRows}{")
    for _, row in physical.iterrows():
        lines.append(
            f"{_latex_name(str(row['policy']))} & "
            f"{row['normalized_value_mean']:.3f}\\,$\\pm$\\,{row['normalized_value_ci95']:.3f} & "
            f"{row['energy_j_mean'] / 1000.0:.1f} & "
            f"{row['drop_pct_mean']:.1f} & "
            f"{row['mean_decision_ms_mean'] * 1000.0:.1f}\\\\"
        )
    lines.append("}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_value_runtime(main: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for _, row in main.iterrows():
        policy = str(row["policy"])
        ax.errorbar(
            row["mean_decision_ms_mean"],
            row["normalized_value_mean"],
            yerr=row["normalized_value_ci95"],
            marker="o",
            ms=6,
            capsize=2.5,
            color=COLORS.get(policy, "#666666"),
            label=_display_name(policy),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Mean decision time (ms, log scale)")
    ax.set_ylabel("Normalized realized value")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_cumulative(events: pd.DataFrame, output: Path) -> None:
    grid = np.linspace(0.0, 24.0, 97)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for policy in POLICY_ORDER:
        policy_rows = events[events["policy"] == policy]
        traces = []
        for _, seed_rows in policy_rows.groupby("seed"):
            seed_rows = seed_rows.sort_values("arrival_h")
            x = seed_rows["arrival_h"].to_numpy()
            y = seed_rows["cum_value"].to_numpy()
            idx = np.searchsorted(x, grid, side="right") - 1
            traces.append(np.where(idx >= 0, y[np.maximum(idx, 0)], 0.0))
        if not traces:
            continue
        values = np.vstack(traces)
        mean = values.mean(axis=0)
        ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
        ax.plot(
            grid,
            mean,
            color=COLORS[policy],
            lw=2.0 if policy == "Heuristic-TVD" else 1.35,
            label=_display_name(policy),
        )
        ax.fill_between(grid, mean - ci, mean + ci, color=COLORS[policy], alpha=0.10)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Cumulative realized value")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=5.8, ncols=3)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_robustness(main_df: pd.DataFrame, robust_df: pd.DataFrame, physical_df: pd.DataFrame, output: Path) -> None:
    frames = [main_df.assign(scenario="fixed")]
    scenario_col = "scenario" if "scenario" in robust_df.columns else "robustness_variant"
    frames.extend(part.assign(scenario=scenario) for scenario, part in robust_df.groupby(scenario_col))
    frames.append(physical_df.assign(scenario="physical-RF"))
    combined = pd.concat(frames, ignore_index=True)
    grouped = combined.groupby(["scenario", "policy"], as_index=False)["normalized_value"].mean()
    scenarios = ["fixed", "no-pre", "tle", "physical-RF"]
    width = 0.085
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for idx, policy in enumerate(POLICY_ORDER):
        vals = []
        for scenario in scenarios:
            match = grouped[(grouped["scenario"] == scenario) & (grouped["policy"] == policy)]
            vals.append(float(match["normalized_value"].iloc[0]) if not match.empty else np.nan)
        ax.bar(
            x + (idx - (len(POLICY_ORDER) - 1) / 2) * width,
            vals,
            width=width,
            color=COLORS[policy],
            label=_display_name(policy),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Fixed", "No PRE", "TLE", "Physical RF"])
    ax.set_ylim(0.6, 1.03)
    ax.set_ylabel("Normalized realized value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=5.8, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.34))
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> int:
    results = resolve_project_path("results")
    figures = resolve_project_path("figures")
    figures.mkdir(parents=True, exist_ok=True)

    def select(name: str) -> Path:
        expanded = results / f"sota_baseline_expanded_{name}.csv"
        return expanded if expanded.exists() else results / f"sota_baseline_{name}.csv"

    main_df = _paper_numerical_rows(pd.read_csv(select("main_metrics")))
    events = _paper_numerical_rows(pd.read_csv(select("timeseries")), renormalize=False)
    robust_df = _paper_numerical_rows(pd.read_csv(select("robustness_metrics")))
    oracle_df = _paper_numerical_rows(pd.read_csv(select("oracle_metrics")), renormalize=False)
    physical_df = _paper_numerical_rows(pd.read_csv(select("physical_link_metrics")))
    eurosat_path = results / "eurosat_baseline_main_metrics.csv"
    eurosat_df = _paper_numerical_rows(pd.read_csv(eurosat_path)) if eurosat_path.exists() else None

    main_summary = _summary(
        main_df,
        ["normalized_value", "total_value", "energy_j", "median_latency_min", "mean_decision_ms"],
    )
    oracle_summary = _summary(oracle_df, ["optimality_gap_pct", "mean_decision_ms"])
    physical_summary = _summary(
        physical_df,
        ["normalized_value", "total_value", "energy_j", "drop_pct", "mean_decision_ms"],
    )
    eurosat_summary = (
        _summary(
            eurosat_df,
            ["normalized_value", "total_value", "energy_j", "on_pct", "pre_pct", "down_pct"],
        )
        if eurosat_df is not None
        else None
    )
    main_summary.to_csv(results / "sota_baseline_main_summary.csv", index=False)
    oracle_summary.to_csv(results / "sota_baseline_oracle_summary.csv", index=False)
    physical_summary.to_csv(results / "sota_baseline_physical_link_ci_summary.csv", index=False)
    if eurosat_summary is not None:
        eurosat_summary.to_csv(results / "eurosat_baseline_main_summary.csv", index=False)
    _write_latex(main_summary, oracle_summary, physical_summary, eurosat_summary, results / "sota_generated_tables.tex")
    _plot_value_runtime(main_summary, figures / "sota_baseline_value_runtime.png")
    _plot_cumulative(events, figures / "sota_baseline_cumulative_value.png")
    _plot_robustness(main_df, robust_df, physical_df, figures / "sota_baseline_robustness.png")
    print("wrote SOTA summaries, LaTeX rows, and figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
