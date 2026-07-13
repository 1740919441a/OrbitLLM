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
    "PBR-adapted",
    "PeakMem-inspired",
    "Heuristic-TVD",
]

LEGACY_POLICY_ORDER = [
    policy for policy in POLICY_ORDER if policy not in {"PBR-adapted", "PeakMem-inspired"}
]

ORIGIN = {
    "Deadline-Energy": "Rule-based reference",
    "Energy-WSRPT": "Index-based reference",
    "ActiveInf-inspired": "He'24, IEEE TMC",
    "LEO-DRL-inspired": "Tang'23, IEEE Satellite",
    "Hierarchical-RA-inspired": "Gao'24, IEEE IoT-J",
    "DRL-UA-inspired": "Zhang'24, IEEE TGCN",
    "QoS-Multihop-projected": "Zhao'24, IEEE IoT-J",
    "DelayCost-adapted": "Li'25, FGCS",
    "PBR-adapted": "Peng'25, JSA",
    "PeakMem-inspired": "Lee'26, JSA",
    "Heuristic-TVD": "This work (2026)",
}

DISPLAY_NAME = {
    "ActiveInf-inspired": "ActiveInf",
    "LEO-DRL-inspired": "LEO-DRL",
    "Hierarchical-RA-inspired": "Hierarchical-RA",
    "DRL-UA-inspired": "DRL-UA",
    "QoS-Multihop-projected": "QoS-Multihop",
    "DelayCost-adapted": "DelayCost",
    "PBR-adapted": "PBR",
    "PeakMem-inspired": "PeakMem",
}


def _summary(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for policy, group in df.groupby("policy"):
        row: dict[str, float | str] = {"policy": policy}
        for metric in metrics:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    order = {name: idx for idx, name in enumerate(POLICY_ORDER)}
    return pd.DataFrame(rows).sort_values("policy", key=lambda col: col.map(order).fillna(999))


def _paper_numerical_rows(df: pd.DataFrame, *, renormalize: bool = True) -> pd.DataFrame:
    """Exclude DPP adaptations from paper-facing numerical evidence."""
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


def _policy_name(policy: str) -> str:
    if policy == "Heuristic-TVD":
        return r"\textbf{Heuristic-TVD}"
    return DISPLAY_NAME.get(policy, policy)


def _fmt(mean: float, ci: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f}\\,$\\pm$\\,{ci:.{digits}f}"


def _maybe_bold(value: str, enabled: bool) -> str:
    return rf"\textbf{{{value}}}" if enabled else value


def _best_policies(summary: pd.DataFrame, metric: str, *, maximize: bool) -> set[str]:
    values = pd.to_numeric(summary[f"{metric}_mean"], errors="coerce")
    target = float(values.max() if maximize else values.min())
    mask = np.isclose(values.to_numpy(dtype=float), target, rtol=1e-9, atol=1e-12)
    return set(summary.loc[mask, "policy"].astype(str))


def _metric_row(
    summary: pd.DataFrame,
    policy: str,
    *,
    dataset: bool = False,
    best_by_metric: dict[str, set[str]] | None = None,
) -> str:
    best = best_by_metric or {}
    row = summary[summary["policy"] == policy].iloc[0]
    norm_value = _maybe_bold(
        _fmt(float(row["normalized_value_mean"]), float(row["normalized_value_ci"])),
        policy in best.get("normalized_value", set()),
    )
    total_value = _maybe_bold(
        _fmt(float(row["total_value_mean"]), float(row["total_value_ci"]), 2 if dataset else 1),
        policy in best.get("total_value", set()),
    )
    energy = _maybe_bold(
        f"{float(row['energy_j_mean']) / 1000.0:.{2 if dataset else 1}f}",
        policy in best.get("energy_j", set()),
    )
    latency_metric = "p95_latency_min" if dataset else "median_latency_min"
    latency = _maybe_bold(
        f"{float(row[f'{latency_metric}_mean']):.{1 if dataset else 2}f}",
        policy in best.get(latency_metric, set()),
    )
    decision = _maybe_bold(
        f"{float(row['mean_decision_ms_mean']) * 1000.0:.1f}",
        policy in best.get("mean_decision_ms", set()),
    )
    if dataset:
        return (
            f"{_policy_name(policy)} & {ORIGIN[policy]} & "
            f"{norm_value} & "
            f"{total_value} & "
            f"{energy} & {latency} & {decision}\\\\"
        )
    return (
        f"{_policy_name(policy)} & {ORIGIN[policy]} & "
        f"{norm_value} & "
        f"{total_value} & "
        f"{energy} & {latency} & {decision}\\\\"
    )


def _mechanism_rows() -> list[str]:
    # E=explicit, P=partial/related resource treatment, --=not modeled.
    rows = [
        ("Deadline-Energy", "--", "Reference policy", "E", "--", "--", "--", "--", "--", "--", "--", "--", "--"),
        ("Energy-WSRPT", "--", "Reference policy", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--"),
        ("Lyapunov DPP", "2023", "Applied Sciences", "E", "--", "--", "--", "--", "--", "P", "--", "--", "--"),
        ("Active Inference", "2024", "IEEE TMC", "--", "E", "--", "P", "P", "--", "P", "--", "--", "--"),
        ("Tang DRL", "2023", "IEEE Satellite", "E", "--", "--", "--", "--", "--", "P", "--", "--", "--"),
        ("Hierarchical RA", "2024", "IEEE IoT-J", "E", "--", "--", "--", "--", "--", "P", "--", "--", "E"),
        ("DRL-UA", "2024", "IEEE TGCN", "E", "--", "--", "--", "--", "--", "P", "--", "--", "P"),
        ("QoS-Multihop", "2024", "IEEE IoT-J", "E", "--", "--", "--", "--", "--", "E", "--", "--", "E"),
        ("Delay-Cost", "2025", "FGCS", "E", "--", "--", "--", "--", "--", "P", "--", "--", "--"),
        ("PBR", "2025", "JSA", "E", "--", "--", "--", "--", "--", "--", "--", "--", "--"),
        ("EH offloading", "2025", "JSA", "--", "--", "--", "--", "--", "--", "--", "--", "E", "--"),
        ("Peak-memory sched.", "2026", "JSA", "--", "--", "--", "E", "--", "--", "--", "P", "--", "--"),
        (r"\textbf{TVD (this work)}", "2026", "This work", "E", "E", "E", "E", "E", "E", "E", "E", "E", "--"),
    ]
    return [" & ".join(row) + r"\\" for row in rows]


def _write_encoded_pre_figure(results: Path) -> None:
    detail = pd.read_csv(results / "mobilesam_pre_encoded_per_image.csv")
    detail = detail[
        (detail["backend"] == "mobile_sam")
        & (detail["selector"].isin(["sam_top1", "sam_top3", "sam_top5", "sam_all_filtered"]))
    ].copy()
    summary = (
        detail.groupby("selector", as_index=False)
        .agg(
            rho_bytes=("rho_bytes_vs_source", "mean"),
            eta_prob=("eta_prob", "mean"),
            eta_ci=("eta_prob", lambda values: 1.96 * values.std(ddof=1) / np.sqrt(len(values))),
        )
    )
    order = ["sam_top1", "sam_top3", "sam_top5", "sam_all_filtered"]
    summary["selector"] = pd.Categorical(summary["selector"], categories=order, ordered=True)
    summary = summary.sort_values("selector")
    labels = {"sam_top1": "top-1", "sam_top3": "top-3", "sam_top5": "top-5", "sam_all_filtered": "all"}
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ax.errorbar(
        summary["rho_bytes"],
        summary["eta_prob"],
        yerr=summary["eta_ci"],
        color="#176B87",
        marker="o",
        linewidth=1.6,
        capsize=2.5,
    )
    for row in summary.itertuples(index=False):
        ax.annotate(
            labels[str(row.selector)],
            (float(row.rho_bytes), float(row.eta_prob)),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Actual PRE JPEG bytes / source JPEG bytes")
    ax.set_ylabel("Per-image utility retention")
    ax.set_xlim(0.10, 0.46)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=0.4)
    output = resolve_project_path("figures/eurosat_pre_encoded_utility.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    results = resolve_project_path("results")
    main = _paper_numerical_rows(pd.read_csv(results / "jsa_recent_main_metrics.csv"))
    oracle_base = _paper_numerical_rows(
        pd.read_csv(results / "sota_baseline_expanded_oracle_metrics.csv"), renormalize=False
    )
    oracle_recent = pd.read_csv(results / "jsa_recent_oracle_metrics.csv")
    oracle = pd.concat(
        [
            oracle_base[~oracle_base["policy"].isin({"PBR-adapted", "PeakMem-inspired"})],
            oracle_recent[oracle_recent["policy"].isin({"PBR-adapted", "PeakMem-inspired"})],
        ],
        ignore_index=True,
    )
    physical = _paper_numerical_rows(
        pd.read_csv(results / "sota_baseline_expanded_physical_link_metrics.csv")
    )
    eurosat = _paper_numerical_rows(pd.read_csv(results / "eurosat_baseline_main_metrics.csv"))

    main_summary = _summary(main, ["normalized_value", "total_value", "energy_j", "median_latency_min", "mean_decision_ms"])
    oracle_summary = _summary(oracle, ["optimality_gap_pct", "mean_decision_ms"])
    physical_summary = _summary(physical, ["normalized_value", "total_value", "energy_j", "drop_pct", "mean_decision_ms"])
    eurosat_summary = _summary(eurosat, ["normalized_value", "total_value", "energy_j", "p95_latency_min", "mean_decision_ms"])
    main_summary.to_csv(results / "sota_baseline_expanded_main_summary.csv", index=False)
    oracle_summary.to_csv(results / "sota_baseline_expanded_oracle_summary.csv", index=False)
    physical_summary.to_csv(results / "sota_baseline_expanded_physical_summary.csv", index=False)
    eurosat_summary.to_csv(results / "eurosat_baseline_main_summary.csv", index=False)

    main_best = {
        "normalized_value": _best_policies(main_summary, "normalized_value", maximize=True),
        "total_value": _best_policies(main_summary, "total_value", maximize=True),
        "energy_j": _best_policies(main_summary, "energy_j", maximize=False),
        "median_latency_min": _best_policies(main_summary, "median_latency_min", maximize=False),
        "mean_decision_ms": _best_policies(main_summary, "mean_decision_ms", maximize=False),
    }
    physical_best = {
        "normalized_value": _best_policies(physical_summary, "normalized_value", maximize=True),
        "energy_j": _best_policies(physical_summary, "energy_j", maximize=False),
        "drop_pct": _best_policies(physical_summary, "drop_pct", maximize=False),
        "mean_decision_ms": _best_policies(physical_summary, "mean_decision_ms", maximize=False),
    }
    eurosat_best = {
        "normalized_value": _best_policies(eurosat_summary, "normalized_value", maximize=True),
        "total_value": _best_policies(eurosat_summary, "total_value", maximize=True),
        "energy_j": _best_policies(eurosat_summary, "energy_j", maximize=False),
        "p95_latency_min": _best_policies(eurosat_summary, "p95_latency_min", maximize=False),
        "mean_decision_ms": _best_policies(eurosat_summary, "mean_decision_ms", maximize=False),
    }

    lines = [
        r"\newcommand{\SotaExpandedMainRows}{",
        *[_metric_row(main_summary, policy, best_by_metric=main_best) for policy in POLICY_ORDER],
        r"}",
        r"\newcommand{\SotaExpandedOracleRows}{",
    ]
    oracle_best = {
        "optimality_gap_pct": _best_policies(oracle_summary, "optimality_gap_pct", maximize=False),
        "mean_decision_ms": _best_policies(oracle_summary, "mean_decision_ms", maximize=False),
    }
    for policy in POLICY_ORDER:
        row = oracle_summary[oracle_summary["policy"] == policy].iloc[0]
        gap = _maybe_bold(
            _fmt(float(row["optimality_gap_pct_mean"]), float(row["optimality_gap_pct_ci"]), 2),
            policy in oracle_best["optimality_gap_pct"],
        )
        decision = _maybe_bold(
            f"{float(row['mean_decision_ms_mean']) * 1000.0:.1f}",
            policy in oracle_best["mean_decision_ms"],
        )
        lines.append(
            f"{_policy_name(policy)} & {ORIGIN[policy]} & "
            f"{gap} & {decision}\\\\"
        )
    lines.extend([r"}", r"\newcommand{\SotaExpandedPhysicalRows}{"])
    for policy in LEGACY_POLICY_ORDER:
        row = physical_summary[physical_summary["policy"] == policy].iloc[0]
        value = _maybe_bold(
            _fmt(float(row["normalized_value_mean"]), float(row["normalized_value_ci"])),
            policy in physical_best["normalized_value"],
        )
        energy = _maybe_bold(
            f"{float(row['energy_j_mean']) / 1000.0:.1f}",
            policy in physical_best["energy_j"],
        )
        drop = _maybe_bold(
            f"{float(row['drop_pct_mean']):.1f}",
            policy in physical_best["drop_pct"],
        )
        decision = _maybe_bold(
            f"{float(row['mean_decision_ms_mean']) * 1000.0:.1f}",
            policy in physical_best["mean_decision_ms"],
        )
        lines.append(
            f"{_policy_name(policy)} & {ORIGIN[policy]} & "
            f"{value} & {energy} & {drop} & {decision}\\\\"
        )
    lines.extend([r"}", r"\newcommand{\EuroSATMainRows}{"])
    lines.extend(
        _metric_row(eurosat_summary, policy, dataset=True, best_by_metric=eurosat_best)
        for policy in LEGACY_POLICY_ORDER
    )
    lines.extend([r"}", r"\newcommand{\SotaMechanismRows}{", *_mechanism_rows(), r"}"])
    (results / "sota_expanded_generated_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_encoded_pre_figure(results)
    print("wrote expanded SOTA summaries and LaTeX rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
