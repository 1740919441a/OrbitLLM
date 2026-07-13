from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_project_path
from .profile import load_profile
from .simulate import (
    SOTA_BASELINE_POLICIES,
    DownlinkScheduler,
    _simulate_action_sequence,
    generate_tasks,
    generate_windows,
    run_policy,
    run_sota_baseline_suite,
    task_costs,
)


JSA_MAIN_ADDITIONS = ["PBR-adapted", "PeakMem-inspired"]


def _normalize(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["normalized_value"] = out.groupby(groups)["total_value"].transform(
        lambda values: values / max(float(values.max()), 1e-9)
    )
    return out


def _summary(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = [
        "normalized_value",
        "total_value",
        "energy_j",
        "median_latency_min",
        "p95_latency_min",
        "mean_decision_ms",
        "oom_count",
        "energy_violation_count",
        "on_pct",
        "pre_pct",
        "down_pct",
        "drop_pct",
        "min_energy_j",
    ]
    rows: list[dict[str, Any]] = []
    for key, part in df.groupby(groups, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key_tuple))
        n = max(len(part), 1)
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95"] = (
                1.96 * float(values.std(ddof=1)) / np.sqrt(n) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_main(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    base = run_sota_baseline_suite(cfg, profile)
    rows: list[dict[str, Any]] = base.to_dict("records")
    for seed in [int(value) for value in cfg["seeds"]]:
        tasks = generate_tasks(cfg, seed)
        windows = generate_windows(cfg)
        for policy in JSA_MAIN_ADDITIONS:
            metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
            metrics.update(
                {
                    "seed": seed,
                    "horizon_h": cfg["horizon_h"],
                    "task_count": len(tasks),
                    "allow_pre": 1,
                    "contact_model": "fixed-window",
                    "training_seed_count": 0,
                    "training_q_states": 0,
                    "ua_training_seed_count": 0,
                    "ua_training_q_states": 0,
                }
            )
            rows.append(metrics)
    return _normalize(pd.DataFrame(rows), ["seed", "contact_model", "allow_pre"])


def run_soc(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    policies = ["Deadline-Energy", "EH-DORA-adapted", "Adaptive-TVD", "Heuristic-TVD"]
    rows: list[dict[str, Any]] = []
    for charge_w in [2.0, 5.0, 10.0]:
        cfg_s = copy.deepcopy(cfg)
        cfg_s["soc"]["initial_soc_fraction"] = 0.45
        cfg_s["soc"]["charge_power_w"] = charge_w
        for seed in [int(value) for value in cfg_s["seeds"]]:
            tasks = generate_tasks(cfg_s, seed)
            windows = generate_windows(cfg_s)
            for policy in policies:
                metrics, _ = run_policy(
                    policy, tasks, windows, profile, cfg_s, energy_model="soc"
                )
                metrics.update({"seed": seed, "charge_power_w": charge_w})
                rows.append(metrics)
    return _normalize(pd.DataFrame(rows), ["seed", "charge_power_w"])


def run_memory(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    policies = ["ActiveInf-inspired", "PeakMem-inspired", "PBR-adapted", "Heuristic-TVD"]
    rows: list[dict[str, Any]] = []
    for memory_gb in [6.0, 8.0, 12.0, 16.0]:
        cfg_s = copy.deepcopy(cfg)
        cfg_s["satellite"]["memory_gb"] = memory_gb
        for seed in [int(value) for value in cfg_s["seeds"]]:
            tasks = generate_tasks(cfg_s, seed)
            windows = generate_windows(cfg_s)
            for policy in policies:
                metrics, _ = run_policy(policy, tasks, windows, profile, cfg_s)
                metrics.update({"seed": seed, "memory_gb": memory_gb})
                rows.append(metrics)
    return _normalize(pd.DataFrame(rows), ["seed", "memory_gb"])


def run_oracle(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    policies = [
        "QoS-Multihop-projected",
        "PBR-adapted",
        "PeakMem-inspired",
        "Heuristic-TVD",
    ]
    cfg_s = copy.deepcopy(cfg)
    cfg_s["horizon_h"] = 3.0
    rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in cfg_s["seeds"]]:
        tasks = generate_tasks(cfg_s, seed, horizon_h=3.0, limit=8)
        windows = generate_windows(cfg_s, horizon_h=3.0)
        scheduler = DownlinkScheduler(windows, 3.0 * 3600.0)
        costs = [task_costs(task, profile, cfg_s) for task in tasks]
        oracle_value = max(
            _simulate_action_sequence(
                actions,
                tasks,
                windows,
                profile,
                cfg_s,
                scheduler,
                0.0,
                float(cfg_s["satellite"]["energy_budget_j"]),
                costs,
            )
            for actions in itertools.product(("ON", "PRE", "DOWN"), repeat=len(tasks))
        )
        for policy in policies:
            metrics, _ = run_policy(policy, tasks, windows, profile, cfg_s)
            value = float(metrics["total_value"])
            rows.append(
                {
                    "seed": seed,
                    "policy": policy,
                    "task_count": len(tasks),
                    "oracle_value": oracle_value,
                    "policy_value": value,
                    "optimality_gap_pct": 100.0
                    * max(0.0, oracle_value - value)
                    / max(oracle_value, 1e-9),
                    "mean_decision_ms": metrics["mean_decision_ms"],
                    "oom_count": metrics["oom_count"],
                    "energy_violation_count": metrics["energy_violation_count"],
                }
            )
    return pd.DataFrame(rows)


def _plot(main: pd.DataFrame, soc: pd.DataFrame, memory: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))

    main_order = [
        "DRL-UA-inspired",
        "QoS-Multihop-projected",
        "PBR-adapted",
        "PeakMem-inspired",
        "Heuristic-TVD",
    ]
    main_s = main.groupby("policy")["normalized_value"].mean().reindex(main_order).dropna()
    axes[0].bar(np.arange(len(main_s)), 100.0 * main_s.values, color="#3973A8")
    axes[0].set_xticks(np.arange(len(main_s)))
    axes[0].set_xticklabels(main_s.index, rotation=28, ha="right")
    axes[0].set_ylabel("Normalized value (%)")
    axes[0].set_title("Recent JSA-oriented baselines")

    for policy in ["Deadline-Energy", "EH-DORA-adapted", "Adaptive-TVD", "Heuristic-TVD"]:
        part = soc[soc["policy"] == policy].groupby("charge_power_w")["normalized_value"].mean()
        axes[1].plot(part.index, 100.0 * part.values, marker="o", label=policy)
    axes[1].set_xlabel("Harvest power (W)")
    axes[1].set_ylabel("Normalized value (%)")
    axes[1].set_title("Energy-harvesting dynamics")
    axes[1].legend(fontsize=7)

    for policy in ["ActiveInf-inspired", "PeakMem-inspired", "PBR-adapted", "Heuristic-TVD"]:
        part = memory[memory["policy"] == policy].groupby("memory_gb")["normalized_value"].mean()
        axes[2].plot(part.index, 100.0 * part.values, marker="o", label=policy)
    axes[2].set_xlabel("Memory ceiling (GB)")
    axes[2].set_ylabel("Normalized value (%)")
    axes[2].set_title("Peak-memory sensitivity")
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(numeric.mean())
    ci = 1.96 * float(numeric.std(ddof=1)) / np.sqrt(len(numeric)) if len(numeric) > 1 else 0.0
    return mean, ci


def write_latex(soc: pd.DataFrame, memory: pd.DataFrame, output: Path) -> None:
    lines = ["% Auto-generated by orbitllm.jsa_recent_baselines."]
    lines.append(r"\newcommand{\JsaRecentSocRows}{%")
    for charge_w in sorted(soc["charge_power_w"].unique()):
        part = soc[soc["charge_power_w"] == charge_w]
        cells = []
        values: dict[str, tuple[float, float]] = {}
        for policy in ["EH-DORA-adapted", "Heuristic-TVD", "Adaptive-TVD"]:
            mean, ci = _mean_ci(part[part["policy"] == policy]["normalized_value"])
            values[policy] = (mean, ci)
        best = max(mean for mean, _ in values.values())
        for policy in ["EH-DORA-adapted", "Heuristic-TVD", "Adaptive-TVD"]:
            mean, ci = values[policy]
            cell = f"{100.0 * mean:.2f}$\\pm${100.0 * ci:.2f}"
            cells.append(rf"\textbf{{{cell}}}" if np.isclose(mean, best, rtol=1e-9, atol=1e-12) else cell)
        lines.append(f"{charge_w:.0f} & " + " & ".join(cells) + r"\\")
    lines.append("}")
    lines.append(r"\newcommand{\JsaRecentMemoryRows}{%")
    for memory_gb in sorted(memory["memory_gb"].unique()):
        part = memory[memory["memory_gb"] == memory_gb]
        cells = []
        values: dict[str, float] = {}
        for policy in ["ActiveInf-inspired", "PBR-adapted", "PeakMem-inspired", "Heuristic-TVD"]:
            mean, _ = _mean_ci(part[part["policy"] == policy]["normalized_value"])
            values[policy] = mean
        best = max(values.values())
        for policy in ["ActiveInf-inspired", "PBR-adapted", "PeakMem-inspired", "Heuristic-TVD"]:
            cell = f"{100.0 * values[policy]:.1f}"
            cells.append(rf"\textbf{{{cell}}}" if np.isclose(values[policy], best) else cell)
        lines.append(f"{memory_gb:.0f} & " + " & ".join(cells) + r"\\")
    lines.append("}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    profile = load_profile(args.profile or cfg["profile_path"])
    main = run_main(cfg, profile)
    soc = run_soc(cfg, profile)
    memory = run_memory(cfg, profile)
    oracle = run_oracle(cfg, profile)

    outputs = {
        "main": resolve_project_path(args.main_output),
        "soc": resolve_project_path(args.soc_output),
        "memory": resolve_project_path(args.memory_output),
    }
    for key, frame in [("main", main), ("soc", soc), ("memory", memory)]:
        outputs[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(outputs[key], index=False)

    _summary(main, ["policy"]).to_csv(resolve_project_path(args.main_summary), index=False)
    _summary(soc, ["charge_power_w", "policy"]).to_csv(
        resolve_project_path(args.soc_summary), index=False
    )
    _summary(memory, ["memory_gb", "policy"]).to_csv(
        resolve_project_path(args.memory_summary), index=False
    )
    oracle_path = resolve_project_path(args.oracle_output)
    oracle_path.parent.mkdir(parents=True, exist_ok=True)
    oracle.to_csv(oracle_path, index=False)
    _summary(
        oracle.rename(
            columns={
                "policy_value": "total_value",
                "optimality_gap_pct": "normalized_value",
            }
        ).assign(
            energy_j=np.nan,
            median_latency_min=np.nan,
            p95_latency_min=np.nan,
            on_pct=np.nan,
            pre_pct=np.nan,
            down_pct=np.nan,
            drop_pct=np.nan,
            min_energy_j=np.nan,
        ),
        ["policy"],
    ).rename(
        columns={
            "normalized_value_mean": "optimality_gap_pct_mean",
            "normalized_value_ci95": "optimality_gap_pct_ci95",
        }
    ).to_csv(resolve_project_path(args.oracle_summary), index=False)
    _plot(main, soc, memory, resolve_project_path(args.figure_output))
    write_latex(soc, memory, resolve_project_path(args.latex_output))
    print(
        f"wrote main={len(main)} soc={len(soc)} memory={len(memory)} oracle={len(oracle)} rows",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run recent JSA baseline adaptations.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--main-output", default="results/jsa_recent_main_metrics.csv")
    parser.add_argument("--main-summary", default="results/jsa_recent_main_summary.csv")
    parser.add_argument("--soc-output", default="results/jsa_recent_soc_metrics.csv")
    parser.add_argument("--soc-summary", default="results/jsa_recent_soc_summary.csv")
    parser.add_argument("--memory-output", default="results/jsa_recent_memory_metrics.csv")
    parser.add_argument("--memory-summary", default="results/jsa_recent_memory_summary.csv")
    parser.add_argument("--oracle-output", default="results/jsa_recent_oracle_metrics.csv")
    parser.add_argument("--oracle-summary", default="results/jsa_recent_oracle_summary.csv")
    parser.add_argument("--figure-output", default="figures/jsa_recent_baselines.png")
    parser.add_argument("--latex-output", default="results/jsa_recent_generated_tables.tex")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
