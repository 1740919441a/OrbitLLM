from __future__ import annotations

import argparse
import copy
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, load_config, resolve_project_path
from .profile import load_profile, write_default_profile
from .simulate import (
    DownlinkScheduler,
    Window,
    generate_tasks,
    generate_windows,
    run_policy,
    task_costs,
    value_at,
)


def _load_cfg_profile(config: str | None, profile_path: str | None) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = load_config(config)
    ensure_output_dirs(cfg)
    profile_ref = profile_path or cfg["profile_path"]
    if not resolve_project_path(profile_ref).exists():
        write_default_profile(profile_ref)
    return cfg, load_profile(profile_ref)


def _write_incremental(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = resolve_project_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def run_epsilon(args: argparse.Namespace) -> None:
    cfg, profile = _load_cfg_profile(args.config, args.profile)
    eps_values = [float(v) for v in args.epsilons.split(",")]
    rows: list[dict[str, Any]] = []
    for epsilon in eps_values:
        for seed in cfg["seeds"]:
            cfg_s = copy.deepcopy(cfg)
            cfg_s.setdefault("heuristic", {})
            cfg_s["heuristic"]["epsilon"] = epsilon
            tasks = generate_tasks(cfg_s, int(seed))
            windows = generate_windows(cfg_s)
            t0 = time.perf_counter()
            metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
            metrics.update({"seed": seed, "epsilon": epsilon, "runtime_s": time.perf_counter() - t0})
            rows.append(metrics)
            _write_incremental(rows, args.output)
            print(f"epsilon={epsilon:g} seed={seed} value={metrics['total_value']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby("seed")["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    df.to_csv(resolve_project_path(args.output), index=False)


def _fragmented_windows(horizon_h: float, count: int, duration_s: float, rate_mbps: float) -> list[Window]:
    horizon_s = horizon_h * 3600.0
    if count <= 0:
        return []
    spacing = max(duration_s + 1.0, horizon_s / count)
    windows: list[Window] = []
    for idx in range(count):
        start = min(idx * spacing, max(0.0, horizon_s - duration_s))
        rate = rate_mbps * 1e6 * (0.75 + 0.25 * (1.0 + np.sin(idx * 1.618)) / 2.0)
        windows.append(Window(float(start), float(min(start + duration_s, horizon_s)), float(rate)))
    return windows


def run_overhead(args: argparse.Namespace) -> None:
    cfg, profile = _load_cfg_profile(args.config, args.profile)
    window_counts = [int(v) for v in args.window_counts.split(",")]
    task_counts = [int(v) for v in args.task_counts.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for window_count in window_counts:
            windows = _fragmented_windows(args.horizon_h, window_count, args.window_duration_s, args.rate_mbps)
            for task_count in task_counts:
                cfg_s = copy.deepcopy(cfg)
                cfg_s["horizon_h"] = args.horizon_h
                cfg_s["tasks"]["count_per_24h"] = task_count
                tasks = generate_tasks(cfg_s, seed, horizon_h=args.horizon_h, limit=task_count)
                t0 = time.perf_counter()
                metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
                wall_s = time.perf_counter() - t0
                metrics.update(
                    {
                        "seed": seed,
                        "window_count": window_count,
                        "task_count": task_count,
                        "window_duration_s": args.window_duration_s,
                        "horizon_h": args.horizon_h,
                        "wall_s": wall_s,
                        "decision_us_mean": float(metrics["mean_decision_ms"]) * 1000.0,
                        "contacts_scanned_proxy": window_count,
                    }
                )
                rows.append(metrics)
                _write_incremental(rows, args.output)
                print(
                    f"overhead seed={seed} windows={window_count} tasks={task_count} "
                    f"decision={metrics['mean_decision_ms']*1000.0:.2f}us wall={wall_s:.2f}s",
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(resolve_project_path(args.output), index=False)


def _exact_oracle(
    tasks: list[Any],
    windows: list[Window],
    profile: pd.DataFrame,
    cfg: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    scheduler = DownlinkScheduler(windows, cfg["horizon_h"] * 3600.0)
    actions = ("ON", "PRE", "DOWN")
    costs_by_task = [task_costs(task, profile, cfg) for task in tasks]
    best_value = -1.0
    best_sequence: tuple[str, ...] | None = None
    evaluated = 0
    t0 = time.perf_counter()
    status = "optimal"
    memory_limit = float(cfg["satellite"]["memory_gb"])
    ground_latency_s = float(cfg.get("ground", {}).get("latency_s", 0.0))
    initial_energy_left = float(cfg["satellite"]["energy_budget_j"])
    for seq in itertools.product(actions, repeat=len(tasks)):
        sim_scheduler = scheduler.clone()
        sim_compute_ready = 0.0
        sim_energy_left = initial_energy_left
        value = 0.0
        for idx, (action, task) in enumerate(zip(seq, tasks)):
            costs = costs_by_task[idx]
            if action in {"ON", "PRE"} and (costs[action].memory_gb > memory_limit or costs[action].energy_j > sim_energy_left):
                action = "DOWN"
            cost = costs[action]
            if action == "ON":
                completion = max(task.arrival_s, sim_compute_ready) + cost.latency_s
                sim_compute_ready = completion
                sim_energy_left -= cost.energy_j
                value += value_at(task, completion)
            elif action == "PRE":
                pre_done = max(task.arrival_s, sim_compute_ready) + cost.latency_s
                sim_compute_ready = pre_done
                sim_energy_left -= cost.energy_j
                completion = sim_scheduler.schedule(pre_done, cost.bits, commit=True) + ground_latency_s
                value += value_at(task, completion) * cost.value_factor
            else:
                completion = sim_scheduler.schedule(task.arrival_s, cost.bits, commit=True) + ground_latency_s
                value += value_at(task, completion)
        evaluated += 1
        if value > best_value:
            best_value = value
            best_sequence = seq
        if timeout_s > 0 and time.perf_counter() - t0 > timeout_s:
            status = "timeout"
            break
    return {
        "oracle_value": best_value,
        "oracle_sequence": "".join(a[0] for a in best_sequence) if best_sequence else "",
        "evaluated_sequences": evaluated,
        "total_sequences": 3 ** len(tasks),
        "runtime_s": time.perf_counter() - t0,
        "status": status,
    }


def run_oracle(args: argparse.Namespace) -> None:
    cfg, profile = _load_cfg_profile(args.config, args.profile)
    task_counts = [int(v) for v in args.task_counts.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    rows: list[dict[str, Any]] = []
    for task_count in task_counts:
        for seed in seeds:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["horizon_h"] = args.horizon_h
            tasks = generate_tasks(cfg_s, seed, horizon_h=args.horizon_h, limit=task_count)
            windows = generate_windows(cfg_s, horizon_h=args.horizon_h)
            heur, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
            oracle = _exact_oracle(tasks, windows, profile, cfg_s, args.timeout_s)
            gap = max(0.0, (float(oracle["oracle_value"]) - float(heur["total_value"])) / max(float(oracle["oracle_value"]), 1e-9))
            row = {
                "seed": seed,
                "task_count": task_count,
                "horizon_h": args.horizon_h,
                "heuristic_value": float(heur["total_value"]),
                "gap": gap,
                **oracle,
            }
            rows.append(row)
            _write_incremental(rows, args.output)
            print(
                f"oracle tasks={task_count} seed={seed} status={oracle['status']} "
                f"gap={gap*100:.2f}% runtime={oracle['runtime_s']:.1f}s",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(resolve_project_path(args.output), index=False)


def run_absolute(args: argparse.Namespace) -> None:
    cfg, profile = _load_cfg_profile(args.config, args.profile)
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Priority-Aware", "Deadline-Energy", "Energy-WSRPT", "Heuristic-TVD"]
    rows: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        for policy in policies:
            metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
            metrics.update({"seed": seed})
            rows.append(metrics)
            _write_incremental(rows, args.output)
            print(f"absolute policy={policy} seed={seed} value={metrics['total_value']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby("seed")["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    df.to_csv(resolve_project_path(args.output), index=False)
    summary = (
        df.groupby("policy", as_index=False)[
            ["total_value", "normalized_value", "energy_j", "useful_tasks", "median_latency_min", "p95_latency_min", "on_pct", "pre_pct", "down_pct", "drop_pct"]
        ]
        .mean()
        .sort_values("normalized_value", ascending=False)
    )
    summary.to_csv(resolve_project_path(args.summary_output), index=False)


def run_wsrpt(args: argparse.Namespace) -> None:
    cfg, profile = _load_cfg_profile(args.config, args.profile)
    cfg.setdefault("wsrpt", {})
    cfg["wsrpt"]["alpha_energy"] = args.alpha_energy
    cfg["wsrpt"]["alpha_bandwidth"] = args.alpha_bandwidth
    policies = ["Deadline-Energy", "Priority-Aware", "Energy-WSRPT", "Heuristic-TVD"]
    rows: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        for policy in policies:
            metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
            metrics.update({"seed": seed, "alpha_energy": args.alpha_energy, "alpha_bandwidth": args.alpha_bandwidth})
            rows.append(metrics)
            _write_incremental(rows, args.output)
            print(f"wsrpt policy={policy} seed={seed} value={metrics['total_value']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby("seed")["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    df.to_csv(resolve_project_path(args.output), index=False)
    summary = (
        df.groupby("policy", as_index=False)[["normalized_value", "total_value", "energy_j", "mean_decision_ms", "on_pct", "pre_pct", "down_pct"]]
        .mean()
        .sort_values("normalized_value", ascending=False)
    )
    summary.to_csv(resolve_project_path(args.summary_output), index=False)


def run_first_batch(args: argparse.Namespace) -> None:
    run_epsilon(args)
    run_overhead(args)
    run_oracle(args)
    run_absolute(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OrbitLLM local/remote supplemental batch experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None)
        p.add_argument("--profile", default=None)

    eps = sub.add_parser("epsilon")
    add_common(eps)
    eps.add_argument("--epsilons", default="0.001,0.01,0.1,1")
    eps.add_argument("--output", default="results/epsilon_sensitivity_local_metrics.csv")
    eps.set_defaults(func=run_epsilon)

    overhead = sub.add_parser("overhead")
    add_common(overhead)
    overhead.add_argument("--window-counts", default="1024,2048,4096")
    overhead.add_argument("--task-counts", default="2000,5000,10000")
    overhead.add_argument("--seeds", default="0")
    overhead.add_argument("--horizon-h", type=float, default=24.0)
    overhead.add_argument("--window-duration-s", type=float, default=20.0)
    overhead.add_argument("--rate-mbps", type=float, default=35.0)
    overhead.add_argument("--output", default="results/worst_case_overhead_extreme_metrics.csv")
    overhead.set_defaults(func=run_overhead)

    oracle = sub.add_parser("oracle")
    add_common(oracle)
    oracle.add_argument("--task-counts", default="10,11,12")
    oracle.add_argument("--seeds", default="0")
    oracle.add_argument("--horizon-h", type=float, default=6.0)
    oracle.add_argument("--timeout-s", type=float, default=10800.0)
    oracle.add_argument("--output", default="results/exhaustive_oracle_10_12_local_metrics.csv")
    oracle.set_defaults(func=run_oracle)

    absolute = sub.add_parser("absolute")
    add_common(absolute)
    absolute.add_argument("--output", default="results/absolute_with_wsrpt_metrics.csv")
    absolute.add_argument("--summary-output", default="results/absolute_with_wsrpt_summary.csv")
    absolute.set_defaults(func=run_absolute)

    wsrpt = sub.add_parser("wsrpt")
    add_common(wsrpt)
    wsrpt.add_argument("--alpha-energy", type=float, default=1.0)
    wsrpt.add_argument("--alpha-bandwidth", type=float, default=0.15)
    wsrpt.add_argument("--output", default="results/energy_wsrpt_metrics.csv")
    wsrpt.add_argument("--summary-output", default="results/energy_wsrpt_summary.csv")
    wsrpt.set_defaults(func=run_wsrpt)

    first = sub.add_parser("first-batch")
    add_common(first)
    first.add_argument("--epsilons", default="0.001,0.01,0.1,1")
    first.add_argument("--window-counts", default="1024,2048")
    first.add_argument("--task-counts", default="2000,5000")
    first.add_argument("--seeds", default="0")
    first.add_argument("--horizon-h", type=float, default=24.0)
    first.add_argument("--window-duration-s", type=float, default=20.0)
    first.add_argument("--rate-mbps", type=float, default=35.0)
    first.add_argument("--timeout-s", type=float, default=10800.0)
    first.add_argument("--output", default="results/first_batch_placeholder.csv")
    first.add_argument("--summary-output", default="results/absolute_with_wsrpt_summary.csv")
    first.set_defaults(func=run_first_batch)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
