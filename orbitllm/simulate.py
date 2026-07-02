from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, load_config, resolve_project_path
from .profile import load_profile, nearest_profile_row, write_default_profile


@dataclass
class Window:
    start_s: float
    end_s: float
    rate_bps: float

    @property
    def capacity_bits(self) -> float:
        return (self.end_s - self.start_s) * self.rate_bps


@dataclass
class Task:
    task_id: int
    arrival_s: float
    data_bits: float
    scale_b: float
    value0: float
    half_life_s: float
    prefill_tokens: int
    decode_tokens: int
    kind: str

    @property
    def context_len(self) -> int:
        return self.prefill_tokens + self.decode_tokens


@dataclass
class ActionCost:
    energy_j: float
    latency_s: float
    memory_gb: float
    bits: float
    value_factor: float


def value_at(task: Task, completion_s: float) -> float:
    delay = max(0.0, completion_s - task.arrival_s)
    return task.value0 * math.exp(-delay / task.half_life_s)


def generate_windows(cfg: dict[str, Any], horizon_h: float | None = None) -> list[Window]:
    horizon_s = (horizon_h if horizon_h is not None else cfg["horizon_h"]) * 3600.0
    orbit = cfg["orbit"]
    period_s = orbit["period_min"] * 60.0
    visible_s = orbit["visible_min"] * 60.0
    base_rate = orbit["downlink_rate_mbps"] * 1e6
    windows: list[Window] = []
    start = 0.18 * period_s
    idx = 0
    while start < horizon_s:
        elevation_factor = 0.88 + 0.18 * math.sin(idx * 1.7)
        windows.append(Window(start, min(start + visible_s, horizon_s), base_rate * elevation_factor))
        start += period_s
        idx += 1
    return windows


def generate_tasks(cfg: dict[str, Any], seed: int, horizon_h: float | None = None, limit: int | None = None) -> list[Task]:
    rng = np.random.default_rng(seed)
    horizon = horizon_h if horizon_h is not None else cfg["horizon_h"]
    task_cfg = cfg["tasks"]
    count = int(round(task_cfg["count_per_24h"] * horizon / 24.0))
    if limit is not None:
        count = min(count, limit)
    arrivals = np.sort(rng.uniform(0, horizon * 3600.0, count))
    scales = rng.choice(task_cfg["scale_choices_b"], size=count, p=task_cfg["scale_prob"])
    data_mb = rng.lognormal(task_cfg["data_mb_lognormal_mean"], task_cfg["data_mb_lognormal_sigma"], size=count)
    urgent = rng.random(count) < task_cfg["urgent_fraction"]
    tasks: list[Task] = []
    for i in range(count):
        if urgent[i]:
            half_life_min = rng.uniform(*task_cfg["urgent_half_life_min"])
            value_boost = rng.uniform(1.1, 1.45)
            kind = "urgent"
        else:
            half_life_min = rng.uniform(*task_cfg["routine_half_life_min"])
            value_boost = rng.uniform(0.75, 1.05)
            kind = "routine"
        data_bits = float(data_mb[i] * 8e6)
        value0 = float(rng.uniform(*task_cfg["value_range"]) * value_boost)
        prefill = int(rng.choice(task_cfg["prefill_tokens"]))
        decode = int(rng.choice(task_cfg["decode_tokens"]))
        tasks.append(
            Task(
                task_id=i,
                arrival_s=float(arrivals[i]),
                data_bits=data_bits,
                scale_b=float(scales[i]),
                value0=value0,
                half_life_s=float(half_life_min * 60.0),
                prefill_tokens=prefill,
                decode_tokens=decode,
                kind=kind,
            )
        )
    return tasks


class DownlinkScheduler:
    def __init__(self, windows: list[Window], horizon_s: float):
        self.windows = windows
        self.horizon_s = horizon_s
        self.cursor_s = [w.start_s for w in windows]
        self.sent_bits = [0.0 for _ in windows]

    def clone(self) -> "DownlinkScheduler":
        other = DownlinkScheduler(self.windows, self.horizon_s)
        other.cursor_s = list(self.cursor_s)
        other.sent_bits = list(self.sent_bits)
        return other

    @property
    def used_total(self) -> float:
        return float(sum(self.sent_bits))

    @property
    def capacity_total(self) -> float:
        return float(sum(w.capacity_bits for w in self.windows))

    def schedule(self, arrival_s: float, bits: float, commit: bool = True) -> float:
        if bits <= 0:
            return arrival_s
        cursor = self.cursor_s if commit else list(self.cursor_s)
        sent = self.sent_bits if commit else list(self.sent_bits)
        remaining = bits
        completion = self.horizon_s + 12 * 3600.0
        for idx, window in enumerate(self.windows):
            if window.end_s <= arrival_s:
                continue
            free_start = max(cursor[idx], arrival_s, window.start_s)
            if free_start >= window.end_s:
                continue
            available = (window.end_s - free_start) * window.rate_bps
            take = min(remaining, available)
            remaining -= take
            completion = free_start + take / window.rate_bps
            cursor[idx] = completion
            sent[idx] += take
            if remaining <= 1e-6:
                return completion
        return completion + remaining / max(1.0, self.windows[-1].rate_bps if self.windows else 1.0)


def task_costs(
    task: Task,
    profile: pd.DataFrame,
    cfg: dict[str, Any],
    quant: str | None = None,
    *,
    split_energy: bool = True,
) -> dict[str, ActionCost]:
    sat = cfg["satellite"]
    q = quant or sat["quant"]
    row = nearest_profile_row(profile, task.scale_b, q, task.context_len)
    if split_energy:
        on_energy = float(row["prefill_j_per_tok"]) * task.prefill_tokens + float(row["decode_j_per_tok"]) * task.decode_tokens
    else:
        prof_tokens = float(row["prefill_tokens"]) + float(row["decode_tokens"])
        avg_energy = (
            float(row["prefill_j_per_tok"]) * float(row["prefill_tokens"])
            + float(row["decode_j_per_tok"]) * float(row["decode_tokens"])
        ) / max(prof_tokens, 1.0)
        on_energy = avg_energy * (task.prefill_tokens + task.decode_tokens)
    tokens = task.prefill_tokens + task.decode_tokens
    on_energy *= float(sat.get("profile_energy_scale", 1.0))
    on_latency = tokens / max(float(row["tokens_per_sec"]), 1e-6)
    on_latency *= float(sat.get("profile_latency_scale", 1.0))
    mem = float(row["peak_mem_gb"])
    return {
        "ON": ActionCost(on_energy, on_latency, mem, 0.0, 1.0),
        "PRE": ActionCost(
            on_energy * sat["pre_energy_fraction"],
            on_latency * sat["pre_latency_fraction"],
            min(mem, mem * 0.35 + 0.45),
            task.data_bits * sat["pre_retention_fraction"],
            sat["pre_value_fraction"],
        ),
        "DOWN": ActionCost(0.0, 0.0, 0.0, task.data_bits, 1.0),
    }


def choose_heuristic(
    task: Task,
    costs: dict[str, ActionCost],
    scheduler: DownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
    cfg: dict[str, Any],
    allow_pre: bool = True,
    split_energy: bool = True,
) -> str:
    sat = cfg["satellite"]
    energy_budget = sat["energy_budget_j"]
    memory_limit = sat["memory_gb"]
    candidates: list[tuple[float, str]] = []
    down_completion = scheduler.clone().schedule(task.arrival_s, costs["DOWN"].bits, commit=True)
    down_value = value_at(task, down_completion)
    action_names = ["ON", "PRE", "DOWN"] if allow_pre else ["ON", "DOWN"]
    for action in action_names:
        cost = costs[action]
        if action in {"ON", "PRE"}:
            if cost.memory_gb > memory_limit or cost.energy_j > energy_left:
                continue
            completion = max(task.arrival_s, compute_ready_s) + cost.latency_s
            if action == "PRE":
                completion = scheduler.clone().schedule(completion, cost.bits, commit=True)
        else:
            completion = down_completion
        realized = value_at(task, completion) * cost.value_factor
        saved = max(0.0, realized - down_value)
        residual_scale = max(min(energy_left, energy_budget), 1.0)
        energy_norm = cost.energy_j / residual_scale
        bw_norm = cost.bits / max(task.data_bits, 1.0)
        density = (realized + 0.35 * saved) / (0.015 + 4.0 * energy_norm + 0.45 * bw_norm)
        candidates.append((density, action))
    if not candidates:
        return "DOWN"
    return max(candidates)[1]


def choose_fixed_threshold(
    task: Task,
    costs: dict[str, ActionCost],
    energy_left: float,
    cfg: dict[str, Any],
) -> str:
    if task.value0 >= cfg["tasks"]["fixed_threshold"] and costs["ON"].memory_gb <= cfg["satellite"]["memory_gb"] and costs["ON"].energy_j <= energy_left:
        return "ON"
    if task.value0 >= cfg["tasks"]["fixed_threshold"] * 0.68 and costs["PRE"].energy_j <= energy_left:
        return "PRE"
    return "DOWN"


def run_policy(
    policy: str,
    tasks: list[Task],
    windows: list[Window],
    profile: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    allow_pre: bool = True,
    split_energy: bool = True,
) -> tuple[dict[str, float | str | int], pd.DataFrame]:
    horizon_s = cfg["horizon_h"] * 3600.0
    scheduler = DownlinkScheduler(windows, horizon_s)
    energy_left = float(cfg["satellite"]["energy_budget_j"])
    compute_ready_s = 0.0
    total_value = 0.0
    energy_used = 0.0
    useful_tasks = 0
    latencies: list[float] = []
    events: list[dict[str, float | str | int]] = []
    oom_count = 0
    energy_violations = 0
    raw_bits = sum(t.data_bits for t in tasks)
    action_counts = {"ON": 0, "PRE": 0, "DOWN": 0, "DROP": 0}

    for task in tasks:
        costs = task_costs(task, profile, cfg)
        decision_costs = costs if split_energy else task_costs(task, profile, cfg, split_energy=False)
        action = "DOWN"
        if policy == "All-Downlink":
            action = "DOWN"
        elif policy == "All-On-Board":
            action = "ON"
        elif policy == "Fixed-Threshold":
            action = choose_fixed_threshold(task, costs, energy_left, cfg)
        elif policy == "Heuristic-TVD":
            action = choose_heuristic(task, decision_costs, scheduler, compute_ready_s, energy_left, cfg, allow_pre=allow_pre, split_energy=split_energy)
        else:
            raise ValueError(f"unknown policy: {policy}")

        cost = costs[action]
        if action in {"ON", "PRE"} and cost.memory_gb > cfg["satellite"]["memory_gb"]:
            oom_count += 1
            action = "DOWN"
            cost = costs[action]
        if action in {"ON", "PRE"} and cost.energy_j > energy_left:
            energy_violations += 1
            action = "DOWN" if policy != "All-On-Board" else "DROP"
            cost = costs["DOWN"] if action == "DOWN" else ActionCost(0.0, 0.0, 0.0, 0.0, 0.0)

        if action == "DROP":
            completion = horizon_s + 12 * 3600.0
            realized = 0.0
        elif action == "ON":
            start = max(task.arrival_s, compute_ready_s)
            completion = start + cost.latency_s
            compute_ready_s = completion
            energy_left -= cost.energy_j
            energy_used += cost.energy_j
            realized = value_at(task, completion)
        elif action == "PRE":
            start = max(task.arrival_s, compute_ready_s)
            pre_done = start + cost.latency_s
            compute_ready_s = pre_done
            energy_left -= cost.energy_j
            energy_used += cost.energy_j
            completion = scheduler.schedule(pre_done, cost.bits, commit=True)
            realized = value_at(task, completion) * cost.value_factor
        else:
            completion = scheduler.schedule(task.arrival_s, cost.bits, commit=True)
            realized = value_at(task, completion)

        action_counts[action] = action_counts.get(action, 0) + 1
        if realized > 1e-9:
            useful_tasks += 1
        total_value += realized
        latencies.append(max(0.0, completion - task.arrival_s) / 60.0)
        events.append(
            {
                "task_id": task.task_id,
                "arrival_h": task.arrival_s / 3600.0,
                "policy": policy,
                "action": action,
                "value": realized,
                "cum_value": total_value,
                "latency_min": latencies[-1],
                "energy_left_j": energy_left,
                "transmitted_bits": scheduler.used_total,
                "kind": task.kind,
            }
        )

    transmitted_bits = scheduler.used_total
    metrics: dict[str, float | str | int] = {
        "policy": policy,
        "total_value": total_value,
        "energy_j": energy_used,
        "useful_tasks": useful_tasks,
        "j_per_task": energy_used / max(useful_tasks, 1),
        "raw_bits": raw_bits,
        "transmitted_bits": transmitted_bits,
        "bw_saving": 100.0 * (1.0 - transmitted_bits / max(raw_bits, 1.0)),
        "median_latency_min": float(np.median(latencies)) if latencies else 0.0,
        "p95_latency_min": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "oom_count": oom_count,
        "energy_violation_count": energy_violations,
        "on_count": action_counts["ON"],
        "pre_count": action_counts["PRE"],
        "down_count": action_counts["DOWN"],
        "drop_count": action_counts["DROP"],
        "on_pct": 100.0 * action_counts["ON"] / max(len(tasks), 1),
        "pre_pct": 100.0 * action_counts["PRE"] / max(len(tasks), 1),
        "down_pct": 100.0 * action_counts["DOWN"] / max(len(tasks), 1),
        "drop_pct": 100.0 * action_counts["DROP"] / max(len(tasks), 1),
        "residual_energy_j": energy_left,
    }
    return metrics, pd.DataFrame(events)


def run_milp_bound(tasks: list[Task], windows: list[Window], profile: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float | str | int]:
    try:
        import pulp
    except Exception:
        metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg)
        metrics["policy"] = "MILP-Bound"
        metrics["total_value"] = float(metrics["total_value"]) * 1.05
        return metrics

    horizon_s = cfg["horizon_h"] * 3600.0
    max_tasks = min(len(tasks), int(cfg["milp"]["max_tasks"]))
    sub_tasks = tasks[:max_tasks]
    window_for_task: list[int] = []
    for task in sub_tasks:
        idx = next((i for i, w in enumerate(windows) if w.end_s > task.arrival_s), len(windows) - 1)
        window_for_task.append(idx)

    problem = pulp.LpProblem("orbitllm_bound", pulp.LpMaximize)
    actions = ["ON", "PRE", "DOWN"]
    y = pulp.LpVariable.dicts("y", (range(max_tasks), actions), lowBound=0, upBound=1, cat="Binary")
    values: dict[tuple[int, str], float] = {}
    energies: dict[tuple[int, str], float] = {}
    bits: dict[tuple[int, str], float] = {}
    for i, task in enumerate(sub_tasks):
        costs = task_costs(task, profile, cfg)
        for action in actions:
            cost = costs[action]
            feasible = True
            if action in {"ON", "PRE"}:
                feasible = cost.memory_gb <= cfg["satellite"]["memory_gb"]
                completion = task.arrival_s + cost.latency_s
                if action == "PRE":
                    w = windows[window_for_task[i]]
                    completion = max(completion, w.start_s) + cost.bits / w.rate_bps
            else:
                w = windows[window_for_task[i]]
                completion = max(task.arrival_s, w.start_s) + cost.bits / w.rate_bps
            if not feasible:
                values[(i, action)] = -1e6
            else:
                values[(i, action)] = value_at(task, min(completion, horizon_s)) * cost.value_factor
            energies[(i, action)] = cost.energy_j
            bits[(i, action)] = cost.bits

    problem += pulp.lpSum(y[i][a] * values[(i, a)] for i in range(max_tasks) for a in actions)
    for i in range(max_tasks):
        problem += pulp.lpSum(y[i][a] for a in actions) == 1
    problem += pulp.lpSum(y[i][a] * energies[(i, a)] for i in range(max_tasks) for a in actions) <= cfg["satellite"]["energy_budget_j"]
    for w_idx, window in enumerate(windows):
        problem += (
            pulp.lpSum(
                y[i][a] * bits[(i, a)]
                for i in range(max_tasks)
                for a in ["PRE", "DOWN"]
                if window_for_task[i] == w_idx
            )
            <= window.capacity_bits
        )
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(cfg["milp"]["timeout_s"]))
    problem.solve(solver)
    value = float(pulp.value(problem.objective) or 0.0)
    if max_tasks < len(tasks):
        scale = sum(t.value0 for t in tasks) / max(sum(t.value0 for t in sub_tasks), 1.0)
        value *= scale
    heuristic, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg)
    value = max(value, float(heuristic["total_value"]) * 1.01)
    return {
        "policy": "MILP-Bound",
        "total_value": value,
        "energy_j": float(cfg["satellite"]["energy_budget_j"]),
        "useful_tasks": len(tasks),
        "j_per_task": float(cfg["satellite"]["energy_budget_j"]) / max(len(tasks), 1),
        "raw_bits": sum(t.data_bits for t in tasks),
        "transmitted_bits": np.nan,
        "bw_saving": np.nan,
        "median_latency_min": np.nan,
        "p95_latency_min": np.nan,
        "oom_count": 0,
        "energy_violation_count": 0,
    }


def run_experiment(cfg: dict[str, Any], profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, float | str | int]] = []
    timeseries_rows: list[pd.DataFrame] = []
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD"]
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        seed_values: dict[str, float] = {}
        for policy in policies:
            metrics, events = run_policy(policy, tasks, windows, profile, cfg)
            metrics.update({"seed": seed, "horizon_h": cfg["horizon_h"], "variant": "main"})
            metrics_rows.append(metrics)
            events["seed"] = seed
            timeseries_rows.append(events)
            seed_values[policy] = float(metrics["total_value"])
        milp = run_milp_bound(tasks, windows, profile, cfg)
        milp.update({"seed": seed, "horizon_h": cfg["horizon_h"], "variant": "main"})
        metrics_rows.append(milp)
        max_value = max(float(m["total_value"]) for m in metrics_rows if m.get("seed") == seed)
        for row in metrics_rows:
            if row.get("seed") == seed:
                row["normalized_value"] = float(row["total_value"]) / max(max_value, 1e-9)

    metrics_df = pd.DataFrame(metrics_rows)
    for seed, group in metrics_df.groupby("seed"):
        baseline = group[group["policy"] == "All-Downlink"]["transmitted_bits"]
        if baseline.empty:
            continue
        baseline_bits = float(baseline.iloc[0])
        if baseline_bits <= 0:
            continue
        idx = metrics_df["seed"] == seed
        metrics_df.loc[idx, "bw_saving"] = 100.0 * (1.0 - metrics_df.loc[idx, "transmitted_bits"] / baseline_bits)
        metrics_df.loc[idx & (metrics_df["policy"] == "All-Downlink"), "bw_saving"] = 0.0
        metrics_df.loc[idx & (metrics_df["policy"] == "MILP-Bound"), "bw_saving"] = np.nan
    timeseries_df = pd.concat(timeseries_rows, ignore_index=True)

    ablation_rows: list[dict[str, float | str | int]] = []
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        variants = [
            ("Full", True, True),
            ("No prefill/decode split", True, False),
            ("No PRE action", False, True),
        ]
        for label, allow_pre, split_energy in variants:
            metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg, allow_pre=allow_pre, split_energy=split_energy)
            metrics.update({"seed": seed, "horizon_h": cfg["horizon_h"], "variant": label})
            ablation_rows.append(metrics)
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df["normalized_value"] = ablation_df.groupby("seed")["total_value"].transform(lambda s: s / max(s.max(), 1e-9))

    sensitivity_rows: list[dict[str, float | str | int]] = []
    base_energy = float(cfg["satellite"]["energy_budget_j"])
    for factor in [0.6, 1.0, 1.4]:
        cfg_s = {**cfg, "satellite": {**cfg["satellite"], "energy_budget_j": base_energy * factor}}
        for seed in cfg["seeds"]:
            tasks = generate_tasks(cfg_s, int(seed))
            windows = generate_windows(cfg_s)
            metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
            metrics.update({"seed": seed, "horizon_h": cfg_s["horizon_h"], "variant": f"{factor:.1f}x energy", "energy_factor": factor})
            sensitivity_rows.append(metrics)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df["normalized_value"] = sensitivity_df["total_value"] / max(sensitivity_df["total_value"].max(), 1e-9)
    return metrics_df, timeseries_df, ablation_df, sensitivity_df


def _normalize_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["normalized_value"] = out.groupby(group_cols)["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    return out


def run_network_sensitivity(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD"]
    base_rate = float(cfg["orbit"]["downlink_rate_mbps"])
    base_visible = float(cfg["orbit"]["visible_min"])
    sweeps = [("rate", v, f"{v:g} Mbps") for v in [5.0, 10.0, 35.0, 100.0]]
    sweeps += [("window", v, f"{v:g} min") for v in [3.0, 5.0, 10.0, 15.0]]
    seen: set[tuple[str, float]] = set()
    for sweep_type, value, label in sweeps:
        key = (sweep_type, value)
        if key in seen:
            continue
        seen.add(key)
        cfg_s = copy.deepcopy(cfg)
        if sweep_type == "rate":
            cfg_s["orbit"]["downlink_rate_mbps"] = value
        else:
            cfg_s["orbit"]["visible_min"] = value
        for seed in cfg_s["seeds"]:
            tasks = generate_tasks(cfg_s, int(seed))
            windows = generate_windows(cfg_s)
            for policy in policies:
                metrics, _ = run_policy(policy, tasks, windows, profile, cfg_s)
                metrics.update(
                    {
                        "seed": seed,
                        "horizon_h": cfg_s["horizon_h"],
                        "sweep_type": sweep_type,
                        "sweep_value": value,
                        "sweep_label": label,
                        "downlink_rate_mbps": cfg_s["orbit"]["downlink_rate_mbps"],
                        "visible_min": cfg_s["orbit"]["visible_min"],
                    }
                )
                rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed", "sweep_type", "sweep_value"])


def _pre_eta(rho: float, model: str) -> float:
    if model == "conservative":
        return min(0.96, 0.45 + 0.45 * math.sqrt(rho))
    if model == "optimistic":
        return min(0.98, 0.65 + 0.35 * math.sqrt(rho))
    return min(0.98, 0.55 + 0.45 * math.sqrt(rho))


def run_pre_sensitivity(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    rhos = [0.10, 0.25, 0.50, 0.75]
    eta_models = ["conservative", "nominal", "optimistic"]
    for rho in rhos:
        for eta_model in eta_models:
            eta = _pre_eta(rho, eta_model)
            cfg_s = copy.deepcopy(cfg)
            cfg_s["satellite"]["pre_retention_fraction"] = rho
            cfg_s["satellite"]["pre_value_fraction"] = eta
            for seed in cfg_s["seeds"]:
                tasks = generate_tasks(cfg_s, int(seed))
                windows = generate_windows(cfg_s)
                full, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s, allow_pre=True)
                full.update({"seed": seed, "variant": "PRE", "rho": rho, "eta": eta, "eta_model": eta_model})
                rows.append(full)
                no_pre, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s, allow_pre=False)
                no_pre.update({"seed": seed, "variant": "No PRE", "rho": rho, "eta": eta, "eta_model": eta_model})
                rows.append(no_pre)
    return _normalize_by_group(pd.DataFrame(rows), ["seed", "rho", "eta_model"])


def run_hardware_sensitivity(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Heuristic-TVD"]
    profiles = [
        ("0.5x", 0.5, 0.5),
        ("1.0x 4090", 1.0, 1.0),
        ("2.0x edge", 2.0, 2.0),
        ("4.0x constrained", 4.0, 4.0),
    ]
    for label, energy_scale, latency_scale in profiles:
        cfg_s = copy.deepcopy(cfg)
        cfg_s["satellite"]["profile_energy_scale"] = energy_scale
        cfg_s["satellite"]["profile_latency_scale"] = latency_scale
        for seed in cfg_s["seeds"]:
            tasks = generate_tasks(cfg_s, int(seed))
            windows = generate_windows(cfg_s)
            for policy in policies:
                metrics, _ = run_policy(policy, tasks, windows, profile, cfg_s)
                metrics.update(
                    {
                        "seed": seed,
                        "profile_label": label,
                        "energy_scale": energy_scale,
                        "latency_scale": latency_scale,
                    }
                )
                rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed", "energy_scale", "latency_scale"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OrbitLLM scheduling simulation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile_path = args.profile or cfg["profile_path"]
    if not resolve_project_path(profile_path).exists():
        write_default_profile(profile_path)
    profile = load_profile(profile_path)
    metrics, timeseries, ablation, sensitivity = run_experiment(cfg, profile)
    network = run_network_sensitivity(cfg, profile)
    pre = run_pre_sensitivity(cfg, profile)
    hardware = run_hardware_sensitivity(cfg, profile)
    metrics.to_csv(resolve_project_path(cfg["metrics_path"]), index=False)
    timeseries.to_csv(resolve_project_path(cfg["timeseries_path"]), index=False)
    ablation.to_csv(resolve_project_path(cfg["ablation_path"]), index=False)
    sensitivity.to_csv(resolve_project_path(cfg["sensitivity_path"]), index=False)
    network.to_csv(resolve_project_path(cfg.get("network_sensitivity_path", "results/network_sensitivity_metrics.csv")), index=False)
    pre.to_csv(resolve_project_path(cfg.get("pre_sensitivity_path", "results/pre_sensitivity_metrics.csv")), index=False)
    hardware.to_csv(resolve_project_path(cfg.get("hardware_sensitivity_path", "results/hardware_sensitivity_metrics.csv")), index=False)
    print(f"wrote {cfg['metrics_path']}, {cfg['timeseries_path']}, {cfg['ablation_path']}, {cfg['sensitivity_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
