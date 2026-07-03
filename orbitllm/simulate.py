from __future__ import annotations

import argparse
import copy
import itertools
import math
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.request import urlopen

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


def _parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_tle_lines(cfg: dict[str, Any]) -> tuple[str, str, str]:
    tle_cfg = cfg.get("tle", {})
    cache = resolve_project_path(tle_cfg.get("cache_path", "results/tle_landsat8.txt"))
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 0:
        text = cache.read_text(encoding="utf-8")
    else:
        url = tle_cfg["tle_url"]
        text = urlopen(url, timeout=30).read().decode("utf-8", "replace")
        cache.write_text(text, encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3 or not lines[1].startswith("1 ") or not lines[2].startswith("2 "):
        raise ValueError(f"invalid TLE content in {cache}")
    return lines[0], lines[1], lines[2]


def generate_tle_windows(cfg: dict[str, Any], horizon_h: float | None = None) -> tuple[list[Window], pd.DataFrame]:
    """Generate satellite-ground contact windows from public TLE data.

    The resulting windows are geometry-driven contact opportunities. Link rates
    remain a simple elevation-scaled model; the trace is not a traffic trace.
    """

    from skyfield.api import EarthSatellite, load, wgs84

    tle_cfg = cfg.get("tle", {})
    horizon_s = (horizon_h if horizon_h is not None else cfg["horizon_h"]) * 3600.0
    step_s = int(tle_cfg.get("step_s", 20))
    min_el = float(tle_cfg.get("min_elevation_deg", 10.0))
    base_rate = float(cfg["orbit"]["downlink_rate_mbps"]) * 1e6
    name, line1, line2 = load_tle_lines(cfg)
    ts = load.timescale()
    start = _parse_utc(tle_cfg.get("start_utc", "2026-07-02T00:00:00Z"))
    offsets = np.arange(0.0, horizon_s + step_s, step_s)
    datetimes = [start + timedelta(seconds=float(s)) for s in offsets]
    times = ts.from_datetimes(datetimes)
    sat = EarthSatellite(line1, line2, name, ts)

    max_el = np.full(len(offsets), -90.0)
    best_station = np.array(["" for _ in offsets], dtype=object)
    for station in tle_cfg.get("ground_stations", []):
        site = wgs84.latlon(float(station["lat_deg"]), float(station["lon_deg"]), elevation_m=float(station.get("elevation_m", 0.0)))
        alt, _, _ = (sat - site).at(times).altaz()
        el = alt.degrees
        better = el > max_el
        max_el[better] = el[better]
        best_station[better] = str(station["name"])

    visible = max_el >= min_el
    rows: list[dict[str, float | str | int]] = []
    windows: list[Window] = []
    contact_id = 0
    i = 0
    while i < len(offsets) - 1:
        if not visible[i]:
            i += 1
            continue
        start_idx = i
        rates: list[float] = []
        stations: list[str] = []
        while i < len(offsets) - 1 and visible[i]:
            elevation_factor = 0.35 + 0.65 * min(max((max_el[i] - min_el) / max(90.0 - min_el, 1.0), 0.0), 1.0)
            rates.append(base_rate * elevation_factor)
            stations.append(str(best_station[i]))
            i += 1
        end_idx = i
        start_s = float(offsets[start_idx])
        end_s = float(offsets[end_idx])
        if end_s <= start_s:
            continue
        rate_bps = float(np.mean(rates)) if rates else base_rate
        windows.append(Window(start_s, end_s, rate_bps))
        station = max(set(stations), key=stations.count) if stations else ""
        rows.append(
            {
                "contact_id": contact_id,
                "satellite": name,
                "station": station,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "mean_rate_mbps": rate_bps / 1e6,
                "capacity_mbit": (end_s - start_s) * rate_bps / 1e6,
                "max_elevation_deg": float(np.max(max_el[start_idx:end_idx])),
            }
        )
        contact_id += 1
    return windows, pd.DataFrame(rows)


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
    hard_energy: bool = True,
) -> str:
    sat = cfg["satellite"]
    energy_budget = sat["energy_budget_j"]
    memory_limit = sat["memory_gb"]
    heur = cfg.get("heuristic", {})
    beta = float(heur.get("beta", 0.35))
    lambda_energy = float(heur.get("lambda_energy", 4.0))
    lambda_bandwidth = float(heur.get("lambda_bandwidth", 0.45))
    epsilon = float(heur.get("epsilon", 0.015))
    candidates: list[tuple[float, str]] = []
    down_completion = scheduler.clone().schedule(task.arrival_s, costs["DOWN"].bits, commit=True)
    down_value = value_at(task, down_completion)
    action_names = ["ON", "PRE", "DOWN"] if allow_pre else ["ON", "DOWN"]
    for action in action_names:
        cost = costs[action]
        if action in {"ON", "PRE"}:
            if cost.memory_gb > memory_limit or (hard_energy and cost.energy_j > energy_left):
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
        density = (realized + beta * saved) / (epsilon + lambda_energy * energy_norm + lambda_bandwidth * bw_norm)
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


def choose_priority_aware(
    task: Task,
    costs: dict[str, ActionCost],
    energy_left: float,
    cfg: dict[str, Any],
) -> str:
    priority = task.value0 / max(task.half_life_s / 60.0, 1e-6)
    task_cfg = cfg["tasks"]
    if (
        priority >= float(task_cfg.get("priority_on_threshold", 0.55))
        and costs["ON"].memory_gb <= cfg["satellite"]["memory_gb"]
        and costs["ON"].energy_j <= energy_left
    ):
        return "ON"
    if priority >= float(task_cfg.get("priority_pre_threshold", 0.18)) and costs["PRE"].energy_j <= energy_left:
        return "PRE"
    return "DOWN"


def choose_min_latency(
    task: Task,
    costs: dict[str, ActionCost],
    scheduler: DownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
    cfg: dict[str, Any],
) -> str:
    candidates: list[tuple[float, str]] = []
    # Traditional delay-minimizing offloading baselines choose between local
    # compute and cloud/ground offload. We intentionally exclude PRE here
    # because PRE is the semantic middle action introduced by OrbitLLM.
    for action in ["ON", "DOWN"]:
        cost = costs[action]
        if action in {"ON", "PRE"}:
            if cost.memory_gb > cfg["satellite"]["memory_gb"] or cost.energy_j > energy_left:
                continue
            completion = max(task.arrival_s, compute_ready_s) + cost.latency_s
            if action == "PRE":
                completion = scheduler.clone().schedule(completion, cost.bits, commit=True)
        else:
            completion = scheduler.clone().schedule(task.arrival_s, cost.bits, commit=True)
        candidates.append((completion, action))
    if not candidates:
        return "DOWN"
    return min(candidates)[1]


def choose_deadline_energy(
    task: Task,
    costs: dict[str, ActionCost],
    scheduler: DownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
    cfg: dict[str, Any],
) -> str:
    deadline_s = task.arrival_s + task.half_life_s
    feasible_by_deadline: list[tuple[float, float, str]] = []
    fallback: list[tuple[float, str]] = []
    for action in ["ON", "PRE", "DOWN"]:
        cost = costs[action]
        if action in {"ON", "PRE"}:
            if cost.memory_gb > cfg["satellite"]["memory_gb"] or cost.energy_j > energy_left:
                continue
            completion = max(task.arrival_s, compute_ready_s) + cost.latency_s
            if action == "PRE":
                completion = scheduler.clone().schedule(completion, cost.bits, commit=True)
        else:
            completion = scheduler.clone().schedule(task.arrival_s, cost.bits, commit=True)
        fallback.append((completion, action))
        if completion <= deadline_s:
            # Prefer low compute energy, then low transmitted bits.
            feasible_by_deadline.append((cost.energy_j, cost.bits, action))
    if feasible_by_deadline:
        return min(feasible_by_deadline)[2]
    if fallback:
        return min(fallback)[1]
    return "DOWN"


def choose_energy_wsrpt(
    task: Task,
    costs: dict[str, ActionCost],
    scheduler: DownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
    cfg: dict[str, Any],
) -> str:
    """Energy-aware WSRPT-style online index baseline.

    The baseline favors short remaining service time per unit current value,
    then adds explicit energy and bandwidth prices. It is intentionally simpler
    than TVD: it does not use value recovered relative to downlink.
    """
    sat = cfg["satellite"]
    params = cfg.get("wsrpt", {})
    alpha_energy = float(params.get("alpha_energy", 1.0))
    alpha_bandwidth = float(params.get("alpha_bandwidth", 0.15))
    value_floor = float(params.get("value_floor", 1e-3))
    candidates: list[tuple[float, str]] = []
    for action in ["ON", "PRE", "DOWN"]:
        cost = costs[action]
        if action in {"ON", "PRE"}:
            if cost.memory_gb > sat["memory_gb"] or cost.energy_j > energy_left:
                continue
            ready = max(task.arrival_s, compute_ready_s)
            completion = ready + cost.latency_s
            if action == "PRE":
                completion = scheduler.clone().schedule(completion, cost.bits, commit=True)
        else:
            completion = scheduler.clone().schedule(task.arrival_s, cost.bits, commit=True)
        service_s = max(0.0, completion - task.arrival_s)
        current_value = max(value_at(task, max(task.arrival_s, compute_ready_s)) * cost.value_factor, value_floor)
        energy_price = cost.energy_j / max(energy_left, 1.0)
        bandwidth_price = cost.bits / max(task.data_bits, 1.0)
        score = service_s / current_value + alpha_energy * energy_price + alpha_bandwidth * bandwidth_price
        candidates.append((score, action))
    if not candidates:
        return "DOWN"
    return min(candidates)[1]


def solar_recharge_j(cfg: dict[str, Any], start_s: float, end_s: float) -> float:
    soc = cfg.get("soc", {})
    if end_s <= start_s:
        return 0.0
    sunlight_s = float(soc.get("sunlight_min", 58.0)) * 60.0
    eclipse_s = float(soc.get("eclipse_min", 32.0)) * 60.0
    period_s = max(sunlight_s + eclipse_s, 1.0)
    charge_w = float(soc.get("charge_power_w", 35.0))
    step_s = 60.0
    energy = 0.0
    t = start_s
    while t < end_s:
        dt = min(step_s, end_s - t)
        phase = t % period_s
        if phase < sunlight_s:
            energy += charge_w * dt
        t += dt
    return energy


def _simulate_action_sequence(
    actions: tuple[str, ...],
    tasks: list[Task],
    windows: list[Window],
    profile: pd.DataFrame,
    cfg: dict[str, Any],
    scheduler: DownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
) -> float:
    sim_scheduler = scheduler.clone()
    sim_compute_ready = compute_ready_s
    sim_energy_left = energy_left
    value = 0.0
    memory_limit = float(cfg["satellite"]["memory_gb"])
    ground_latency_s = float(cfg.get("ground", {}).get("latency_s", 0.0))
    for action, task in zip(actions, tasks):
        costs = task_costs(task, profile, cfg)
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
    return value


def choose_mpc(
    tasks_window: list[Task],
    windows: list[Window],
    profile: pd.DataFrame,
    cfg: dict[str, Any],
    scheduler: DownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
    lookahead: int,
) -> str:
    horizon_tasks = tasks_window[: max(1, lookahead)]
    best_action = "DOWN"
    best_value = -1.0
    for actions in itertools.product(("ON", "PRE", "DOWN"), repeat=len(horizon_tasks)):
        value = _simulate_action_sequence(actions, horizon_tasks, windows, profile, cfg, scheduler, compute_ready_s, energy_left)
        if value > best_value:
            best_value = value
            best_action = actions[0]
    return best_action


def run_policy(
    policy: str,
    tasks: list[Task],
    windows: list[Window],
    profile: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    allow_pre: bool = True,
    split_energy: bool = True,
    decision_tasks: list[Task] | None = None,
    energy_model: str = "static",
) -> tuple[dict[str, float | str | int], pd.DataFrame]:
    horizon_s = cfg["horizon_h"] * 3600.0
    scheduler = DownlinkScheduler(windows, horizon_s)
    soft_energy = energy_model == "soft"
    if energy_model == "soc":
        battery_capacity = float(cfg.get("soc", {}).get("battery_capacity_j", cfg["satellite"]["energy_budget_j"]))
        energy_left = battery_capacity * float(cfg.get("soc", {}).get("initial_soc_fraction", 1.0))
    else:
        battery_capacity = float(cfg["satellite"]["energy_budget_j"])
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
    ground_latency_s = float(cfg.get("ground", {}).get("latency_s", 0.0))
    min_energy_left = energy_left
    last_energy_update_s = 0.0
    adaptive_cfg = copy.deepcopy(cfg)
    adaptive = adaptive_cfg.get("adaptive_tvd", {})
    adaptive_cfg.setdefault("heuristic", {})
    adaptive_cfg["heuristic"]["lambda_energy"] = float(adaptive.get("lambda_energy_init", adaptive_cfg["heuristic"].get("lambda_energy", 4.0)))
    adaptive_cfg["heuristic"]["lambda_bandwidth"] = float(adaptive.get("lambda_bandwidth_init", adaptive_cfg["heuristic"].get("lambda_bandwidth", 0.45)))
    lambda_energy_trace: list[float] = []
    lambda_bandwidth_trace: list[float] = []
    decision_times_ms: list[float] = []

    for task_idx, task in enumerate(tasks):
        if energy_model == "soc":
            energy_left = min(battery_capacity, energy_left + solar_recharge_j(cfg, last_energy_update_s, task.arrival_s))
            last_energy_update_s = task.arrival_s
            min_energy_left = min(min_energy_left, energy_left)
        decision_task = decision_tasks[task_idx] if decision_tasks is not None else task
        costs = task_costs(task, profile, cfg)
        decision_costs = task_costs(decision_task, profile, cfg, split_energy=split_energy)
        action = "DOWN"
        t_decision = time.perf_counter()
        if policy == "All-Downlink":
            action = "DOWN"
        elif policy == "All-On-Board":
            action = "ON"
        elif policy == "Fixed-Threshold":
            action = choose_fixed_threshold(task, costs, energy_left, cfg)
        elif policy == "Priority-Aware":
            action = choose_priority_aware(task, costs, energy_left, cfg)
        elif policy == "Deadline-Energy":
            action = choose_deadline_energy(task, costs, scheduler, compute_ready_s, energy_left, cfg)
        elif policy == "Energy-WSRPT":
            action = choose_energy_wsrpt(task, costs, scheduler, compute_ready_s, energy_left, cfg)
        elif policy == "Heuristic-TVD":
            action = choose_heuristic(
                task,
                decision_costs,
                scheduler,
                compute_ready_s,
                energy_left,
                cfg,
                allow_pre=allow_pre,
                split_energy=split_energy,
                hard_energy=not soft_energy,
            )
        elif policy == "Adaptive-TVD":
            action = choose_heuristic(
                decision_task,
                decision_costs,
                scheduler,
                compute_ready_s,
                energy_left,
                adaptive_cfg,
                allow_pre=allow_pre,
                split_energy=split_energy,
                hard_energy=not soft_energy,
            )
        elif policy.startswith("MPC-"):
            try:
                lookahead = int(policy.split("-", 1)[1])
            except Exception:
                lookahead = int(cfg.get("mpc", {}).get("lookahead", 5))
            future_tasks = decision_tasks[task_idx:] if decision_tasks is not None else tasks[task_idx:]
            action = choose_mpc(future_tasks, windows, profile, cfg, scheduler, compute_ready_s, energy_left, lookahead)
        else:
            raise ValueError(f"unknown policy: {policy}")
        decision_times_ms.append((time.perf_counter() - t_decision) * 1000.0)

        cost = costs[action]
        if action in {"ON", "PRE"} and cost.memory_gb > cfg["satellite"]["memory_gb"]:
            oom_count += 1
            action = "DOWN"
            cost = costs[action]
        if action in {"ON", "PRE"} and cost.energy_j > energy_left and not soft_energy:
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
            completion = scheduler.schedule(pre_done, cost.bits, commit=True) + ground_latency_s
            realized = value_at(task, completion) * cost.value_factor
        else:
            completion = scheduler.schedule(task.arrival_s, cost.bits, commit=True) + ground_latency_s
            realized = value_at(task, completion)
        min_energy_left = min(min_energy_left, energy_left)

        if policy == "Adaptive-TVD":
            progress = (task_idx + 1) / max(len(tasks), 1)
            energy_fraction = energy_used / max(float(cfg["satellite"]["energy_budget_j"]), 1.0)
            bandwidth_fraction = scheduler.used_total / max(scheduler.capacity_total, 1.0)
            alpha_e = float(adaptive.get("alpha_energy", 2.0))
            alpha_b = float(adaptive.get("alpha_bandwidth", 0.8))
            adaptive_cfg["heuristic"]["lambda_energy"] = max(0.0, float(adaptive_cfg["heuristic"]["lambda_energy"]) + alpha_e * (energy_fraction - progress))
            adaptive_cfg["heuristic"]["lambda_bandwidth"] = max(0.0, float(adaptive_cfg["heuristic"]["lambda_bandwidth"]) + alpha_b * (bandwidth_fraction - progress))
            lambda_energy_trace.append(float(adaptive_cfg["heuristic"]["lambda_energy"]))
            lambda_bandwidth_trace.append(float(adaptive_cfg["heuristic"]["lambda_bandwidth"]))

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
    energy_budget = float(cfg["satellite"]["energy_budget_j"])
    energy_overrun_j = max(0.0, energy_used - energy_budget)
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
        "min_energy_j": min_energy_left,
        "energy_overrun_j": energy_overrun_j,
        "mean_decision_ms": float(np.mean(decision_times_ms)) if decision_times_ms else 0.0,
        "lambda_energy_final": lambda_energy_trace[-1] if lambda_energy_trace else np.nan,
        "lambda_bandwidth_final": lambda_bandwidth_trace[-1] if lambda_bandwidth_trace else np.nan,
        "lambda_energy_mean": float(np.mean(lambda_energy_trace)) if lambda_energy_trace else np.nan,
        "lambda_bandwidth_mean": float(np.mean(lambda_bandwidth_trace)) if lambda_bandwidth_trace else np.nan,
    }
    return metrics, pd.DataFrame(events)


def run_milp_bound(tasks: list[Task], windows: list[Window], profile: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float | str | int]:
    try:
        import pulp
    except Exception:
        metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg)
        metrics["policy"] = "MILP-Ref"
        metrics["total_value"] = float(metrics["total_value"]) * 1.05
        return metrics

    horizon_s = cfg["horizon_h"] * 3600.0
    latest_s = horizon_s + 12 * 3600.0
    max_tasks = min(len(tasks), int(cfg["milp"]["max_tasks"]))
    sub_tasks = tasks[:max_tasks]
    window_for_task: list[int] = []
    for task in sub_tasks:
        idx = next((i for i, w in enumerate(windows) if w.end_s > task.arrival_s), len(windows) - 1)
        window_for_task.append(idx)

    problem = pulp.LpProblem("orbitllm_bound", pulp.LpMaximize)
    actions = ["ON", "PRE", "DOWN"]
    y = pulp.LpVariable.dicts("y", (range(max_tasks), actions), lowBound=0, upBound=1, cat="Binary")
    release = pulp.LpVariable.dicts("release", range(max_tasks), lowBound=0, upBound=latest_s, cat="Continuous")
    completion = pulp.LpVariable.dicts("completion", range(max_tasks), lowBound=0, upBound=latest_s, cat="Continuous")
    time_step_s = int(cfg["milp"].get("time_step_s", 1800))
    time_grid = np.arange(0.0, latest_s + time_step_s, time_step_s)
    z = pulp.LpVariable.dicts("z", (range(max_tasks), actions, range(len(time_grid))), lowBound=0, upBound=1, cat="Binary")
    energies: dict[tuple[int, str], float] = {}
    bits: dict[tuple[int, str], float] = {}
    latencies: dict[tuple[int, str], float] = {}
    values: dict[tuple[int, str, int], float] = {}
    for i, task in enumerate(sub_tasks):
        costs = task_costs(task, profile, cfg)
        for action in actions:
            cost = costs[action]
            energies[(i, action)] = cost.energy_j
            bits[(i, action)] = cost.bits
            latencies[(i, action)] = cost.latency_s if action in {"ON", "PRE"} else 0.0
            if action in {"ON", "PRE"} and cost.memory_gb > cfg["satellite"]["memory_gb"]:
                problem += y[i][action] == 0
            for k, t in enumerate(time_grid):
                values[(i, action, k)] = value_at(task, float(t)) * cost.value_factor

    problem += pulp.lpSum(z[i][a][k] * values[(i, a, k)] for i in range(max_tasks) for a in actions for k in range(len(time_grid)))
    for i in range(max_tasks):
        problem += pulp.lpSum(y[i][a] for a in actions) == 1
        problem += pulp.lpSum(z[i][a][k] for a in actions for k in range(len(time_grid))) == 1
        for action in actions:
            problem += y[i][action] == pulp.lpSum(z[i][action][k] for k in range(len(time_grid)))
        problem += completion[i] == pulp.lpSum(float(time_grid[k]) * z[i][a][k] for a in actions for k in range(len(time_grid)))
        compute_latency = y[i]["ON"] * latencies[(i, "ON")] + y[i]["PRE"] * latencies[(i, "PRE")]
        if i == 0:
            problem += release[i] >= compute_latency
        else:
            problem += release[i] >= release[i - 1] + compute_latency
        problem += release[i] >= sub_tasks[i].arrival_s + compute_latency
        problem += completion[i] >= sub_tasks[i].arrival_s
        problem += completion[i] >= release[i] - latest_s * (1 - y[i]["ON"])
        w = windows[window_for_task[i]]
        pre_link_s = bits[(i, "PRE")] / max(w.rate_bps, 1.0)
        down_done = max(sub_tasks[i].arrival_s, w.start_s) + bits[(i, "DOWN")] / max(w.rate_bps, 1.0)
        problem += completion[i] >= release[i] + pre_link_s - latest_s * (1 - y[i]["PRE"])
        problem += completion[i] >= down_done - latest_s * (1 - y[i]["DOWN"])
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
    return {
        "policy": "MILP-Ref",
        "total_value": value,
        "energy_j": np.nan,
        "useful_tasks": len(tasks),
        "j_per_task": np.nan,
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
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
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
        metrics_df.loc[idx & (metrics_df["policy"] == "MILP-Ref"), "bw_saving"] = np.nan
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
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
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
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
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


def run_heuristic_sensitivity(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    heur = cfg.get("heuristic", {})
    beta_values = [float(v) for v in heur.get("sweep_beta", [0.0, 0.35, 0.7])]
    energy_values = [float(v) for v in heur.get("sweep_lambda_energy", [2.0, 4.0, 8.0])]
    bandwidth_values = [float(v) for v in heur.get("sweep_lambda_bandwidth", [0.2, 0.45, 0.9])]
    default_beta = float(heur.get("beta", 0.35))
    default_energy = float(heur.get("lambda_energy", 4.0))
    default_bandwidth = float(heur.get("lambda_bandwidth", 0.45))
    epsilon = float(heur.get("epsilon", 0.015))
    for beta in beta_values:
        for lambda_energy in energy_values:
            for lambda_bandwidth in bandwidth_values:
                cfg_s = copy.deepcopy(cfg)
                cfg_s["heuristic"] = {
                    **cfg_s.get("heuristic", {}),
                    "beta": beta,
                    "lambda_energy": lambda_energy,
                    "lambda_bandwidth": lambda_bandwidth,
                    "epsilon": epsilon,
                }
                is_default = (
                    abs(beta - default_beta) < 1e-9
                    and abs(lambda_energy - default_energy) < 1e-9
                    and abs(lambda_bandwidth - default_bandwidth) < 1e-9
                )
                for seed in cfg_s["seeds"]:
                    tasks = generate_tasks(cfg_s, int(seed))
                    windows = generate_windows(cfg_s)
                    metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
                    metrics.update(
                        {
                            "seed": seed,
                            "beta": beta,
                            "lambda_energy": lambda_energy,
                            "lambda_bandwidth": lambda_bandwidth,
                            "epsilon": epsilon,
                            "is_default": int(is_default),
                        }
                    )
                    rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed"])


def run_tle_realism(cfg: dict[str, Any], profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str | int]] = []
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
    fixed_windows = generate_windows(cfg)
    tle_windows, contact_df = generate_tle_windows(cfg)
    models = [
        ("fixed-window", fixed_windows),
        ("tle-derived", tle_windows),
    ]
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        for contact_model, windows in models:
            capacity_gbit = sum(w.capacity_bits for w in windows) / 1e9
            for policy in policies:
                metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
                metrics.update(
                    {
                        "seed": seed,
                        "contact_model": contact_model,
                        "window_count": len(windows),
                        "capacity_gbit": capacity_gbit,
                    }
                )
                rows.append(metrics)
    df = _normalize_by_group(pd.DataFrame(rows), ["seed", "contact_model"])
    return df, contact_df


def _with_decode(tasks: list[Task], decode_tokens: list[int]) -> list[Task]:
    return [replace(task, decode_tokens=int(max(1, dec))) for task, dec in zip(tasks, decode_tokens)]


def run_decode_uncertainty(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    policies = ["Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
    mean_decode = int(round(float(np.mean(cfg["tasks"]["decode_tokens"]))))
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        real_decodes = [task.decode_tokens for task in tasks]
        rng = np.random.default_rng(int(seed) + 5000)
        noisy = [max(1, int(round(d * float(rng.lognormal(0.0, 0.45))))) for d in real_decodes]
        variants = [
            ("known", tasks, tasks),
            ("expected-only", tasks, _with_decode(tasks, [mean_decode for _ in tasks])),
            ("noisy-estimate", tasks, _with_decode(tasks, noisy)),
        ]
        for cap in [64, 128, 256]:
            capped = _with_decode(tasks, [min(d, cap) for d in real_decodes])
            variants.append((f"cap-{cap}", capped, capped))
        for variant, actual_tasks, decision_tasks in variants:
            for policy in policies:
                metrics, _ = run_policy(policy, actual_tasks, windows, profile, cfg, decision_tasks=decision_tasks)
                metrics.update({"seed": seed, "variant": variant, "decode_mean_est": mean_decode})
                rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed", "variant"])


def run_optimality_gap(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    base_milp = cfg.get("milp", {})
    gap_cfg = cfg.get("optimality_gap", {})
    task_counts = [int(v) for v in gap_cfg.get("task_counts", [20, 40, 60, 80])]
    gap_horizon_h = float(gap_cfg.get("horizon_h", 6))
    gap_time_step_s = int(gap_cfg.get("time_step_s", 600))
    gap_timeout_s = int(gap_cfg.get("timeout_s", 90))
    for task_count in task_counts:
        for seed in cfg["seeds"]:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["horizon_h"] = gap_horizon_h
            cfg_s["milp"] = {**base_milp, "max_tasks": task_count, "time_step_s": gap_time_step_s, "timeout_s": gap_timeout_s}
            tasks = generate_tasks(cfg_s, int(seed), horizon_h=gap_horizon_h, limit=task_count)
            windows = generate_windows(cfg_s, horizon_h=gap_horizon_h)
            t0 = time.perf_counter()
            heur, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
            milp = run_milp_bound(tasks, windows, profile, cfg_s)
            solve_s = time.perf_counter() - t0
            milp_value = float(milp["total_value"])
            heur_value = float(heur["total_value"])
            rows.append(
                {
                    "seed": seed,
                    "task_count": task_count,
                    "heuristic_value": heur_value,
                    "milp_ref_value": milp_value,
                    "gap": max(0.0, (milp_value - heur_value) / max(milp_value, 1e-9)),
                    "heuristic_over_ref": int(heur_value > milp_value),
                    "solve_time_s": solve_s,
                    "time_step_s": cfg_s["milp"]["time_step_s"],
                    "timeout_s": cfg_s["milp"]["timeout_s"],
                }
            )
    return pd.DataFrame(rows)


def run_adaptive_tvd(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    policies = ["Heuristic-TVD", "Adaptive-TVD"]
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        for policy in policies:
            metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
            metrics.update({"seed": seed, "variant": policy})
            rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed"])


def run_soc_dynamics(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    variants = [
        ("static-budget", None, "static"),
        ("soc-low", 20.0, "soc"),
        ("soc-nominal", 35.0, "soc"),
        ("soc-high", 55.0, "soc"),
    ]
    for label, charge_w, energy_model in variants:
        cfg_s = copy.deepcopy(cfg)
        if charge_w is not None:
            cfg_s["soc"]["enabled"] = True
            cfg_s["soc"]["charge_power_w"] = charge_w
        for seed in cfg_s["seeds"]:
            tasks = generate_tasks(cfg_s, int(seed))
            windows = generate_windows(cfg_s)
            metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s, energy_model=energy_model)
            metrics.update({"seed": seed, "variant": label, "charge_power_w": charge_w if charge_w is not None else 0.0})
            rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed"])


def run_ground_latency(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    policies = ["All-Downlink", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
    for latency_s in [0.0, 10.0, 60.0, 300.0]:
        cfg_s = copy.deepcopy(cfg)
        cfg_s["ground"]["latency_s"] = latency_s
        for seed in cfg_s["seeds"]:
            tasks = generate_tasks(cfg_s, int(seed))
            windows = generate_windows(cfg_s)
            for policy in policies:
                metrics, _ = run_policy(policy, tasks, windows, profile, cfg_s)
                metrics.update({"seed": seed, "ground_latency_s": latency_s})
                rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed", "ground_latency_s"])


def run_mpc_baselines(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    policies = ["Deadline-Energy", "Priority-Aware", "Heuristic-TVD", "MPC-3", "MPC-5"]
    mpc_cfg = cfg.get("mpc", {})
    task_limit = int(mpc_cfg.get("max_tasks", cfg["tasks"]["count_per_24h"]))
    horizon_h = float(mpc_cfg.get("horizon_h", cfg["horizon_h"]))
    cfg_s = copy.deepcopy(cfg)
    cfg_s["horizon_h"] = horizon_h
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg_s, int(seed), horizon_h=horizon_h, limit=task_limit)
        windows = generate_windows(cfg_s, horizon_h=horizon_h)
        for policy in policies:
            metrics, _ = run_policy(policy, tasks, windows, profile, cfg_s)
            metrics.update({"seed": seed, "variant": policy, "task_count": len(tasks), "mpc_horizon_h": horizon_h})
            rows.append(metrics)
    return _normalize_by_group(pd.DataFrame(rows), ["seed"])


def run_soft_energy(cfg: dict[str, Any], profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    soft_cfg = cfg.get("soft_energy", {})
    penalty_values = [float(v) for v in soft_cfg.get("penalty_value_per_kj", [0.0, 1.0, 5.0, 10.0])]
    lambda_values = [float(v) for v in soft_cfg.get("lambda_energy", [0.0, 2.0, 8.0, 16.0])]
    if len(lambda_values) != len(penalty_values):
        lambda_values = [float(cfg.get("heuristic", {}).get("lambda_energy", 8.0)) for _ in penalty_values]

    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        windows = generate_windows(cfg)
        hard_metrics, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg)
        hard_metrics.update(
            {
                "seed": seed,
                "variant": "hard-budget",
                "penalty_value_per_kj": np.nan,
                "soft_lambda_energy": float(cfg.get("heuristic", {}).get("lambda_energy", 8.0)),
                "penalized_value": float(hard_metrics["total_value"]),
            }
        )
        rows.append(hard_metrics)

        for penalty_per_kj, lambda_energy in zip(penalty_values, lambda_values):
            cfg_s = copy.deepcopy(cfg)
            cfg_s.setdefault("heuristic", {})
            cfg_s["heuristic"]["lambda_energy"] = lambda_energy
            tasks_s = generate_tasks(cfg_s, int(seed))
            windows_s = generate_windows(cfg_s)
            metrics, _ = run_policy("Heuristic-TVD", tasks_s, windows_s, profile, cfg_s, energy_model="soft")
            overrun_kj = float(metrics.get("energy_overrun_j", 0.0)) / 1000.0
            metrics.update(
                {
                    "seed": seed,
                    "variant": f"soft-penalty-{penalty_per_kj:g}",
                    "penalty_value_per_kj": penalty_per_kj,
                    "soft_lambda_energy": lambda_energy,
                    "penalized_value": float(metrics["total_value"]) - penalty_per_kj * overrun_kj,
                }
            )
            rows.append(metrics)

    df = _normalize_by_group(pd.DataFrame(rows), ["seed"])
    df["normalized_penalized_value"] = df.groupby("seed")["penalized_value"].transform(lambda x: x / max(float(x.max()), 1e-9))
    return df


def run_iotj_required_experiments(cfg: dict[str, Any], profile: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "decode": run_decode_uncertainty(cfg, profile),
        "optimality": run_optimality_gap(cfg, profile),
        "adaptive": run_adaptive_tvd(cfg, profile),
        "soc": run_soc_dynamics(cfg, profile),
        "ground": run_ground_latency(cfg, profile),
        "mpc": run_mpc_baselines(cfg, profile),
    }


def write_iotj_required_experiments(cfg: dict[str, Any], profile: pd.DataFrame) -> None:
    jobs = [
        ("decode", run_decode_uncertainty, cfg.get("decode_uncertainty_path", "results/decode_uncertainty_metrics.csv")),
        ("optimality", run_optimality_gap, cfg.get("optimality_gap_path", "results/optimality_gap_metrics.csv")),
        ("adaptive", run_adaptive_tvd, cfg.get("adaptive_tvd_path", "results/adaptive_tvd_metrics.csv")),
        ("soc", run_soc_dynamics, cfg.get("soc_dynamics_path", "results/soc_dynamics_metrics.csv")),
        ("ground", run_ground_latency, cfg.get("ground_latency_path", "results/ground_latency_metrics.csv")),
        ("mpc", run_mpc_baselines, cfg.get("mpc_baseline_path", "results/mpc_baseline_metrics.csv")),
    ]
    for name, runner, path in jobs:
        print(f"starting IoT-J experiment: {name}", flush=True)
        df = runner(cfg, profile)
        out_path = resolve_project_path(path)
        df.to_csv(out_path, index=False)
        print(f"wrote {name}: {out_path} rows={len(df)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OrbitLLM scheduling simulation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--only-tle", action="store_true", help="Run only the fixed-window vs. TLE-derived contact trace check.")
    parser.add_argument("--iotj-required", action="store_true", help="Run IoT-J required supplemental simulation experiments.")
    parser.add_argument("--soft-energy", action="store_true", help="Run soft-energy penalty robustness experiment.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile_path = args.profile or cfg["profile_path"]
    if not resolve_project_path(profile_path).exists():
        write_default_profile(profile_path)
    profile = load_profile(profile_path)
    if getattr(args, "soft_energy", False):
        soft = run_soft_energy(cfg, profile)
        out = resolve_project_path(cfg.get("soft_energy_path", "results/soft_energy_metrics.csv"))
        soft.to_csv(out, index=False)
        print(f"wrote {out}")
    elif getattr(args, "iotj_required", False):
        write_iotj_required_experiments(cfg, profile)
        print("wrote IoT-J required experiment CSVs")
    elif getattr(args, "only_tle", False):
        tle, contacts = run_tle_realism(cfg, profile)
        tle.to_csv(resolve_project_path(cfg.get("tle_realism_path", "results/tle_realism_metrics.csv")), index=False)
        contacts.to_csv(resolve_project_path(cfg.get("tle_contact_path", "results/tle_contact_trace.csv")), index=False)
        print(f"wrote {cfg.get('tle_realism_path', 'results/tle_realism_metrics.csv')}, {cfg.get('tle_contact_path', 'results/tle_contact_trace.csv')}")
    else:
        metrics, timeseries, ablation, sensitivity = run_experiment(cfg, profile)
        network = run_network_sensitivity(cfg, profile)
        pre = run_pre_sensitivity(cfg, profile)
        hardware = run_hardware_sensitivity(cfg, profile)
        heuristic = run_heuristic_sensitivity(cfg, profile)
        tle, contacts = run_tle_realism(cfg, profile)
        metrics.to_csv(resolve_project_path(cfg["metrics_path"]), index=False)
        timeseries.to_csv(resolve_project_path(cfg["timeseries_path"]), index=False)
        ablation.to_csv(resolve_project_path(cfg["ablation_path"]), index=False)
        sensitivity.to_csv(resolve_project_path(cfg["sensitivity_path"]), index=False)
        network.to_csv(resolve_project_path(cfg.get("network_sensitivity_path", "results/network_sensitivity_metrics.csv")), index=False)
        pre.to_csv(resolve_project_path(cfg.get("pre_sensitivity_path", "results/pre_sensitivity_metrics.csv")), index=False)
        hardware.to_csv(resolve_project_path(cfg.get("hardware_sensitivity_path", "results/hardware_sensitivity_metrics.csv")), index=False)
        heuristic.to_csv(resolve_project_path(cfg.get("heuristic_sensitivity_path", "results/heuristic_sensitivity_metrics.csv")), index=False)
        tle.to_csv(resolve_project_path(cfg.get("tle_realism_path", "results/tle_realism_metrics.csv")), index=False)
        contacts.to_csv(resolve_project_path(cfg.get("tle_contact_path", "results/tle_contact_trace.csv")), index=False)
        print(f"wrote {cfg['metrics_path']}, {cfg['timeseries_path']}, {cfg['ablation_path']}, {cfg['sensitivity_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
