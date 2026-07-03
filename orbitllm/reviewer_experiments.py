from __future__ import annotations

import argparse
import copy
import csv
import math
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import PowerSampler, measure_idle_power
from .config import ensure_output_dirs, load_config, resolve_project_path
from .profile import load_profile, write_default_profile
from .simulate import (
    ActionCost,
    Task,
    Window,
    choose_deadline_energy,
    choose_fixed_threshold,
    choose_heuristic,
    choose_priority_aware,
    generate_tasks,
    generate_tle_windows,
    generate_windows,
    run_milp_bound,
    run_policy,
    task_costs,
    value_at,
)


def _first_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "cuda"


def _load_eurosat_images(zip_path: Path, limit: int, seed: int) -> list[Any]:
    from PIL import Image
    import io

    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png")) and not n.endswith("/")]
        if not names:
            raise ValueError(f"no image files found in {zip_path}")
        idx = rng.choice(len(names), size=min(limit, len(names)), replace=False)
        images = []
        for i in idx:
            with zf.open(names[int(i)]) as f:
                images.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
    return images


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def _prepare_vlm_inputs(processor: Any, image: Any, prompt: str, device: str) -> Any:
    if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        except Exception:
            inputs = processor(images=image, text=text, return_tensors="pt")
    else:
        inputs = processor(images=image, text=prompt, return_tensors="pt")
    return _move_inputs(inputs, device)


def load_vlm(model_name: str, dtype: str = "fp16") -> tuple[Any, Any]:
    import torch
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = str(getattr(config, "model_type", "")).lower()
    torch_dtype = torch.float16 if dtype.lower() in {"fp16", "float16"} else torch.float32
    kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    errors: list[str] = []
    cls_order = ["AutoModelForImageTextToText", "AutoModelForVision2Seq"]
    if "qwen" in model_type:
        cls_order = ["Qwen2_5_VLForConditionalGeneration"] + cls_order
    for cls_name in cls_order:
        try:
            module = __import__("transformers", fromlist=[cls_name])
            loader = getattr(module, cls_name)
            model = loader.from_pretrained(model_name, **kwargs)
            model.eval()
            return processor, model
        except Exception as exc:
            errors.append(f"{cls_name} load failed: {exc}")
    raise RuntimeError("Could not load VLM model. " + " | ".join(errors[-5:]))


def run_vlm_profile(args: argparse.Namespace) -> None:
    import torch

    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_zip = resolve_project_path(args.eurosat_zip)
    images = _load_eurosat_images(data_zip, args.samples, args.seed)
    processor, model = load_vlm(args.model, args.dtype)
    device = _first_device(model)
    prompt = args.prompt
    idle_power_w = measure_idle_power(args.idle_sample_s)
    rows: list[dict[str, object]] = []
    print(f"VLM profile start model={args.model} samples={len(images)} idle={idle_power_w:.3f}W device={device}", flush=True)

    for idx, image in enumerate(images):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        inputs = _prepare_vlm_inputs(processor, image, prompt, device)
        input_tokens = int(inputs["input_ids"].shape[-1]) if isinstance(inputs, dict) and "input_ids" in inputs else 1

        prefill_repeats = max(1, int(args.prefill_repeats))
        generate_repeats = max(1, int(args.generate_repeats))
        with PowerSampler() as sampler:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for _ in range(prefill_repeats):
                try:
                    with torch.no_grad():
                        _ = model(**inputs)
                except Exception:
                    # Encoder-decoder captioning models often need decoder inputs
                    # for forward(). A one-token greedy generation is a robust
                    # prefill proxy for image+prompt processing.
                    with torch.no_grad():
                        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            generated_tokens_total = 0
            for _ in range(generate_repeats):
                with torch.no_grad():
                    generated = model.generate(**inputs, max_new_tokens=args.decode_tokens, do_sample=False)
                try:
                    out_len = int(generated.shape[-1])
                    if "input_ids" in inputs and out_len > int(inputs["input_ids"].shape[-1]):
                        generated_tokens_total += out_len - int(inputs["input_ids"].shape[-1])
                    else:
                        generated_tokens_total += out_len
                except Exception:
                    generated_tokens_total += args.decode_tokens
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t2 = time.perf_counter()

        prefill_energy_total, prefill_power = sampler.energy_between(t0, t1)
        total_gen_energy_total, gen_power = sampler.energy_between(t1, t2)
        prefill_active_total, prefill_active_power = sampler.energy_between(t0, t1, idle_power_w=idle_power_w)
        gen_active_total, gen_active_power = sampler.energy_between(t1, t2, idle_power_w=idle_power_w)
        prefill_energy = prefill_energy_total / prefill_repeats
        total_gen_energy = total_gen_energy_total / generate_repeats
        prefill_active = prefill_active_total / prefill_repeats
        gen_active = gen_active_total / generate_repeats
        decode_energy = max(0.0, total_gen_energy - prefill_energy)
        if decode_energy <= 0.0:
            decode_energy = max(0.0, total_gen_energy)
        decode_active = max(0.0, gen_active - prefill_active)
        decode_time = max(1e-9, t2 - t1)
        actual_decode_tokens = max(1.0, generated_tokens_total / generate_repeats)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        row = {
            "model": args.model,
            "scale_b": args.scale_b,
            "modality": "image_text",
            "quant": args.dtype.upper(),
            "image_size": args.image_size,
            "context_len": input_tokens + actual_decode_tokens,
            "prefill_tokens": input_tokens,
            "decode_tokens": actual_decode_tokens,
            "vision_prefill_j_per_image": prefill_energy,
            "prefill_j_per_tok": prefill_energy / max(input_tokens, 1),
            "decode_j_per_tok": decode_energy / max(actual_decode_tokens, 1),
            "prefill_active_j_per_tok": prefill_active / max(input_tokens, 1),
            "decode_active_j_per_tok": decode_active / max(actual_decode_tokens, 1),
            "tokens_per_sec": generated_tokens_total / max(decode_time, 1e-9),
            "peak_mem_gb": peak_mem,
            "gpu_power_w": max(prefill_power, gen_power),
            "idle_power_w": idle_power_w,
            "prefill_avg_power_w": prefill_power,
            "decode_avg_power_w": gen_power,
            "prefill_active_avg_power_w": prefill_active_power,
            "decode_active_avg_power_w": gen_active_power,
            "seed": args.seed,
            "sample_id": idx,
            "prefill_repeats": prefill_repeats,
            "generate_repeats": generate_repeats,
            "run_id": f"vlm_4090_{int(time.time())}",
        }
        rows.append(row)
        if (idx + 1) % max(1, args.flush_every) == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"wrote partial VLM profile rows={len(rows)} to {out}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    mean = df.mean(numeric_only=True).to_dict()
    summary = {
        "model": args.model,
        "scale_b": args.scale_b,
        "modality": "image_text",
        "quant": args.dtype.upper(),
        "image_size": args.image_size,
        "context_len": round(mean.get("context_len", 0.0), 2),
        "prefill_tokens": round(mean.get("prefill_tokens", 0.0), 2),
        "decode_tokens": round(mean.get("decode_tokens", float(args.decode_tokens)), 2),
        "vision_prefill_j_per_image": mean.get("vision_prefill_j_per_image", 0.0),
        "prefill_j_per_tok": mean.get("prefill_j_per_tok", 0.0),
        "decode_j_per_tok": mean.get("decode_j_per_tok", 0.0),
        "tokens_per_sec": mean.get("tokens_per_sec", 0.0),
        "peak_mem_gb": mean.get("peak_mem_gb", 0.0),
        "gpu_power_w": mean.get("gpu_power_w", 0.0),
        "idle_power_w": mean.get("idle_power_w", 0.0),
        "seed": args.seed,
        "run_id": "vlm_4090_mean",
    }
    summary_path = resolve_project_path(args.summary_output)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"wrote VLM profile {out} and summary {summary_path}", flush=True)


@dataclass
class PhysicalDownlinkScheduler:
    windows: list[Window]
    horizon_s: float
    radio_circuit_w: float
    pa_static_w: float
    pa_peak_w: float
    pa_efficiency: float
    base_rate_bps: float

    def __post_init__(self) -> None:
        self.cursor_s = [w.start_s for w in self.windows]
        self.sent_bits = [0.0 for _ in self.windows]
        self.comm_energy_j = 0.0

    def clone(self) -> "PhysicalDownlinkScheduler":
        other = PhysicalDownlinkScheduler(
            self.windows,
            self.horizon_s,
            self.radio_circuit_w,
            self.pa_static_w,
            self.pa_peak_w,
            self.pa_efficiency,
            self.base_rate_bps,
        )
        other.cursor_s = list(self.cursor_s)
        other.sent_bits = list(self.sent_bits)
        other.comm_energy_j = self.comm_energy_j
        return other

    @property
    def used_total(self) -> float:
        return float(sum(self.sent_bits))

    @property
    def capacity_total(self) -> float:
        return float(sum(w.capacity_bits for w in self.windows))

    def _power_for_rate(self, rate_bps: float) -> float:
        rate_norm = min(max(rate_bps / max(self.base_rate_bps, 1.0), 0.05), 3.0)
        rf_pa = self.pa_static_w + self.pa_peak_w * (rate_norm**2)
        return self.radio_circuit_w + rf_pa / max(self.pa_efficiency, 1e-3)

    def schedule_energy(self, arrival_s: float, bits: float, commit: bool = True) -> tuple[float, float, float]:
        if bits <= 0:
            return arrival_s, 0.0, 0.0
        cursor = self.cursor_s if commit else list(self.cursor_s)
        sent = self.sent_bits if commit else list(self.sent_bits)
        remaining = bits
        completion = self.horizon_s + 12 * 3600.0
        energy = 0.0
        tx_time = 0.0
        for idx, window in enumerate(self.windows):
            if window.end_s <= arrival_s:
                continue
            free_start = max(cursor[idx], arrival_s, window.start_s)
            if free_start >= window.end_s:
                continue
            available = (window.end_s - free_start) * window.rate_bps
            take = min(remaining, available)
            dt = take / max(window.rate_bps, 1.0)
            remaining -= take
            tx_time += dt
            energy += self._power_for_rate(window.rate_bps) * dt
            completion = free_start + dt
            cursor[idx] = completion
            sent[idx] += take
            if remaining <= 1e-6:
                if commit:
                    self.comm_energy_j += energy
                return completion, energy, tx_time
        if self.windows:
            fallback_rate = max(1.0, self.windows[-1].rate_bps)
        else:
            fallback_rate = max(1.0, self.base_rate_bps)
        dt = remaining / fallback_rate
        energy += self._power_for_rate(fallback_rate) * dt
        completion += dt
        tx_time += dt
        if commit:
            self.comm_energy_j += energy
        return completion, energy, tx_time

    def schedule(self, arrival_s: float, bits: float, commit: bool = True) -> float:
        completion, _, _ = self.schedule_energy(arrival_s, bits, commit=commit)
        return completion


def _physical_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    link = cfg.get("physical_link", {})
    return {
        "radio_circuit_w": float(link.get("radio_circuit_w", 6.0)),
        "pa_static_w": float(link.get("pa_static_w", 2.0)),
        "pa_peak_w": float(link.get("pa_peak_w", 16.0)),
        "pa_efficiency": float(link.get("pa_efficiency", 0.35)),
        "base_rate_bps": float(cfg["orbit"]["downlink_rate_mbps"]) * 1e6,
    }


def choose_heuristic_physical(
    task: Task,
    costs: dict[str, ActionCost],
    scheduler: PhysicalDownlinkScheduler,
    compute_ready_s: float,
    energy_left: float,
    cfg: dict[str, Any],
) -> str:
    sat = cfg["satellite"]
    heur = cfg.get("heuristic", {})
    beta = float(heur.get("beta", 0.35))
    lambda_energy = float(heur.get("lambda_energy", 4.0))
    lambda_bandwidth = float(heur.get("lambda_bandwidth", 0.45))
    epsilon = float(heur.get("epsilon", 0.015))
    memory_limit = float(sat["memory_gb"])
    down_completion, down_comm_e, _ = scheduler.clone().schedule_energy(task.arrival_s, costs["DOWN"].bits, commit=True)
    down_value = value_at(task, down_completion)
    candidates: list[tuple[float, str]] = []
    for action in ["ON", "PRE", "DOWN"]:
        cost = costs[action]
        compute_e = cost.energy_j
        comm_e = 0.0
        if action == "ON":
            completion = max(task.arrival_s, compute_ready_s) + cost.latency_s
        elif action == "PRE":
            pre_done = max(task.arrival_s, compute_ready_s) + cost.latency_s
            completion, comm_e, _ = scheduler.clone().schedule_energy(pre_done, cost.bits, commit=True)
        else:
            completion = down_completion
            comm_e = down_comm_e
        total_e = compute_e + comm_e
        if cost.memory_gb > memory_limit or total_e > energy_left:
            continue
        realized = value_at(task, completion) * cost.value_factor
        saved = max(0.0, realized - down_value)
        density = (realized + beta * saved) / (
            epsilon
            + lambda_energy * total_e / max(min(energy_left, float(sat["energy_budget_j"])), 1.0)
            + lambda_bandwidth * cost.bits / max(task.data_bits, 1.0)
        )
        candidates.append((density, action))
    return max(candidates)[1] if candidates else "DOWN"


def run_policy_physical(policy: str, tasks: list[Task], windows: list[Window], profile: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float | str | int]:
    horizon_s = cfg["horizon_h"] * 3600.0
    params = _physical_cfg(cfg)
    scheduler = PhysicalDownlinkScheduler(windows, horizon_s, **params)
    energy_left = float(cfg["satellite"]["energy_budget_j"])
    compute_ready_s = 0.0
    compute_energy = 0.0
    total_value = 0.0
    action_counts = {"ON": 0, "PRE": 0, "DOWN": 0, "DROP": 0}
    latencies: list[float] = []
    raw_bits = sum(t.data_bits for t in tasks)
    oom_count = 0
    energy_violation_count = 0
    for task in tasks:
        costs = task_costs(task, profile, cfg)
        if policy == "All-Downlink":
            action = "DOWN"
        elif policy == "All-On-Board":
            action = "ON"
        elif policy == "Fixed-Threshold":
            action = choose_fixed_threshold(task, costs, energy_left, cfg)
        elif policy == "Deadline-Energy":
            # Use the original baseline choice, then enforce physical energy at execution.
            action = choose_deadline_energy(task, costs, scheduler, compute_ready_s, energy_left, cfg)  # type: ignore[arg-type]
        elif policy == "Priority-Aware":
            action = choose_priority_aware(task, costs, energy_left, cfg)
        else:
            action = choose_heuristic_physical(task, costs, scheduler, compute_ready_s, energy_left, cfg)
        cost = costs[action]
        if cost.memory_gb > float(cfg["satellite"]["memory_gb"]):
            oom_count += 1
            action = "DOWN"
            cost = costs[action]
        if action == "ON":
            total_e = cost.energy_j
            if total_e > energy_left:
                energy_violation_count += 1
                action = "DROP" if policy == "All-On-Board" else "DOWN"
                cost = costs[action] if action != "DROP" else ActionCost(0, 0, 0, 0, 0)
        if action == "PRE":
            pre_done = max(task.arrival_s, compute_ready_s) + cost.latency_s
            _, comm_e_check, _ = scheduler.clone().schedule_energy(pre_done, cost.bits, commit=True)
            if cost.energy_j + comm_e_check > energy_left:
                energy_violation_count += 1
                action = "DOWN"
                cost = costs[action]
        if action == "DOWN":
            _, comm_e_check, _ = scheduler.clone().schedule_energy(task.arrival_s, cost.bits, commit=True)
            if comm_e_check > energy_left:
                energy_violation_count += 1
                action = "DROP"
                cost = ActionCost(0, 0, 0, 0, 0)

        if action == "DROP":
            completion = horizon_s + 12 * 3600.0
            realized = 0.0
        elif action == "ON":
            completion = max(task.arrival_s, compute_ready_s) + cost.latency_s
            compute_ready_s = completion
            energy_left -= cost.energy_j
            compute_energy += cost.energy_j
            realized = value_at(task, completion)
        elif action == "PRE":
            pre_done = max(task.arrival_s, compute_ready_s) + cost.latency_s
            compute_ready_s = pre_done
            completion, comm_e, _ = scheduler.schedule_energy(pre_done, cost.bits, commit=True)
            energy_left -= cost.energy_j + comm_e
            compute_energy += cost.energy_j
            realized = value_at(task, completion) * cost.value_factor
        else:
            completion, comm_e, _ = scheduler.schedule_energy(task.arrival_s, cost.bits, commit=True)
            energy_left -= comm_e
            realized = value_at(task, completion)
        action_counts[action] += 1
        total_value += realized
        latencies.append(max(0.0, completion - task.arrival_s) / 60.0)
    comm_energy = scheduler.comm_energy_j
    total_energy = compute_energy + comm_energy
    return {
        "policy": policy,
        "total_value": total_value,
        "compute_energy_j": compute_energy,
        "comm_energy_j": comm_energy,
        "energy_j": total_energy,
        "useful_tasks": sum(1 for x in latencies if x < 12 * 60),
        "raw_bits": raw_bits,
        "transmitted_bits": scheduler.used_total,
        "bw_saving": 100.0 * (1.0 - scheduler.used_total / max(raw_bits, 1.0)),
        "median_latency_min": float(np.median(latencies)) if latencies else 0.0,
        "p95_latency_min": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "oom_count": oom_count,
        "energy_violation_count": energy_violation_count,
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


def run_physical_link(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    cfg["physical_link"] = {
        "radio_circuit_w": args.radio_circuit_w,
        "pa_static_w": args.pa_static_w,
        "pa_peak_w": args.pa_peak_w,
        "pa_efficiency": args.pa_efficiency,
    }
    profile_path = args.profile or cfg["profile_path"]
    if not resolve_project_path(profile_path).exists():
        write_default_profile(profile_path)
    profile = load_profile(profile_path)
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
    rows = []
    for seed in cfg["seeds"]:
        tasks = generate_tasks(cfg, int(seed))
        fixed = generate_windows(cfg)
        tle, contacts = generate_tle_windows(cfg)
        for contact_model, windows in [("fixed-window-original", fixed), ("tle-elevation-dynamic", tle)]:
            for policy in policies:
                if contact_model == "fixed-window-original":
                    metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
                    metrics["compute_energy_j"] = metrics["energy_j"]
                    metrics["comm_energy_j"] = 0.0
                else:
                    metrics = run_policy_physical(policy, tasks, windows, profile, cfg)
                metrics.update(
                    {
                        "seed": seed,
                        "contact_model": contact_model,
                        "window_count": len(windows),
                        "capacity_gbit": sum(w.capacity_bits for w in windows) / 1e9,
                    }
                )
                rows.append(metrics)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby(["seed", "contact_model"])["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    out = resolve_project_path(args.output)
    df.to_csv(out, index=False)
    mix = (
        df.groupby(["contact_model", "policy"], as_index=False)[["on_pct", "pre_pct", "down_pct", "drop_pct", "compute_energy_j", "comm_energy_j", "energy_j", "normalized_value"]]
        .mean()
        .sort_values(["contact_model", "policy"])
    )
    mix.to_csv(resolve_project_path(args.mix_output), index=False)
    contacts.to_csv(resolve_project_path(args.contact_output), index=False)
    print(f"wrote physical link metrics {out} rows={len(df)}", flush=True)


def _sample_decode_uncertainty(tasks: list[Task], seed: int) -> list[int]:
    rng = np.random.default_rng(seed + 9000)
    return [max(1, int(round(t.decode_tokens * float(rng.lognormal(0.0, 0.55))))) for t in tasks]


def run_guardband_cap(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile_path = args.profile or cfg["profile_path"]
    if not resolve_project_path(profile_path).exists():
        write_default_profile(profile_path)
    profile = load_profile(profile_path)
    gammas = [float(v) for v in args.gammas.split(",")]
    caps = [None if v.strip().lower() in {"none", "uncapped"} else int(v) for v in args.decode_caps.split(",")]
    rows = []
    for seed in cfg["seeds"]:
        base_tasks = generate_tasks(cfg, int(seed))
        actual_decodes = _sample_decode_uncertainty(base_tasks, int(seed))
        actual_tasks = [replace(t, decode_tokens=d) for t, d in zip(base_tasks, actual_decodes)]
        windows = generate_windows(cfg)
        full_metrics, _ = run_policy("Heuristic-TVD", actual_tasks, windows, profile, cfg)
        full_value = float(full_metrics["total_value"])
        for gamma in gammas:
            for cap in caps:
                cfg_s = copy.deepcopy(cfg)
                cfg_s["satellite"]["memory_gb"] = float(cfg["satellite"]["memory_gb"]) * (1.0 - gamma)
                capped_decodes = [d if cap is None else min(d, cap) for d in actual_decodes]
                capped_tasks = [replace(t, decode_tokens=d) for t, d in zip(base_tasks, capped_decodes)]
                metrics, events = run_policy("Heuristic-TVD", capped_tasks, windows, profile, cfg_s)
                action_by_task = dict(zip(events["task_id"], events["action"]))
                value_by_task = dict(zip(events["task_id"], events["value"]))
                early = 0
                truncated_tokens = 0
                total_actual_tokens = 0
                value_loss_factor = 0.0
                truncation_value_loss = 0.0
                for task, actual_d, capped_d in zip(base_tasks, actual_decodes, capped_decodes):
                    action = action_by_task.get(task.task_id, "DOWN")
                    if action in {"ON", "PRE"}:
                        total_actual_tokens += actual_d
                        if capped_d < actual_d:
                            early += 1
                            truncated_tokens += actual_d - capped_d
                            retention = (capped_d / max(actual_d, 1)) ** args.value_loss_alpha
                            value_loss_factor += 1.0 - retention
                            truncation_value_loss += float(value_by_task.get(task.task_id, 0.0)) * (1.0 - retention)
                adjusted_value = max(0.0, float(metrics["total_value"]) - truncation_value_loss)
                value_loss = max(0.0, full_value - adjusted_value)
                metrics.update(
                    {
                        "seed": seed,
                        "gamma": gamma,
                        "decode_cap": "uncapped" if cap is None else cap,
                        "early_stop_count": early,
                        "early_stop_rate": early / max(int(metrics["on_count"]) + int(metrics["pre_count"]), 1),
                        "truncated_token_ratio": truncated_tokens / max(total_actual_tokens, 1),
                        "truncation_value_loss": truncation_value_loss,
                        "truncation_value_loss_pct": 100.0 * truncation_value_loss / max(float(metrics["total_value"]), 1e-9),
                        "adjusted_total_value": adjusted_value,
                        "value_loss": value_loss,
                        "value_loss_pct": 100.0 * value_loss / max(full_value, 1e-9),
                        "mean_retention_loss_for_stopped": value_loss_factor / max(early, 1),
                        "full_uncapped_value": full_value,
                    }
                )
                rows.append(metrics)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby("seed")["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    df["normalized_adjusted_value"] = df.groupby("seed")["adjusted_total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    out = resolve_project_path(args.output)
    df.to_csv(out, index=False)
    print(f"wrote guardband/decode-cap ablation {out} rows={len(df)}", flush=True)


def run_milp_granularity(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile_path = args.profile or cfg["profile_path"]
    if not resolve_project_path(profile_path).exists():
        write_default_profile(profile_path)
    profile = load_profile(profile_path)
    task_counts = [int(v) for v in args.task_counts.split(",")]
    steps = [int(v) for v in args.time_steps.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")] if args.seeds else [int(v) for v in cfg["seeds"]]
    rows = []
    for task_count in task_counts:
        for step_s in steps:
            for seed in seeds:
                cfg_s = copy.deepcopy(cfg)
                cfg_s["horizon_h"] = args.horizon_h
                cfg_s["milp"] = {**cfg_s.get("milp", {}), "max_tasks": task_count, "time_step_s": step_s, "timeout_s": args.timeout_s}
                tasks = generate_tasks(cfg_s, seed, horizon_h=args.horizon_h, limit=task_count)
                windows = generate_windows(cfg_s, horizon_h=args.horizon_h)
                grid_count = int(math.ceil((args.horizon_h * 3600.0 + 12 * 3600.0) / step_s)) + 1
                t0 = time.perf_counter()
                heur, _ = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
                milp = run_milp_bound(tasks, windows, profile, cfg_s)
                runtime = time.perf_counter() - t0
                rows.append(
                    {
                        "seed": seed,
                        "task_count": task_count,
                        "time_step_s": step_s,
                        "grid_count": grid_count,
                        "estimated_binary_vars": task_count * 3 * grid_count + task_count * 3,
                        "timeout_s": args.timeout_s,
                        "heuristic_value": float(heur["total_value"]),
                        "milp_ref_value": float(milp["total_value"]),
                        "milp_minus_heuristic_pct": 100.0 * (float(milp["total_value"]) - float(heur["total_value"])) / max(float(milp["total_value"]), 1e-9),
                        "heuristic_over_ref": int(float(heur["total_value"]) > float(milp["total_value"])),
                        "runtime_s": runtime,
                    }
                )
                pd.DataFrame(rows).to_csv(resolve_project_path(args.output), index=False)
                print(f"milp granularity task={task_count} step={step_s} seed={seed} runtime={runtime:.1f}s", flush=True)
    out = resolve_project_path(args.output)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote MILP granularity {out} rows={len(rows)}", flush=True)


EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def _class_prompt(name: str) -> str:
    spaced = {
        "AnnualCrop": "annual crop land",
        "HerbaceousVegetation": "herbaceous vegetation",
        "PermanentCrop": "permanent crop land",
        "SeaLake": "sea or lake",
    }.get(name, name.lower())
    return f"a satellite photo of {spaced}"


def _load_eurosat_labeled(zip_path: Path, limit: int, seed: int) -> tuple[list[Any], list[str]]:
    from PIL import Image
    import io

    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            parts = name.replace("\\", "/").split("/")
            label = next((p for p in parts if p in EUROSAT_CLASSES), None)
            if label:
                rows.append((name, label))
        if not rows:
            raise ValueError(f"no labeled EuroSAT images found in {zip_path}")
        pick = rng.choice(len(rows), size=min(limit, len(rows)), replace=False)
        images: list[Any] = []
        labels: list[str] = []
        for idx in pick:
            name, label = rows[int(idx)]
            with zf.open(name) as f:
                images.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
                labels.append(label)
    return images, labels


def _mask_image(image: Any, rho: float, selector: str, rng: np.random.Generator, grid: int = 8) -> Any:
    if rho >= 0.999:
        return image
    from PIL import Image

    arr = np.array(image).copy()
    h, w = arr.shape[:2]
    ph = max(1, h // grid)
    pw = max(1, w // grid)
    patches: list[tuple[float, int, int]] = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * ph, h if gy == grid - 1 else (gy + 1) * ph
            x0, x1 = gx * pw, w if gx == grid - 1 else (gx + 1) * pw
            patch = arr[y0:y1, x0:x1]
            score = float(np.var(patch)) if selector == "variance" else float(rng.random())
            patches.append((score, gy, gx))
    keep = max(1, int(round(len(patches) * rho)))
    selected = {(gy, gx) for _, gy, gx in sorted(patches, reverse=True)[:keep]}
    fill = np.array([127, 127, 127], dtype=arr.dtype)
    for _, gy, gx in patches:
        if (gy, gx) in selected:
            continue
        y0, y1 = gy * ph, h if gy == grid - 1 else (gy + 1) * ph
        x0, x1 = gx * pw, w if gx == grid - 1 else (gx + 1) * pw
        arr[y0:y1, x0:x1] = fill
    return Image.fromarray(arr)


def _clip_feature_tensor(output: Any) -> Any:
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (tuple, list)):
        return output[1] if len(output) > 1 and output[1] is not None else output[0]
    return output


def run_clip_pre_calibrate(args: argparse.Namespace) -> None:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    out = resolve_project_path(args.output)
    summary_out = resolve_project_path(args.summary_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    images, labels = _load_eurosat_labeled(resolve_project_path(args.eurosat_zip), args.samples, args.seed)
    prompts = [_class_prompt(c) for c in EUROSAT_CLASSES]
    label_to_idx = {c: i for i, c in enumerate(EUROSAT_CLASSES)}
    y_true = np.array([label_to_idx[x] for x in labels], dtype=int)
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = CLIPModel.from_pretrained(args.model, local_files_only=args.local_files_only)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = _clip_feature_tensor(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    idle_power_w = measure_idle_power(args.idle_sample_s)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    rhos = [float(v) for v in args.rhos.split(",")]
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    variants: list[tuple[str, float]] = [("full", 1.0)]
    for selector in selectors:
        for rho in rhos:
            if rho < 0.999:
                variants.append((selector, rho))

    full_acc = 0.0
    for selector, rho in variants:
        masked = [_mask_image(img, rho, selector, rng, grid=args.grid) for img in images]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        preds: list[int] = []
        with PowerSampler() as sampler:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            for start in range(0, len(masked), args.batch_size):
                batch = masked[start : start + args.batch_size]
                inputs = processor(images=batch, return_tensors="pt").to(device)
                with torch.no_grad():
                    image_features = _clip_feature_tensor(model.get_image_features(**inputs))
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits = image_features @ text_features.T
                preds.extend(logits.argmax(dim=-1).detach().cpu().numpy().tolist())
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
        energy_j, avg_power = sampler.energy_between(t0, t1)
        active_energy_j, active_power = sampler.energy_between(t0, t1, idle_power_w=idle_power_w)
        acc = float((np.array(preds, dtype=int) == y_true).mean())
        if selector == "full":
            full_acc = max(acc, 1e-9)
        eta = acc / max(full_acc, 1e-9)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        row = {
            "model": args.model,
            "selector": selector,
            "rho": rho,
            "samples": len(masked),
            "accuracy": acc,
            "eta": eta,
            "energy_j_per_image": energy_j / max(len(masked), 1),
            "active_j_per_image": active_energy_j / max(len(masked), 1),
            "images_per_sec": len(masked) / max(t1 - t0, 1e-9),
            "peak_mem_gb": peak_mem,
            "gpu_power_w": avg_power,
            "active_power_w": active_power,
            "idle_power_w": idle_power_w,
            "seed": args.seed,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"clip-pre selector={selector} rho={rho:g} acc={acc:.4f} eta={eta:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    df.to_csv(summary_out, index=False)
    print(f"wrote CLIP PRE calibration {out} rows={len(df)}", flush=True)


def _make_orin_profile(base: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = base.copy()
    rows["model"] = rows["model"].astype(str).str.replace("/root/autodl-tmp/models/", "Orin-literature/", regex=False)
    rows["tokens_per_sec"] = rows["tokens_per_sec"].astype(float) * float(args.throughput_scale)
    rows["prefill_j_per_tok"] = rows["prefill_j_per_tok"].astype(float) * float(args.energy_scale)
    rows["decode_j_per_tok"] = rows["decode_j_per_tok"].astype(float) * float(args.energy_scale)
    rows["gpu_power_w"] = float(args.power_w)
    rows["run_id"] = "orin_agx_literature_scaled_arya2025"
    return rows


def run_orin_portability(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    base = load_profile(args.profile or cfg["profile_path"])
    orin = _make_orin_profile(base, args)
    profile_out = resolve_project_path(args.profile_output)
    orin.to_csv(profile_out, index=False)
    policies = ["All-Downlink", "All-On-Board", "Fixed-Threshold", "Deadline-Energy", "Priority-Aware", "Heuristic-TVD"]
    rows = []
    for label, profile in [("RTX4090-measured", base), ("OrinAGX-literature", orin)]:
        for seed in cfg["seeds"]:
            tasks = generate_tasks(cfg, int(seed))
            windows = generate_windows(cfg)
            for policy in policies:
                metrics, _ = run_policy(policy, tasks, windows, profile, cfg)
                metrics.update(
                    {
                        "seed": seed,
                        "hardware_profile": label,
                        "throughput_scale": 1.0 if label.startswith("RTX") else float(args.throughput_scale),
                        "energy_scale": 1.0 if label.startswith("RTX") else float(args.energy_scale),
                        "profile_source": "measured" if label.startswith("RTX") else "literature-derived",
                    }
                )
                rows.append(metrics)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby(["seed", "hardware_profile"])["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    out = resolve_project_path(args.output)
    df.to_csv(out, index=False)
    mix = df.groupby(["hardware_profile", "policy"], as_index=False)[
        ["normalized_value", "energy_j", "median_latency_min", "p95_latency_min", "on_pct", "pre_pct", "down_pct", "drop_pct"]
    ].mean()
    mix.to_csv(resolve_project_path(args.mix_output), index=False)
    print(f"wrote Orin portability {out} rows={len(df)} profile={profile_out}", flush=True)


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if n == 1:
        return mean, 0.0
    tcrit = 2.262 if n == 10 else 1.96
    ci = float(tcrit * vals.std(ddof=1) / math.sqrt(n))
    return mean, ci


def run_statistics_ci(args: argparse.Namespace) -> None:
    df = pd.read_csv(resolve_project_path(args.metrics))
    metrics = [m.strip() for m in args.metrics_cols.split(",") if m.strip()]
    rows = []
    for policy, group in df.groupby("policy"):
        if args.exclude_milp and policy == "MILP-Ref":
            continue
        for metric in metrics:
            if metric not in group.columns:
                continue
            mean, ci = _mean_ci(group[metric])
            rows.append({"policy": policy, "metric": metric, "mean": mean, "ci95": ci, "n": int(group[metric].dropna().shape[0])})
    out = resolve_project_path(args.output)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote statistical CI summary {out} rows={len(rows)}", flush=True)


def run_decode_prior_sweep(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile = load_profile(args.profile or cfg["profile_path"])
    sigmas = [float(v) for v in args.sigmas.split(",")]
    rows = []
    for sigma in sigmas:
        for seed in cfg["seeds"]:
            rng = np.random.default_rng(int(seed) + 17000)
            tasks = generate_tasks(cfg, int(seed))
            actual_decodes = [max(1, int(round(t.decode_tokens * float(rng.lognormal(0.0, sigma))))) for t in tasks]
            actual_tasks = [replace(t, decode_tokens=d) for t, d in zip(tasks, actual_decodes)]
            expected = int(round(float(np.mean(cfg["tasks"]["decode_tokens"]))))
            expected_tasks = [replace(t, decode_tokens=expected) for t in tasks]
            noisy_decodes = [max(1, int(round(d * float(rng.lognormal(0.0, sigma / 2.0))))) for d in actual_decodes]
            noisy_tasks = [replace(t, decode_tokens=d) for t, d in zip(tasks, noisy_decodes)]
            windows = generate_windows(cfg)
            for label, decision_tasks in [("known", actual_tasks), ("expected-only", expected_tasks), ("noisy-prior", noisy_tasks)]:
                metrics, _ = run_policy("Heuristic-TVD", actual_tasks, windows, profile, cfg, decision_tasks=decision_tasks)
                metrics.update({"seed": seed, "decode_sigma": sigma, "scheduler_info": label})
                rows.append(metrics)
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby(["seed", "decode_sigma"])["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    out = resolve_project_path(args.output)
    df.to_csv(out, index=False)
    print(f"wrote decode prior sweep {out} rows={len(df)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reviewer-requested OrbitLLM supplemental experiments.")
    sub = parser.add_subparsers(dest="experiment", required=True)

    vlm = sub.add_parser("vlm-profile")
    vlm.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    vlm.add_argument("--scale-b", type=float, default=3.0)
    vlm.add_argument("--dtype", default="fp16")
    vlm.add_argument("--eurosat-zip", default="EuroSAT.zip")
    vlm.add_argument("--samples", type=int, default=80)
    vlm.add_argument("--decode-tokens", type=int, default=64)
    vlm.add_argument("--image-size", type=int, default=224)
    vlm.add_argument("--idle-sample-s", type=float, default=5.0)
    vlm.add_argument("--prefill-repeats", type=int, default=8)
    vlm.add_argument("--generate-repeats", type=int, default=4)
    vlm.add_argument("--seed", type=int, default=0)
    vlm.add_argument("--flush-every", type=int, default=5)
    vlm.add_argument("--prompt", default="Describe this remote-sensing image in one concise operational sentence.")
    vlm.add_argument("--output", default="results/profile_vlm_4090.csv")
    vlm.add_argument("--summary-output", default="results/profile_vlm_4090_summary.csv")
    vlm.set_defaults(func=run_vlm_profile)

    phy = sub.add_parser("physical-link")
    phy.add_argument("--config", default=None)
    phy.add_argument("--profile", default=None)
    phy.add_argument("--radio-circuit-w", type=float, default=6.0)
    phy.add_argument("--pa-static-w", type=float, default=2.0)
    phy.add_argument("--pa-peak-w", type=float, default=16.0)
    phy.add_argument("--pa-efficiency", type=float, default=0.35)
    phy.add_argument("--output", default="results/physical_link_metrics.csv")
    phy.add_argument("--mix-output", default="results/physical_link_action_mix.csv")
    phy.add_argument("--contact-output", default="results/physical_link_tle_contacts.csv")
    phy.set_defaults(func=run_physical_link)

    guard = sub.add_parser("guardband-cap")
    guard.add_argument("--config", default=None)
    guard.add_argument("--profile", default=None)
    guard.add_argument("--gammas", default="0,0.05,0.10,0.15,0.20")
    guard.add_argument("--decode-caps", default="64,128,256,512,uncapped")
    guard.add_argument("--value-loss-alpha", type=float, default=0.75)
    guard.add_argument("--output", default="results/guardband_decode_cap_ablation.csv")
    guard.set_defaults(func=run_guardband_cap)

    milp = sub.add_parser("milp-granularity")
    milp.add_argument("--config", default=None)
    milp.add_argument("--profile", default=None)
    milp.add_argument("--task-counts", default="20,40")
    milp.add_argument("--time-steps", default="1800,600,300")
    milp.add_argument("--seeds", default="0,1,2")
    milp.add_argument("--horizon-h", type=float, default=6.0)
    milp.add_argument("--timeout-s", type=int, default=60)
    milp.add_argument("--output", default="results/milp_granularity_metrics.csv")
    milp.set_defaults(func=run_milp_granularity)

    clip = sub.add_parser("clip-pre-calibrate")
    clip.add_argument("--model", default="models/clip-vit-base-patch32")
    clip.add_argument("--eurosat-zip", default="EuroSAT.zip")
    clip.add_argument("--samples", type=int, default=1000)
    clip.add_argument("--batch-size", type=int, default=32)
    clip.add_argument("--rhos", default="1.0,0.75,0.50,0.25")
    clip.add_argument("--selectors", default="random,variance")
    clip.add_argument("--grid", type=int, default=8)
    clip.add_argument("--idle-sample-s", type=float, default=5.0)
    clip.add_argument("--seed", type=int, default=0)
    clip.add_argument("--local-files-only", action="store_true")
    clip.add_argument("--output", default="results/vlm_pre_calibration_metrics.csv")
    clip.add_argument("--summary-output", default="results/vlm_pre_eta_summary.csv")
    clip.set_defaults(func=run_clip_pre_calibrate)

    orin = sub.add_parser("orin-portability")
    orin.add_argument("--config", default=None)
    orin.add_argument("--profile", default=None)
    orin.add_argument("--throughput-scale", type=float, default=0.20)
    orin.add_argument("--energy-scale", type=float, default=2.7)
    orin.add_argument("--power-w", type=float, default=45.0)
    orin.add_argument("--profile-output", default="results/profile_orin_literature.csv")
    orin.add_argument("--output", default="results/hardware_portability_metrics.csv")
    orin.add_argument("--mix-output", default="results/hardware_portability_action_mix.csv")
    orin.set_defaults(func=run_orin_portability)

    ci = sub.add_parser("statistics-ci")
    ci.add_argument("--metrics", default="results/simulation_metrics.csv")
    ci.add_argument("--metrics-cols", default="total_value,normalized_value,energy_j,median_latency_min,p95_latency_min,on_pct,pre_pct,down_pct,drop_pct")
    ci.add_argument("--exclude-milp", action="store_true")
    ci.add_argument("--output", default="results/statistical_ci_summary.csv")
    ci.set_defaults(func=run_statistics_ci)

    dec = sub.add_parser("decode-prior-sweep")
    dec.add_argument("--config", default=None)
    dec.add_argument("--profile", default=None)
    dec.add_argument("--sigmas", default="0,0.25,0.50,0.75,1.0")
    dec.add_argument("--output", default="results/decode_prior_sensitivity.csv")
    dec.set_defaults(func=run_decode_prior_sweep)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
