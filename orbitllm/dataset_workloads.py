from __future__ import annotations

import argparse
import copy
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, load_config, resolve_project_path
from .profile import load_profile
from .simulate import (
    SOTA_BASELINE_POLICIES,
    Task,
    generate_tle_windows,
    generate_windows,
    run_policy,
)


def _zip_key(image_path: str) -> str:
    normalized = image_path.replace("\\", "/")
    marker = "data/eurosat/"
    return normalized.split(marker, 1)[1] if marker in normalized else normalized.lstrip("/")


def load_eurosat_records(eurosat_zip: Path, pre_rows: Path) -> pd.DataFrame:
    calibration = pd.read_csv(pre_rows)
    sample = calibration[
        (calibration["backend"] == "mobile_sam")
        & (calibration["selector"] == "sam_top3")
    ].copy()
    if sample.empty:
        raise ValueError("MobileSAM sam_top3 rows are required for the EuroSAT workload")
    with zipfile.ZipFile(eurosat_zip) as archive:
        sizes = {
            entry.filename: entry.file_size
            for entry in archive.infolist()
            if entry.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        }
    sample["zip_key"] = sample["image"].map(_zip_key)
    sample["image_bytes"] = sample["zip_key"].map(sizes)
    sample = sample.dropna(subset=["image_bytes"]).copy()
    if sample.empty:
        raise ValueError("No MobileSAM-calibrated EuroSAT image was found in EuroSAT.zip")
    sample["image_bytes"] = sample["image_bytes"].astype(int)
    if "rho_bytes_vs_source" in sample.columns:
        sample["pre_payload_fraction"] = pd.to_numeric(
            sample["rho_bytes_vs_source"], errors="coerce"
        )
        sample["pre_payload_model"] = "measured-jpeg-bytes-vs-source"
    else:
        sample["pre_payload_fraction"] = pd.to_numeric(sample["rho"], errors="coerce")
        sample["pre_payload_model"] = "mask-pixel-rho-proxy"
    if "eta_prob" in sample.columns:
        sample["pre_utility_factor"] = pd.to_numeric(sample["eta_prob"], errors="coerce")
        sample["pre_utility_model"] = "per-image-clip-probability-retention"
    else:
        sample["pre_utility_factor"] = np.nan
        sample["pre_utility_model"] = "configured-global-factor"
    sample = sample.dropna(subset=["pre_payload_fraction"]).copy()
    sample["pre_payload_fraction"] = sample["pre_payload_fraction"].clip(lower=0.0, upper=1.0)
    sample["pre_utility_factor"] = sample["pre_utility_factor"].clip(lower=0.0, upper=1.0)
    return sample.reset_index(drop=True)


def build_eurosat_qwenvl_profile(summary_path: Path, output_path: Path) -> pd.DataFrame:
    """Convert the measured Qwen2.5-VL SkyScript summary into simulator schema."""
    source = pd.read_csv(summary_path)
    if source.empty:
        raise ValueError("Qwen2.5-VL profile summary is empty")
    row = source.iloc[0]
    profile = pd.DataFrame(
        [
            {
                "model": "Qwen2.5-VL-3B-Instruct (SkyScript-measured)",
                "scale_b": float(row["scale_b"]),
                "quant": str(row["quant"]),
                "context_len": int(round(float(row["context_len"]))),
                "prefill_tokens": int(round(float(row["prefill_tokens"]))),
                "decode_tokens": int(round(float(row["decode_tokens"]))),
                "prefill_j_per_tok": float(row["prefill_j_per_tok"]),
                "decode_j_per_tok": float(row["decode_j_per_tok"]),
                "tokens_per_sec": float(row["tokens_per_sec"]),
                "peak_mem_gb": float(row["peak_mem_gb"]),
                "gpu_power_w": float(row["gpu_power_w"]),
                "seed": int(row.get("seed", 0)),
                "run_id": str(row.get("run_id", "skyscript_qwenvl_summary")),
            }
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)
    return profile


def generate_eurosat_tasks(
    cfg: dict[str, Any],
    records: pd.DataFrame,
    seed: int,
    *,
    horizon_h: float | None = None,
    task_limit: int | None = None,
) -> tuple[list[Task], pd.DataFrame]:
    """Build a dataset-driven workload from measured EuroSAT PRE artifacts.

    Source image bytes come from the EuroSAT archive. New calibration files
    provide each PRE image's encoded-byte ratio and CLIP true-class probability
    retention. Older files remain supported through mask-pixel and configured
    global-value fallbacks.
    Arrival times and a common service-level half-life remain controlled inputs,
    because EuroSAT is not a timestamped satellite traffic trace.
    """
    settings = cfg.get("eurosat_workload", {})
    horizon_s = float(horizon_h if horizon_h is not None else cfg["horizon_h"]) * 3600.0
    count = int(task_limit if task_limit is not None else settings.get("task_count", 260))
    if count > len(records):
        raise ValueError(f"requested {count} EuroSAT tasks but only {len(records)} calibrated images are available")
    rng = np.random.default_rng(int(seed) + 27000)
    chosen = records.iloc[rng.choice(len(records), size=count, replace=False)].copy().reset_index(drop=True)
    chosen["arrival_s"] = np.sort(rng.uniform(0.0, horizon_s, size=count))
    chosen["seed"] = int(seed)
    chosen["half_life_s"] = float(settings.get("half_life_min", 60.0)) * 60.0
    chosen["value0"] = float(settings.get("value0", 1.0))
    chosen["prefill_tokens"] = int(settings.get("prefill_tokens", 1))
    chosen["decode_tokens"] = int(settings.get("decode_tokens", 25))
    fallback_pre_value = float(
        settings.get("pre_value_factor", cfg["satellite"]["pre_value_fraction"])
    )
    chosen["pre_value_factor"] = chosen["pre_utility_factor"].fillna(fallback_pre_value)
    tasks = [
        Task(
            task_id=idx,
            arrival_s=float(row.arrival_s),
            data_bits=float(row.image_bytes) * 8.0,
            scale_b=float(settings.get("scale_b", 3.0)),
            value0=float(row.value0),
            half_life_s=float(row.half_life_s),
            prefill_tokens=int(row.prefill_tokens),
            decode_tokens=int(row.decode_tokens),
            kind=f"EuroSAT:{row.label}",
            pre_retention_fraction=float(row.pre_payload_fraction),
            pre_value_factor=float(row.pre_value_factor),
            source_id=str(row.image),
        )
        for idx, row in chosen.iterrows()
    ]
    return tasks, chosen


def _train_dataset_learner(
    policy: str,
    cfg: dict[str, Any],
    profile: pd.DataFrame,
    records: pd.DataFrame,
    *,
    horizon_h: float,
    task_limit: int,
) -> dict[str, Any]:
    settings_key = "drl_ua" if policy == "DRL-UA-inspired" else "leo_drl"
    settings = cfg.get(settings_key, {})
    seeds = [int(v) for v in settings.get("training_seeds", [])]
    context: dict[str, Any] = {
        "training": True,
        "epsilon": float(settings.get("epsilon", 0.15)),
        "q_table": {},
        "rng": np.random.default_rng(20260715 if policy == "DRL-UA-inspired" else 20260712),
        "training_seeds": tuple(seeds),
    }
    for _ in range(max(1, int(settings.get("training_epochs", 4)))):
        for seed in seeds:
            tasks, _ = generate_eurosat_tasks(cfg, records, seed, horizon_h=horizon_h, task_limit=task_limit)
            windows = generate_windows(cfg, horizon_h=horizon_h)
            run_policy(policy, tasks, windows, profile, cfg, policy_context=context)
    context["training"] = False
    context["epsilon"] = 0.0
    return context


def run_eurosat_baseline_suite(
    cfg: dict[str, Any],
    profile: pd.DataFrame,
    records: pd.DataFrame,
    *,
    seeds: list[int] | None = None,
    horizon_h: float | None = None,
    task_limit: int | None = None,
    use_tle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all recent policies on a real-image-payload workload."""
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["satellite"]["quant"] = "FP16"
    settings = eval_cfg.get("eurosat_workload", {})
    pre_profile_path = settings.get("pre_profile_path")
    if pre_profile_path:
        pre_profile_rows = pd.read_csv(resolve_project_path(str(pre_profile_path)))
        if pre_profile_rows.empty:
            raise ValueError("EuroSAT PRE profile is empty")
        pre_row = pre_profile_rows.iloc[0]
        eval_cfg["satellite"]["pre_profile"] = {
            "backend": f"{pre_row['backend']} {pre_row['selector']} encoded-byte calibration",
            "source_csv": str(pre_profile_path),
            "active_energy_j_per_task": float(pre_row["pre_energy_j_per_image"]),
            "latency_s_per_task": float(pre_row["pre_latency_ms_per_image"]) / 1000.0,
            "peak_mem_gb": float(pre_row["peak_mem_gb"]),
        }
    if "energy_budget_j" in settings:
        eval_cfg["satellite"]["energy_budget_j"] = float(settings["energy_budget_j"])
    if "downlink_rate_mbps" in settings:
        eval_cfg["orbit"]["downlink_rate_mbps"] = float(settings["downlink_rate_mbps"])
    if "visible_min" in settings:
        eval_cfg["orbit"]["visible_min"] = float(settings["visible_min"])
    eval_horizon = float(horizon_h if horizon_h is not None else eval_cfg["horizon_h"])
    eval_limit = int(task_limit if task_limit is not None else settings.get("task_count", 260))
    test_seeds = [int(v) for v in (seeds if seeds is not None else eval_cfg["seeds"])]
    leo_context = _train_dataset_learner("LEO-DRL-inspired", eval_cfg, profile, records, horizon_h=eval_horizon, task_limit=eval_limit)
    ua_context = _train_dataset_learner("DRL-UA-inspired", eval_cfg, profile, records, horizon_h=eval_horizon, task_limit=eval_limit)
    shared_windows = generate_tle_windows(eval_cfg, horizon_h=eval_horizon)[0] if use_tle else generate_windows(eval_cfg, horizon_h=eval_horizon)
    rows: list[dict[str, Any]] = []
    trace_frames: list[pd.DataFrame] = []
    for seed in test_seeds:
        tasks, trace = generate_eurosat_tasks(eval_cfg, records, seed, horizon_h=eval_horizon, task_limit=eval_limit)
        trace_frames.append(trace)
        for policy in SOTA_BASELINE_POLICIES:
            context = None
            if policy == "LEO-DRL-inspired":
                context = {"training": False, "epsilon": 0.0, "q_table": leo_context["q_table"]}
            elif policy == "DRL-UA-inspired":
                context = {"training": False, "epsilon": 0.0, "q_table": ua_context["q_table"]}
            metrics, _ = run_policy(policy, tasks, shared_windows, profile, eval_cfg, policy_context=context)
            metrics.update(
                {
                    "seed": seed,
                    "workload": "EuroSAT-image-bytes+MobileSAM-encoded-bytes+per-image-eta",
                    "contact_model": "tle" if use_tle else "fixed-window",
                    "task_count": len(tasks),
                    "training_seed_count": len(leo_context["training_seeds"]),
                    "training_q_states": len(leo_context["q_table"]),
                    "ua_training_seed_count": len(ua_context["training_seeds"]),
                    "ua_training_q_states": len(ua_context["q_table"]),
                }
            )
            rows.append(metrics)
    result = pd.DataFrame(rows)
    result["normalized_value"] = result.groupby(["seed", "contact_model"])["total_value"].transform(
        lambda values: values / max(float(values.max()), 1e-9)
    )
    return result, pd.concat(trace_frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dataset-driven EuroSAT scheduling baselines.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--eurosat-zip", default="EuroSAT.zip")
    parser.add_argument("--pre-rows", default="results/mobilesam_pre_encoded_per_image.csv")
    parser.add_argument("--vlm-summary", default="results/skyscript_qwen_vl_profile_summary.csv")
    parser.add_argument("--profile-output", default="results/profile_eurosat_qwenvl.csv")
    parser.add_argument("--output", default="results/eurosat_baseline_metrics.csv")
    parser.add_argument("--trace-output", default="results/eurosat_dataset_trace.csv")
    parser.add_argument("--tle", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile = build_eurosat_qwenvl_profile(
        resolve_project_path(args.vlm_summary), resolve_project_path(args.profile_output)
    )
    records = load_eurosat_records(resolve_project_path(args.eurosat_zip), resolve_project_path(args.pre_rows))
    metrics, trace = run_eurosat_baseline_suite(cfg, profile, records, use_tle=args.tle)
    metrics.to_csv(resolve_project_path(args.output), index=False)
    trace.to_csv(resolve_project_path(args.trace_output), index=False)
    print(f"wrote {args.output} rows={len(metrics)} and {args.trace_output} rows={len(trace)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
