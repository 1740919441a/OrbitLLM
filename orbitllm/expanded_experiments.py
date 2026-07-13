from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ensure_output_dirs, load_config, resolve_project_path
from .dataset_workloads import (
    build_eurosat_qwenvl_profile,
    load_eurosat_records,
    run_eurosat_baseline_suite,
)
from .expanded_reporting import main as write_expanded_reporting
from .profile import load_profile, write_default_profile
from .reviewer_experiments import run_sota_physical_link
from .simulate import (
    run_sota_baseline_suite,
    run_sota_baseline_timeseries,
    run_sota_oracle_comparison,
)
from .sota_reporting import main as write_sota_figures


def _write_csv(frame: pd.DataFrame, relative_path: str) -> Path:
    output = resolve_project_path(relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {relative_path} rows={len(frame)}", flush=True)
    return output


def run_all(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    profile_path = args.profile or cfg["profile_path"]
    if not resolve_project_path(profile_path).exists():
        write_default_profile(profile_path)
    profile = load_profile(profile_path)

    smoke = run_sota_baseline_suite(
        cfg,
        profile,
        seeds=[0, 1],
        horizon_h=6.0,
        task_limit=60,
    )
    _write_csv(smoke, "results/sota_baseline_expanded_smoke_metrics.csv")

    main = run_sota_baseline_suite(cfg, profile)
    _write_csv(main, "results/sota_baseline_expanded_main_metrics.csv")
    _write_csv(
        run_sota_baseline_timeseries(cfg, profile),
        "results/sota_baseline_expanded_timeseries.csv",
    )

    no_pre = run_sota_baseline_suite(cfg, profile, allow_pre=False)
    no_pre["scenario"] = "no-pre"
    tle = run_sota_baseline_suite(cfg, profile, use_tle=True)
    tle["scenario"] = "tle"
    _write_csv(
        pd.concat([no_pre, tle], ignore_index=True),
        "results/sota_baseline_expanded_robustness_metrics.csv",
    )

    oracle = run_sota_oracle_comparison(cfg, profile, horizon_h=3.0, task_limit=8)
    _write_csv(oracle, "results/sota_baseline_expanded_oracle_metrics.csv")

    if not args.skip_physical:
        run_sota_physical_link(
            argparse.Namespace(
                config=args.config,
                profile=profile_path,
                radio_circuit_w=6.0,
                pa_static_w=2.0,
                pa_peak_w=16.0,
                pa_efficiency=0.35,
                output="results/sota_baseline_expanded_physical_link_metrics.csv",
                summary_output="results/sota_baseline_expanded_physical_link_summary.csv",
                contact_output="results/sota_baseline_expanded_physical_link_contacts.csv",
            )
        )

    vlm_profile = build_eurosat_qwenvl_profile(
        resolve_project_path(args.vlm_summary),
        resolve_project_path("results/profile_eurosat_qwenvl.csv"),
    )
    records = load_eurosat_records(
        resolve_project_path(args.eurosat_zip),
        resolve_project_path(args.pre_rows),
    )
    eurosat_main, eurosat_trace = run_eurosat_baseline_suite(cfg, vlm_profile, records)
    _write_csv(eurosat_main, "results/eurosat_baseline_main_metrics.csv")
    _write_csv(eurosat_trace, "results/eurosat_dataset_trace.csv")
    eurosat_tle, eurosat_tle_trace = run_eurosat_baseline_suite(
        cfg,
        vlm_profile,
        records,
        use_tle=True,
    )
    _write_csv(eurosat_tle, "results/eurosat_baseline_tle_metrics.csv")
    _write_csv(eurosat_tle_trace, "results/eurosat_dataset_tle_trace.csv")

    write_expanded_reporting()
    write_sota_figures()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the expanded recent-baseline and EuroSAT experiment package."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--eurosat-zip", default="EuroSAT.zip")
    parser.add_argument("--pre-rows", default="results/mobilesam_pre_encoded_per_image.csv")
    parser.add_argument(
        "--vlm-summary",
        default="results/skyscript_qwen_vl_profile_summary.csv",
    )
    parser.add_argument(
        "--skip-physical",
        action="store_true",
        help="Skip the TLE/elevation/RF physical-link run.",
    )
    args = parser.parse_args(argv)
    run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
