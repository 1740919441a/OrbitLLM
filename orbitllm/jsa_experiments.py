import argparse
import copy
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config, resolve_project_path
from .profile import load_profile
from .simulate import generate_tasks, generate_windows, run_policy


def _ci95(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def _summary(df: pd.DataFrame, groups: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_ci95"] = _ci95(g[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _action_mix(events: pd.DataFrame) -> dict[str, float]:
    counts = events["action"].value_counts(normalize=True) * 100.0 if not events.empty else pd.Series(dtype=float)
    return {
        "on_pct": float(counts.get("ON", 0.0)),
        "pre_pct": float(counts.get("PRE", 0.0)),
        "down_pct": float(counts.get("DOWN", 0.0)),
        "drop_pct": float(counts.get("DROP", 0.0)),
    }


def _load_cfg_profile(config: str | None, profile: str | None) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = load_config(config)
    profile_path = profile or cfg["profile_path"]
    return cfg, load_profile(profile_path)


def _select_quant(profile: pd.DataFrame, preferred: str) -> str:
    available = {str(q) for q in profile["quant"].dropna().unique()}
    for q in [preferred, "INT4", "INT8", "FP16"]:
        if q in available:
            return q
    if not available:
        raise ValueError("profile has no quant rows")
    return sorted(available)[0]


def _scaled_profile(base: pd.DataFrame, label: str, energy: float, latency: float, power_w: float | None = None, memory: float = 1.0) -> pd.DataFrame:
    p = base.copy()
    p["model"] = f"{label}/" + p["model"].astype(str).map(lambda x: x.split("/")[-1])
    p["prefill_j_per_tok"] = p["prefill_j_per_tok"].astype(float) * energy
    p["decode_j_per_tok"] = p["decode_j_per_tok"].astype(float) * energy
    p["tokens_per_sec"] = p["tokens_per_sec"].astype(float) / max(latency, 1e-9)
    p["peak_mem_gb"] = p["peak_mem_gb"].astype(float) * memory
    if power_w is not None:
        p["gpu_power_w"] = power_w
    else:
        p["gpu_power_w"] = p["gpu_power_w"].astype(float) * energy / max(latency, 1e-9)
    p["run_id"] = f"jsa_scaled_{label}"
    p["seed"] = 0
    return p


def run_profile_replaceability(args: argparse.Namespace) -> None:
    cfg, base = _load_cfg_profile(args.config, args.profile)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    policies = ["Deadline-Energy", "Priority-Aware", "Energy-WSRPT", "Heuristic-TVD"]

    profiles: list[tuple[str, str, pd.DataFrame, dict[str, Any]]] = [
        ("RTX4090 measured", "measured", base, {}),
        ("Edge-NPU synthetic", "scaled", _scaled_profile(base, "edge_npu", energy=0.45, latency=0.8, power_w=35.0), {"memory_gb": 12}),
        ("FPGA-low-power synthetic", "scaled", _scaled_profile(base, "fpga_lp", energy=0.35, latency=2.8, power_w=18.0), {"memory_gb": 8}),
    ]

    orin_path = resolve_project_path("results/profile_orin_literature.csv")
    if orin_path.exists():
        profiles.append(("Orin AGX literature", "literature-derived", pd.read_csv(orin_path), {}))
    proxy_path = resolve_project_path("results/profile_local_4090_60w_qwen_1p7b_3b.csv")
    if proxy_path.exists():
        profiles.append(("60W mobile proxy", "derived-proxy", pd.read_csv(proxy_path), {}))

    rows: list[dict[str, Any]] = []
    for label, source, profile, overrides in profiles:
        for seed in seeds:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["seeds"] = [seed]
            cfg_s.setdefault("satellite", {}).update(overrides)
            cfg_s["satellite"]["quant"] = _select_quant(profile, str(cfg_s["satellite"].get("quant", "INT4")))
            tasks = generate_tasks(cfg_s, seed)
            windows = generate_windows(cfg_s)
            for policy in policies:
                metrics, events = run_policy(policy, tasks, windows, profile, cfg_s)
                rows.append({
                    **metrics,
                    **_action_mix(events),
                    "seed": seed,
                    "profile_label": label,
                    "profile_source": source,
                    "memory_gb": cfg_s["satellite"]["memory_gb"],
                    "quant_used": cfg_s["satellite"]["quant"],
                })

    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby(["seed", "profile_label"])["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = _summary(df, ["profile_label", "profile_source", "policy"], ["normalized_value", "energy_j", "median_latency_min", "mean_decision_ms", "on_pct", "pre_pct", "down_pct", "drop_pct"])
    summary_out = resolve_project_path(args.summary_output)
    summary.to_csv(summary_out, index=False)

    fig_out = resolve_project_path(args.figure_output)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    h = summary[summary["policy"] == "Heuristic-TVD"].copy()
    h = h.sort_values("normalized_value_mean", ascending=False)
    labels = h["profile_label"].tolist()
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    colors = {"on_pct_mean": "#D8902F", "pre_pct_mean": "#2B9C69", "down_pct_mean": "#3E79B8", "drop_pct_mean": "#8C8C8C"}
    names = {"on_pct_mean": "ON", "pre_pct_mean": "PRE", "down_pct_mean": "DOWN", "drop_pct_mean": "DROP"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for col in ["on_pct_mean", "pre_pct_mean", "down_pct_mean", "drop_pct_mean"]:
        vals = h[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottoms, color=colors[col], label=names[col])
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Action mix (%)")
    ax.set_title("Profile-replaceable scheduler response")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}, {summary_out}, {fig_out}")


def run_scalability(args: argparse.Namespace) -> None:
    cfg, profile = _load_cfg_profile(args.config, args.profile)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    task_counts = [int(x) for x in args.task_counts.split(",") if x.strip()]
    window_counts = [int(x) for x in args.window_counts.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    for tasks_n in task_counts:
        for windows_n in window_counts:
            for seed in seeds:
                cfg_s = copy.deepcopy(cfg)
                cfg_s["seeds"] = [seed]
                cfg_s["horizon_h"] = float(args.horizon_h)
                cfg_s["tasks"]["count_per_24h"] = tasks_n
                cfg_s["orbit"]["period_min"] = max(1.0, (float(args.horizon_h) * 60.0) / windows_n)
                cfg_s["orbit"]["visible_min"] = max(0.1, min(float(args.window_duration_s) / 60.0, cfg_s["orbit"]["period_min"] * 0.8))
                tasks = generate_tasks(cfg_s, seed)
                windows = generate_windows(cfg_s)
                t0 = time.perf_counter()
                metrics, events = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s)
                wall_s = time.perf_counter() - t0
                rows.append({
                    **metrics,
                    **_action_mix(events),
                    "seed": seed,
                    "tasks": tasks_n,
                    "windows": len(windows),
                    "requested_windows": windows_n,
                    "wall_s": wall_s,
                    "decisions_per_s": tasks_n / max(wall_s, 1e-9),
                })

    df = pd.DataFrame(rows)
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = _summary(df, ["tasks", "requested_windows"], ["mean_decision_ms", "wall_s", "decisions_per_s", "total_value", "energy_j"])
    summary_out = resolve_project_path(args.summary_output)
    summary.to_csv(summary_out, index=False)

    fig_out = resolve_project_path(args.figure_output)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for tasks_n, g in summary.groupby("tasks"):
        g = g.sort_values("requested_windows")
        ax.plot(g["requested_windows"], g["mean_decision_ms_mean"] * 1000.0, marker="o", label=f"{tasks_n} tasks")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Fragmented contact windows")
    ax.set_ylabel("Mean decision time (us)")
    ax.set_title("Heuristic-TVD online scalability")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}, {summary_out}, {fig_out}")


def _active_profile(base: pd.DataFrame) -> pd.DataFrame:
    phase_path = resolve_project_path("results/profile_power_phases.csv")
    if not phase_path.exists():
        return _scaled_profile(base, "active_proxy", energy=0.55, latency=1.0, power_w=None)
    phase = pd.read_csv(phase_path)
    p = phase.copy()
    p["prefill_j_per_tok"] = p["prefill_active_j_per_tok"].astype(float)
    p["decode_j_per_tok"] = p["decode_active_j_per_tok"].astype(float)
    p["run_id"] = "active_only_profile_from_phase_measurement"
    return p[base.columns]


def run_profile_ablation(args: argparse.Namespace) -> None:
    cfg, base = _load_cfg_profile(args.config, args.profile)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variants: list[tuple[str, pd.DataFrame, bool, bool]] = [
        ("Full profile", base, True, True),
        ("Merged prefill/decode cost", base, True, False),
        ("No PRE action", base, False, True),
        ("Active-only energy profile", _active_profile(base), True, True),
    ]
    rows: list[dict[str, Any]] = []
    for label, profile, allow_pre, split_energy in variants:
        for seed in seeds:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["seeds"] = [seed]
            tasks = generate_tasks(cfg_s, seed)
            windows = generate_windows(cfg_s)
            metrics, events = run_policy("Heuristic-TVD", tasks, windows, profile, cfg_s, allow_pre=allow_pre, split_energy=split_energy)
            rows.append({
                **metrics,
                **_action_mix(events),
                "seed": seed,
                "variant": label,
                "allow_pre": allow_pre,
                "split_energy": split_energy,
            })
    df = pd.DataFrame(rows)
    df["normalized_value"] = df.groupby("seed")["total_value"].transform(lambda s: s / max(float(s.max()), 1e-9))
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = _summary(df, ["variant"], ["normalized_value", "energy_j", "median_latency_min", "mean_decision_ms", "on_pct", "pre_pct", "down_pct"])
    summary_out = resolve_project_path(args.summary_output)
    summary.to_csv(summary_out, index=False)

    fig_out = resolve_project_path(args.figure_output)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    order = ["Full profile", "Merged prefill/decode cost", "No PRE action", "Active-only energy profile"]
    s = summary.set_index("variant").loc[[x for x in order if x in summary["variant"].values]]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.bar(np.arange(len(s)), s["normalized_value_mean"] * 100.0, color=["#22577A", "#D8902F", "#A23E48", "#2B9C69"][: len(s)])
    ax.errorbar(np.arange(len(s)), s["normalized_value_mean"] * 100.0, yerr=s["normalized_value_ci95"] * 100.0, fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax.set_xticks(np.arange(len(s)))
    ax.set_xticklabels(s.index, rotation=16, ha="right")
    ax.set_ylabel("Normalized value (%)")
    ax.set_title("Profile-field ablation")
    ax.set_ylim(0, max(105, float((s["normalized_value_mean"] * 100.0).max()) + 8))
    fig.tight_layout()
    fig.savefig(fig_out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}, {summary_out}, {fig_out}")


def write_latex(args: argparse.Namespace) -> None:
    repl = pd.read_csv(resolve_project_path(args.replaceability_summary))
    scal = pd.read_csv(resolve_project_path(args.scalability_summary))
    abl = pd.read_csv(resolve_project_path(args.ablation_summary))
    remote_profile_path = resolve_project_path("results/jsa_remote_serving_profile.csv")
    out = resolve_project_path(args.output)

    heur = repl[repl["policy"] == "Heuristic-TVD"].copy()
    order = ["RTX4090 measured", "Orin AGX literature", "60W mobile proxy", "Edge-NPU synthetic", "FPGA-low-power synthetic"]
    heur["order"] = heur["profile_label"].map({k: i for i, k in enumerate(order)}).fillna(99)
    heur = heur.sort_values("order")
    replace_rows = []
    for _, r in heur.iterrows():
        mix = f"{r['on_pct_mean']:.1f}/{r['pre_pct_mean']:.1f}/{r['down_pct_mean']:.1f}/{r['drop_pct_mean']:.1f}"
        replace_rows.append(f"{r['profile_label']} & {r['normalized_value_mean']*100:.1f} & {r['energy_j_mean']/1000:.1f} & {r['median_latency_min_mean']:.1f} & {mix}\\\\")

    scal_rows = []
    selected = scal.sort_values(["tasks", "requested_windows"])
    for _, r in selected.iterrows():
        scal_rows.append(f"{int(r['tasks'])} & {int(r['requested_windows'])} & {r['mean_decision_ms_mean']*1000:.1f} & {r['wall_s_mean']:.2f} & {r['decisions_per_s_mean']:.0f}\\\\")

    abl_order = ["Full profile", "Merged prefill/decode cost", "No PRE action", "Active-only energy profile"]
    abl = abl.set_index("variant").loc[[x for x in abl_order if x in abl["variant"].values]].reset_index()
    abl_rows = []
    for _, r in abl.iterrows():
        mix = f"{r['on_pct_mean']:.1f}/{r['pre_pct_mean']:.1f}/{r['down_pct_mean']:.1f}"
        abl_rows.append(f"{r['variant']} & {r['normalized_value_mean']*100:.1f}$\\pm${r['normalized_value_ci95']*100:.1f} & {r['energy_j_mean']/1000:.1f} & {mix}\\\\")

    remote_rows: list[str] = []
    if remote_profile_path.exists():
        remote = pd.read_csv(remote_profile_path)
        for _, r in remote.sort_values("context_len").iterrows():
            remote_rows.append(
                f"{int(r['context_len'])} & {float(r['prefill_j_per_tok'])*1000:.2f} & "
                f"{float(r['decode_j_per_tok']):.2f} & {float(r['tokens_per_sec']):.1f} & "
                f"{float(r['peak_mem_gb']):.2f} & {float(r['idle_power_w']):.1f}\\\\"
            )
    if not remote_rows:
        remote_rows.append("\\multicolumn{6}{c}{Not run}\\\\")

    text = (
        "% Auto-generated by orbitllm.jsa_experiments; do not edit numbers manually.\n"
        "\\newcommand{\\JsaProfileReplaceRows}{%\n" + "\n".join(replace_rows) + "\n}\n\n"
        "\\newcommand{\\JsaScalabilityRows}{%\n" + "\n".join(scal_rows) + "\n}\n\n"
        "\\newcommand{\\JsaProfileAblationRows}{%\n" + "\n".join(abl_rows) + "\n}\n"
        "\\newcommand{\\JsaRemoteServingRows}{%\n" + "\n".join(remote_rows) + "\n}\n"
    )
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")


def run_all(args: argparse.Namespace) -> None:
    run_profile_replaceability(args)
    run_scalability(args)
    run_profile_ablation(args)
    write_latex(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="JSA-targeted OrbitLLM supplemental experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None)
        p.add_argument("--profile", default=None)
        p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")

    repl = sub.add_parser("profile-replaceability")
    common(repl)
    repl.add_argument("--output", default="results/jsa_profile_replaceability_metrics.csv")
    repl.add_argument("--summary-output", default="results/jsa_profile_replaceability_summary.csv")
    repl.add_argument("--figure-output", default="figures/jsa_profile_replaceability.png")
    repl.set_defaults(func=run_profile_replaceability)

    scal = sub.add_parser("scalability")
    common(scal)
    scal.add_argument("--task-counts", default="260,1000,2500,5000")
    scal.add_argument("--window-counts", default="16,64,256,1024")
    scal.add_argument("--horizon-h", type=float, default=24.0)
    scal.add_argument("--window-duration-s", type=float, default=20.0)
    scal.add_argument("--output", default="results/jsa_scheduler_scalability_metrics.csv")
    scal.add_argument("--summary-output", default="results/jsa_scheduler_scalability_summary.csv")
    scal.add_argument("--figure-output", default="figures/jsa_scheduler_scalability.png")
    scal.set_defaults(func=run_scalability)

    abl = sub.add_parser("profile-ablation")
    common(abl)
    abl.add_argument("--output", default="results/jsa_profile_ablation_metrics.csv")
    abl.add_argument("--summary-output", default="results/jsa_profile_ablation_summary.csv")
    abl.add_argument("--figure-output", default="figures/jsa_profile_ablation.png")
    abl.set_defaults(func=run_profile_ablation)

    latex = sub.add_parser("write-latex")
    latex.add_argument("--replaceability-summary", default="results/jsa_profile_replaceability_summary.csv")
    latex.add_argument("--scalability-summary", default="results/jsa_scheduler_scalability_summary.csv")
    latex.add_argument("--ablation-summary", default="results/jsa_profile_ablation_summary.csv")
    latex.add_argument("--output", default="results/jsa_generated_tables.tex")
    latex.set_defaults(func=write_latex)

    allp = sub.add_parser("all")
    common(allp)
    allp.add_argument("--task-counts", default="260,1000,2500,5000")
    allp.add_argument("--window-counts", default="16,64,256,1024")
    allp.add_argument("--horizon-h", type=float, default=24.0)
    allp.add_argument("--window-duration-s", type=float, default=20.0)
    allp.add_argument("--output", default="results/jsa_profile_replaceability_metrics.csv")
    allp.add_argument("--summary-output", default="results/jsa_profile_replaceability_summary.csv")
    allp.add_argument("--figure-output", default="figures/jsa_profile_replaceability.png")
    allp.add_argument("--replaceability-summary", default="results/jsa_profile_replaceability_summary.csv")
    allp.add_argument("--scalability-summary", default="results/jsa_scheduler_scalability_summary.csv")
    allp.add_argument("--ablation-summary", default="results/jsa_profile_ablation_summary.csv")
    allp.add_argument("--ablation-output", default="results/jsa_profile_ablation_metrics.csv")
    allp.add_argument("--ablation-summary-output", default="results/jsa_profile_ablation_summary.csv")
    allp.add_argument("--ablation-figure-output", default="figures/jsa_profile_ablation.png")
    allp.add_argument("--scalability-output", default="results/jsa_scheduler_scalability_metrics.csv")
    allp.add_argument("--scalability-summary-output", default="results/jsa_scheduler_scalability_summary.csv")
    allp.add_argument("--scalability-figure-output", default="figures/jsa_scheduler_scalability.png")
    allp.add_argument("--latex-output", default="results/jsa_generated_tables.tex")

    def run_all_adapter(ns: argparse.Namespace) -> None:
        ns.output = ns.output
        ns.summary_output = ns.summary_output
        ns.figure_output = ns.figure_output
        run_profile_replaceability(ns)
        ns.output = ns.scalability_output
        ns.summary_output = ns.scalability_summary_output
        ns.figure_output = ns.scalability_figure_output
        run_scalability(ns)
        ns.output = ns.ablation_output
        ns.summary_output = ns.ablation_summary_output
        ns.figure_output = ns.ablation_figure_output
        run_profile_ablation(ns)
        ns.output = ns.latex_output
        write_latex(ns)

    allp.set_defaults(func=run_all_adapter)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
