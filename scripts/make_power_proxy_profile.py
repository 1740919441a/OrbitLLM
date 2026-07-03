from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a conservative power-capped proxy profile from a measured GPU profile.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-power-w", type=float, default=60.0)
    parser.add_argument("--scales", default="1.7,3.0")
    parser.add_argument("--quant", default="FP16")
    parser.add_argument("--contexts", default="512,1024,2048")
    parser.add_argument("--energy-guard", type=float, default=1.10)
    args = parser.parse_args()

    scales = {float(v) for v in args.scales.split(",")}
    contexts = {int(v) for v in args.contexts.split(",")}
    df = pd.read_csv(args.input)
    df = df[(df["scale_b"].astype(float).isin(scales)) & (df["quant"].str.upper() == args.quant.upper()) & (df["context_len"].astype(int).isin(contexts))].copy()
    if df.empty:
        raise SystemExit("no matching rows for proxy profile")

    keys = ["model", "scale_b", "quant", "context_len", "prefill_tokens", "decode_tokens"]
    numeric = ["prefill_j_per_tok", "decode_j_per_tok", "tokens_per_sec", "peak_mem_gb", "gpu_power_w"]
    df = df.groupby(keys, as_index=False)[numeric].mean()

    source_power = df["gpu_power_w"].clip(lower=args.target_power_w)
    speed_scale = (args.target_power_w / source_power).clip(upper=1.0)
    # Conservative proxy: lower power reduces throughput approximately linearly,
    # while per-token energy is kept slightly worse than the measured anchor.
    df["tokens_per_sec"] = df["tokens_per_sec"] * speed_scale
    df["prefill_j_per_tok"] = df["prefill_j_per_tok"] * args.energy_guard
    df["decode_j_per_tok"] = df["decode_j_per_tok"] * args.energy_guard
    df["gpu_power_w"] = args.target_power_w
    df["run_id"] = f"power_proxy_{args.target_power_w:g}w_from_4090_energy_guard_{args.energy_guard:g}"
    df["seed"] = 0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} rows={len(df)}")
    print(df[["scale_b", "quant", "context_len", "tokens_per_sec", "prefill_j_per_tok", "decode_j_per_tok", "peak_mem_gb", "gpu_power_w"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
