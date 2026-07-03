from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path
from typing import Any

from .config import load_config, resolve_project_path
from .profile import PROFILE_COLUMNS


POWER_PHASE_COLUMNS = [
    "idle_power_w",
    "prefill_wall_j_per_tok",
    "decode_wall_j_per_tok",
    "prefill_active_j_per_tok",
    "decode_active_j_per_tok",
    "prefill_avg_power_w",
    "decode_avg_power_w",
]


class PowerSampler:
    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any = None
        self._handle: Any = None

    def __enter__(self) -> "PowerSampler":
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._handle = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            power = 0.0
            if self._nvml is not None and self._handle is not None:
                try:
                    power = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
                except Exception:
                    power = 0.0
            self.samples.append((time.perf_counter(), power))
            time.sleep(self.interval_s)

    def energy_between(self, start: float, end: float, idle_power_w: float = 0.0) -> tuple[float, float]:
        points = [(t, p) for t, p in self.samples if start <= t <= end and p > 0]
        if len(points) < 2:
            return 0.0, 0.0
        energy = 0.0
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            p0_adj = max(0.0, p0 - idle_power_w)
            p1_adj = max(0.0, p1 - idle_power_w)
            energy += (t1 - t0) * (p0_adj + p1_adj) / 2.0
        avg_power = energy / max(end - start, 1e-9)
        return energy, avg_power


def measure_idle_power(duration_s: float = 5.0, interval_s: float = 0.05) -> float:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return 0.0
    values: list[float] = []
    end = time.perf_counter() + duration_s
    while time.perf_counter() < end:
        try:
            values.append(float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0)
        except Exception:
            pass
        time.sleep(interval_s)
    return float(sum(values) / max(len(values), 1))


def load_model(model_name: str, quant: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if quant.upper() == "FP16":
        kwargs["torch_dtype"] = torch.float16
    elif quant.upper() in {"INT8", "INT4"}:
        from transformers import BitsAndBytesConfig

        if quant.upper() == "INT8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return tokenizer, model


def make_prompt(tokenizer: Any, context_len: int) -> Any:
    import torch

    base = "Analyze this remote-sensing observation and produce a concise operational report. "
    text = base * max(8, context_len // 12)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_len)
    if encoded["input_ids"].shape[1] < context_len:
        pad = encoded["input_ids"][0, -1:].repeat(1, context_len - encoded["input_ids"].shape[1])
        encoded["input_ids"] = torch.cat([encoded["input_ids"], pad], dim=1)
        mask_pad = torch.ones_like(pad)
        encoded["attention_mask"] = torch.cat([encoded["attention_mask"], mask_pad], dim=1)
    return encoded


def run_one(
    tokenizer: Any,
    model: Any,
    model_name: str,
    scale_b: float,
    quant: str,
    context_len: int,
    decode_tokens: int,
    prefill_repeats: int,
    seed: int,
    idle_power_w: float,
) -> dict[str, object]:
    import torch

    encoded = make_prompt(tokenizer, context_len)
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(**encoded, use_cache=True)
    with PowerSampler() as sampler:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = None
            for _ in range(max(prefill_repeats, 1)):
                out = model(**encoded, use_cache=True)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        if out is None:
            raise RuntimeError("prefill benchmark produced no output")
        next_token = out.logits[:, -1:].argmax(dim=-1)
        past = out.past_key_values
        decode_start = time.perf_counter()
        for _ in range(decode_tokens):
            with torch.no_grad():
                out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.perf_counter()
    prefill_energy, avg_power_prefill = sampler.energy_between(t0, t1)
    decode_energy, avg_power_decode = sampler.energy_between(decode_start, t2)
    prefill_active_energy, active_power_prefill = sampler.energy_between(t0, t1, idle_power_w=idle_power_w)
    decode_active_energy, active_power_decode = sampler.energy_between(decode_start, t2, idle_power_w=idle_power_w)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    prefill_work_tokens = max(context_len * max(prefill_repeats, 1), 1)
    prefill_j = prefill_energy / prefill_work_tokens if prefill_energy > 0 else avg_power_prefill * (t1 - t0) / prefill_work_tokens
    decode_j = decode_energy / max(decode_tokens, 1) if decode_energy > 0 else avg_power_decode * (t2 - decode_start) / max(decode_tokens, 1)
    prefill_active_j = prefill_active_energy / prefill_work_tokens if prefill_active_energy > 0 else active_power_prefill * (t1 - t0) / prefill_work_tokens
    decode_active_j = decode_active_energy / max(decode_tokens, 1) if decode_active_energy > 0 else active_power_decode * (t2 - decode_start) / max(decode_tokens, 1)
    return {
        "model": model_name,
        "scale_b": scale_b,
        "quant": quant.upper(),
        "context_len": context_len,
        "prefill_tokens": context_len,
        "decode_tokens": decode_tokens,
        "prefill_j_per_tok": round(float(prefill_j), 8),
        "decode_j_per_tok": round(float(decode_j), 8),
        "tokens_per_sec": round(float(decode_tokens / max(t2 - decode_start, 1e-9)), 4),
        "peak_mem_gb": round(float(peak_mem), 4),
        "gpu_power_w": round(float(max(avg_power_prefill, avg_power_decode)), 3),
        "idle_power_w": round(float(idle_power_w), 3),
        "prefill_wall_j_per_tok": round(float(prefill_j), 8),
        "decode_wall_j_per_tok": round(float(decode_j), 8),
        "prefill_active_j_per_tok": round(float(prefill_active_j), 8),
        "decode_active_j_per_tok": round(float(decode_active_j), 8),
        "prefill_avg_power_w": round(float(avg_power_prefill), 3),
        "decode_avg_power_w": round(float(avg_power_decode), 3),
        "seed": seed,
        "run_id": f"remote_4090_{int(time.time())}",
    }


def run_model_quant(spec: dict[str, Any], quant: str, contexts: list[int], decode_tokens: int, prefill_repeats: int, runs: int, idle_sample_s: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tokenizer, model = load_model(spec["model"], quant)
    idle_power_w = measure_idle_power(idle_sample_s)
    print(f"idle_power {spec['model']} {quant}: {idle_power_w:.3f} W", flush=True)
    for context_len in contexts:
        for seed in range(runs):
            rows.append(run_one(tokenizer, model, spec["model"], float(spec["scale_b"]), quant, int(context_len), decode_tokens, prefill_repeats, seed, idle_power_w))
            print(f"benchmarked {spec['model']} {quant} ctx={context_len} run={seed}", flush=True)
    del model
    del tokenizer
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark LLM profile anchors on a GPU host.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default="results/profile_remote.csv")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--quants", nargs="*", default=None)
    parser.add_argument("--contexts", nargs="*", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    bench = cfg["benchmark"]
    model_specs = bench["models"]
    if args.models:
        model_specs = [m for m in model_specs if m["model"] in set(args.models)]
    quants = args.quants or bench["quants"]
    contexts = args.contexts or bench["context_lengths"]
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for spec in model_specs:
        for quant in quants:
            rows.extend(
                run_model_quant(
                    spec,
                    quant,
                    [int(c) for c in contexts],
                    int(bench["decode_tokens"]),
                    int(bench.get("prefill_repeats", 8)),
                    int(bench["runs"]),
                    float(bench.get("idle_sample_s", 5.0)),
                )
            )
            with out.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=PROFILE_COLUMNS + POWER_PHASE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
