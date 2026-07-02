# OrbitLLM Artifact Guide

This artifact contains the code and data needed to reproduce the OrbitLLM workshop results.

## Code Layout

- `orbitllm/benchmark.py`: RTX 4090 LLM profiling CLI.
- `orbitllm/simulate.py`: task generation, policies, MILP bound, and all sensitivity sweeps.
- `orbitllm/plot.py`: figure and LaTeX table generation from CSV outputs.
- `orbitllm/profile.py`: measured-profile loading and aggregation.
- `orbitllm/cli.py`: unified command line interface.
- `configs/default.json`: default experiment configuration.
- `tests/test_simulation.py`: unit tests for value decay, costs, bandwidth, and constraints.

## Result Layout

- `results/profile_remote_fp16.csv`, `results/profile_remote_int8.csv`, `results/profile_remote_int4.csv`: remote RTX 4090 benchmark outputs.
- `results/profile_anchor.csv`: aggregated measured 4090 profile used by the simulator.
- `results/simulation_metrics.csv`: main 24-hour experiment.
- `results/network_sensitivity_metrics.csv`: downlink/contact sweep.
- `results/pre_sensitivity_metrics.csv`: PRE retention/utility sweep.
- `results/hardware_sensitivity_metrics.csv`: hardware-profile scaling sweep.
- `results/generated_tables.tex`: paper table macros.
- `figures/*.png`: generated paper figures.

## Reproduce

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Regenerate simulation outputs:

```bash
python -m orbitllm.cli simulate
```

Regenerate figures and tables:

```bash
python -m orbitllm.cli plot
```

Build the paper:

```bash
latexmk -pdf main.tex
```

## Open-Source Notes

The source code, configuration files, generated CSV results, and plotting scripts can be released publicly. Large model weights are not redistributed; users should download them from the original Qwen model repositories and place them in their own model directory.
