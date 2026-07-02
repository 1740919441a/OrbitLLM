# OrbitLLM Artifact Guide

This artifact contains the code needed to reproduce the OrbitLLM workshop experiments. The paper text and LaTeX sources are intentionally not included in the public code repository.

## Code Layout

- `orbitllm/benchmark.py`: RTX 4090 LLM profiling CLI.
- `orbitllm/simulate.py`: task generation, policies, MILP bound, and all sensitivity sweeps.
- `orbitllm/plot.py`: figure and LaTeX table generation from CSV outputs.
- `orbitllm/profile.py`: measured-profile loading and aggregation.
- `orbitllm/cli.py`: unified command line interface.
- `configs/default.json`: default experiment configuration.
- `tests/test_simulation.py`: unit tests for value decay, costs, bandwidth, and constraints.

## Result Layout

Generated CSV results and figures are produced locally by the commands below. They are not required to be committed to the public code repository.

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

## Open-Source Notes

The source code, configuration files, tests, and plotting scripts can be released publicly. Large model weights are not redistributed; users should download them from the original Qwen model repositories and place them in their own model directory.
