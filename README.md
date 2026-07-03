# OrbitLLM

OrbitLLM is a profile-anchored scheduling framework for large-model inference in LEO/NTN mobile edge systems. This repository contains the experiment code: an RTX 4090 profiling pipeline, a discrete-event simulator, MILP upper bound, online scheduling baselines, sensitivity sweeps, and plotting scripts.

This artifact is code-only. Paper source files, compiled PDFs, experiment logs, generated figures/results, datasets, and model weights are intentionally excluded.

## Layout

- `orbitllm/`: benchmark, simulator, profile loader, and plotting code.
- `configs/default.json`: default experiment configuration.
- `orbitllm/batch_experiments.py`: supplemental sweeps such as epsilon sensitivity, exact small-instance oracle, and Energy-WSRPT.
- `orbitllm/reviewer_experiments.py`: reviewer-requested physical-link, guardband/cap, VLM profile, and portability experiments.
- `scripts/`: lightweight helper scripts, including guarded local power-limit attempts and proxy-profile generation.
- `tests/`: unit tests for value decay, bandwidth accounting, costs, and constraints.
- `ARTIFACT.md`: reproduction guide.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m orbitllm.cli simulate
python -m orbitllm.cli plot
```

Large model weights are not redistributed. To rerun profiling, download the Qwen models from their original repositories and update the model paths in `configs/default.json`.
