# OrbitLLM

OrbitLLM is a profile-anchored scheduling framework for large-model inference in LEO/NTN mobile edge systems. The artifact contains an RTX 4090 profiling pipeline, a discrete-event simulator, MILP upper bound, online scheduling baselines, sensitivity sweeps, plotting scripts, and the LaTeX source for the workshop paper.

## Layout

- `orbitllm/`: benchmark, simulator, profile loader, and plotting code.
- `configs/default.json`: default experiment configuration.
- `tests/`: unit tests for value decay, bandwidth accounting, costs, and constraints.
- `results/`: generated CSV files, including measured RTX 4090 profile anchors and simulation outputs.
- `figures/`: generated paper figures.
- `sections/`: LaTeX section files.
- `ARTIFACT.md`: reproduction guide.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m orbitllm.cli simulate
python -m orbitllm.cli plot
latexmk -pdf main.tex
```

Large model weights are not redistributed. To rerun profiling, download the Qwen models from their original repositories and update the model paths in `configs/default.json`.
