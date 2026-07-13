# OrbitLLM Replication Package

This package contains the non-proprietary material needed to regenerate the
reported simulator tables and figures:

- event-driven simulator and plotting code;
- experiment configuration files;
- measured/profile-derived CSV tables and generated simulation metrics;
- the frozen Landsat-8 TLE input used for the contact-trace check.

It intentionally excludes paper source files, build logs, model weights,
Hugging Face caches, and third-party image datasets. Download Qwen, BLIP,
Qwen2.5-VL, EuroSAT, and SkyScript from their original providers under their
respective licenses.

Run the following commands from the package root after installing
requirements.txt:

    python -m orbitllm.simulate --config configs/default.json
    python -m orbitllm.simulate --config configs/default.json --iotj-required
    python -m orbitllm.simulate --config configs/default.json --soft-energy
    python -m orbitllm.simulate --config configs/default.json --sota-baselines
    python -m orbitllm.reviewer_experiments sota-physical-link
    python -m orbitllm.sota_reporting
    python -m orbitllm.expanded_experiments
    python -m orbitllm.jsa_experiments all
    python -m orbitllm.jsa_recent_baselines
    python -m orbitllm.plot --config configs/default.json

The JSA main numerical comparison contains eleven policies: ActiveInf-inspired,
LEO-DRL-inspired, Hierarchical-RA-inspired, DRL-UA-inspired,
QoS-Multihop-projected, DelayCost-adapted, PBR-adapted, PeakMem-inspired, two
internal rule anchors, and Heuristic-TVD. EH-DORA-adapted is evaluated in the
recharge-aware SoC experiment. Adapted, inspired, and projected implementations are
OrbitLLM-interface comparisons, not faithful reproductions of complete source
systems. The Lyapunov DPP publication is discussed only as a source-level
mechanism reference in the manuscript and is excluded from numerical outputs.
Learning policies train on seeds 100--109 and evaluate on disjoint seeds 0--9.

The JSA-oriented additions produce separate main, exact-oracle, recharge-aware
SoC, and peak-memory CSV files. PBR-adapted uses a bounded five-task search;
EH-DORA-adapted accounts for expected solar recharge and an eclipse reserve;
PeakMem-inspired explicitly prices peak-memory and decode-growth pressure.

The EuroSAT workload uses actual source JPG byte counts, re-encoded MobileSAM
PRE JPEG byte counts, class labels, and per-image CLIP probability-retention
utility. Arrival times and SLA/value fields remain controlled; this is a
dataset-driven payload validation, not an operational satellite traffic trace.

The package is intended for a versioned archive at submission. Tag the matching
GitHub release and obtain a permanent archive DOI before final publication.
