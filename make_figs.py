#!/usr/bin/env python3
"""Compatibility wrapper for regenerating OrbitLLM paper figures.

The authoritative plotting code lives in ``orbitllm.plot`` and reads the CSV
outputs produced by ``python -m orbitllm.cli simulate``.
"""

from orbitllm.plot import main


if __name__ == "__main__":
    raise SystemExit(main())
