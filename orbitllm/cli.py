from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orbitllm", description="OrbitLLM experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("benchmark", help="Run GPU LLM profiling anchors")
    sub.add_parser("simulate", help="Run scheduling simulations")
    sub.add_parser("plot", help="Generate paper figures and tables")
    sub.add_parser("precalibrate", help="Run a small public-dataset PRE calibration smoke test")
    args, rest = parser.parse_known_args(argv)
    if args.command == "benchmark":
        from .benchmark import main as bench_main

        return bench_main(rest)
    if args.command == "simulate":
        from .simulate import main as sim_main

        return sim_main(rest)
    if args.command == "plot":
        from .plot import main as plot_main

        return plot_main(rest)
    if args.command == "precalibrate":
        from .precalibrate import main as pre_main

        return pre_main(rest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
