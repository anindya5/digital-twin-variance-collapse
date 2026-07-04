"""Command-line interface for the variance-collapse experiment."""
from __future__ import annotations

import argparse

from .pipeline.orchestrator import ExperimentOrchestrator

CLI_DOC = """CLI orchestrator for testing the paper's variance-collapse hypothesis on
Twin-2K-500.

Typical usage:

    python run_experiment.py fetch-data
    python run_experiment.py build-plan
    python run_experiment.py simulate --condition multi_respondent
    # optional, costs ~n_simulated_per_group x more API calls:
    # python run_experiment.py simulate --condition independent_coarse
    python run_experiment.py analyze

Or just:

    python run_experiment.py all

Run `python run_experiment.py self-test` first (no API key required) to sanity
check the whole pipeline end-to-end on synthetic data.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=CLI_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch-data", help="Download required Twin-2K-500 files")
    sub.add_parser("build-plan", help="Build archetype groups + select questions")

    p_sim = sub.add_parser("simulate", help="Run an LLM simulation condition")
    p_sim.add_argument(
        "--condition",
        choices=["multi_respondent", "independent_coarse", "persistent_batched"],
        required=True,
    )
    p_sim.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")

    sub.add_parser("analyze", help="Compute metrics + significance tests + plots")
    sub.add_parser(
        "paper-section",
        help="Render a paper-ready draft section from current results/ (gitignored)",
    )

    p_all = sub.add_parser("all", help="fetch-data -> build-plan -> simulate multi_respondent -> analyze")
    p_all.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")

    sub.add_parser("self-test", help="Verify the pipeline on synthetic data (no API key needed)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = ExperimentOrchestrator()

    if args.cmd == "fetch-data":
        orchestrator.fetch_data()
    elif args.cmd == "build-plan":
        orchestrator.build_plan()
    elif args.cmd == "simulate":
        orchestrator.simulate(args.condition, skip_confirm=args.yes)
    elif args.cmd == "analyze":
        orchestrator.analyze()
    elif args.cmd == "paper-section":
        orchestrator.paper_section()
    elif args.cmd == "all":
        orchestrator.run_all(skip_confirm=args.yes)
    elif args.cmd == "self-test":
        orchestrator.self_test()
    else:
        parser.error(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
