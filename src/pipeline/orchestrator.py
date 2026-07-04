"""Top-level experiment orchestrator wiring pipeline steps together."""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import Config, load_config
from .analysis import run_analysis
from .fetch import fetch_data
from .plan import build_plan, load_plan
from .self_test import run_self_test
from .simulation_run import run_simulation


class ExperimentOrchestrator:
    """Coordinates fetch, plan, simulate, and analyze steps for one experiment run."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()

    def fetch_data(self) -> None:
        fetch_data(self.cfg)

    def build_plan(self) -> None:
        build_plan(self.cfg)

    def load_plan(self):
        return load_plan(self.cfg)

    def simulate(self, condition: str, *, skip_confirm: bool = False) -> Path | None:
        archetypes, questions = self.load_plan()
        return run_simulation(
            self.cfg, condition, archetypes, questions, skip_confirm=skip_confirm
        )

    def analyze(self):
        archetypes, questions = self.load_plan()
        return run_analysis(self.cfg, archetypes, questions)

    def run_all(self, *, skip_confirm: bool = False) -> None:
        self.fetch_data()
        self.build_plan()
        self.simulate("multi_respondent", skip_confirm=skip_confirm)
        self.analyze()

    def self_test(self) -> None:
        run_self_test()

    def paper_section(self) -> Path:
        try:
            from ..paper_section import write_paper_section
        except ImportError:
            sys.exit(
                "The paper-section command requires src/paper_section.py, which is "
                "not distributed with the replication codebase. All analysis "
                "outputs are in results/ after running `analyze`."
            )
        out_path = write_paper_section(self.cfg)
        print(f"Wrote paper-ready draft section to {out_path}")
        print("(gitignored -- copy/adapt whatever you need into your manuscript)")
        return out_path
