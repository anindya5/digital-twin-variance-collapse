"""Tests for the experiment orchestrator and CLI smoke paths."""
from __future__ import annotations

from src.cli import build_parser
from src.pipeline import ExperimentOrchestrator
from src.pipeline.self_test import run_self_test


def test_orchestrator_instantiates():
    orch = ExperimentOrchestrator()
    assert orch.cfg is not None


def test_cli_parser_has_all_commands():
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "cmd")
    assert sub.choices is not None
    assert {
        "fetch-data",
        "build-plan",
        "simulate",
        "analyze",
        "all",
        "self-test",
    } <= set(sub.choices)


def test_self_test_passes(capsys):
    run_self_test()
    out = capsys.readouterr().out
    assert "Self-test passed" in out
