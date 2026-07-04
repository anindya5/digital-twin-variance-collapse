"""Run a single LLM simulation condition and write results to CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..simulate import (
    responses_to_records,
    run_independent_coarse,
    run_multi_respondent,
    run_persistent_batched,
)
from .confirm import confirm_proceed
from .constants import SIMULATED_CONDITION_FILES


def _describe_condition(cfg: Config, condition: str, n_archetypes: int, n_questions: int) -> tuple[int, str]:
    n_per_group = cfg.n_simulated_per_group
    if condition == "multi_respondent":
        n_calls = n_archetypes * n_questions
        return n_calls, (
            f"About to make up to {n_calls} API calls (1 per archetype x question, "
            f"each producing {n_per_group} simulated answers)."
        )
    if condition == "persistent_batched":
        n_calls = n_archetypes
        return n_calls, (
            f"About to make up to {n_calls} API calls (1 per archetype; each "
            f"instantiates {n_per_group} personas answering all {n_questions} questions)."
        )
    n_calls = n_archetypes * n_questions * n_per_group
    return n_calls, (
        f"About to make up to {n_calls} API calls (1 per archetype x question x "
        f"simulated twin). This is {n_per_group}x more expensive than "
        "multi_respondent -- make sure that's intended."
    )


def run_simulation(
    cfg: Config,
    condition: str,
    archetypes,
    questions,
    *,
    skip_confirm: bool = False,
) -> Path:
    _, message = _describe_condition(cfg, condition, len(archetypes), len(questions))
    print(message)
    if not skip_confirm and not confirm_proceed():
        raise SystemExit(0)

    runners = {
        "multi_respondent": run_multi_respondent,
        "persistent_batched": run_persistent_batched,
        "independent_coarse": run_independent_coarse,
    }
    responses = runners[condition](cfg, archetypes, questions)

    out_path = cfg.results_dir / SIMULATED_CONDITION_FILES[condition]
    df = pd.DataFrame(responses_to_records(responses))
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} simulated responses to {out_path}")
    return out_path
