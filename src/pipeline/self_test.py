"""Synthetic end-to-end pipeline check (no network or API)."""
from __future__ import annotations

import json
import random

import pandas as pd

from ..archetypes import Archetype
from ..metrics import build_comparison_table, paired_condition_test
from ..question_selection import QuestionSpec


def run_self_test() -> None:
    rng = random.Random(0)
    archetypes = [Archetype(f"arch_{i}", {}, {"QID13": "30-49"}, list(range(20))) for i in range(5)]
    questions = [
        QuestionSpec(f"QID{i}", f"Question {i}?", ["A", "B", "C", "D"], "Block", f"QID{i}", f"Q{i}")
        for i in range(3)
    ]
    n_options_by_question = {q.question_id: q.n_options for q in questions}

    rows = []
    for a in archetypes:
        for q in questions:
            for _pid in a.pids:
                rows.append(
                    {
                        "archetype_id": a.archetype_id,
                        "question_id": q.question_id,
                        "source": "human",
                        "selected_position": rng.randint(1, 4),
                    }
                )
            fixed = rng.randint(1, 4)
            for _ in range(10):
                pos = fixed if rng.random() < 0.9 else rng.randint(1, 4)
                rows.append(
                    {
                        "archetype_id": a.archetype_id,
                        "question_id": q.question_id,
                        "source": "baseline_precomputed",
                        "selected_position": pos,
                    }
                )
            for _ in range(10):
                rows.append(
                    {
                        "archetype_id": a.archetype_id,
                        "question_id": q.question_id,
                        "source": "multi_respondent",
                        "selected_position": rng.randint(1, 4),
                    }
                )

    long_df = pd.DataFrame(rows)
    comparison_df = build_comparison_table(long_df, n_options_by_question)
    assert len(comparison_df) > 0, "comparison table is empty"

    result = paired_condition_test(
        comparison_df, "variance_ratio", "multi_respondent", "baseline_precomputed"
    )
    print(json.dumps(result, indent=2))
    assert result["mean_a"] > result["mean_b"], (
        "expected the synthetic multi_respondent condition (more spread) to show a higher "
        "variance_ratio than the synthetic near-collapsed baseline condition"
    )
    print("\nSelf-test passed: data loading -> archetypes -> metrics -> stats pipeline is wired correctly.")
