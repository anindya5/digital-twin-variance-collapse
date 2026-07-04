"""Shared fixtures for evaluation and pipeline tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def four_option_question() -> dict[str, int]:
    return {"Q1": 4}


@pytest.fixture
def collapsed_vs_spread_long_df() -> pd.DataFrame:
    """Two archetypes, one question: humans spread; sim_collapsed vs sim_spread."""
    rows = []
    for arch in ("a1", "a2"):
        human = [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 2, 3, 1]
        rows += [
            {"archetype_id": arch, "question_id": "Q1", "source": "human", "selected_position": p}
            for p in human
        ]
        rows += [
            {"archetype_id": arch, "question_id": "Q1", "source": "independent_coarse", "selected_position": 1}
            for _ in range(10)
        ]
        rows += [
            {"archetype_id": arch, "question_id": "Q1", "source": "multi_respondent", "selected_position": p}
            for p in [1, 2, 3, 4, 1, 2, 3, 4, 2, 3]
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def respondent_matrix_with_structure() -> pd.DataFrame:
    """50 respondents, 4 questions, with deliberate Q1–Q2 correlation."""
    rng = np.random.default_rng(0)
    n = 50
    q1 = rng.integers(1, 5, size=n)
    q2 = np.clip(q1 + rng.integers(-1, 2, size=n), 1, 4)
    mat = pd.DataFrame(
        {
            "Q1": q1,
            "Q2": q2,
            "Q3": rng.integers(1, 5, size=n),
            "Q4": rng.integers(1, 5, size=n),
            "archetype_id": "a1",
        }
    )
    return mat


@pytest.fixture
def collapsed_respondent_matrix(respondent_matrix_with_structure: pd.DataFrame) -> pd.DataFrame:
    """Same shape but every answer is 1 — structure should collapse."""
    mat = respondent_matrix_with_structure.copy()
    for col in ["Q1", "Q2", "Q3", "Q4"]:
        mat[col] = 1
    return mat
