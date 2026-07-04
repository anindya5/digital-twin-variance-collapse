"""Tests for Self-Correlation Distance and structure recovery."""
from __future__ import annotations

import pandas as pd

from src.self_correlation import compare_structures, correlation_matrix


def test_correlation_matrix_finds_structure(respondent_matrix_with_structure):
    corr = correlation_matrix(respondent_matrix_with_structure, demean_by_archetype=False)
    assert pd.notna(corr.loc["Q1", "Q2"])
    assert abs(corr.loc["Q1", "Q2"]) > 0.2


def test_correlation_matrix_collapsed_pairs_are_undefined(collapsed_respondent_matrix):
    corr = correlation_matrix(collapsed_respondent_matrix, demean_by_archetype=False)
    assert pd.isna(corr.loc["Q1", "Q2"])


def test_compare_structures_recovers_matching_matrix(respondent_matrix_with_structure):
    summary, pairs = compare_structures(
        respondent_matrix_with_structure,
        {"same": respondent_matrix_with_structure.copy()},
        demean_by_archetype=False,
    )
    row = summary.iloc[0]
    assert row["SCD"] < 0.01
    assert row["structure_recovery_r"] > 0.99
    assert row["undefined_share"] == 0.0
    assert len(pairs) > 0


def test_compare_structures_collapsed_sim_has_high_undefined_share(
    respondent_matrix_with_structure, collapsed_respondent_matrix
):
    summary, _ = compare_structures(
        respondent_matrix_with_structure,
        {"collapsed": collapsed_respondent_matrix},
        demean_by_archetype=False,
    )
    row = summary.iloc[0]
    assert row["undefined_share"] > 0.5
    assert row["valid_pairs"] < row["total_pairs"]
