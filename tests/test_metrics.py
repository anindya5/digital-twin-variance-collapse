"""Tests for variance / precision-recall metrics."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.metrics import (
    build_comparison_table,
    compute_distribution_stats,
    paired_condition_test,
    safe_ratio,
)


def test_compute_distribution_stats_collapsed_has_zero_variance():
    stats = compute_distribution_stats([1, 1, 1, 1], n_options=4)
    assert stats is not None
    assert stats.variance == 0.0
    assert stats.norm_entropy == 0.0
    assert stats.mode == 1


def test_compute_distribution_stats_spread_has_positive_variance_and_entropy():
    stats = compute_distribution_stats([1, 2, 3, 4], n_options=4)
    assert stats is not None
    assert stats.variance > 0
    assert stats.norm_entropy > 0.5
    assert stats.unique_ratio == 1.0


def test_safe_ratio_handles_zero_denominator():
    assert safe_ratio(0, 0) is None
    assert safe_ratio(1, 0) == math.inf
    assert safe_ratio(2, 4) == 0.5


def test_build_comparison_table_variance_ratio_detects_collapse(collapsed_vs_spread_long_df):
    n_options = {"Q1": 4}
    table = build_comparison_table(collapsed_vs_spread_long_df, n_options)

    collapsed = table[table.source == "independent_coarse"]
    spread = table[table.source == "multi_respondent"]

    assert (collapsed["variance_ratio"] == 0.0).all()
    assert spread["variance_ratio"].mean() > collapsed["variance_ratio"].mean()
    assert collapsed["mode_match"].mean() == 1.0
    assert spread["entropy_ratio"].mean() > collapsed["entropy_ratio"].mean()


def test_paired_condition_test_prefers_spread_over_collapsed(collapsed_vs_spread_long_df):
    table = build_comparison_table(collapsed_vs_spread_long_df, {"Q1": 4})
    result = paired_condition_test(table, "variance_ratio", "multi_respondent", "independent_coarse")
    assert result["n_pairs"] == 2
    assert result["mean_a"] > result["mean_b"]
