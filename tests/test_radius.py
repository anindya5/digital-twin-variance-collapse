"""Tests for RADIUS alignment metrics and KL divergence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.radius import (
    compute_radius_tables,
    distribution_homogeneity,
    kl_divergence,
    rank_correlation,
    radius_paired_tests,
    radius_summary,
    top_rank_match,
    total_variation_distance,
)


def test_total_variation_distance_identical_is_zero():
    h = np.array([1, 1, 2, 2, 3, 3])
    assert total_variation_distance(h, h.copy(), n_options=4) == 0.0


def test_total_variation_distance_max_when_no_overlap():
    h = np.array([1, 1, 1, 1])
    a = np.array([4, 4, 4, 4])
    assert total_variation_distance(h, a, n_options=4) == 1.0


def test_kl_divergence_infinite_when_simulator_zeros_human_mass():
    h = np.array([1, 2, 2, 3])
    a = np.array([1, 1, 1, 1])
    smoothed, is_inf = kl_divergence(h, a, n_options=4)
    assert is_inf == 1.0
    assert smoothed > 0


def test_kl_divergence_finite_when_distributions_match():
    h = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    a = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    smoothed, is_inf = kl_divergence(h, a, n_options=4)
    assert is_inf == 0.0
    assert smoothed < 0.01


def test_rank_correlation_perfect_for_monotonic_orderings():
    h = np.array([1] * 10 + [2] * 5 + [3] * 2 + [4])
    a = np.array([1] * 8 + [2] * 6 + [3] * 3 + [4] * 2 + [4])
    rc = rank_correlation(h, a, n_options=4)
    assert rc is not None
    assert rc > 0.95


def test_rank_correlation_undefined_for_constant_counts():
    h = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    a = np.array([1, 2, 3, 4, 1, 2, 3, 4])
    assert rank_correlation(h, a, n_options=4) is None


def test_top_rank_match_accepts_agent_modal_in_human_tie_group():
    rng = np.random.default_rng(0)
    h = np.array([1, 1, 2, 2, 2, 3, 3, 4])
    a = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    assert top_rank_match(h, a, n_options=4, rng=rng) == 1.0


def test_distribution_homogeneity_accepts_identical_distributions():
    rng = np.random.default_rng(0)
    h = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4])
    a = np.array([1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4])
    assert distribution_homogeneity(h, a, n_options=4, rng=rng) == 1.0


def test_compute_radius_tables_separates_collapsed_and_spread(collapsed_vs_spread_long_df):
    per_df, pooled_df = compute_radius_tables(
        collapsed_vs_spread_long_df, {"Q1": 4}, seed=42
    )
    assert len(per_df) == 4
    assert len(pooled_df) == 2

    summary = radius_summary(per_df).set_index("source")
    assert summary.loc["multi_respondent", "TVD"] < summary.loc["independent_coarse", "TVD"]
    assert summary.loc["multi_respondent", "DH"] >= summary.loc["independent_coarse", "DH"]
    assert summary.loc["independent_coarse", "KL_inf"] >= summary.loc["multi_respondent", "KL_inf"]


def test_radius_paired_tests_run_on_per_archetype_table(collapsed_vs_spread_long_df):
    per_df, _ = compute_radius_tables(collapsed_vs_spread_long_df, {"Q1": 4}, seed=42)
    tests = radius_paired_tests(
        per_df, "multi_respondent", "independent_coarse", ["archetype_id", "question_id"]
    )
    tvd = next(t for t in tests if t["metric"] == "TVD")
    assert tvd["n_pairs"] == 2
    assert tvd["mean_a"] < tvd["mean_b"]
