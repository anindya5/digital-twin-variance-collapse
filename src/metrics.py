"""Precision/recall-style metrics for comparing response distributions.

For a set of 1-based option-position answers over an ordinal/nominal scale
with k options, we compute per (archetype, question, source) group:

  - n              number of answers observed
  - mean           average position (only meaningful as a rough proxy; we
                   mainly use it for the "precision" central-tendency check)
  - mode           most common position ("precision" target -- do we get the
                   modal / typical answer right?)
  - variance       sample variance of positions ("recall" target -- how much
                   spread is there?)
  - norm_entropy   Shannon entropy of the position distribution, normalized
                   by log(k) so it's comparable across questions with
                   different numbers of options (0 = everyone picks the same
                   option, 1 = perfectly uniform across all options)
  - unique_ratio   distinct positions used / n

We then compare a simulated-condition group against the matched real-human
group from the same archetype+question with:

  - variance_ratio   sim_variance / human_variance  (recall proxy; 1.0 = the
                     LLM reproduces exactly as much spread as real humans do)
  - entropy_ratio    sim_norm_entropy / human_norm_entropy
  - mode_match       1 if sim_mode == human_mode else 0  (precision proxy)
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DistributionStats:
    n: int
    mean: float
    mode: int
    variance: float
    norm_entropy: float
    unique_ratio: float


def compute_distribution_stats(positions: list[int], n_options: int) -> DistributionStats | None:
    positions = [p for p in positions if p is not None and not (isinstance(p, float) and math.isnan(p))]
    if len(positions) == 0:
        return None
    arr = np.array(positions, dtype=float)
    counts = Counter(positions)
    mode = counts.most_common(1)[0][0]
    probs = np.array(list(counts.values()), dtype=float) / len(positions)
    entropy = float(-(probs * np.log(probs)).sum())
    max_entropy = math.log(n_options) if n_options > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    variance = float(arr.var(ddof=1)) if len(positions) > 1 else 0.0
    return DistributionStats(
        n=len(positions),
        mean=float(arr.mean()),
        mode=int(mode),
        variance=variance,
        norm_entropy=norm_entropy,
        unique_ratio=len(counts) / len(positions),
    )


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None if numerator == 0 else float("inf")
    return numerator / denominator


def build_comparison_table(
    long_df: pd.DataFrame,
    n_options_by_question: dict[str, int],
) -> pd.DataFrame:
    """`long_df` must have columns: archetype_id, question_id, source,
    selected_position (source in {"human", "multi_respondent",
    "independent_coarse", "baseline_precomputed"}).

    Returns one row per (archetype_id, question_id, source) with distribution
    stats, plus variance_ratio / entropy_ratio / mode_match relative to the
    "human" source for the same (archetype_id, question_id).
    """
    rows = []
    grouped = long_df.groupby(["archetype_id", "question_id", "source"])["selected_position"]
    for (archetype_id, question_id, source), series in grouped:
        stats_ = compute_distribution_stats(series.tolist(), n_options_by_question[question_id])
        if stats_ is None:
            continue
        rows.append(
            {
                "archetype_id": archetype_id,
                "question_id": question_id,
                "source": source,
                **stats_.__dict__,
            }
        )
    df = pd.DataFrame(rows)

    human = df[df.source == "human"].set_index(["archetype_id", "question_id"])
    out_rows = []
    for _, row in df.iterrows():
        key = (row["archetype_id"], row["question_id"])
        if key not in human.index:
            continue
        h = human.loc[key]
        row = row.to_dict()
        row["human_variance"] = h["variance"]
        row["human_norm_entropy"] = h["norm_entropy"]
        row["human_mode"] = h["mode"]
        row["variance_ratio"] = safe_ratio(row["variance"], h["variance"])
        row["entropy_ratio"] = safe_ratio(row["norm_entropy"], h["norm_entropy"])
        row["mode_match"] = int(row["mode"] == h["mode"])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def paired_condition_test(
    comparison_df: pd.DataFrame, metric: str, source_a: str, source_b: str
) -> dict:
    """Wilcoxon signed-rank test on `metric`, paired by (archetype_id,
    question_id), comparing source_a vs source_b (e.g. multi_respondent vs
    independent_coarse). Returns summary stats + the test result.
    """
    a = comparison_df[comparison_df.source == source_a].set_index(["archetype_id", "question_id"])[metric]
    b = comparison_df[comparison_df.source == source_b].set_index(["archetype_id", "question_id"])[metric]
    joined = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    joined = joined[np.isfinite(joined["a"]) & np.isfinite(joined["b"])]
    if len(joined) < 2:
        return {"n_pairs": len(joined), "error": "not enough paired observations"}

    diffs = joined["a"] - joined["b"]
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(joined["a"], joined["b"])
    except ValueError:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    ttest_stat, ttest_p = stats.ttest_rel(joined["a"], joined["b"])

    return {
        "metric": metric,
        "source_a": source_a,
        "source_b": source_b,
        "n_pairs": len(joined),
        "mean_a": float(joined["a"].mean()),
        "mean_b": float(joined["b"].mean()),
        "median_a": float(joined["a"].median()),
        "median_b": float(joined["b"].median()),
        "mean_diff_a_minus_b": float(diffs.mean()),
        "wilcoxon_stat": float(wilcoxon_stat),
        "wilcoxon_p": float(wilcoxon_p),
        "ttest_stat": float(ttest_stat),
        "ttest_p": float(ttest_p),
    }
