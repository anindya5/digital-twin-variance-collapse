"""RADIUS alignment suite (Lajewska, Missault, Davidson & Mansour,
arXiv:2603.19002) applied to our Twin-2K-500 experiment.

RADIUS evaluates survey simulation on two complementary dimensions, each with
statistical significance built in:

  Ranking alignment (maps to the paper's "precision"):
    - Top Rank Match (TRM): binary -- does the simulator's most common answer
      fall inside the humans' *statistically tied* top-choice group? The tie
      group is built by bootstrap-resampling the human votes (n=1000) and
      clustering options whose vote-share confidence intervals overlap the
      top option's interval.
    - Rank Correlation (RC): Spearman correlation between the human and
      simulated option orderings (by vote count), normalized to [0, 1] via
      (rho + 1) / 2.

  Distribution alignment (maps to the paper's "recall"):
    - Total Variation Distance (TVD): 0.5 * sum_i |H_i - A_i| over option
      proportions -- the probability mass that must move to turn the
      simulated distribution into the human one. Lower is better.
    - Distribution Homogeneity (DH): binary -- 1 if a chi-square test of
      homogeneity cannot distinguish the simulated from the human
      distribution (p >= 0.05). Where expected cell counts are too small for
      the chi-square approximation (common in our per-archetype tables), we
      fall back to a Monte Carlo permutation test on the same statistic.

Survey-level scores are means of question-level scores; conditions are
compared with paired t-tests (per RADIUS) plus Wilcoxon signed-rank tests.

We compute the suite at two granularities:
  - per (archetype, question): keeps our identical-profile design central,
    but human/agent samples are small (agent n = 10);
  - pooled per question: all archetype members' human answers vs. all
    simulated answers for that question -- matches RADIUS's own
    population-level setting and is the statistically robust headline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAP_RESAMPLES = 1000
PERMUTATION_RESAMPLES = 2000
ALPHA = 0.05


def _counts(positions: np.ndarray, n_options: int) -> np.ndarray:
    return np.bincount(positions - 1, minlength=n_options).astype(float)


def top_rank_match(
    human_positions: np.ndarray,
    agent_positions: np.ndarray,
    n_options: int,
    rng: np.random.Generator,
) -> float:
    """TRM with bootstrap tie-grouping of the human top choice."""
    human_counts = _counts(human_positions, n_options)
    n_h = len(human_positions)

    # Bootstrap the human vote shares to get per-option CIs.
    probs = human_counts / n_h
    resamples = rng.multinomial(n_h, probs, size=BOOTSTRAP_RESAMPLES) / n_h
    lo = np.percentile(resamples, 2.5, axis=0)
    hi = np.percentile(resamples, 97.5, axis=0)

    top = int(np.argmax(human_counts))
    # Options whose CI overlaps the top option's CI form the tied top group.
    top_group = {i for i in range(n_options) if hi[i] >= lo[top] and lo[i] <= hi[top] and human_counts[i] > 0}
    top_group.add(top)

    agent_counts = _counts(agent_positions, n_options)
    agent_top = int(np.argmax(agent_counts))
    return 1.0 if agent_top in top_group else 0.0


def rank_correlation(
    human_positions: np.ndarray, agent_positions: np.ndarray, n_options: int
) -> float | None:
    """Normalized Spearman correlation between option orderings by vote count."""
    h = _counts(human_positions, n_options)
    a = _counts(agent_positions, n_options)
    if np.all(h == h[0]) or np.all(a == a[0]):
        return None  # correlation undefined for a constant ranking
    rho, _ = stats.spearmanr(h, a)
    if np.isnan(rho):
        return None
    return (rho + 1.0) / 2.0


def total_variation_distance(
    human_positions: np.ndarray, agent_positions: np.ndarray, n_options: int
) -> float:
    h = _counts(human_positions, n_options)
    a = _counts(agent_positions, n_options)
    return 0.5 * float(np.abs(h / h.sum() - a / a.sum()).sum())


KL_SMOOTHING_ALPHA = 0.5  # Jeffreys prior on the simulated distribution


def kl_divergence(
    human_positions: np.ndarray, agent_positions: np.ndarray, n_options: int
) -> tuple[float, float]:
    """Forward KL divergence D_KL(P_Train || P_LLM) in nats, where
    P_Train(C|b) is the empirical human choice distribution for background b
    and P_LLM(C|b) is the empirical simulated choice distribution.

    Forward KL penalizes the simulator for assigning low/zero probability to
    options real humans chose -- the recall dimension. Returns
    (smoothed_kl, is_infinite_unsmoothed):
      - smoothed_kl uses additive (Jeffreys, alpha=0.5) smoothing on P_LLM
        only, keeping P_Train as the raw empirical distribution;
      - is_infinite_unsmoothed is 1.0 when the unsmoothed KL diverges, i.e.
        the simulator puts zero mass on at least one option with human
        support. This divergence rate is itself a signature of variance
        collapse.
    """
    h = _counts(human_positions, n_options)
    a = _counts(agent_positions, n_options)
    p = h / h.sum()
    is_inf = 1.0 if np.any((p > 0) & (a == 0)) else 0.0

    q = (a + KL_SMOOTHING_ALPHA) / (a.sum() + KL_SMOOTHING_ALPHA * n_options)
    mask = p > 0
    smoothed = float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return smoothed, is_inf


def _chi2_statistic(table: np.ndarray) -> float:
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / table.sum()
    mask = expected > 0
    return float((((table - expected) ** 2)[mask] / expected[mask]).sum())


def distribution_homogeneity(
    human_positions: np.ndarray,
    agent_positions: np.ndarray,
    n_options: int,
    rng: np.random.Generator,
) -> float:
    """DH = 1 if human and simulated distributions are statistically
    indistinguishable (p >= ALPHA). Uses the chi-square test of homogeneity,
    with a Monte Carlo permutation fallback when expected counts are too
    small for the asymptotic approximation.
    """
    h = _counts(human_positions, n_options)
    a = _counts(agent_positions, n_options)
    keep = (h + a) > 0
    table = np.vstack([h[keep], a[keep]])
    if table.shape[1] < 2:
        return 1.0  # both groups concentrated on the same single option

    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / table.sum()

    if expected.min() >= 5:
        _, p, _, _ = stats.chi2_contingency(table)
    else:
        # Permutation test: pool all answers, reshuffle group labels.
        observed = _chi2_statistic(table)
        pooled = np.concatenate([human_positions, agent_positions])
        n_h = len(human_positions)
        count_ge = 0
        for _ in range(PERMUTATION_RESAMPLES):
            perm = rng.permutation(pooled)
            t = np.vstack(
                [_counts(perm[:n_h], n_options)[keep], _counts(perm[n_h:], n_options)[keep]]
            )
            if _chi2_statistic(t) >= observed:
                count_ge += 1
        p = (count_ge + 1) / (PERMUTATION_RESAMPLES + 1)

    return 1.0 if p >= ALPHA else 0.0


def _score_pair(
    human_positions: np.ndarray,
    agent_positions: np.ndarray,
    n_options: int,
    rng: np.random.Generator,
) -> dict:
    kl_smoothed, kl_inf = kl_divergence(human_positions, agent_positions, n_options)
    return {
        "n_human": len(human_positions),
        "n_agent": len(agent_positions),
        "TRM": top_rank_match(human_positions, agent_positions, n_options, rng),
        "RC": rank_correlation(human_positions, agent_positions, n_options),
        "TVD": total_variation_distance(human_positions, agent_positions, n_options),
        "DH": distribution_homogeneity(human_positions, agent_positions, n_options, rng),
        "KL": kl_smoothed,
        "KL_inf": kl_inf,
    }


def compute_radius_tables(
    long_df: pd.DataFrame,
    n_options_by_question: dict[str, int],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every simulated condition against the human ground truth.

    `long_df` columns: archetype_id, question_id, source, selected_position
    (source "human" is the reference; all other sources are scored).

    Returns (per_archetype_df, pooled_df).
    """
    rng = np.random.default_rng(seed)
    conditions = [s for s in long_df.source.unique() if s != "human"]

    human = long_df[long_df.source == "human"]
    per_rows = []
    pooled_rows = []

    for question_id, k in n_options_by_question.items():
        h_q = human[human.question_id == question_id]
        if h_q.empty:
            continue
        for condition in conditions:
            a_q = long_df[(long_df.source == condition) & (long_df.question_id == question_id)]
            if a_q.empty:
                continue

            # Pooled (population-level, RADIUS-canonical).
            pooled_rows.append(
                {
                    "question_id": question_id,
                    "source": condition,
                    **_score_pair(
                        h_q.selected_position.to_numpy(),
                        a_q.selected_position.to_numpy(),
                        k,
                        rng,
                    ),
                }
            )

            # Per-archetype (keeps the identical-profile design central).
            for archetype_id, a_g in a_q.groupby("archetype_id"):
                h_g = h_q[h_q.archetype_id == archetype_id]
                if len(h_g) < 2:
                    continue
                per_rows.append(
                    {
                        "archetype_id": archetype_id,
                        "question_id": question_id,
                        "source": condition,
                        **_score_pair(
                            h_g.selected_position.to_numpy(),
                            a_g.selected_position.to_numpy(),
                            k,
                            rng,
                        ),
                    }
                )

    return pd.DataFrame(per_rows), pd.DataFrame(pooled_rows)


def radius_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Survey-level RADIUS + KL scores: mean of question-level scores per condition."""
    return (
        df.groupby("source")[["TRM", "RC", "TVD", "DH", "KL", "KL_inf"]]
        .mean()
        .reset_index()
        .sort_values("source")
    )


def radius_paired_tests(
    df: pd.DataFrame, source_a: str, source_b: str, keys: list[str]
) -> list[dict]:
    """Paired t-tests (per RADIUS) + Wilcoxon on each metric between two
    conditions, paired on `keys` (e.g. ["question_id"] for pooled scores).
    """
    results = []
    a = df[df.source == source_a].set_index(keys)
    b = df[df.source == source_b].set_index(keys)
    for metric in ["TRM", "RC", "TVD", "DH", "KL"]:
        joined = a[metric].to_frame("a").join(b[metric].to_frame("b"), how="inner").dropna()
        if len(joined) < 2:
            results.append({"metric": metric, "n_pairs": len(joined), "error": "too few pairs"})
            continue
        t_stat, t_p = stats.ttest_rel(joined["a"], joined["b"])
        try:
            w_stat, w_p = stats.wilcoxon(joined["a"], joined["b"])
        except ValueError:  # all differences zero
            w_stat, w_p = float("nan"), float("nan")
        results.append(
            {
                "metric": metric,
                "source_a": source_a,
                "source_b": source_b,
                "n_pairs": len(joined),
                "mean_a": float(joined["a"].mean()),
                "mean_b": float(joined["b"].mean()),
                "ttest_stat": float(t_stat),
                "ttest_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
            }
        )
    return results
