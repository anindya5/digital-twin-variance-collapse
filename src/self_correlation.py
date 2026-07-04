"""Self-Correlation Distance: structural alignment of the inter-question
correlation matrix.

Real respondents exhibit internal covariance: their answer to question A
predicts their answer to question B through shared underlying beliefs. A
simulator can match every marginal distribution and still answer each
question in a vacuum, destroying that structure. We measure this by
comparing the human inter-question correlation matrix with each condition's.

For each source we build a respondent x question matrix (humans linked by
pid; simulated conditions by (archetype_id, person_index); the precomputed
baseline by pid) and compute pairwise-complete Spearman correlations over
the 105 question pairs, requiring MIN_OVERLAP respondents and nonzero
variance on both items -- pairs where collapsed answers make the correlation
undefined are themselves diagnostic and reported as `undefined_share`.

Summary statistics per condition, over pairs valid in both matrices:
  - SCD (Self-Correlation Distance): mean |r_human - r_sim|, lower is better;
  - structure recovery: Pearson correlation between the human and simulated
    off-diagonal correlation vectors (Mantel-style), higher is better, with
    a Mantel permutation test (question labels of the simulated matrix
    permuted jointly) for significance;
  - mean |r| of each matrix, to show how much structure exists at all.

Computed on raw pooled matrices (between- plus within-archetype covariance)
and on within-archetype demeaned matrices (per archetype-question cell means
removed), which isolate the individual-level "belief system" covariance that
demographic conditioning alone cannot explain.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

MIN_OVERLAP = 30
MANTEL_PERMUTATIONS = 999
RNG_SEED = 42


def correlation_matrix(
    mat: pd.DataFrame, demean_by_archetype: bool = False
) -> pd.DataFrame:
    """Pairwise-complete Spearman correlation matrix over question columns.

    `mat` has one row per respondent, question columns plus `archetype_id`.
    Cells with insufficient overlap or zero variance are NaN.
    """
    qids = sorted(c for c in mat.columns if c != "archetype_id")
    vals = mat[qids].astype(float)
    if demean_by_archetype:
        vals = vals - vals.groupby(mat["archetype_id"]).transform("mean")

    out = pd.DataFrame(np.nan, index=qids, columns=qids)
    for q1, q2 in itertools.combinations(qids, 2):
        both = vals[[q1, q2]].dropna()
        if len(both) < MIN_OVERLAP or both[q1].std() == 0 or both[q2].std() == 0:
            continue
        r = stats.spearmanr(both[q1], both[q2]).statistic
        out.loc[q1, q2] = r
        out.loc[q2, q1] = r
    return out


def _offdiag_vector(corr: pd.DataFrame) -> pd.Series:
    qids = list(corr.index)
    return pd.Series(
        {(q1, q2): corr.loc[q1, q2] for q1, q2 in itertools.combinations(qids, 2)}
    )


def _mantel_p(human_corr: pd.DataFrame, sim_corr: pd.DataFrame, observed_r: float) -> float:
    """Two-sided Mantel permutation test: permute the simulated matrix's
    question labels (rows and columns jointly) and recompute the structure-
    recovery correlation against the human matrix.
    """
    rng = np.random.default_rng(RNG_SEED)
    h_vec = _offdiag_vector(human_corr)
    qids = list(sim_corr.index)
    count_ge = 0
    for _ in range(MANTEL_PERMUTATIONS):
        perm = rng.permutation(qids)
        relabeled = pd.DataFrame(sim_corr.to_numpy(), index=perm, columns=perm)
        relabeled = relabeled.loc[qids, qids]
        joined = pd.concat(
            [h_vec.rename("h"), _offdiag_vector(relabeled).rename("s")], axis=1
        ).dropna()
        if len(joined) < 3:
            continue
        r_perm = stats.pearsonr(joined["h"], joined["s"]).statistic
        if abs(r_perm) >= abs(observed_r):
            count_ge += 1
    return (count_ge + 1) / (MANTEL_PERMUTATIONS + 1)


def compare_structures(
    human_mat: pd.DataFrame,
    sim_mats: dict[str, pd.DataFrame],
    demean_by_archetype: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, pairs_df) comparing each simulated condition's
    correlation structure against the human one.
    """
    human_corr = correlation_matrix(human_mat, demean_by_archetype)
    h_vec = _offdiag_vector(human_corr)
    n_total_pairs = len(h_vec)

    summary_rows = []
    pair_rows = []
    for source, mat in sim_mats.items():
        sim_corr = correlation_matrix(mat, demean_by_archetype)
        s_vec = _offdiag_vector(sim_corr)
        joined = pd.concat([h_vec.rename("r_human"), s_vec.rename("r_sim")], axis=1)

        for (q1, q2), row in joined.iterrows():
            pair_rows.append(
                {
                    "source": source,
                    "question_1": q1,
                    "question_2": q2,
                    "r_human": row["r_human"],
                    "r_sim": row["r_sim"],
                }
            )

        valid = joined.dropna()
        scd = float((valid["r_human"] - valid["r_sim"]).abs().mean())
        recovery = (
            float(stats.pearsonr(valid["r_human"], valid["r_sim"]).statistic)
            if len(valid) > 2
            else float("nan")
        )
        mantel_p = (
            _mantel_p(human_corr, sim_corr, recovery) if np.isfinite(recovery) else float("nan")
        )
        summary_rows.append(
            {
                "source": source,
                "SCD": scd,
                "structure_recovery_r": recovery,
                "mantel_p": mantel_p,
                "mean_abs_r_human": float(valid["r_human"].abs().mean()),
                "mean_abs_r_sim": float(valid["r_sim"].abs().mean()),
                "valid_pairs": int(len(valid)),
                "human_valid_pairs": int(h_vec.notna().sum()),
                "undefined_share": float(s_vec.isna().mean()),
                "total_pairs": n_total_pairs,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)
