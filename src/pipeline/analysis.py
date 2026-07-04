"""Compute metrics, write tables, reports, and plots."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import Config
from ..metrics import build_comparison_table, paired_condition_test
from ..radius import compute_radius_tables, radius_paired_tests, radius_summary
from ..self_correlation import compare_structures
from .dataframes import baseline_long_df, human_long_df, load_simulated_long_frames, person_question_matrices
from .plots import make_structure_plot, make_variance_plot


def _independent_label(cfg: Config) -> str:
    return "baseline_precomputed" if cfg.independent_source == "precomputed" else "independent_coarse"


def _append_variance_summary(report_lines: list[str], comparison_df: pd.DataFrame, independent_label: str) -> None:
    for metric in ["variance_ratio", "entropy_ratio", "mode_match"]:
        for source in [s for s in [independent_label, "multi_respondent"] if s in comparison_df.source.unique()]:
            raw = comparison_df[comparison_df.source == source][metric].dropna()
            n_infinite = int(np.isinf(raw).sum())
            sub = raw[np.isfinite(raw)]
            if len(sub):
                note = (
                    f", {n_infinite} cases where humans unanimously agreed but the model didn't "
                    f"(ratio=inf, excluded from mean)"
                    if n_infinite
                    else ""
                )
                report_lines.append(
                    f"- **{source}** mean {metric}: {sub.mean():.3f} "
                    f"(median {sub.median():.3f}, n={len(sub)}{note})"
                )

    if independent_label in comparison_df.source.unique() and "multi_respondent" in comparison_df.source.unique():
        report_lines.append(f"\n## Statistical comparison: multi_respondent vs {independent_label}")
        for metric in ["variance_ratio", "entropy_ratio", "mode_match"]:
            result = paired_condition_test(comparison_df, metric, "multi_respondent", independent_label)
            report_lines.append(f"\n### {metric}\n```\n{json.dumps(result, indent=2)}\n```")


def _append_radius_summary(report_lines: list[str], per_arch_df: pd.DataFrame, pooled_df: pd.DataFrame) -> None:
    report_lines.append("\n## RADIUS alignment suite")
    for label, df_, keys in [
        ("Pooled per question (population-level, RADIUS-canonical)", pooled_df, ["question_id"]),
        ("Per archetype-question (identical-profile design)", per_arch_df, ["archetype_id", "question_id"]),
    ]:
        report_lines.append(f"\n### {label}")
        summary = radius_summary(df_)
        report_lines.append(
            "\n| Condition | TRM (up) | RC (up) | TVD (down) | DH (up) | KL nats (down) | KL divergent share |\n"
            "|---|---|---|---|---|---|---|"
        )
        for _, row in summary.iterrows():
            report_lines.append(
                f"| {row['source']} | {row['TRM']:.3f} | {row['RC']:.3f} | {row['TVD']:.3f} | {row['DH']:.3f} "
                f"| {row['KL']:.3f} | {row['KL_inf']:.3f} |"
            )
        for pair in [
            ("multi_respondent", "independent_coarse"),
            ("persistent_batched", "multi_respondent"),
            ("persistent_batched", "independent_coarse"),
        ]:
            if set(pair) <= set(df_.source.unique()):
                tests = radius_paired_tests(df_, pair[0], pair[1], keys)
                report_lines.append(f"\nPaired tests, {pair[0]} vs {pair[1]}:")
                report_lines.append("```\n" + json.dumps(tests, indent=2) + "\n```")


def _append_scd_summary(report_lines: list[str], scd_summary: pd.DataFrame) -> None:
    report_lines.append("\n## Self-Correlation Distance (inter-question structure)")
    report_lines.append(
        "\nSCD = mean |r_human - r_sim| over valid question pairs (down); "
        "structure recovery = Pearson r between the human and simulated "
        "off-diagonal correlation vectors, with Mantel permutation p (up); "
        "undefined share = fraction of question pairs whose simulated correlation "
        "is undefined (insufficient overlap or zero variance)."
    )
    for granularity, sub in scd_summary.groupby("granularity"):
        report_lines.append(f"\n### {granularity}")
        report_lines.append(
            "\n| Condition | SCD (down) | Structure recovery r (up) | Mantel p | "
            "mean abs r sim | mean abs r human | valid pairs | undefined share |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        for _, row in sub.iterrows():
            report_lines.append(
                f"| {row['source']} | {row['SCD']:.3f} | {row['structure_recovery_r']:.3f} | "
                f"{row['mantel_p']:.3f} | {row['mean_abs_r_sim']:.3f} | {row['mean_abs_r_human']:.3f} | "
                f"{row['valid_pairs']} | {row['undefined_share']:.3f} |"
            )


def run_analysis(cfg: Config, archetypes, questions) -> pd.DataFrame:
    n_options_by_question = {q.question_id: q.n_options for q in questions}
    independent_label = _independent_label(cfg)

    print("Loading human ground truth (wave4) ...")
    frames = [human_long_df(cfg, archetypes, questions)]
    frames.extend(load_simulated_long_frames(cfg))

    print("Loading precomputed GPT-4.1-mini independent-twin baseline ...")
    frames.append(baseline_long_df(cfg, archetypes, questions))

    long_df = pd.concat(frames, ignore_index=True)
    comparison_df = build_comparison_table(long_df, n_options_by_question)
    comparison_path = cfg.results_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Wrote per (archetype, question, source) comparison table to {comparison_path}")

    report_lines = ["# Variance-collapse experiment: results summary\n"]
    report_lines.append(
        f"- Archetypes: {len(archetypes)} | Questions: {len(questions)} | "
        f"Independent-condition source: `{independent_label}`\n"
    )
    _append_variance_summary(report_lines, comparison_df, independent_label)

    print("Computing RADIUS alignment suite (TRM / RC / TVD / DH) ...")
    per_arch_df, pooled_df = compute_radius_tables(long_df, n_options_by_question)
    per_arch_df.to_csv(cfg.results_dir / "radius_per_archetype.csv", index=False)
    pooled_df.to_csv(cfg.results_dir / "radius_pooled.csv", index=False)
    print(f"Wrote RADIUS tables to {cfg.results_dir}/radius_per_archetype.csv and radius_pooled.csv")
    _append_radius_summary(report_lines, per_arch_df, pooled_df)

    print("Computing self-correlation distance (inter-question structure) ...")
    matrices = person_question_matrices(cfg, archetypes, questions)
    human_mat = matrices.pop("human")
    scd_frames, pair_frames = [], []
    for demean, granularity in [(False, "pooled_raw"), (True, "within_archetype")]:
        summary_df, pairs_df = compare_structures(human_mat, matrices, demean_by_archetype=demean)
        summary_df.insert(0, "granularity", granularity)
        pairs_df.insert(0, "granularity", granularity)
        scd_frames.append(summary_df)
        pair_frames.append(pairs_df)
    scd_summary = pd.concat(scd_frames, ignore_index=True)
    scd_pairs = pd.concat(pair_frames, ignore_index=True)
    scd_summary.to_csv(cfg.results_dir / "self_correlation_summary.csv", index=False)
    scd_pairs.to_csv(cfg.results_dir / "self_correlation_pairs.csv", index=False)
    print(f"Wrote self-correlation tables to {cfg.results_dir}/self_correlation_*.csv")
    _append_scd_summary(report_lines, scd_summary)

    make_structure_plot(cfg, scd_pairs)

    report_path = cfg.results_dir / "summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote human-readable summary to {report_path}")

    make_variance_plot(cfg, comparison_df, independent_label)
    return comparison_df
