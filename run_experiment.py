#!/usr/bin/env python3
"""CLI orchestrator for testing the paper's variance-collapse hypothesis on
Twin-2K-500.

Typical usage:

    python run_experiment.py fetch-data
    python run_experiment.py build-plan
    python run_experiment.py simulate --condition multi_respondent
    # optional, costs ~n_simulated_per_group x more API calls:
    # python run_experiment.py simulate --condition independent_coarse
    python run_experiment.py analyze

Or just:

    python run_experiment.py all

Run `python run_experiment.py self-test` first (no API key required) to sanity
check the whole pipeline end-to-end on synthetic data.
"""
from __future__ import annotations

import argparse
import json
import random
import sys

import pandas as pd

from src.archetypes import archetypes_from_records, archetypes_to_records, build_archetypes
from src.baseline import load_precomputed_independent_baseline
from src.config import load_config
from src.data_loading import (
    fetch_all,
    load_question_catalog,
    load_wave1_3_responses,
    load_wave4_mapping,
    load_wave4_responses,
)
from src.metrics import build_comparison_table, paired_condition_test
from src.radius import compute_radius_tables, radius_paired_tests, radius_summary
from src.self_correlation import compare_structures
from src.question_selection import questions_from_records, questions_to_records, select_questions
from src.simulate import responses_to_records, run_independent_coarse, run_multi_respondent

PLAN_PATH_NAME = "plan.json"


def cmd_fetch_data(args):
    cfg = load_config()
    print(f"Fetching required files from {cfg.hf_repo} into {cfg.dataset_cache_dir} ...")
    fetch_all(cfg)
    print("Done.")


def cmd_build_plan(args):
    cfg = load_config()
    print("Loading question catalog, wave1-3 responses, wave4 mapping ...")
    catalog = load_question_catalog(cfg)
    wave1_3 = load_wave1_3_responses(cfg)
    mapping = load_wave4_mapping(cfg)

    print("Building archetype groups from shared demographic profiles ...")
    archetypes = build_archetypes(wave1_3, catalog, cfg)
    print(f"  -> {len(archetypes)} archetypes, sizes {[a.size for a in archetypes[:5]]} ...")

    print("Selecting target wave4 questions ...")
    questions = select_questions(catalog, mapping, cfg)
    print(f"  -> {len(questions)} questions across blocks: "
          f"{sorted(set(q.block_name for q in questions))}")

    plan = {
        "archetypes": archetypes_to_records(archetypes),
        "questions": questions_to_records(questions),
    }
    plan_path = cfg.results_dir / PLAN_PATH_NAME
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Wrote plan to {plan_path}")


def _load_plan(cfg):
    plan_path = cfg.results_dir / PLAN_PATH_NAME
    if not plan_path.exists():
        sys.exit(f"No plan found at {plan_path}. Run `build-plan` first.")
    with open(plan_path) as f:
        plan = json.load(f)
    archetypes = archetypes_from_records(plan["archetypes"])
    questions = questions_from_records(plan["questions"])
    return archetypes, questions


def cmd_simulate(args):
    cfg = load_config()
    archetypes, questions = _load_plan(cfg)

    n_calls_multi = len(archetypes) * len(questions)
    n_calls_indep = len(archetypes) * len(questions) * cfg.n_simulated_per_group

    if args.condition == "multi_respondent":
        print(f"About to make up to {n_calls_multi} API calls (1 per archetype x question, "
              f"each producing {cfg.n_simulated_per_group} simulated answers).")
        if not args.yes and not _confirm():
            return
        responses = run_multi_respondent(cfg, archetypes, questions)
        out_path = cfg.results_dir / "simulated_multi_respondent.csv"
    else:
        print(f"About to make up to {n_calls_indep} API calls (1 per archetype x question x "
              f"simulated twin). This is {cfg.n_simulated_per_group}x more expensive than "
              "multi_respondent -- make sure that's intended.")
        if not args.yes and not _confirm():
            return
        responses = run_independent_coarse(cfg, archetypes, questions)
        out_path = cfg.results_dir / "simulated_independent_coarse.csv"

    df = pd.DataFrame(responses_to_records(responses))
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} simulated responses to {out_path}")


def _confirm() -> bool:
    reply = input("Proceed? [y/N] ").strip().lower()
    return reply == "y"


def _human_long_df(cfg, archetypes, questions) -> pd.DataFrame:
    wave4 = load_wave4_responses(cfg)
    pid_to_archetype: dict[int, str] = {}
    for a in archetypes:
        for pid in a.pids:
            pid_to_archetype[pid] = a.archetype_id

    rows = []
    for q in questions:
        if q.catalog_csv_column not in wave4.columns:
            continue
        sub = wave4[["pid", q.catalog_csv_column]].dropna()
        for _, r in sub.iterrows():
            pid = int(r["pid"])
            archetype_id = pid_to_archetype.get(pid)
            if archetype_id is None:
                continue
            rows.append(
                {
                    "archetype_id": archetype_id,
                    "question_id": q.question_id,
                    "source": "human",
                    "selected_position": int(r[q.catalog_csv_column]),
                }
            )
    return pd.DataFrame(rows)


def _person_question_matrices(cfg, archetypes, questions) -> dict[str, pd.DataFrame]:
    """Respondent x question matrices for the self-correlation analysis.

    Humans and the precomputed baseline are linked across questions by pid;
    the simulated conditions by (archetype_id, person_index). Each matrix has
    one row per respondent, one column per question_id, plus `archetype_id`.
    """
    pid_to_archetype = {p: a.archetype_id for a in archetypes for p in a.pids}
    cols = {q.question_id: q.catalog_csv_column for q in questions}

    def from_wide(wide: pd.DataFrame) -> pd.DataFrame:
        sub = wide[wide.pid.isin(pid_to_archetype)].set_index("pid")
        mat = pd.DataFrame(
            {qid: pd.to_numeric(sub[c], errors="coerce") for qid, c in cols.items() if c in sub.columns}
        )
        mat["archetype_id"] = [pid_to_archetype[p] for p in mat.index]
        return mat

    matrices = {"human": from_wide(load_wave4_responses(cfg))}

    mapping = load_wave4_mapping(cfg)
    matrices["baseline_precomputed"] = from_wide(
        load_precomputed_independent_baseline(cfg, mapping)
    )

    for filename, source in [
        ("simulated_independent_coarse.csv", "independent_coarse"),
        ("simulated_multi_respondent.csv", "multi_respondent"),
    ]:
        path = cfg.results_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        mat = df.pivot_table(
            index=["archetype_id", "person_index"], columns="question_id", values="selected_position"
        )
        mat["archetype_id"] = mat.index.get_level_values(0)
        matrices[source] = mat

    return matrices


def _baseline_long_df(cfg, archetypes, questions) -> pd.DataFrame:
    mapping = load_wave4_mapping(cfg)
    baseline = load_precomputed_independent_baseline(cfg, mapping)
    pid_to_archetype: dict[int, str] = {}
    for a in archetypes:
        for pid in a.pids:
            pid_to_archetype[pid] = a.archetype_id

    rows = []
    for q in questions:
        if q.catalog_csv_column not in baseline.columns:
            continue
        sub = baseline[["pid", q.catalog_csv_column]].dropna()
        for _, r in sub.iterrows():
            pid = int(r["pid"])
            archetype_id = pid_to_archetype.get(pid)
            if archetype_id is None:
                continue
            rows.append(
                {
                    "archetype_id": archetype_id,
                    "question_id": q.question_id,
                    "source": "baseline_precomputed",
                    "selected_position": int(r[q.catalog_csv_column]),
                }
            )
    return pd.DataFrame(rows)


def cmd_analyze(args):
    cfg = load_config()
    archetypes, questions = _load_plan(cfg)
    n_options_by_question = {q.question_id: q.n_options for q in questions}

    print("Loading human ground truth (wave4) ...")
    human_df = _human_long_df(cfg, archetypes, questions)

    frames = [human_df]

    multi_path = cfg.results_dir / "simulated_multi_respondent.csv"
    if multi_path.exists():
        multi_df = pd.read_csv(multi_path)[["archetype_id", "question_id", "selected_position"]]
        multi_df["source"] = "multi_respondent"
        frames.append(multi_df)
    else:
        print(f"WARNING: {multi_path} not found -- run `simulate --condition multi_respondent` first.")

    indep_path = cfg.results_dir / "simulated_independent_coarse.csv"
    if indep_path.exists():
        indep_df = pd.read_csv(indep_path)[["archetype_id", "question_id", "selected_position"]]
        indep_df["source"] = "independent_coarse"
        frames.append(indep_df)

    # Always include the dataset's precomputed GPT-4.1-mini baseline so the
    # RADIUS evaluation covers all three conditions (it's free/local).
    print("Loading precomputed GPT-4.1-mini independent-twin baseline ...")
    frames.append(_baseline_long_df(cfg, archetypes, questions))

    long_df = pd.concat(frames, ignore_index=True)
    comparison_df = build_comparison_table(long_df, n_options_by_question)
    comparison_path = cfg.results_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Wrote per (archetype, question, source) comparison table to {comparison_path}")

    independent_label = (
        "baseline_precomputed" if cfg.independent_source == "precomputed" else "independent_coarse"
    )

    report_lines = ["# Variance-collapse experiment: results summary\n"]
    report_lines.append(
        f"- Archetypes: {len(archetypes)} | Questions: {len(questions)} | "
        f"Independent-condition source: `{independent_label}`\n"
    )

    import numpy as np

    for metric in ["variance_ratio", "entropy_ratio", "mode_match"]:
        for source in [s for s in [independent_label, "multi_respondent"] if s in comparison_df.source.unique()]:
            raw = comparison_df[comparison_df.source == source][metric].dropna()
            n_infinite = int(np.isinf(raw).sum())
            sub = raw[np.isfinite(raw)]
            if len(sub):
                note = f", {n_infinite} cases where humans unanimously agreed but the model didn't (ratio=inf, excluded from mean)" if n_infinite else ""
                report_lines.append(
                    f"- **{source}** mean {metric}: {sub.mean():.3f} (median {sub.median():.3f}, n={len(sub)}{note})"
                )

    if independent_label in comparison_df.source.unique() and "multi_respondent" in comparison_df.source.unique():
        report_lines.append("\n## Statistical comparison: multi_respondent vs " + independent_label)
        for metric in ["variance_ratio", "entropy_ratio", "mode_match"]:
            result = paired_condition_test(comparison_df, metric, "multi_respondent", independent_label)
            report_lines.append(f"\n### {metric}\n```\n{json.dumps(result, indent=2)}\n```")

    # ---- RADIUS alignment suite (arXiv:2603.19002) ----
    print("Computing RADIUS alignment suite (TRM / RC / TVD / DH) ...")
    per_arch_df, pooled_df = compute_radius_tables(long_df, n_options_by_question)
    per_arch_df.to_csv(cfg.results_dir / "radius_per_archetype.csv", index=False)
    pooled_df.to_csv(cfg.results_dir / "radius_pooled.csv", index=False)
    print(f"Wrote RADIUS tables to {cfg.results_dir}/radius_per_archetype.csv and radius_pooled.csv")

    report_lines.append("\n## RADIUS alignment suite")
    for label, df_, keys in [
        ("Pooled per question (population-level, RADIUS-canonical)", pooled_df, ["question_id"]),
        ("Per archetype-question (identical-profile design)", per_arch_df, ["archetype_id", "question_id"]),
    ]:
        report_lines.append(f"\n### {label}")
        summary = radius_summary(df_)
        report_lines.append(
            "\n| Condition | TRM (up) | RC (up) | TVD (down) | DH (up) | KL nats (down) | KL divergent share |\n|---|---|---|---|---|---|---|"
        )
        for _, row in summary.iterrows():
            report_lines.append(
                f"| {row['source']} | {row['TRM']:.3f} | {row['RC']:.3f} | {row['TVD']:.3f} | {row['DH']:.3f} "
                f"| {row['KL']:.3f} | {row['KL_inf']:.3f} |"
            )
        if {"multi_respondent", "independent_coarse"} <= set(df_.source.unique()):
            tests = radius_paired_tests(df_, "multi_respondent", "independent_coarse", keys)
            report_lines.append("\nPaired tests, multi_respondent vs independent_coarse:")
            report_lines.append("```\n" + json.dumps(tests, indent=2) + "\n```")

    # ---- Self-Correlation Distance (inter-question structural alignment) ----
    print("Computing self-correlation distance (inter-question structure) ...")
    matrices = _person_question_matrices(cfg, archetypes, questions)
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
            "\n| Condition | SCD (down) | Structure recovery r (up) | Mantel p | mean abs r sim | mean abs r human | valid pairs | undefined share |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        for _, row in sub.iterrows():
            report_lines.append(
                f"| {row['source']} | {row['SCD']:.3f} | {row['structure_recovery_r']:.3f} | "
                f"{row['mantel_p']:.3f} | {row['mean_abs_r_sim']:.3f} | {row['mean_abs_r_human']:.3f} | "
                f"{row['valid_pairs']} | {row['undefined_share']:.3f} |"
            )

    _make_structure_plot(cfg, scd_pairs)

    report_path = cfg.results_dir / "summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Wrote human-readable summary to {report_path}")

    _make_plots(cfg, comparison_df, independent_label)


def _make_plots(cfg, comparison_df, independent_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [s for s in [independent_label, "multi_respondent"] if s in comparison_df.source.unique()]
    if not sources:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    import numpy as np

    def _clean(series, cap=3):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        return s.clip(upper=cap)

    data = [_clean(comparison_df[comparison_df.source == s]["variance_ratio"]) for s in sources]
    axes[0].boxplot(data, tick_labels=sources)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("variance_ratio (model / human, capped at 3)")
    axes[0].set_title("Recall proxy: variance ratio vs. real human variance")

    data2 = [comparison_df[comparison_df.source == s]["norm_entropy"].dropna() for s in sources] + [
        comparison_df[comparison_df.source == "human"]["norm_entropy"].dropna()
    ]
    labels2 = sources + ["human"]
    axes[1].boxplot(data2, tick_labels=labels2)
    axes[1].set_ylabel("normalized entropy of answers")
    axes[1].set_title("Response diversity within archetype")

    fig.tight_layout()
    plot_path = cfg.results_dir / "variance_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote plot to {plot_path}")


def _make_structure_plot(cfg, scd_pairs):
    """Scatter of simulated vs human inter-question correlations, one panel
    per condition, within-archetype granularity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = scd_pairs[scd_pairs.granularity == "within_archetype"]
    sources = [
        s for s in ["independent_coarse", "multi_respondent", "baseline_precomputed"]
        if s in sub.source.unique()
    ]
    if not sources:
        return

    display_names = {
        "independent_coarse": "Independent prompting",
        "multi_respondent": "Multi-respondent prompting",
        "baseline_precomputed": "Rich-persona baseline",
    }
    fig, axes = plt.subplots(1, len(sources), figsize=(4.2 * len(sources), 4.2), sharex=True, sharey=True)
    if len(sources) == 1:
        axes = [axes]
    for ax, source in zip(axes, sources):
        d = sub[sub.source == source].dropna(subset=["r_human", "r_sim"])
        ax.axhline(0, color="lightgray", linewidth=0.8)
        ax.axvline(0, color="lightgray", linewidth=0.8)
        ax.plot([-1, 1], [-1, 1], color="gray", linestyle="--", linewidth=1, label="perfect recovery")
        ax.scatter(d.r_human, d.r_sim, s=18, alpha=0.7)
        ax.set_xlim(-0.6, 0.9)
        ax.set_ylim(-0.6, 0.9)
        ax.set_title(f"{display_names.get(source, source)} (n={len(d)} pairs)")
        ax.set_xlabel("human inter-question r")
    axes[0].set_ylabel("simulated inter-question r")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Within-archetype inter-question correlation structure: simulated vs. human")
    fig.tight_layout()
    plot_path = cfg.results_dir / "self_correlation_scatter.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Wrote plot to {plot_path}")


def cmd_paper_section(args):
    # Manuscript-drafting helper; src/paper_section.py is intentionally not
    # part of the public replication codebase (see .gitignore).
    try:
        from src.paper_section import write_paper_section
    except ImportError:
        sys.exit(
            "The paper-section command requires src/paper_section.py, which is "
            "not distributed with the replication codebase. All analysis "
            "outputs are in results/ after running `analyze`."
        )
    cfg = load_config()
    out_path = write_paper_section(cfg)
    print(f"Wrote paper-ready draft section to {out_path}")
    print("(gitignored -- copy/adapt whatever you need into your manuscript)")


def cmd_all(args):
    cmd_fetch_data(args)
    cmd_build_plan(args)
    args.condition = "multi_respondent"
    cmd_simulate(args)
    cmd_analyze(args)


def cmd_self_test(args):
    """Run the full metrics/analysis pipeline on fabricated data, with no
    network or API calls, to verify the plumbing before spending real money.
    """
    from src.archetypes import Archetype
    from src.question_selection import QuestionSpec

    rng = random.Random(0)
    archetypes = [Archetype(f"arch_{i}", {}, {"QID13": "30-49"}, list(range(20))) for i in range(5)]
    questions = [QuestionSpec(f"QID{i}", f"Question {i}?", ["A", "B", "C", "D"], "Block", f"QID{i}", f"Q{i}") for i in range(3)]
    n_options_by_question = {q.question_id: q.n_options for q in questions}

    rows = []
    for a in archetypes:
        for q in questions:
            # Real humans: spread out across options.
            for pid in a.pids:
                rows.append({"archetype_id": a.archetype_id, "question_id": q.question_id,
                             "source": "human", "selected_position": rng.randint(1, 4)})
            # Baseline independent condition: near-collapsed (mostly the same answer).
            fixed = rng.randint(1, 4)
            for i in range(10):
                pos = fixed if rng.random() < 0.9 else rng.randint(1, 4)
                rows.append({"archetype_id": a.archetype_id, "question_id": q.question_id,
                             "source": "baseline_precomputed", "selected_position": pos})
            # Multi-respondent condition: more spread, by construction.
            for i in range(10):
                rows.append({"archetype_id": a.archetype_id, "question_id": q.question_id,
                             "source": "multi_respondent", "selected_position": rng.randint(1, 4)})

    long_df = pd.DataFrame(rows)
    comparison_df = build_comparison_table(long_df, n_options_by_question)
    assert len(comparison_df) > 0, "comparison table is empty"

    result = paired_condition_test(comparison_df, "variance_ratio", "multi_respondent", "baseline_precomputed")
    print(json.dumps(result, indent=2))
    assert result["mean_a"] > result["mean_b"], (
        "expected the synthetic multi_respondent condition (more spread) to show a higher "
        "variance_ratio than the synthetic near-collapsed baseline condition"
    )
    print("\nSelf-test passed: data loading -> archetypes -> metrics -> stats pipeline is wired correctly.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch-data", help="Download required Twin-2K-500 files").set_defaults(func=cmd_fetch_data)
    sub.add_parser("build-plan", help="Build archetype groups + select questions").set_defaults(func=cmd_build_plan)

    p_sim = sub.add_parser("simulate", help="Run an LLM simulation condition")
    p_sim.add_argument("--condition", choices=["multi_respondent", "independent_coarse"], required=True)
    p_sim.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    p_sim.set_defaults(func=cmd_simulate)

    sub.add_parser("analyze", help="Compute metrics + significance tests + plots").set_defaults(func=cmd_analyze)
    sub.add_parser("paper-section", help="Render a paper-ready draft section from current results/ (gitignored)").set_defaults(func=cmd_paper_section)

    p_all = sub.add_parser("all", help="fetch-data -> build-plan -> simulate multi_respondent -> analyze")
    p_all.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    p_all.set_defaults(func=cmd_all)

    sub.add_parser("self-test", help="Verify the pipeline on synthetic data (no API key needed)").set_defaults(func=cmd_self_test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
