"""Diagnostic plots written to results/."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config


def make_variance_plot(cfg: Config, comparison_df: pd.DataFrame, independent_label: str) -> None:
    sources = [s for s in [independent_label, "multi_respondent"] if s in comparison_df.source.unique()]
    if not sources:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

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


def make_structure_plot(cfg: Config, scd_pairs: pd.DataFrame) -> None:
    sub = scd_pairs[scd_pairs.granularity == "within_archetype"]
    sources = [
        s
        for s in ["independent_coarse", "multi_respondent", "persistent_batched", "baseline_precomputed"]
        if s in sub.source.unique()
    ]
    if not sources:
        return

    display_names = {
        "independent_coarse": "Independent prompting",
        "multi_respondent": "Multi-respondent prompting",
        "persistent_batched": "Persistent batched personas",
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
