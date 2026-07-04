"""Build long-format and respondent x question matrices for analysis."""
from __future__ import annotations

import pandas as pd

from ..baseline import load_precomputed_independent_baseline
from ..config import Config
from ..data_loading import load_wave4_mapping, load_wave4_responses
from .constants import SIMULATED_CONDITION_FILES


def human_long_df(cfg: Config, archetypes, questions) -> pd.DataFrame:
    wave4 = load_wave4_responses(cfg)
    pid_to_archetype = {pid: a.archetype_id for a in archetypes for pid in a.pids}

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


def baseline_long_df(cfg: Config, archetypes, questions) -> pd.DataFrame:
    mapping = load_wave4_mapping(cfg)
    baseline = load_precomputed_independent_baseline(cfg, mapping)
    pid_to_archetype = {pid: a.archetype_id for a in archetypes for pid in a.pids}

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


def person_question_matrices(cfg: Config, archetypes, questions) -> dict[str, pd.DataFrame]:
    """Respondent x question matrices for the self-correlation analysis."""
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
    matrices["baseline_precomputed"] = from_wide(load_precomputed_independent_baseline(cfg, mapping))

    for source, filename in SIMULATED_CONDITION_FILES.items():
        path = cfg.results_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        mat = df.pivot_table(
            index=["archetype_id", "person_index"],
            columns="question_id",
            values="selected_position",
        )
        mat["archetype_id"] = mat.index.get_level_values(0)
        matrices[source] = mat

    return matrices


def load_simulated_long_frames(cfg: Config) -> list[pd.DataFrame]:
    frames = []
    for source, filename in SIMULATED_CONDITION_FILES.items():
        path = cfg.results_dir / filename
        if not path.exists():
            if source == "multi_respondent":
                print(f"WARNING: {path} not found -- run `simulate --condition multi_respondent` first.")
            continue
        df = pd.read_csv(path)[["archetype_id", "question_id", "selected_position"]]
        df["source"] = source
        frames.append(df)
    return frames
