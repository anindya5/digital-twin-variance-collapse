"""Load the dataset's precomputed GPT-4.1-mini independent-twin simulation.

These CSVs are exported straight from the simulation pipeline in Qualtrics
"formatted" shape: row 0 is short column codes (incl. `TWIN_ID`), row 1 is the
full question text (metadata, not data), and real rows start at row 2. We skip
row 1 and rename `TWIN_ID` -> `pid` so it joins cleanly against the rest of
the dataset.
"""
from __future__ import annotations

import pandas as pd

from .config import Config
from .data_loading import _fetch


def _load_formatted_csv(cfg: Config, filename: str) -> pd.DataFrame:
    path = _fetch(cfg, f"{cfg.baseline_dir}/{filename}")
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    df = df.rename(columns={"TWIN_ID": "pid"})
    df["pid"] = pd.to_numeric(df["pid"], errors="coerce")
    df = df.dropna(subset=["pid"])
    df["pid"] = df["pid"].astype(int)
    return df


def load_precomputed_independent_baseline(cfg: Config, mapping: list[dict]) -> pd.DataFrame:
    """Return a DataFrame indexed by pid with QID-style columns (matching
    wave4_response.csv's column naming), containing GPT-4.1-mini's
    independently-simulated answer for each question, as 1-based option
    positions where resolvable.
    """
    llm_df = _load_formatted_csv(cfg, cfg.baseline_llm_file)

    col_to_qid_col = {m["formatted_column"]: m["catalog_csv_column"] for m in mapping}

    out = pd.DataFrame({"pid": llm_df["pid"]})
    for formatted_col, qid_col in col_to_qid_col.items():
        if formatted_col in llm_df.columns:
            # Values are 1-based option positions, sometimes round-tripped as
            # floats (e.g. "1.0") through the CSV export. Coerce to nullable ints.
            out[qid_col] = pd.to_numeric(llm_df[formatted_col], errors="coerce").round().astype("Int64")
    return out
