"""Download and load the pieces of Twin-2K-500 this experiment needs.

We deliberately avoid pulling the full ~700MB dataset. Only these small,
clean files are used:

  - question_catalog_and_human_response_csv/question_catalog.json
      Metadata (text, options, type, block) for every question.
  - question_catalog_and_human_response_csv/wave1_3_response.csv
      Numeric wave1-3 answers (incl. the 14 Demographics questions we use
      to build "identical profile" archetypes).
  - question_catalog_and_human_response_csv/wave4_response.csv
      Numeric wave4 answers -- our human ground truth for the target questions.
  - LLM_simulation_results/wave4_formatted_to_catalog_mapping.json
      Maps the "formatted" baseline CSV columns back to QuestionIDs.
  - LLM_simulation_results/GPT4.1-mini-simulation-llm-vs-human/*.csv
      The dataset's own precomputed independent-twin simulation (GPT-4.1-mini),
      reused as the "independent prompting" baseline condition.

All files are cached under `dataset.cache_dir` (default `data/raw/`) via
`huggingface_hub.hf_hub_download`, so re-running is instant after the first
fetch and works offline afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from .config import Config

QUESTION_CATALOG_PATH = "question_catalog_and_human_response_csv/question_catalog.json"
WAVE1_3_RESPONSE_PATH = "question_catalog_and_human_response_csv/wave1_3_response.csv"
WAVE4_RESPONSE_PATH = "question_catalog_and_human_response_csv/wave4_response.csv"
WAVE4_RESPONSE_LABEL_PATH = "question_catalog_and_human_response_csv/wave4_response_label.csv"


def _fetch(cfg: Config, repo_relpath: str) -> Path:
    local_path = hf_hub_download(
        repo_id=cfg.hf_repo,
        repo_type="dataset",
        filename=repo_relpath,
        local_dir=str(cfg.dataset_cache_dir),
    )
    return Path(local_path)


def fetch_all(cfg: Config) -> None:
    """Pre-download every file this experiment touches (nice for `fetch-data`)."""
    paths = [
        QUESTION_CATALOG_PATH,
        WAVE1_3_RESPONSE_PATH,
        WAVE4_RESPONSE_PATH,
        WAVE4_RESPONSE_LABEL_PATH,
        cfg.mapping_file,
        f"{cfg.baseline_dir}/{cfg.baseline_human_file}",
        f"{cfg.baseline_dir}/{cfg.baseline_llm_file}",
    ]
    for p in paths:
        local = _fetch(cfg, p)
        print(f"  fetched {p} -> {local}")


def load_question_catalog(cfg: Config) -> list[dict]:
    path = _fetch(cfg, QUESTION_CATALOG_PATH)
    with open(path) as f:
        return json.load(f)


def load_wave1_3_responses(cfg: Config) -> pd.DataFrame:
    path = _fetch(cfg, WAVE1_3_RESPONSE_PATH)
    return pd.read_csv(path)


def load_wave4_responses(cfg: Config) -> pd.DataFrame:
    path = _fetch(cfg, WAVE4_RESPONSE_PATH)
    return pd.read_csv(path)


def load_wave4_response_labels(cfg: Config) -> pd.DataFrame:
    path = _fetch(cfg, WAVE4_RESPONSE_LABEL_PATH)
    return pd.read_csv(path)


def load_wave4_mapping(cfg: Config) -> list[dict]:
    path = _fetch(cfg, cfg.mapping_file)
    with open(path) as f:
        return json.load(f)
