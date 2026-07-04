"""Load config.yaml + .env into a single, typed-ish config object."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def hf_repo(self) -> str:
        return self.raw["dataset"]["hf_repo"]

    @property
    def dataset_cache_dir(self) -> Path:
        return REPO_ROOT / self.raw["dataset"]["cache_dir"]

    @property
    def baseline_dir(self) -> str:
        return self.raw["dataset"]["baseline_dir"]

    @property
    def baseline_human_file(self) -> str:
        return self.raw["dataset"]["baseline_human_file"]

    @property
    def baseline_llm_file(self) -> str:
        return self.raw["dataset"]["baseline_llm_file"]

    @property
    def mapping_file(self) -> str:
        return self.raw["dataset"]["mapping_file"]

    @property
    def archetype_attributes(self) -> list[str]:
        return list(self.raw["archetype"]["attributes"])

    @property
    def archetype_min_group_size(self) -> int:
        return int(self.raw["archetype"]["min_group_size"])

    @property
    def archetype_n_groups(self) -> int:
        return int(self.raw["archetype"]["n_groups"])

    @property
    def n_simulated_per_group(self) -> int:
        return int(self.raw["archetype"]["n_simulated_per_group"])

    @property
    def archetype_seed(self) -> int:
        return int(self.raw["archetype"]["random_seed"])

    @property
    def n_questions(self) -> int:
        return int(self.raw["questions"]["n_questions"])

    @property
    def allowed_question_types(self) -> list[str]:
        return list(self.raw["questions"]["allowed_types"])

    @property
    def min_options(self) -> int:
        return int(self.raw["questions"]["min_options"])

    @property
    def max_options(self) -> int:
        return int(self.raw["questions"]["max_options"])

    @property
    def max_per_block(self) -> int:
        return int(self.raw["questions"]["max_per_block"])

    @property
    def question_seed(self) -> int:
        return int(self.raw["questions"]["random_seed"])

    @property
    def model(self) -> str:
        return os.environ.get("ANTHROPIC_MODEL", self.raw["simulation"]["model"])

    @property
    def simulation_seed(self) -> int | None:
        seed = self.raw["simulation"].get("random_seed")
        return int(seed) if seed is not None else None

    @property
    def temperature(self) -> float:
        return float(self.raw["simulation"]["temperature"])

    @property
    def max_tokens(self) -> int:
        return int(self.raw["simulation"]["max_tokens"])

    @property
    def request_timeout_s(self) -> float:
        return float(self.raw["simulation"]["request_timeout_s"])

    @property
    def max_retries(self) -> int:
        return int(self.raw["simulation"]["max_retries"])

    @property
    def max_concurrency(self) -> int:
        return int(self.raw["simulation"].get("max_concurrency", 8))

    @property
    def independent_source(self) -> str:
        return self.raw["simulation"]["independent_source"]

    @property
    def cache_dir(self) -> Path:
        return REPO_ROOT / self.raw["paths"]["cache_dir"]

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / self.raw["paths"]["results_dir"]


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> Config:
    load_dotenv(REPO_ROOT / ".env")
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw=raw)
    cfg.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    return cfg
