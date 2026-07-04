"""Build, persist, and load the experiment plan (archetypes + questions)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..archetypes import archetypes_from_records, archetypes_to_records, build_archetypes
from ..config import Config
from ..data_loading import load_question_catalog, load_wave1_3_responses, load_wave4_mapping
from ..question_selection import questions_from_records, questions_to_records, select_questions
from .constants import PLAN_FILENAME


def plan_path(cfg: Config) -> Path:
    return cfg.results_dir / PLAN_FILENAME


def build_plan(cfg: Config) -> None:
    print("Loading question catalog, wave1-3 responses, wave4 mapping ...")
    catalog = load_question_catalog(cfg)
    wave1_3 = load_wave1_3_responses(cfg)
    mapping = load_wave4_mapping(cfg)

    print("Building archetype groups from shared demographic profiles ...")
    archetypes = build_archetypes(wave1_3, catalog, cfg)
    print(f"  -> {len(archetypes)} archetypes, sizes {[a.size for a in archetypes[:5]]} ...")

    print("Selecting target wave4 questions ...")
    questions = select_questions(catalog, mapping, cfg)
    print(
        f"  -> {len(questions)} questions across blocks: "
        f"{sorted(set(q.block_name for q in questions))}"
    )

    save_plan(cfg, archetypes, questions)


def save_plan(cfg, archetypes, questions) -> Path:
    path = plan_path(cfg)
    payload = {
        "archetypes": archetypes_to_records(archetypes),
        "questions": questions_to_records(questions),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote plan to {path}")
    return path


def load_plan(cfg: Config):
    path = plan_path(cfg)
    if not path.exists():
        sys.exit(f"No plan found at {path}. Run `build-plan` first.")
    with open(path) as f:
        plan = json.load(f)
    return archetypes_from_records(plan["archetypes"]), questions_from_records(plan["questions"])
