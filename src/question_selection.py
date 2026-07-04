"""Select a diverse set of wave4 questions to simulate.

We restrict to single-select multiple-choice (MC/SAVR, MC/SAHR) questions
with a small, bounded option set. These give a clean, discrete response
distribution (needed for variance/entropy comparisons) and happen to cover
most of Twin-2K-500's behavioral-economics heuristics & biases replications
(anchoring, framing, sunk cost, WTA/WTP, Allais, proportion dominance, ...)
-- exactly the kind of items where real humans sharing a demographic profile
are known to disagree, making them a strong testbed for the paper's
variance-collapse claim.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .config import Config


@dataclass
class QuestionSpec:
    question_id: str
    question_text: str
    options: list[str]
    block_name: str
    catalog_csv_column: str  # column name in wave1_3_response.csv / wave4_response.csv
    formatted_column: str | None  # column name in the precomputed baseline CSVs

    @property
    def n_options(self) -> int:
        return len(self.options)

    def options_block(self) -> str:
        return "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(self.options))


def select_questions(
    catalog: list[dict],
    mapping: list[dict],
    cfg: Config,
) -> list[QuestionSpec]:
    cat_by_qid = {q["QuestionID"]: q for q in catalog}
    formatted_by_qid = {m["QuestionID"]: m["formatted_column"] for m in mapping}

    candidates: list[QuestionSpec] = []
    seen_qids: set[str] = set()
    for m in mapping:
        qid = m["QuestionID"]
        if qid in seen_qids:
            continue
        q = cat_by_qid.get(qid)
        if not q or q.get("is_descriptive"):
            continue
        if q["QuestionType"] not in cfg.allowed_question_types:
            continue
        options = q.get("Options") or []
        if not (cfg.min_options <= len(options) <= cfg.max_options):
            continue
        text = (q.get("QuestionText") or "").strip()
        if not text:
            continue
        csv_col = m["catalog_csv_column"]
        # catalog_csv_column may itself be "QID_sub"; only take plain single-column MC items here.
        if csv_col != qid:
            continue
        candidates.append(
            QuestionSpec(
                question_id=qid,
                question_text=text,
                options=options,
                block_name=q.get("BlockName", "Unknown"),
                catalog_csv_column=csv_col,
                formatted_column=formatted_by_qid.get(qid),
            )
        )
        seen_qids.add(qid)

    rng = random.Random(cfg.question_seed)
    rng.shuffle(candidates)

    selected: list[QuestionSpec] = []
    per_block_count: dict[str, int] = {}
    for c in candidates:
        if len(selected) >= cfg.n_questions:
            break
        if per_block_count.get(c.block_name, 0) >= cfg.max_per_block:
            continue
        selected.append(c)
        per_block_count[c.block_name] = per_block_count.get(c.block_name, 0) + 1

    # Backfill if block-diversity cap left us short (e.g. too few distinct blocks).
    if len(selected) < cfg.n_questions:
        for c in candidates:
            if len(selected) >= cfg.n_questions:
                break
            if c not in selected:
                selected.append(c)

    return selected


def questions_to_records(questions: list[QuestionSpec]) -> list[dict]:
    return [
        {
            "question_id": q.question_id,
            "question_text": q.question_text,
            "options": q.options,
            "block_name": q.block_name,
            "catalog_csv_column": q.catalog_csv_column,
            "formatted_column": q.formatted_column,
        }
        for q in questions
    ]


def questions_from_records(records: list[dict]) -> list[QuestionSpec]:
    return [
        QuestionSpec(
            question_id=r["question_id"],
            question_text=r["question_text"],
            options=r["options"],
            block_name=r["block_name"],
            catalog_csv_column=r["catalog_csv_column"],
            formatted_column=r.get("formatted_column"),
        )
        for r in records
    ]
