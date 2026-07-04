"""Build 'identical profile' archetype groups from wave1-3 demographics.

This operationalizes the paper's setup: "digital twins constructed with
identical profile attributes" -- we group real Twin-2K-500 participants who
match *exactly* on a handful of coarse demographic questions, then treat each
group as one archetype. Because these are real, distinct human beings who
happen to share the same surface profile, their real wave4 answers give us a
ground-truth measurement of how much genuine response variance *should*
exist within an "identical" persona -- the paper's "recall" target that LLM
twins are hypothesized to under-produce.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

from .config import Config

ATTRIBUTE_LABELS = {
    "QID13": "Age",
    "QID12": "Sex assigned at birth",
    "QID11": "US region",
    "QID14": "Highest education completed",
    "QID15": "Race or origin",
    "QID21": "Household income (past year)",
    "QID24": "Employment status",
}


@dataclass
class Archetype:
    archetype_id: str
    attribute_values: dict[str, int]     # {QID: 1-based option position}
    attribute_labels: dict[str, str]     # {QID: human-readable option text}
    pids: list[int]

    @property
    def size(self) -> int:
        return len(self.pids)

    def profile_description(self) -> str:
        parts = [
            f"{ATTRIBUTE_LABELS.get(qid, qid)}: {label}"
            for qid, label in self.attribute_labels.items()
        ]
        return "; ".join(parts)


def _catalog_by_qid(catalog: list[dict]) -> dict[str, dict]:
    return {q["QuestionID"]: q for q in catalog}


def build_archetypes(
    wave1_3_df: pd.DataFrame,
    catalog: list[dict],
    cfg: Config,
) -> list[Archetype]:
    attributes = cfg.archetype_attributes
    cat_by_qid = _catalog_by_qid(catalog)

    df = wave1_3_df[["pid"] + attributes].dropna()
    for a in attributes:
        df[a] = df[a].astype(int)

    groups = df.groupby(attributes)["pid"].apply(list)
    groups = groups[groups.apply(len) >= cfg.archetype_min_group_size]

    archetypes: list[Archetype] = []
    for key, pids in groups.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        attribute_values = dict(zip(attributes, key_tuple))
        attribute_labels = {}
        for qid, position in attribute_values.items():
            options = cat_by_qid.get(qid, {}).get("Options", [])
            label = options[position - 1] if 0 < position <= len(options) else str(position)
            attribute_labels[qid] = label
        archetype_id = "arch_" + "_".join(f"{qid}-{v}" for qid, v in attribute_values.items())
        archetypes.append(
            Archetype(
                archetype_id=archetype_id,
                attribute_values=attribute_values,
                attribute_labels=attribute_labels,
                pids=sorted(int(p) for p in pids),
            )
        )

    rng = random.Random(cfg.archetype_seed)
    if len(archetypes) > cfg.archetype_n_groups:
        archetypes = rng.sample(archetypes, cfg.archetype_n_groups)

    archetypes.sort(key=lambda a: -a.size)
    return archetypes


def archetypes_to_records(archetypes: list[Archetype]) -> list[dict]:
    return [
        {
            "archetype_id": a.archetype_id,
            "attribute_values": a.attribute_values,
            "attribute_labels": a.attribute_labels,
            "profile_description": a.profile_description(),
            "size": a.size,
            "pids": a.pids,
        }
        for a in archetypes
    ]


def archetypes_from_records(records: list[dict]) -> list[Archetype]:
    return [
        Archetype(
            archetype_id=r["archetype_id"],
            attribute_values=r["attribute_values"],
            attribute_labels=r["attribute_labels"],
            pids=r["pids"],
        )
        for r in records
    ]
