"""Run the two new-API-call simulation conditions over every
(archetype, question) pair and return tidy long-format results.

Calls are dispatched concurrently (bounded by `simulation.max_concurrency`)
since a full run can be thousands of API calls. Each individual call is
still cached to disk by `llm_client.call_tool`, so an interrupted run can
simply be restarted and will skip everything already completed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from .archetypes import Archetype
from .config import Config
from .llm_client import call_tool
from .prompts import build_independent_prompt, build_multi_respondent_prompt
from .question_selection import QuestionSpec


@dataclass
class SimResponse:
    archetype_id: str
    question_id: str
    condition: str  # "multi_respondent" | "independent_coarse"
    person_index: int
    selected_position: int


def _run_concurrent(tasks, worker, max_workers: int, desc: str) -> list:
    """Run `worker(task)` over `tasks` with a bounded thread pool, showing a
    progress bar, and returning results in completion order (order doesn't
    matter downstream since everything is tagged with archetype/question ids).
    Exceptions from individual calls are logged and skipped rather than
    aborting the whole run.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            task = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.extend(result) if isinstance(result, list) else results.append(result)
            except Exception as e:  # noqa: BLE001 - keep going on individual failures
                tqdm.write(f"  [{desc}] FAILED for {task}: {e}")
    return results


def run_multi_respondent(
    cfg: Config, archetypes: list[Archetype], questions: list[QuestionSpec]
) -> list[SimResponse]:
    n = cfg.n_simulated_per_group
    pairs = [(a, q) for a in archetypes for q in questions]

    def worker(pair):
        archetype, question = pair
        system, user, tool = build_multi_respondent_prompt(archetype, question, n)
        parsed = call_tool(cfg, system, user, tool, cache_subdir="multi_respondent")
        return [
            SimResponse(
                archetype_id=archetype.archetype_id,
                question_id=question.question_id,
                condition="multi_respondent",
                person_index=int(r["person_index"]),
                selected_position=int(r["selected_position"]),
            )
            for r in parsed["responses"]
        ]

    return _run_concurrent(pairs, worker, cfg.max_concurrency, "multi_respondent")


def run_independent_coarse(
    cfg: Config, archetypes: list[Archetype], questions: list[QuestionSpec]
) -> list[SimResponse]:
    n = cfg.n_simulated_per_group
    tasks = [
        (archetype, question, person_index)
        for archetype in archetypes
        for question in questions
        for person_index in range(1, n + 1)
    ]

    def worker(task):
        archetype, question, person_index = task
        system, user, tool = build_independent_prompt(archetype, question)
        # Identical profile/question every time by design -- this is the exact
        # "same profile in, same call out" setup the paper diagnoses. The
        # respondent tag only exists so each of the n calls gets its own cache
        # entry (and therefore its own fresh sample from the model) instead of
        # all n collapsing onto a single cached response.
        parsed = call_tool(
            cfg, system, f"{user}\n\n[[respondent #{person_index} of {n}]]", tool,
            cache_subdir="independent_coarse",
        )
        return SimResponse(
            archetype_id=archetype.archetype_id,
            question_id=question.question_id,
            condition="independent_coarse",
            person_index=person_index,
            selected_position=int(parsed["selected_position"]),
        )

    return _run_concurrent(tasks, worker, cfg.max_concurrency, "independent_coarse")


def responses_to_records(responses: list[SimResponse]) -> list[dict]:
    return [r.__dict__ for r in responses]
