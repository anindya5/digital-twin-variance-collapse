"""Prompt + tool-schema construction for the two simulation conditions.

Both conditions are given the *exact same information*: only the archetype's
shared coarse demographic profile (never an individuating backstory). This
keeps the comparison clean -- the only thing that differs between conditions
is whether the model is asked to answer for one simulated twin per API call
("independent prompting") or for all N simulated twins in a single call
("multi-respondent prompting"), which is precisely the manipulation the paper
proposes as a mitigation for variance collapse.
"""
from __future__ import annotations

from .archetypes import Archetype
from .question_selection import QuestionSpec

SYSTEM_PROMPT = (
    "You are simulating how real survey respondents with a given demographic "
    "profile would answer a survey question. You are not the assistant here; "
    "you are role-playing plausible, realistic individual human respondents. "
    "Different people who share the same demographic profile still have "
    "different personal histories, moods, risk preferences, and opinions, so "
    "their answers to subjective or judgment-based questions can legitimately "
    "differ. Answer as those individuals would, not as an assistant giving "
    "advice."
)


def build_independent_prompt(archetype: Archetype, question: QuestionSpec) -> tuple[str, str, dict]:
    profile = archetype.profile_description()
    user = (
        f"Simulated person's profile:\n{profile}\n\n"
        f"Survey question:\n{question.question_text}\n\n"
        f"Options:\n{question.options_block()}\n\n"
        "Pick exactly one option, as this specific person plausibly would. "
        "Record your answer using the record_response tool."
    )
    tool = {
        "name": "record_response",
        "description": "Record this simulated person's single chosen answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selected_position": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": question.n_options,
                    "description": "1-based index into the Options list.",
                },
                "selected_text": {
                    "type": "string",
                    "enum": question.options,
                },
            },
            "required": ["selected_position", "selected_text"],
        },
    }
    return SYSTEM_PROMPT, user, tool


def build_persistent_batched_prompt(
    archetype: Archetype, questions: list[QuestionSpec], n: int
) -> tuple[str, str, dict]:
    """Persistent Batched Personas: one call instantiates n distinct
    individuals sharing the archetype profile and answers the ENTIRE
    questionnaire for each of them, so every simulated person's answers can
    cohere across questions (the structural-alignment lever) while the batch
    preserves between-person diversity (the distributional lever).
    """
    profile = archetype.profile_description()

    question_blocks = []
    answer_properties = {}
    for i, q in enumerate(questions, start=1):
        question_blocks.append(
            f"Question q{i}:\n{q.question_text}\nOptions:\n{q.options_block()}"
        )
        answer_properties[f"q{i}"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": q.n_options,
            "description": f"1-based index into question q{i}'s options.",
        }

    user = (
        f"There are {n} different, unrelated real people who all share this exact "
        f"demographic profile:\n{profile}\n\n"
        f"First, silently imagine each of the {n} people as a distinct, "
        "internally consistent individual: their own personality, values, "
        "personal history, risk tolerance, and current mood. Real people who "
        "share only these coarse demographic traits still differ from one "
        "another in all of these ways.\n\n"
        f"Then have each of the {n} people answer the ENTIRE survey below. "
        "Each person's answers should be consistent with that same "
        "individual's character across all questions (a person's answer to "
        "one question should cohere with their answers to the others), while "
        "different people should reflect the natural diversity of real "
        "respondents. Do not make every person answer identically unless you "
        "genuinely believe that reflects reality.\n\n"
        "SURVEY\n======\n\n" + "\n\n".join(question_blocks) + "\n\n"
        f"Record all answers for all {n} people using the record_survey tool, "
        f"with person_index running from 1 to {n}."
    )

    tool = {
        "name": "record_survey",
        "description": f"Record the full survey answers of all {n} simulated people.",
        "input_schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": {
                        "type": "object",
                        "properties": {
                            "person_index": {"type": "integer", "minimum": 1, "maximum": n},
                            "answers": {
                                "type": "object",
                                "properties": answer_properties,
                                "required": list(answer_properties),
                            },
                        },
                        "required": ["person_index", "answers"],
                    },
                }
            },
            "required": ["responses"],
        },
    }
    return SYSTEM_PROMPT, user, tool


def build_multi_respondent_prompt(
    archetype: Archetype, question: QuestionSpec, n: int
) -> tuple[str, str, dict]:
    profile = archetype.profile_description()
    user = (
        f"There are {n} different, unrelated real people who all share this exact "
        f"demographic profile:\n{profile}\n\n"
        f"Survey question:\n{question.question_text}\n\n"
        f"Options:\n{question.options_block()}\n\n"
        f"Independently simulate how each of the {n} people would answer, "
        "reflecting the natural diversity of opinion, mood, and personal "
        "history that exists among real people who share only these coarse "
        "demographic traits. Do not make every person answer identically "
        "unless you genuinely believe that reflects reality. Record all "
        f"{n} answers using the record_responses tool, with person_index "
        f"running from 1 to {n}."
    )
    tool = {
        "name": "record_responses",
        "description": f"Record the {n} simulated people's answers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "responses": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "items": {
                        "type": "object",
                        "properties": {
                            "person_index": {"type": "integer", "minimum": 1, "maximum": n},
                            "selected_position": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": question.n_options,
                            },
                            "selected_text": {"type": "string", "enum": question.options},
                        },
                        "required": ["person_index", "selected_position", "selected_text"],
                    },
                }
            },
            "required": ["responses"],
        },
    }
    return SYSTEM_PROMPT, user, tool
