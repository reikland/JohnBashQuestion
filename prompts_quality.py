from __future__ import annotations

import json
from datetime import date, time as dtime
from typing import Dict, List, Tuple

from models import ALLOWED_RATINGS, ALLOWED_VERIFY_STATUS, FullQuestion, METACULUS_CREDIBLE_SOURCES_URL
from utils import iso_dt, to_model_dict


# ---------------------------
# Prompts (verify/rebalance)
# ---------------------------
SYSTEM_LIGHT = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."
SYSTEM_FORMATTER = "You are a schema enforcer. Return ONLY a SINGLE valid JSON object. No markdown, no prose."


def prompt_verify(full_q: FullQuestion, end_d: date) -> Tuple[List[Dict[str, str]], str]:
    end_iso = iso_dt(end_d, dtime(23, 59))

    schema_hint = r"""{
  "status": "pass",
  "issues": [
    "Example issue description if any."
  ],
  "question": {
    "title": "Will country X's inflation rate exceed 10% in 2027?",
    "type": "binary",
    "resolution_criteria": "...",
    "fine_print": "...",
    "description": "...",
    "question_weight": 1.0,
    "open_time": "2026-01-01T00:00:00+01:00",
    "scheduled_close_time": "2027-12-31T23:59:00+01:00",
    "scheduled_resolve_time": "2028-06-30T23:59:00+01:00",
    "range_min": null,
    "range_max": null,
    "zero_point": null,
    "open_lower_bound": null,
    "open_upper_bound": null,
    "unit": null,
    "group_variable": null,
    "options": null,
    "categories": ["..."],
    "ai_rating": "accept_for_aib",
    "ai_rationale": "..."
  }
}"""

    full_json = json.dumps(to_model_dict(full_q), ensure_ascii=False)

    user = (
        "You are a strict validator and light editor for forecasting questions.\n\n"
        "Task:\n"
        "- Inspect the provided FullQuestion.\n"
        "- If it already satisfies all constraints, return status='pass' and the (possibly lightly cleaned) question.\n"
        "- If it violates any constraint, minimally fix it and return status='fix' plus a short list of issues.\n\n"
        "Conceptual checks (content):\n"
        "- The question must pass Tetlock's clairvoyance test:\n"
        "  A perfect clairvoyant, given only the text and cited sources, can answer it without asking for clarifications.\n"
        "- Avoid vague terms unless tied to numerical thresholds or formal categories.\n"
        "- The event must not obviously be already decided before now.\n\n"
        "Date language checks (reduce ambiguity):\n"
        "- Avoid 'between' and 'by' in text.\n"
        "- Prefer explicit bounds: 'after <DATE>', 'before <DATE>', 'on or before <DATE>'.\n"
        "- Make inclusivity explicit.\n\n"
        "Overspecification checks:\n"
        "- Remove unnecessary restrictions on HOW an event must be announced (e.g., 'press release OR keynote').\n"
        "- Remove enumerations of specific media outlets as gatekeepers.\n"
        f"- If source disputes are mentioned, keep it short and point to credible sources guidance: {METACULUS_CREDIBLE_SOURCES_URL}\n"
        "- Remove redundant fine-print statements that merely restate logical consequences.\n\n"
        "Terminology checks:\n"
        "- Ensure the question uses terminology aligned with the resolution source (e.g., 'spot price' vs 'closing price').\n"
        "- Ensure resolution source is linked when feasible.\n\n"
        "Time constraint:\n"
        f"- scheduled_resolve_time must be <= {end_iso}.\n"
        "- If needed, adjust scheduled_resolve_time so that the question can realistically be resolved using cited sources.\n\n"
        "Type & structure checks:\n"
        "- question.type must be one of: binary | numeric | multiple_choice.\n"
        "- For numeric questions:\n"
        "  * range_min, range_max, and unit must make sense.\n"
        "  * The question must not be disguised binary.\n"
        "- For multiple_choice:\n"
        "  * options must be a non-empty list with at least 4 meaningful, mutually exclusive options.\n"
        "- For non-MCQ:\n"
        "  * options must be null.\n"
        "- categories must be a non-empty list of strings.\n\n"
        "Rating fields:\n"
        f"- VerifyResponse.status must be one of: {sorted(ALLOWED_VERIFY_STATUS)} (exactly 'pass' or 'fix').\n"
        f"- question.ai_rating must be one of: {sorted(ALLOWED_RATINGS)}.\n"
        "- IMPORTANT: Never set question.ai_rating to 'pass' or 'fix'. These values are reserved for VerifyResponse.status only.\n"
        "- ai_rationale should briefly justify ai_rating.\n\n"
        "Output format:\n"
        "- Return ONLY one JSON object matching the VerifyResponse structure shown in the schema hint.\n"
        "- Use null (not empty strings) for missing optional fields inside question.\n\n"
        f"Question to validate:\n{full_json}\n\n"
        f"Schema hint example for VerifyResponse:\n{schema_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_LIGHT},
        {"role": "user", "content": user},
    ]
    return messages, schema_hint


def prompt_format_single(full_q: FullQuestion) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"question":FULL_QUESTION_OBJECT}"""
    full_json = json.dumps(to_model_dict(full_q), ensure_ascii=False)
    user = (
        "Canonicalize this single FullQuestion into the exact schema.\n"
        "Do not change meaning; only fix formatting/schema issues.\n\n"
        "Rules:\n"
        "- options must be null for non-MCQ; list for MCQ\n"
        "- categories must be a non-empty list\n"
        "- use null for missing optional fields\n"
        f"- ai_rating must be one of {sorted(ALLOWED_RATINGS)}\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Input:\n{full_json}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


def prompt_rebalance(
    questions: List[FullQuestion],
    target_binary: int,
    target_numeric: int,
    target_mcq: int,
    end_d: date,
) -> Tuple[List[Dict[str, str]], str]:
    end_iso = iso_dt(end_d, dtime(23, 59))
    schema_hint = """{"questions":[FULL_QUESTION_OBJECTS],"change_log":["..."]}"""
    qs_json = json.dumps([to_model_dict(q) for q in questions], ensure_ascii=False)
    user = (
        "Minimally edit questions to match target type distribution.\n\n"
        f"Targets: binary={target_binary}, numeric={target_numeric}, multiple_choice={target_mcq}\n"
        f"Constraint: scheduled_resolve_time <= {end_iso}\n"
        "No disguised-binary numeric. MCQ >=4 meaningful options.\n\n"
        "Also keep edits disciplined:\n"
        "- Avoid introducing overspecification (do not add unnecessary channel constraints or outlet lists).\n"
        "- Keep date language explicit (avoid 'between'/'by'; prefer 'after/before/on or before').\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Questions:\n{qs_json}"
    )
    return ([{"role": "system", "content": SYSTEM_LIGHT}, {"role": "user", "content": user}], schema_hint)


def prompt_canonicalize_questions_list(questions: List[FullQuestion]) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"questions":[FULL_QUESTION_OBJECTS]}"""
    qs_json = json.dumps([to_model_dict(q) for q in questions], ensure_ascii=False)
    user = (
        "Canonicalize the list of FullQuestion objects.\n"
        "Do NOT change meaning; only ensure schema compliance.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Input:\n{qs_json}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


def prompt_clean_questions(questions: List[FullQuestion]) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"questions":[FULL_QUESTION_OBJECTS],"notes":["optional"]}"""
    qs_json = json.dumps([to_model_dict(q) for q in questions], ensure_ascii=False)
    user = (
        "Return cleaned strict JSON with the same questions. Minimal changes only for schema correctness.\n\n"
        "Cleanup goals (minimal):\n"
        "- Keep date language explicit (avoid 'between'/'by').\n"
        "- Remove accidental overspecification if present.\n"
        "- Keep terminology aligned with cited sources (e.g., 'spot price' vs 'closing price').\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Input:\n{qs_json}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


__all__ = [
    "SYSTEM_LIGHT",
    "prompt_canonicalize_questions_list",
    "prompt_clean_questions",
    "prompt_format_single",
    "prompt_rebalance",
    "prompt_verify",
]
