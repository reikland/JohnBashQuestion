from __future__ import annotations

"""
Post-processing helpers for rebalancing, canonicalizing, and cleaning
question lists after the draft generation phase.
"""
# Keep logic in one place to keep the Streamlit UI lean.


from typing import Any, Dict, List, Tuple

from models import FullQuestion
from openrouter_client import OpenRouterConfig, call_json_strict, validate_or_reformat
from orchestration import compute_targets, count_types
from prompts_quality import prompt_canonicalize_questions_list, prompt_clean_questions, prompt_rebalance


def _validate_question_list(
    cfg: OpenRouterConfig,
    raw_questions: List[FullQuestion],
    *,
    context_label: str,
) -> List[FullQuestion]:
    """Validate a list of FullQuestion objects with a consistent label prefix."""
    validated: List[FullQuestion] = []
    for idx, qd in enumerate(raw_questions, start=1):
        fq = validate_or_reformat(
            cfg,
            qd,
            FullQuestion,
            """FULL_QUESTION_OBJECT""",
            f"{context_label} #{idx}",
            max_attempts=3,
        )
        validated.append(fq)
    return validated


def finalize_questions(
    cfg: OpenRouterConfig,
    questions: List[FullQuestion],
    *,
    end_d,
    do_rebalance: bool,
    use_formatter_after_rebalance: bool,
    do_final_clean: bool,
    tick,
) -> Tuple[List[FullQuestion], Dict[str, Any]]:
    """
    Apply optional rebalance/canonicalize/clean passes and return the new list
    along with a log payload for the UI.
    """
    final_questions = list(questions)
    log: Dict[str, Any] = {"type_counts_draft": count_types(questions)}

    if do_rebalance and final_questions:
        tick("Rebalancing types to ~50/30/20.")
        total = len(final_questions)
        tb, tn, tm = compute_targets(total)
        msgs, hint = prompt_rebalance(final_questions, tb, tn, tm, end_d)
        raw = call_json_strict(cfg, cfg.light_model, msgs, schema_hint=hint)
        edited = raw.get("questions", []) or []
        # validate each FullQuestion with formatter retries
        final_questions = _validate_question_list(
            cfg,
            edited,
            context_label="Rebalance FullQuestion",
        )
        log["rebalance_change_log"] = raw.get("change_log", [])
        log["type_counts_after_rebalance"] = count_types(final_questions)

        if use_formatter_after_rebalance:
            tick("Canonicalizing post-rebalance list (formatter).")
            msgs, hint2 = prompt_canonicalize_questions_list(final_questions)
            canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint2, retries=1)
            canon_list = canon.get("questions", []) or []
            if canon_list:
                final_questions = _validate_question_list(
                    cfg,
                    canon_list,
                    context_label="Post-rebalance canonical FullQuestion",
                )

    if do_final_clean and final_questions:
        tick("Final clean pass (formatter).")
        msgs, hint = prompt_clean_questions(final_questions)
        raw = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
        cleaned = raw.get("questions", []) or []
        final_questions = _validate_question_list(
            cfg,
            cleaned,
            context_label="Final clean FullQuestion",
        )
        log["clean_notes"] = raw.get("notes", [])
        log["type_counts_final"] = count_types(final_questions)

    return final_questions, log


__all__ = ["finalize_questions"]
