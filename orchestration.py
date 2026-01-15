from __future__ import annotations

from typing import Any, Dict, List, Tuple

from models import FullQuestion, ProtoQuestion, VerifyResponse
from openrouter_client import OpenRouterConfig, call_json_strict, validate_or_reformat
from prompts_generation import (
    prompt_build_full,
    prompt_canonicalize_selected_protos,
    prompt_generate_protos,
    prompt_select_k,
)
from prompts_quality import (
    prompt_format_single,
    prompt_verify,
)
from utils import normalize_full_question_fields


# ---------------------------
# Orchestration
# ---------------------------
def compute_targets(total: int) -> Tuple[int, int, int]:
    b = int(round(total * 0.50))
    n = int(round(total * 0.30))
    m = total - b - n
    if m < 0:
        m = 0
        while b + n + m > total and n > 0:
            n -= 1
        while b + n + m > total and b > 0:
            b -= 1
    return b, n, m


def count_types(questions: List[FullQuestion]) -> Dict[str, int]:
    c = {"binary": 0, "numeric": 0, "multiple_choice": 0}
    for q in questions:
        c[q.type] = c.get(q.type, 0) + 1
    return c


def generate_for_topic_iter(
    cfg: OpenRouterConfig,
    topic: str,
    n: int,
    k: int,
    start_d,
    end_d,
    use_formatter_after_selection: bool,
    progress_cb=None,
):
    # 1) protos
    if progress_cb:
        progress_cb(f"Generating {n} protos for topic: {topic}")
    msgs, hint = prompt_generate_protos(topic, n, start_d, end_d)
    raw = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    protos = [
        validate_or_reformat(
            cfg,
            p,
            ProtoQuestion,
            """{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}""",
            "ProtoQuestion",
        )
        for p in (raw.get("protos", []) or [])
    ][:n]
    if not protos:
        raise ValueError(f"No protos generated for topic: {topic}")

    # 2) select k
    if progress_cb:
        progress_cb(f"Selecting top {k} protos for topic: {topic}")
    msgs, hint = prompt_select_k(topic, protos, k)
    raw2 = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    selected_raw = raw2.get("selected", []) or []
    selected = [
        validate_or_reformat(
            cfg,
            p,
            ProtoQuestion,
            """{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}""",
            "Selected ProtoQuestion",
        )
        for p in selected_raw
    ][:k]
    if not selected:
        selected = protos[:k]

    # 3) optional canonicalize selected protos
    if use_formatter_after_selection:
        if progress_cb:
            progress_cb(f"Canonicalizing selected protos (formatter) for topic: {topic}")
        msgs, hint = prompt_canonicalize_selected_protos(selected)
        canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
        canon_list = canon.get("selected", []) or []
        if canon_list:
            selected = [
                validate_or_reformat(cfg, p, ProtoQuestion, hint, "Canonicalized ProtoQuestion")
                for p in canon_list
            ]

    # 4) per selected: build -> verify -> format -> yield
    for i, proto in enumerate(selected, start=1):
        if progress_cb:
            progress_cb(f"Building full card {i}/{len(selected)} for topic: {topic}")
        msgs, hint = prompt_build_full(topic, proto, start_d, end_d)
        raw3 = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)
        fq = validate_or_reformat(cfg, raw3, FullQuestion, hint, f"Build FullQuestion ({topic} #{i})")
        fq = normalize_full_question_fields(fq)

        if progress_cb:
            progress_cb(f"Verifying card {i}/{len(selected)} for topic: {topic}")
        msgs, hint_v = prompt_verify(fq, end_d)
        raw4 = call_json_strict(cfg, cfg.light_model, msgs, schema_hint=hint_v)

        # Retry formatter up to 3 times on validation error.
        vr = validate_or_reformat(
            cfg,
            raw4,
            VerifyResponse,
            hint_v,
            f"VerifyResponse ({topic} #{i})",
            max_attempts=3,
        )
        fq2 = normalize_full_question_fields(vr.question)

        if progress_cb:
            progress_cb(f"Formatting card {i}/{len(selected)} for topic: {topic}")
        msgs, hint_f = prompt_format_single(fq2)
        raw5 = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint_f, retries=1)
        fq_final = validate_or_reformat(
            cfg,
            raw5.get("question", raw5),
            FullQuestion,
            """FULL_QUESTION_OBJECT""",
            f"Formatted FullQuestion ({topic} #{i})",
            max_attempts=3,
        )
        fq_final = normalize_full_question_fields(fq_final)

        yield fq_final


__all__ = ["compute_targets", "count_types", "generate_for_topic_iter"]
