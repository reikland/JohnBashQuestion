from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Tuple

from models import ALLOWED_RATINGS, METACULUS_CREDIBLE_SOURCES_URL, ProtoQuestion
from utils import date_window_phrase_inclusive, dt_range_defaults, to_model_dict
SYSTEM_PRIMARY = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."
def prompt_generate_protos(topic: str, n: int, start_d: date, end_d: date) -> Tuple[List[Dict[str, str]], str]:
    today_iso = date.today().isoformat()
    _, _, resolve_time = dt_range_defaults(start_d, end_d)
    window_phrase = date_window_phrase_inclusive(start_d, end_d)

    schema_hint = r"""{
  "protos": [
    {
      "title": "Will country X's inflation rate exceed 10% in 2027?",
      "suggested_type": "binary",
      "description": "Tracks whether inflation in country X reaches a double-digit level by 2027, which would signal macroeconomic stress.",
      "why_informative": "High inflation has implications for monetary policy, living standards, and financial markets.",
      "candidate_sources": [
        "IMF World Economic Outlook – https://www.imf.org/",
        "World Bank Data – https://data.worldbank.org/"
      ]
    }
  ]
}"""

    user = (
        f"Topic: {topic}\n"
        f"Generate exactly {n} proto-questions.\n"
        f"Each question must be resolvable unambiguously by: {resolve_time}\n"
        f"Current date is: {today_iso}.\n\n"
        "Core goals:\n"
        "- Each question must be decision-relevant or insight-generating, not definitional trivia.\n"
        "- Each question must pass Tetlock's clairvoyance test:\n"
        "  A perfect clairvoyant, given the question text and cited sources, could say clearly 'Yes, it occurred' or 'No, it did not'.\n"
        "- Do NOT write questions about events whose outcomes are already determined before the current date.\n"
        "- Avoid vague language like 'significant', 'substantial', 'major', 'severe', unless tied to a numeric threshold or an official category.\n\n"
        "Date wording (avoid ambiguity):\n"
        "- Avoid using 'between' and 'by' for bounds.\n"
        "- Use explicit inequalities such as 'after <DATE>' / 'before <DATE>' / 'on or before <DATE>'.\n"
        f"- If you need an inclusive window from {start_d.isoformat()} to {end_d.isoformat()}, prefer phrasing like: '{window_phrase}'.\n\n"
        "Avoid overspecification:\n"
        "- Do NOT restrict outcomes to a narrow publication channel (e.g., 'only via press release' / 'only at a keynote') unless essential.\n"
        "- Do NOT enumerate media outlets as gatekeepers; instead rely on a credible-sources principle.\n"
        f"- If disputes are plausible, prefer a short reference to Metaculus credible sources policy: {METACULUS_CREDIBLE_SOURCES_URL}\n\n"
        "Type rules:\n"
        "- suggested_type must be one of: binary | numeric | multiple_choice.\n"
        "- Avoid disguised-binary numeric questions (e.g., numeric ranges where only the sign or a threshold matters).\n"
        "- Use numeric type only when a range of values is genuinely informative.\n"
        "- Use multiple_choice only when there are >= 4 genuinely distinct, plausible options.\n\n"
        "Source rules:\n"
        "- candidate_sources must contain named sources + URLs whenever feasible.\n"
        "- Prefer reputable and stable sources: government statistical offices, major international orgs, established media, etc.\n"
        "- If you cite a specific dataset/series, include a direct URL to that series/page when feasible.\n\n"
        "Output format:\n"
        "- Return ONLY valid JSON, with a single top-level object having a 'protos' array.\n"
        "- Each entry must match the structure shown in the schema hint.\n\n"
        f"Schema hint example:\n{schema_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PRIMARY},
        {"role": "user", "content": user},
    ]
    return messages, schema_hint

def prompt_select_k(topic: str, protos: List[ProtoQuestion], k: int) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"selected":[{PROTO_OBJECTS_SUBSET}],"selection_rationale":"..."}"""
    protos_json = json.dumps([to_model_dict(p) for p in protos], ensure_ascii=False)
    user = (
        f"Topic: {topic}\n"
        f"Select the best {k} proto-questions from the list.\n\n"
        "Criteria: info value, resolvability, non-redundant.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Proto list:\n{protos_json}"
    )
    return ([{"role": "system", "content": SYSTEM_PRIMARY}, {"role": "user", "content": user}], schema_hint)

def prompt_canonicalize_selected_protos(selected: List[ProtoQuestion]) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"selected":[{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}]}"""
    payload = json.dumps([to_model_dict(p) for p in selected], ensure_ascii=False)
    user = (
        "Canonicalize the selected proto-questions.\n"
        "Do NOT change meaning; only ensure schema compliance.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Selected protos:\n{payload}"
    )
    return ([{"role": "system", "content": SYSTEM_PRIMARY}, {"role": "user", "content": user}], schema_hint)

def prompt_build_full(topic: str, proto: ProtoQuestion, start_d: date, end_d: date) -> Tuple[List[Dict[str, str]], str]:
    today_iso = date.today().isoformat()
    open_time, close_time, resolve_time = dt_range_defaults(start_d, end_d)
    window_phrase = date_window_phrase_inclusive(start_d, end_d)

    schema_hint = r"""{
  "title": "Will country X's inflation rate exceed 10% in 2027?",
  "type": "binary",
  "resolution_criteria": "This question resolves to Yes if, according to IMF World Economic Outlook data or World Bank data, the year-on-year CPI inflation rate for country X is strictly above 10.0% in any month of 2027. Otherwise it resolves to No. If both sources disagree, IMF data takes precedence.",
  "fine_print": "If official statistics are temporarily unavailable, use the first subsequent official release that covers the relevant period. If country X undergoes a change in CPI methodology, use the main headline CPI series as reported. Source disputes are handled per Metaculus credible sources guidance.",
  "description": "Tracks whether country X experiences double-digit inflation by 2027, signalling macroeconomic stress and potential policy changes.",
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
  "categories": ["Macroeconomics", "Inflation"],
  "ai_rating": "accept_for_aib",
  "ai_rationale": "Falsifiable, decision-relevant, clear data sources, no major ambiguity."
}"""

    proto_json = json.dumps(to_model_dict(proto), ensure_ascii=False)

    user = (
        f"Topic: {topic}\n"
        f"Current date is: {today_iso}.\n"
        "Build a single full forecasting question object (FullQuestion) from the proto.\n\n"
        "Conceptual requirements:\n"
        "- The question must pass Tetlock's clairvoyance test:\n"
        "  A perfect clairvoyant, given only the question text and the cited sources, can answer Yes/No (for binary),\n"
        "  or give a single numeric value, or select a single option, without ambiguity.\n"
        "- The event must NOT already be decided before the current date.\n"
        "- Avoid vague language such as 'significant', 'major', 'severe', 'substantial', unless tied to a precise numeric threshold or\n"
        "  to an explicit official category.\n\n"
        "Date wording (avoid ambiguity and tighten bounds):\n"
        "- Avoid 'between' and 'by'. Use explicit 'after <DATE>', 'before <DATE>', 'on or before <DATE>'.\n"
        f"- If the event window is the UI window ({start_d.isoformat()}..{end_d.isoformat()} inclusive), prefer phrasing like: '{window_phrase}'.\n"
        "- Be explicit about inclusivity: e.g., 'on or before 2026-05-01' vs 'before 2026-05-01'.\n\n"
        "Avoid overspecification (common failure mode):\n"
        "- Do NOT unnecessarily restrict the event to a specific announcement channel (press release, keynote, newsroom page, etc.).\n"
        "- You may include example channels, but do not make them exclusive unless truly required.\n"
        "- Do NOT list specific media outlets as mandatory/forbidden. Handle disputes via a credible-sources rule instead.\n"
        f"- If needed, reference Metaculus credible sources policy: {METACULUS_CREDIBLE_SOURCES_URL}\n\n"
        "Fine print discipline:\n"
        "- Fine print should cover procedural/edge cases: source precedence, revisions, data gaps, definitional clarifications.\n"
        "- Avoid redundant causal statements like 'If X is delayed, it resolves No' (that is usually implied).\n\n"
        "Source/terminology discipline:\n"
        "- Link directly to the resolution source when feasible.\n"
        "- Use the same terminology as the resolution source (e.g., if the source says 'spot price', do not call it 'closing price').\n\n"
        "Time rules:\n"
        f"- open_time must be >= {open_time}.\n"
        f"- scheduled_close_time must be <= {close_time}.\n"
        f"- scheduled_resolve_time must be <= {resolve_time}.\n"
        "- Ensure the event can realistically be resolved by scheduled_resolve_time using the cited sources.\n\n"
        "Type-specific rules:\n"
        "- type must be one of: binary | numeric | multiple_choice.\n"
        "- For numeric:\n"
        "  * Provide meaningful range_min and range_max.\n"
        "  * Provide a clear unit.\n"
        "  * The question must not be effectively binary.\n"
        "- For multiple_choice:\n"
        "  * Provide at least 4 mutually exclusive, collectively exhaustive, and genuinely plausible options.\n"
        "  * Options must be concise and clearly distinguishable.\n"
        "- For binary:\n"
        "  * The Yes/No condition must be sharp and fully specified in resolution_criteria.\n\n"
        "Resolution criteria & sources:\n"
        "- resolution_criteria must be executable: it should read like instructions to a careful resolve editor.\n"
        "- Explicitly name AND, where possible, link to the primary sources that determine the outcome.\n"
        "- Specify what to do in case of conflicting sources or data revisions (e.g., 'IMF data takes precedence').\n\n"
        "AI rating:\n"
        f"- ai_rating must be one of: {sorted(ALLOWED_RATINGS)}.\n"
        "- Do NOT use 'pass' or 'fix' as ai_rating values. These belong only to VerifyResponse.status.\n"
        "- ai_rationale should briefly justify the rating, focusing on clarity, informativeness, and resolvability.\n\n"
        "Output format:\n"
        "- Return ONLY a single JSON object matching the FullQuestion structure in the schema hint.\n"
        "- Use null for missing optional fields (not empty strings).\n\n"
        f"Proto object:\n{proto_json}\n\n"
        f"Schema hint example for the FullQuestion structure:\n{schema_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PRIMARY},
        {"role": "user", "content": user},
    ]
    return messages, schema_hint


__all__ = [
    "SYSTEM_PRIMARY",
    "prompt_build_full",
    "prompt_canonicalize_selected_protos",
    "prompt_generate_protos",
    "prompt_select_k",
]
