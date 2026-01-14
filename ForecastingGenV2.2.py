# QuestionGenV2.py
# Streamlit all-in-one (incremental + formatter retries on validation errors):
# Topics CSV -> protos -> select k -> (optional canonicalize selected protos)
# -> build full -> verify -> format single -> append to draft CSV incrementally
# -> optional rebalance -> (optional canonicalize post-rebalance) -> optional final clean -> export CSV
#
# Install:
#   pip install streamlit requests pydantic
# Run:
#   streamlit run QuestionGenV2.py

from __future__ import annotations

import ast
import csv
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# --- Pydantic v2/v1 compatibility ---
try:
    from pydantic import BaseModel, Field, ValidationError
    from pydantic import field_validator  # pydantic v2
    PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, ValidationError, validator  # type: ignore
    PYDANTIC_V2 = False

    def field_validator(*fields, **kwargs):  # type: ignore
        def deco(fn):
            return validator(*fields, **kwargs, allow_reuse=True)(fn)
        return deco


PARIS_TZ = ZoneInfo("Europe/Paris")

# Metaculus “credible sources” policy (often sufficient vs enumerating outlets)
METACULUS_CREDIBLE_SOURCES_URL = "https://www.metaculus.com/faq/#definitions"

CSV_COLUMNS = [
    "title",
    "type",
    "resolution_criteria",
    "fine_print",
    "description",
    "question_weight",
    "open_time",
    "scheduled_close_time",
    "scheduled_resolve_time",
    "range_min",
    "range_max",
    "zero_point",
    "open_lower_bound",
    "open_upper_bound",
    "unit",
    "group_variable",
    "options",
    "categories",
    "ai_rating",
    "ai_rationale",
]

ALLOWED_TYPES = {"binary", "numeric", "multiple_choice"}
ALLOWED_RATINGS = {"hard_reject", "soft_reject", "accept_for_aib", "accept_for_main_site"}
ALLOWED_VERIFY_STATUS = {"pass", "fix"}


# ---------------------------
# Schemas
# ---------------------------
class ProtoQuestion(BaseModel):
    title: str
    suggested_type: str = Field(description="binary | numeric | multiple_choice")
    description: str
    why_informative: str
    candidate_sources: List[str] = Field(default_factory=list)

    @field_validator("suggested_type")
    def _type_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_TYPES:
            raise ValueError(f"suggested_type must be one of {sorted(ALLOWED_TYPES)}")
        return v2


class FullQuestion(BaseModel):
    title: str
    type: str
    resolution_criteria: str
    fine_print: str
    description: str
    question_weight: float

    open_time: str
    scheduled_close_time: str
    scheduled_resolve_time: str

    range_min: Optional[float] = None
    range_max: Optional[float] = None
    zero_point: Optional[float] = None
    open_lower_bound: Optional[float] = None
    open_upper_bound: Optional[float] = None
    unit: Optional[str] = None
    group_variable: Optional[str] = None
    options: Optional[List[str]] = None
    categories: List[str]

    ai_rating: str
    ai_rationale: str

    @field_validator("type")
    def _type_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_TYPES:
            raise ValueError(f"type must be one of {sorted(ALLOWED_TYPES)}")
        return v2

    @field_validator("ai_rating")
    def _rating_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_RATINGS:
            raise ValueError(f"ai_rating must be one of {sorted(ALLOWED_RATINGS)}")
        return v2

    @field_validator("categories")
    def _categories_ok(cls, v: List[str]):  # type: ignore
        vv = [str(c).strip() for c in (v or []) if str(c).strip()]
        if not vv:
            raise ValueError("categories must be a non-empty list")
        return vv

    @field_validator("options")
    def _options_ok(cls, v: Optional[List[str]]):  # type: ignore
        if v is None:
            return None
        vv = [str(o).strip() for o in v if str(o).strip()]
        return vv or None


class VerifyResponse(BaseModel):
    status: str = Field(description="pass | fix")
    issues: List[str] = Field(default_factory=list)
    question: FullQuestion

    @field_validator("status")
    def _status_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_VERIFY_STATUS:
            raise ValueError("status must be 'pass' or 'fix'")
        return v2


# ---------------------------
# Utils
# ---------------------------
def iso_dt(d: date, t: dtime = dtime(0, 0)) -> str:
    dt = datetime.combine(d, t).replace(tzinfo=PARIS_TZ)
    return dt.isoformat()


def dt_range_defaults(start_d: date, end_d: date) -> Tuple[str, str, str]:
    open_time = iso_dt(start_d, dtime(0, 0))
    close_time = iso_dt(end_d, dtime(23, 59))
    resolve_time = iso_dt(end_d, dtime(23, 59))
    return open_time, close_time, resolve_time


def date_window_phrase_inclusive(start_d: date, end_d: date) -> str:
    """
    Produces an unambiguous inclusive window phrase without 'between'/'by'.
    Convention: [start_d, end_d] inclusive is described as:
      "after (start_d - 1 day) and on or before end_d"
    """
    s0 = (start_d - timedelta(days=1)).isoformat()
    e = end_d.isoformat()
    return f"after {s0} and on or before {e}"


def read_topics_csv(uploaded_file) -> Tuple[List[str], List[Dict[str, Any]]]:
    data = uploaded_file.getvalue()
    text = data.decode("utf-8", errors="replace")
    sio = io.StringIO(text)
    reader = csv.DictReader(sio)
    if not reader.fieldnames:
        raise ValueError("CSV appears to have no header row.")
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has a header but no data rows.")
    return reader.fieldnames, rows


def pick_topic_column(fieldnames: List[str], rows: List[Dict[str, Any]]) -> str:
    sample = rows[:50]
    best_col, best_score = fieldnames[0], -1
    for col in fieldnames:
        score = sum(1 for r in sample if str(r.get(col, "")).strip())
        if score > best_score:
            best_col, best_score = col, score
    return best_col


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_first_json_block(s: str) -> Optional[str]:
    s = s.strip()
    start = None
    opener = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None or opener is None:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : j + 1]
    return None


def try_parse_json(s: str) -> Any:
    s0 = strip_code_fences(s)

    try:
        return json.loads(s0)
    except Exception:
        pass

    block = extract_first_json_block(s0)
    if block:
        try:
            return json.loads(block)
        except Exception:
            try:
                obj = ast.literal_eval(block)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    try:
        obj = ast.literal_eval(s0)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    raise ValueError("Could not parse valid JSON from the model output.")


def to_model_dict(obj: Any) -> Dict[str, Any]:
    if PYDANTIC_V2 and hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def normalize_full_question_fields(q: FullQuestion) -> FullQuestion:
    d = to_model_dict(q)

    if d.get("type") != "multiple_choice":
        d["options"] = None

    if d.get("type") != "numeric":
        d["range_min"] = None
        d["range_max"] = None
        d["zero_point"] = None
        d["open_lower_bound"] = None
        d["open_upper_bound"] = None
        d["unit"] = None

    if d.get("question_weight") is None:
        d["question_weight"] = 1.0

    return FullQuestion(**d)


def full_question_to_row(q: FullQuestion) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}
    row["title"] = q.title
    row["type"] = q.type
    row["resolution_criteria"] = q.resolution_criteria
    row["fine_print"] = q.fine_print
    row["description"] = q.description
    row["question_weight"] = q.question_weight
    row["open_time"] = q.open_time
    row["scheduled_close_time"] = q.scheduled_close_time
    row["scheduled_resolve_time"] = q.scheduled_resolve_time

    row["range_min"] = "" if q.range_min is None else q.range_min
    row["range_max"] = "" if q.range_max is None else q.range_max
    row["zero_point"] = "" if q.zero_point is None else q.zero_point
    row["open_lower_bound"] = "" if q.open_lower_bound is None else q.open_lower_bound
    row["open_upper_bound"] = "" if q.open_upper_bound is None else q.open_upper_bound
    row["unit"] = "" if q.unit is None else q.unit
    row["group_variable"] = "" if q.group_variable is None else q.group_variable

    row["options"] = "" if not q.options else json.dumps(q.options, ensure_ascii=False)
    row["categories"] = json.dumps(q.categories, ensure_ascii=False)

    row["ai_rating"] = q.ai_rating
    row["ai_rationale"] = q.ai_rationale
    return row


def write_csv(rows: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return output.getvalue()


# ---------------------------
# OpenRouter client
# ---------------------------
@dataclass
class OpenRouterConfig:
    api_key: str
    primary_model: str
    light_model: str
    formatter_model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    use_response_format_json: bool = True
    app_title: str = "Forecast Question Generator"
    http_referer: str = "http://localhost:8501"


class OpenRouterError(RuntimeError):
    pass


def openrouter_chat(
    cfg: OpenRouterConfig,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
) -> str:
    if not cfg.api_key or len(cfg.api_key.strip()) < 20:
        raise OpenRouterError("OpenRouter API key missing/too short. Paste a valid OpenRouter key.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg.http_referer,
        "X-Title": cfg.app_title,
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(cfg.temperature if temperature_override is None else temperature_override),
        "max_tokens": int(cfg.max_tokens if max_tokens_override is None else max_tokens_override),
    }
    if cfg.use_response_format_json:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1200]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise OpenRouterError(f"Unexpected response format: {json.dumps(data)[:1500]}")


def repair_to_json(cfg: OpenRouterConfig, raw_text: str, schema_hint: str) -> Any:
    system = (
        "You are a strict JSON repair tool.\n"
        "Return ONLY valid JSON (no markdown, no prose).\n"
        "Use double quotes, no trailing commas.\n"
        "Extract and return the single best JSON object that matches the schema hint."
    )
    user = (
        f"Schema hint:\n{schema_hint}\n\n"
        f"Input text:\n{raw_text}\n\n"
        "Return ONLY the repaired JSON."
    )
    text = openrouter_chat(
        cfg,
        cfg.formatter_model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens_override=min(cfg.max_tokens, 1200),
        temperature_override=0.0,
    )
    return try_parse_json(text)


def call_json_strict(
    cfg: OpenRouterConfig,
    model: str,
    messages: List[Dict[str, str]],
    schema_hint: str,
    *,
    retries: int = 2,
    sleep_s: float = 0.6,
) -> Any:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        raw = openrouter_chat(cfg, model, messages)
        try:
            return try_parse_json(raw)
        except Exception as e:
            last_err = e

        try:
            return repair_to_json(cfg, raw_text=raw, schema_hint=schema_hint)
        except Exception as e2:
            last_err = e2

        messages = messages + [
            {
                "role": "user",
                "content": (
                    "INVALID OUTPUT. Return ONLY a SINGLE valid JSON object. "
                    "No markdown. No commentary. Use double quotes. Follow the schema exactly."
                ),
            }
        ]
        time.sleep(sleep_s)

    raise ValueError(f"Model did not return valid JSON after retries. Last error: {last_err}")


# ---------------------------
# Validation-reformat loop (requested)
# ---------------------------
T = TypeVar("T", bound=BaseModel)

SYSTEM_FORMATTER = (
    "You are a schema enforcer.\n"
    "Return ONLY a SINGLE valid JSON object. No markdown, no prose.\n"
    "Use double quotes. Ensure it parses with json.loads().\n"
    "Fix ONLY what is needed for schema compliance; do not change meaning."
)

def validate_or_reformat(
    cfg: OpenRouterConfig,
    data: Any,
    model_cls: Type[T],
    schema_hint: str,
    context_label: str,
    *,
    max_attempts: int = 3,
) -> T:
    """
    Try to validate with Pydantic.
    If ValidationError occurs, call formatter_model to fix JSON to the schema, up to max_attempts.
    """
    cur = data
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return model_cls(**cur)  # type: ignore[arg-type]
        except ValidationError as e:
            last_err = e
            err_txt = str(e)

            # Ask formatter to correct it
            user = (
                f"Context: {context_label}\n"
                f"Validation error:\n{err_txt}\n\n"
                f"Required schema hint:\n{schema_hint}\n\n"
                "Important constraints:\n"
                f"- For FullQuestion.ai_rating, allowed: {sorted(ALLOWED_RATINGS)}\n"
                f"- VerifyResponse.status allowed: {sorted(ALLOWED_VERIFY_STATUS)}\n"
                "- Do NOT set ai_rating to 'pass' or 'fix'. Those belong only to VerifyResponse.status.\n"
                "- Return ONLY JSON.\n\n"
                "Input JSON:\n"
                f"{json.dumps(cur, ensure_ascii=False)}"
            )
            cur = call_json_strict(
                cfg,
                cfg.formatter_model,
                [{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}],
                schema_hint=schema_hint,
                retries=1,
            )

        except Exception as e:
            last_err = e
            break

    raise ValueError(f"Failed to validate after {max_attempts} formatter attempts ({context_label}). Last error: {last_err}")


# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PRIMARY = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."
SYSTEM_LIGHT = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."


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
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


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
    start_d: date,
    end_d: date,
    use_formatter_after_selection: bool,
    progress_cb=None,
):
    # 1) protos
    if progress_cb:
        progress_cb(f"Generating {n} protos for topic: {topic}")
    msgs, hint = prompt_generate_protos(topic, n, start_d, end_d)
    raw = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    protos = [validate_or_reformat(cfg, p, ProtoQuestion, """{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}""", "ProtoQuestion") for p in (raw.get("protos", []) or [])][:n]
    if not protos:
        raise ValueError(f"No protos generated for topic: {topic}")

    # 2) select k
    if progress_cb:
        progress_cb(f"Selecting top {k} protos for topic: {topic}")
    msgs, hint = prompt_select_k(topic, protos, k)
    raw2 = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    selected_raw = raw2.get("selected", []) or []
    selected = [validate_or_reformat(cfg, p, ProtoQuestion, """{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}""", "Selected ProtoQuestion") for p in selected_raw][:k]
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
            selected = [validate_or_reformat(cfg, p, ProtoQuestion, hint, "Canonicalized ProtoQuestion") for p in canon_list]

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
        vr = validate_or_reformat(cfg, raw4, VerifyResponse, hint_v, f"VerifyResponse ({topic} #{i})", max_attempts=3)
        fq2 = normalize_full_question_fields(vr.question)

        if progress_cb:
            progress_cb(f"Formatting card {i}/{len(selected)} for topic: {topic}")
        msgs, hint_f = prompt_format_single(fq2)
        raw5 = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint_f, retries=1)
        fq_final = validate_or_reformat(cfg, raw5.get("question", raw5), FullQuestion, """FULL_QUESTION_OBJECT""", f"Formatted FullQuestion ({topic} #{i})", max_attempts=3)
        fq_final = normalize_full_question_fields(fq_final)

        yield fq_final


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Topics -> Forecast Questions CSV", layout="wide")
st.title("Topics CSV → Forecast Questions → CSV Export (with formatter retries)")

colA, colB = st.columns(2)

with colA:
    api_key = st.text_input("OpenRouter API key", type="password").strip()
    primary_model = st.text_input("Primary model (generation)", value="openai/gpt-4.1-mini")
    light_model = st.text_input("Light model (verify / rebalance)", value="openai/gpt-4.1-mini")
    formatter_model = st.text_input("Formatter model (schema enforcement / repair)", value="openai/gpt-4.1-nano")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens per call", min_value=500, max_value=8000, value=2000, step=100)
    use_rf = st.checkbox("Use response_format=json_object (if supported)", value=True)

with colB:
    uploaded = st.file_uploader("Upload topics CSV", type=["csv"])
    n = st.number_input("n = proto questions per topic", min_value=1, max_value=50, value=6, step=1)
    k = st.number_input("k = keep per topic", min_value=1, max_value=20, value=3, step=1)
    start_d = st.date_input("Questions start date", value=date.today())
    end_d = st.date_input("Questions end date (must resolve by this date)", value=date.today())
    do_rebalance = st.checkbox("Rebalance to ~50% binary / 30% numeric / 20% MCQ", value=True)
    use_formatter_after_selection = st.checkbox("Formatter after selecting k protos (recommended)", value=True)
    use_formatter_after_rebalance = st.checkbox("Formatter after rebalancing (recommended)", value=True)
    do_final_clean = st.checkbox("Final clean pass (formatter)", value=True)

st.divider()

topics: List[str] = []
if uploaded:
    try:
        fieldnames, rows = read_topics_csv(uploaded)
        default_col = pick_topic_column(fieldnames, rows)
        chosen_col = st.selectbox("Topic column", options=fieldnames, index=fieldnames.index(default_col))
        topics_raw = [str(r.get(chosen_col, "")).strip() for r in rows]
        topics = [t for t in topics_raw if t]
        st.write(f"Detected {len(topics)} topics.")
        if topics:
            st.dataframe({"topic": topics[: min(30, len(topics))]})
    except Exception as e:
        st.error(f"Failed to read topics CSV: {e}")
        topics = []

run = st.button("Generate CSV", type="primary", disabled=not (api_key and topics and start_d and end_d))

status_box = st.empty()
progress = st.progress(0)

draft_info_ph = st.empty()
draft_preview_ph = st.empty()
final_info_ph = st.empty()
final_preview_ph = st.empty()

def progress_cb(msg: str):
    status_box.info(msg)

if run:
    if end_d < start_d:
        st.error("End date must be >= start date.")
        st.stop()
    if int(k) > int(n):
        st.error("k must be <= n.")
        st.stop()

    cfg = OpenRouterConfig(
        api_key=api_key,
        primary_model=primary_model.strip(),
        light_model=light_model.strip(),
        formatter_model=formatter_model.strip(),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        use_response_format_json=bool(use_rf),
    )

    try:
        # Rough step count for UI progress
        per_topic = 2 + (1 if use_formatter_after_selection else 0) + int(k) * 3
        total_steps = max(1, len(topics)) * per_topic
        if do_rebalance:
            total_steps += 1
            if use_formatter_after_rebalance:
                total_steps += 1
        if do_final_clean:
            total_steps += 1

        done = {"value": 0}
        def tick(msg: str):
            done["value"] += 1
            progress.progress(min(1.0, done["value"] / max(1, total_steps)))
            progress_cb(msg)

        # Draft accumulation (incremental)
        draft_questions: List[FullQuestion] = []
        draft_rows: List[Dict[str, Any]] = []

        for t_i, topic in enumerate(topics, start=1):
            tick(f"=== Topic {t_i}/{len(topics)}: {topic} ===")
            for q in generate_for_topic_iter(
                cfg=cfg,
                topic=topic,
                n=int(n),
                k=int(k),
                start_d=start_d,
                end_d=end_d,
                use_formatter_after_selection=use_formatter_after_selection,
                progress_cb=tick,
            ):
                draft_questions.append(q)
                draft_rows.append(full_question_to_row(q))

                draft_info_ph.markdown(
                    f"**Draft rows:** {len(draft_rows)} | **Draft type counts:** {count_types(draft_questions)}"
                )
                draft_preview_ph.dataframe(draft_rows[-min(20, len(draft_rows)) :], use_container_width=True)

        draft_csv_text = write_csv(draft_rows)

        # Final passes
        final_questions = list(draft_questions)
        log: Dict[str, Any] = {"type_counts_draft": count_types(draft_questions)}

        if do_rebalance and final_questions:
            tick("Rebalancing types to ~50/30/20.")
            total = len(final_questions)
            tb, tn, tm = compute_targets(total)
            msgs, hint = prompt_rebalance(final_questions, tb, tn, tm, end_d)
            raw = call_json_strict(cfg, cfg.light_model, msgs, schema_hint=hint)
            edited = raw.get("questions", []) or []
            # validate each FullQuestion with formatter retries
            tmp: List[FullQuestion] = []
            for idx, qd in enumerate(edited, start=1):
                fq = validate_or_reformat(cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Rebalance FullQuestion #{idx}", max_attempts=3)
                tmp.append(normalize_full_question_fields(fq))
            final_questions = tmp
            log["rebalance_change_log"] = raw.get("change_log", [])
            log["type_counts_after_rebalance"] = count_types(final_questions)

            if use_formatter_after_rebalance:
                tick("Canonicalizing post-rebalance list (formatter).")
                msgs, hint2 = prompt_canonicalize_questions_list(final_questions)
                canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint2, retries=1)
                canon_list = canon.get("questions", []) or []
                if canon_list:
                    tmp2: List[FullQuestion] = []
                    for idx, qd in enumerate(canon_list, start=1):
                        fq = validate_or_reformat(cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Post-rebalance canonical FullQuestion #{idx}", max_attempts=3)
                        tmp2.append(normalize_full_question_fields(fq))
                    final_questions = tmp2

        if do_final_clean and final_questions:
            tick("Final clean pass (formatter).")
            msgs, hint = prompt_clean_questions(final_questions)
            raw = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
            cleaned = raw.get("questions", []) or []
            tmp3: List[FullQuestion] = []
            for idx, qd in enumerate(cleaned, start=1):
                fq = validate_or_reformat(cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Final clean FullQuestion #{idx}", max_attempts=3)
                tmp3.append(normalize_full_question_fields(fq))
            final_questions = tmp3
            log["clean_notes"] = raw.get("notes", [])
            log["type_counts_final"] = count_types(final_questions)

        final_rows = [full_question_to_row(q) for q in final_questions]
        final_csv_text = write_csv(final_rows)

        final_info_ph.markdown(
            f"**Final rows:** {len(final_rows)} | **Final type counts:** {count_types(final_questions)}"
        )
        final_preview_ph.dataframe(final_rows[: min(20, len(final_rows))], use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download DRAFT CSV (incremental)",
                data=draft_csv_text.encode("utf-8"),
                file_name="forecast_questions_DRAFT.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                label="Download FINAL CSV",
                data=final_csv_text.encode("utf-8"),
                file_name="forecast_questions_FINAL.csv",
                mime="text/csv",
            )

        with st.expander("Logs / change log", expanded=False):
            st.json(log)

        st.success("Done.")

    except (OpenRouterError, ValidationError, ValueError) as e:
        st.error(str(e))
        if "draft_rows" in locals() and draft_rows:
            salvage_csv = write_csv(draft_rows)
            st.download_button(
                label="Download SALVAGED DRAFT CSV (partial)",
                data=salvage_csv.encode("utf-8"),
                file_name="forecast_questions_SALVAGED_DRAFT.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        if "draft_rows" in locals() and draft_rows:
            salvage_csv = write_csv(draft_rows)
            st.download_button(
                label="Download SALVAGED DRAFT CSV (partial)",
                data=salvage_csv.encode("utf-8"),
                file_name="forecast_questions_SALVAGED_DRAFT.csv",
                mime="text/csv",
            )
