from __future__ import annotations

import ast
import csv
import io
import json
import re
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from models import CSV_COLUMNS, FullQuestion, PARIS_TZ, PYDANTIC_V2
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
