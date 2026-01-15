from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar

import requests

from models import ALLOWED_RATINGS, ALLOWED_VERIFY_STATUS, ValidationError
from utils import try_parse_json


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
T = TypeVar("T")

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

    for _attempt in range(1, max_attempts + 1):
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

    raise ValueError(
        f"Failed to validate after {max_attempts} formatter attempts ({context_label}). "
        f"Last error: {last_err}"
    )


__all__ = ["OpenRouterConfig", "OpenRouterError", "call_json_strict", "validate_or_reformat"]
