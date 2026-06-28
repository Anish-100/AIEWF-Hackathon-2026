"""Gemini 2.5 Flash claim-detection adapter.

`detect(sentence)` → strict JSON `{is_checkworthy, subject, predicate, value, unit}`.

We use `response_mime_type="application/json"` plus a `response_schema` so the
model is forced to emit parseable JSON. Retry once on parse failure; on second
failure return a non-check-worthy stub so the caller can drop the sentence
cleanly.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

import config

log = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


_SYSTEM_INSTRUCTION = (
    "You decide if a single transcribed sentence contains a verifiable factual claim "
    "about the US labor market (jobs, unemployment, payrolls, wages, labor force "
    "participation, hiring, layoffs, sector employment). "
    "A check-worthy claim makes a specific, verifiable statement with a subject, "
    "predicate, and a numeric or named value (e.g. 'US unemployment rate in May 2025 was 4.1 percent'). "
    "Opinions, predictions, hypotheticals, questions, and small-talk are NOT check-worthy. "
    "Output canonical, stable subjects so the same fact across paraphrases hashes to the same key "
    "(e.g. always 'US unemployment rate May 2025', not 'unemployment in May'). "
    "Always respond with the JSON object — never prose."
)


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "is_checkworthy": {
            "type": "BOOLEAN",
            "description": "True only if the sentence makes a specific, verifiable factual claim about the US labor market.",
        },
        "subject": {
            "type": "STRING",
            "description": "Canonical, stable subject phrase. Empty if not check-worthy.",
        },
        "predicate": {
            "type": "STRING",
            "description": "Relation verb, e.g. 'equals', 'grew_by', 'fell_to'. Empty if not check-worthy.",
        },
        "value": {
            "type": "STRING",
            "description": "Numeric or named value as a string (e.g. '4.1', '139000'). Empty if not check-worthy.",
        },
        "unit": {
            "type": "STRING",
            "description": "Unit, e.g. '%', 'jobs', 'USD'. Empty if not applicable.",
        },
    },
    "required": ["is_checkworthy", "subject", "predicate", "value", "unit"],
}


def _stub_negative() -> dict:
    return {"is_checkworthy": False, "subject": "", "predicate": "", "value": "", "unit": ""}


async def _call_once(sentence: str) -> dict:
    client = _get_client()
    resp = await client.aio.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=sentence,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
            max_output_tokens=256,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise ValueError("empty response")
    return json.loads(text)


async def detect(sentence: str) -> dict:
    sentence = (sentence or "").strip()
    if not sentence:
        return _stub_negative()
    try:
        return await _call_once(sentence)
    except Exception as first:
        log.warning("Flash.detect retry after error: %s", first)
        try:
            return await _call_once(sentence)
        except Exception as second:
            log.error("Flash.detect failed twice; dropping sentence (%s)", second)
            return _stub_negative()
