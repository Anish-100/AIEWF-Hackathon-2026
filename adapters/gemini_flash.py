"""Gemini 2.5 Flash claim-detection adapter.

`detect(sentence)` → `list[dict]`, one entry per verifiable claim found in the
sentence. Empty list means no check-worthy content. Each entry has keys
`subject`, `predicate`, `value`, `unit`.

A single sentence can pack multiple claims ("Unemployment was 4.1% in May and
payrolls rose by 139,000") so the schema returns an array.

We use `response_mime_type="application/json"` plus a `response_schema` so the
model is forced to emit parseable JSON. Retry once on parse failure; on second
failure return an empty list so the caller drops the sentence cleanly.
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
    "You extract verifiable factual claims from a single transcribed sentence. "
    "A check-worthy claim makes a specific, verifiable statement with a subject, predicate, and "
    "a numeric or named value — for example a statistic, measurement, dated event, named entity, "
    "monetary amount, percentage, count, ranking, or quoted attribution. "
    "Claims about the US labor market (jobs, unemployment, payrolls, wages, labor force "
    "participation, hiring, layoffs, sector employment) are common in this stream but NOT the "
    "only valid topic — extract verifiable claims from ANY domain (economics, business, science, "
    "politics, sports, history, geography, etc.) as long as they are specific and checkable. "
    "Opinions, predictions, hypotheticals, questions, and small-talk are NOT check-worthy. "
    "A sentence may contain ZERO, ONE, or MULTIPLE claims — return every distinct claim. "
    "If the sentence has no check-worthy claim, return an empty array. "
    "Use canonical, stable subject phrasing so the same fact across paraphrases hashes to the "
    "same key (always 'US unemployment rate May 2025', never 'unemployment in May'; "
    "'NVIDIA Q3 2025 revenue', not 'their last quarter'). "
    "If the value is non-numeric (a name, place, date), put it in `value` as a string and leave "
    "`unit` empty. "
    "If a PRIOR CONTEXT block is provided, use it ONLY to resolve pronouns ('it', 'they', "
    "'their'), relative dates ('last month', 'this quarter'), and ambiguous references in the "
    "TARGET SENTENCE. Do NOT extract claims from the prior context — only from the target "
    "sentence. "
    "Always respond with the JSON object — never prose."
)


_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "description": "One entry per verifiable claim. Empty if the sentence has none.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject": {
                        "type": "STRING",
                        "description": "Canonical, stable subject phrase.",
                    },
                    "predicate": {
                        "type": "STRING",
                        "description": "Relation verb, e.g. 'equals', 'grew_by', 'fell_to'.",
                    },
                    "value": {
                        "type": "STRING",
                        "description": "Numeric or named value as a string (e.g. '4.1', '139000').",
                    },
                    "unit": {
                        "type": "STRING",
                        "description": "Unit, e.g. '%', 'jobs', 'USD'. Empty if not applicable.",
                    },
                },
                "required": ["subject", "predicate", "value", "unit"],
            },
        },
    },
    "required": ["claims"],
}


def _build_contents(sentence: str, prior_context: list[str] | None) -> str:
    """Compose the user message. When prior context is provided we frame both
    blocks explicitly so the model can resolve pronouns / relative dates
    against context without mistaking context for the extraction target."""
    if not prior_context:
        return sentence
    ctx_block = "\n".join(f"- {line}" for line in prior_context if line)
    return (
        "PRIOR CONTEXT (for reference only — do NOT extract claims from these):\n"
        f"{ctx_block}\n\n"
        "TARGET SENTENCE (extract claims from this ONLY):\n"
        f"{sentence}"
    )


async def _call_once(sentence: str, prior_context: list[str] | None) -> list[dict]:
    client = _get_client()
    resp = await client.aio.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=_build_contents(sentence, prior_context),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.0,
            max_output_tokens=512,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise ValueError("empty response")
    parsed = json.loads(text)
    claims = parsed.get("claims") or []
    # Drop entries missing a subject — they're useless downstream.
    return [c for c in claims if isinstance(c, dict) and str(c.get("subject") or "").strip()]


async def detect(sentence: str, prior_context: list[str] | None = None) -> list[dict]:
    """Extract verifiable claims from `sentence`.

    `prior_context` is an optional list of recent finalized utterances from
    the same speaker (most recent last). The model uses them ONLY to resolve
    pronouns / relative dates in `sentence`. Pass `None` (or empty) to skip
    context — caller's choice.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return []
    try:
        return await _call_once(sentence, prior_context)
    except Exception as first:
        log.warning("Flash.detect retry after error: %s", first)
        try:
            return await _call_once(sentence, prior_context)
        except Exception as second:
            log.error("Flash.detect failed twice; dropping sentence (%s)", second)
            return []
