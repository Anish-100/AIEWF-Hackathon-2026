"""Async research adapter — Gemini 2.5 Flash with Google Search grounding.

Why not Antigravity? The Antigravity managed agent (Interactions API) is built
for multi-step computer-use tasks (browse, write code, execute in a hosted
Linux sandbox). For our use case — "look up ONE number on the web and return
JSON" — its agentic loop adds 30-120 s of latency that's incompatible with
a real-time fact-check. Gemini 2.5 Flash with Google Search grounding hits
the web through Google's index and returns in ~5-10 s.

The interface stays identical (`research(claim) -> ResearchResult`) so the
research_queue and orchestrator wiring don't change. The module is still
named `antigravity` to avoid file churn; flip USE_ANTIGRAVITY=true in
`.env.local` to enable this path.

Docs: https://ai.google.dev/gemini-api/docs/grounding
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from google import genai
from google.genai import types

import config
from core.schemas import Claim

log = logging.getLogger(__name__)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


@dataclass
class ResearchResult:
    verdict: str                       # "true" | "false" | "unverifiable" | "dubious"
    canonical_value: Any
    unit: Optional[str]
    explanation: str
    environment_id: Optional[str] = None  # unused with Search grounding
    raw_text: Optional[str] = None     # for debugging


# Search grounding does NOT combine with response_schema. We use a clear
# prompt and parse the JSON out of the free-form response (with a salvage
# step for fenced code blocks).
_SYSTEM_INSTRUCTION = (
    "You verify ONE factual claim using Google Search. Speed matters — this "
    "is a real-time fact-check, not a research report.\n"
    "\n"
    "Process: do ONE Google search for the most authoritative primary source "
    "for THIS specific claim (government statistics agencies, central banks, "
    "official corporate filings, peer-reviewed data, or major reputable news). "
    "Find the single value. Return JSON.\n"
    "\n"
    "Output ONLY a JSON object with exactly these fields:\n"
    '  "verdict": "true" | "false" | "unverifiable" | "dubious"\n'
    '  "canonical_value": the authoritative value as a string (e.g. "4.1", "228000")\n'
    '  "unit": the unit (e.g. "%", "thousand jobs"). Empty string if N/A.\n'
    '  "explanation": ONE short sentence citing the source by name.\n'
    "\n"
    "verdict rules: \"true\" if claim matches authoritative value within 1%; "
    "\"false\" if it clearly conflicts; \"unverifiable\" if no primary source "
    "is found. NO prose outside the JSON object. Do not wrap the JSON in "
    "Markdown fences."
)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a free-form model response."""
    if not text:
        raise ValueError("empty response")
    # 1) direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2) fenced ```json {...} ```
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3) first bare {...} object
    m = _JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


def _format_input(claim: Claim) -> str:
    parts = [
        f"Subject: {claim.subject or '(missing)'}",
        f"Claimed value: {claim.value} {claim.unit or ''}".strip(),
        f"Spoken context: {claim.raw_text}",
    ]
    return "\n".join(parts)


async def research(claim: Claim, *, environment_id: Optional[str] = None) -> ResearchResult:
    """Resolve a single claim via Gemini Flash + Google Search grounding."""
    client = _get_client()
    log.info("research: starting subject=%r value=%r", claim.subject, claim.value)
    resp = await client.aio.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=_format_input(claim),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("research: empty response")
    parsed = _extract_json(text)
    result = ResearchResult(
        verdict=str(parsed.get("verdict") or "unverifiable"),
        canonical_value=parsed.get("canonical_value"),
        unit=(str(parsed.get("unit") or "").strip() or None),
        explanation=str(parsed.get("explanation") or "").strip(),
        raw_text=text,
    )
    log.info(
        "research: finished subject=%r → verdict=%s value=%r",
        claim.subject, result.verdict, result.canonical_value,
    )
    return result
