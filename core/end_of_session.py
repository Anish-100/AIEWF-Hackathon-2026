"""End-of-session transcript distillation.

After a session ends, replay its full transcript through Flash and extract
durable, fact-checkable claims. Each extracted fact is embedded and written
into the persistent memory with full provenance (which session, which speaker,
which exact quote).

This complements the **per-utterance** detector (which fires during the live
session for low latency) and the Phase 6 **async research** path (which
resolves single MISSes on the open web). The distiller's edge is *full
conversational context* — it can resolve "their revenue" → a canonical subject,
dedupe paraphrases of the same fact, and weight claims that came up repeatedly.

Usage:
    python -m scripts.distill_session [SESSION_ID]
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from google import genai
from google.genai import types

import config
from adapters import gemini_embed
from core import memory
from core.schemas import VerifiedFact

log = logging.getLogger(__name__)

_DISTILLER_SYSTEM = (
    "You are reviewing a transcript of a multi-speaker conversation about the US labor market. "
    "Extract every distinct, verifiable factual claim that was asserted during the conversation. "
    "Use the FULL conversation as context to resolve pronouns and ambiguous references. "
    "Canonicalise subjects so paraphrases collapse to one fact (e.g. 'unemployment in May' and "
    "'May 2025 jobless rate' → 'US unemployment rate May 2025'). "
    "Skip opinions, predictions, questions, and small-talk. "
    "For each fact include the exact transcript quote it came from (verbatim) and which speaker said it. "
    "Output JSON only — never prose."
)

_DISTILL_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "facts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject":          {"type": "STRING"},
                    "predicate":        {"type": "STRING"},
                    "value":            {"type": "STRING"},
                    "unit":             {"type": "STRING"},
                    "supporting_quote": {"type": "STRING"},
                    "speaker":          {"type": "STRING"},
                },
                "required": ["subject", "value", "supporting_quote", "speaker"],
            },
        }
    },
    "required": ["facts"],
}


def _claim_key(subject: str) -> str:
    return hashlib.sha1(subject.strip().lower().encode("utf-8")).hexdigest()[:16]


def _format_transcript(utterances: list[dict]) -> str:
    return "\n".join(
        f"[{u['clip_ts']:.1f}s] {u['speaker_id']}: {u['text']}" for u in utterances
    )


async def distill_session(session_id: str) -> int:
    """Returns number of new facts written to memory."""
    mem = memory.get_memory()
    transcript = mem.get_session_transcript(session_id)
    if not transcript:
        log.warning("distill: session %s has no utterances", session_id)
        return 0
    log.info("distill: session=%s utterances=%d", session_id, len(transcript))

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = await client.aio.models.generate_content(
        model=config.GEMINI_FLASH_MODEL,
        contents=_format_transcript(transcript),
        config=types.GenerateContentConfig(
            system_instruction=_DISTILLER_SYSTEM,
            response_mime_type="application/json",
            response_schema=_DISTILL_SCHEMA,
            temperature=0.0,
            max_output_tokens=4096,
        ),
    )
    try:
        parsed = json.loads(resp.text or "{}")
    except json.JSONDecodeError:
        log.error("distill: model returned unparseable JSON")
        return 0
    facts = parsed.get("facts") or []
    log.info("distill: model proposed %d facts", len(facts))
    if not facts:
        return 0

    subjects = [str(f.get("subject", "")).strip() for f in facts]
    embeddings = await gemini_embed.embed_batch(subjects)

    now = time.time()
    written = 0
    for f, emb in zip(facts, embeddings):
        subject = str(f.get("subject", "")).strip()
        if not subject or not emb:
            continue
        fact = VerifiedFact(
            claim_key=_claim_key(subject),
            subject=subject,
            canonical_value=f.get("value"),
            unit=str(f.get("unit") or "").strip() or None,
            verdict="true",     # provisional: came from observed conversation
            source="session",
            explanation=f"Extracted from session {session_id} (speaker={f.get('speaker','?')})",
            embedding=emb,
            first_seen_ts=now,
            times_seen=1,
        )
        mem.put(
            fact,
            source_session_id=session_id,
            source_speaker=str(f.get("speaker") or "").strip() or None,
            extracted_at=now,
            supporting_quote=str(f.get("supporting_quote") or "").strip() or None,
        )
        written += 1
    log.info("distill: wrote %d facts to memory (store size=%d)", written, mem.size())
    return written


async def distill_latest() -> int:
    """Distill the most recent ENDED session."""
    sessions = memory.get_memory().list_sessions()
    ended = [s for s in sessions if s.get("ended_at")]
    if not ended:
        log.warning("distill: no ended sessions to distill")
        return 0
    return await distill_session(ended[0]["id"])
