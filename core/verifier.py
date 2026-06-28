"""Hot-path verifier.

Given a Claim with `subject`/`predicate`/`value` already filled by the Flash
detector, embed the subject, vector-search memory, and assign a verdict:

- HIT (cosine ≥ SUBJECT_MATCH_THRESHOLD AND value compares cleanly)
    → status="verified", source="kb"|"memory", verdict="true"|"false"
- MISS (no match above threshold)
    → status="researching" (handed to Phase 6 research queue when wired)

Sets `time_to_verdict_ms` so the metrics tracker can show the cold→warm delta.
"""
from __future__ import annotations

import logging
import time

import config
from adapters import gemini_embed
from core import memory
from core.schemas import Claim, VerifiedFact

log = logging.getLogger(__name__)


def _coerce_number(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").replace("$", "").strip()
    if not s:
        return None
    # Strip common unit suffixes: %, k, m, b
    mult = 1.0
    if s[-1:].lower() == "k":
        mult, s = 1e3, s[:-1]
    elif s[-1:].lower() == "m":
        mult, s = 1e6, s[:-1]
    elif s[-1:].lower() == "b":
        mult, s = 1e9, s[:-1]
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _values_match(claim_val, fact_val, tolerance: float) -> bool | None:
    """Return True/False if comparable, None if we can't tell."""
    if claim_val is None or fact_val is None:
        return None
    cn = _coerce_number(claim_val)
    fn = _coerce_number(fact_val)
    if cn is not None and fn is not None:
        if fn == 0:
            return abs(cn) <= tolerance
        rel = abs(cn - fn) / abs(fn)
        return rel <= tolerance
    # Fall back to case-insensitive string equality.
    return str(claim_val).strip().lower() == str(fact_val).strip().lower()


async def verify(claim: Claim) -> tuple[Claim, bool]:
    """Returns (claim_with_verdict_filled, hit_bool)."""
    started_ns = time.perf_counter_ns()
    mem = memory.get_memory()

    # Subject (preferred) or raw text gets embedded.
    embed_text = claim.subject or claim.raw_text
    try:
        emb = await gemini_embed.embed(embed_text)
    except Exception:
        log.exception("verifier: embedding failed")
        claim.status = "researching"
        return claim, False
    claim.embedding = emb

    candidates = mem.vector_search(emb, top_k=5)
    if not candidates:
        claim.status = "researching"
        claim.source = None
        log.info("verifier: MISS (empty memory) for subject=%r", claim.subject)
        return claim, False

    best, score = candidates[0]
    if score < config.SUBJECT_MATCH_THRESHOLD:
        claim.status = "researching"
        claim.source = None
        log.info(
            "verifier: MISS subject=%r best=%r score=%.3f < %.3f",
            claim.subject, best.subject, score, config.SUBJECT_MATCH_THRESHOLD,
        )
        return claim, False

    # We have a likely-same-subject fact; compare values.
    match = _values_match(claim.value, best.canonical_value, config.VALUE_TOLERANCE)
    if match is True:
        claim.verdict = "true"
        claim.status = "verified"
    elif match is False:
        claim.verdict = "false"
        claim.status = "flagged"
    else:
        claim.verdict = "unverifiable"
        claim.status = "verified"

    claim.source = best.source or "memory"
    claim.confidence = float(score)
    claim.explanation = best.explanation
    claim.resolved_at = time.time()
    claim.time_to_verdict_ms = int((time.perf_counter_ns() - started_ns) / 1_000_000)

    # Touch the fact (memory dynamics for the cold→warm story).
    try:
        mem.touch(best.id)
    except Exception:
        log.exception("verifier: touch failed for %s", best.id)

    log.info(
        "verifier: HIT subject=%r → %r value=%r vs %r → %s (score=%.3f, %dms)",
        claim.subject, best.subject, claim.value, best.canonical_value,
        claim.verdict, score, claim.time_to_verdict_ms,
    )
    return claim, True
