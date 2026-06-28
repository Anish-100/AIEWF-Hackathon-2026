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


def coerce_number(x) -> float | None:
    return _coerce_number(x)


def values_match(claim_val, fact_val, claim_unit, fact_unit, tolerance: float) -> bool | None:
    """Public wrapper around _values_match so other modules (contradiction)
    can use the same unit-aware comparison without re-implementing it."""
    return _values_match(claim_val, fact_val, claim_unit, fact_unit, tolerance)


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


def _unit_multiplier(unit) -> float:
    """Extract scale from a unit string. 'thousand jobs' → 1000, 'million' →
    1_000_000, etc. Returns 1.0 if no scale word is present."""
    if not unit:
        return 1.0
    u = str(unit).lower().strip()
    if "trillion" in u:
        return 1e12
    if "billion" in u:
        return 1e9
    if "million" in u:
        return 1e6
    if "thousand" in u:
        return 1e3
    return 1.0


def _values_match(claim_val, fact_val, claim_unit, fact_unit, tolerance: float) -> bool | None:
    """Return True/False if comparable, None if we can't tell. Normalises
    unit scale words ('thousand', 'million', ...) on both sides so the
    KB's `228 thousand jobs` matches a claim of `228000 jobs`.
    """
    if claim_val is None or fact_val is None:
        return None
    cn = _coerce_number(claim_val)
    fn = _coerce_number(fact_val)
    if cn is not None and fn is not None:
        cn *= _unit_multiplier(claim_unit)
        fn *= _unit_multiplier(fact_unit)
        # A small floor so floating-point rounding ("4.1" vs 4.1) doesn't
        # flag identical values as conflicting. `tolerance=0` from env still
        # means "essentially exact", and 1% is well under any real labor stat
        # we care about.
        eff_tol = max(tolerance, 0.01)
        if fn == 0:
            return abs(cn) <= eff_tol
        rel = abs(cn - fn) / abs(fn)
        return rel <= eff_tol
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

    # We have a likely-same-subject fact; compare values (unit-aware).
    match = _values_match(claim.value, best.canonical_value, claim.unit, best.unit, config.VALUE_TOLERANCE)
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
