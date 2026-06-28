"""Multi-speaker contradiction detector.

For every verified claim that has a subject + embedding + value, scan the
in-session claim log for a prior claim with:

  - **same subject** (cosine ≥ SUBJECT_MATCH_THRESHOLD), AND
  - **conflicting value** (per the unit-aware comparator in core.verifier)

When found, emit a `Contradiction` tagged:

  - `same_speaker` — one person changed their story across the session
  - `cross_speaker` — two speakers disagree on the same fact

This is intra-session only. KB conflicts (a single claim vs the curated KB)
are already surfaced by the verifier as a ✗ FALSE card; we don't double-tag
them as contradictions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

import config
from core import verifier
from core.schemas import Claim, Contradiction

log = logging.getLogger(__name__)


@dataclass
class _LoggedClaim:
    id: str
    speaker_id: str
    subject: str
    embedding: np.ndarray
    value: Any
    unit: Optional[str]
    clip_ts: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float((a @ b) / (na * nb))


def _fmt_value(v: Any, unit: Optional[str]) -> str:
    if v is None:
        return "?"
    s = str(v)
    if unit:
        return f"{s} {unit}".strip()
    return s


class ContradictionChecker:
    """Per-session, in-memory log + scanner."""

    def __init__(self) -> None:
        self._log: list[_LoggedClaim] = []

    def reset(self) -> None:
        self._log.clear()

    def record_and_check(self, claim: Claim) -> Optional[Contradiction]:
        """Add the claim to the session log and, if it contradicts a prior
        logged claim, return the Contradiction. The new claim is logged
        whether or not it contradicts (so later claims can be compared against
        it)."""
        if not (claim.subject and claim.embedding and claim.value is not None):
            self._add(claim)
            return None

        emb = np.asarray(claim.embedding, dtype=np.float32)
        for prior in self._log:
            sim = _cosine(emb, prior.embedding)
            if sim < config.SUBJECT_MATCH_THRESHOLD:
                continue
            match = verifier.values_match(
                claim.value, prior.value, claim.unit, prior.unit, config.VALUE_TOLERANCE
            )
            if match is not False:
                # Match=True (same value) or match=None (incomparable) → no conflict.
                continue
            kind = "same_speaker" if claim.speaker_id == prior.speaker_id else "cross_speaker"
            explanation = (
                f"{prior.speaker_id} said {_fmt_value(prior.value, prior.unit)} at "
                f"t={prior.clip_ts:.1f}s; {claim.speaker_id} now says "
                f"{_fmt_value(claim.value, claim.unit)} at t={claim.clip_ts:.1f}s "
                f"(subject similarity {sim:.2f})."
            )
            contradiction = Contradiction(
                subject=claim.subject,
                kind=kind,
                speaker_a_id=prior.speaker_id,
                speaker_b_id=claim.speaker_id,
                claim_a_id=prior.id,
                claim_b_id=claim.id,
                value_a=prior.value,
                value_b=claim.value,
                ts_a=prior.clip_ts,
                ts_b=claim.clip_ts,
                explanation=explanation,
            )
            log.warning(
                "contradiction (%s): %s — %s",
                kind, claim.subject, explanation,
            )
            self._add(claim)
            return contradiction

        self._add(claim)
        return None

    def _add(self, claim: Claim) -> None:
        if not (claim.subject and claim.embedding):
            return
        self._log.append(_LoggedClaim(
            id=claim.id,
            speaker_id=claim.speaker_id,
            subject=claim.subject,
            embedding=np.asarray(claim.embedding, dtype=np.float32),
            value=claim.value,
            unit=claim.unit,
            clip_ts=claim.clip_ts,
        ))


def check_contradiction(claim: Claim) -> Optional[Contradiction]:
    """Module-level convenience for callers without their own checker — uses a
    process-wide singleton. The orchestrator owns its own per-session instance
    so this singleton is mostly for tests and ad-hoc usage."""
    global _singleton
    if _singleton is None:
        _singleton = ContradictionChecker()
    return _singleton.record_and_check(claim)


_singleton: Optional[ContradictionChecker] = None
