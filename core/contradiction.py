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
import re
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
    subject_key: str            # normalised subject for exact-equality match
    embedding: np.ndarray
    value: Any
    unit: Optional[str]
    clip_ts: float


_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize_subject(subject: str) -> str:
    """Canonical form for exact-equality subject comparison.

    Lowercase, strip punctuation, collapse whitespace. Two subjects that differ
    only by a month / quarter / year token will NOT collapse — that's the
    point: 'us unemployment rate may 2025' != 'us unemployment rate march 2025',
    so they won't be flagged as contradictions even though their embeddings
    are ~0.97 cosine similar.
    """
    if not subject:
        return ""
    s = _PUNCT_RE.sub(" ", subject.lower())
    s = _WS_RE.sub(" ", s).strip()
    return s


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
        new_key = _normalize_subject(claim.subject)
        for prior in self._log:
            # Two guards. Embedding similarity is a fast filter; the canonical
            # subject-string equality is the real gate. Embeddings happily
            # consider "US unemployment rate May 2025" and "US unemployment
            # rate March 2025" ~0.97 similar, so without the string-equality
            # check we'd fire spurious "contradictions" between different
            # months of the same metric.
            if prior.subject_key != new_key:
                continue
            sim = _cosine(emb, prior.embedding)
            if sim < config.SUBJECT_MATCH_THRESHOLD:
                # Should be impossible (same canonical subject ⇒ near-identical
                # embedding) but cheap to defend against odd embedding drift.
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
                f"{_fmt_value(claim.value, claim.unit)} at t={claim.clip_ts:.1f}s."
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
            subject_key=_normalize_subject(claim.subject),
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
