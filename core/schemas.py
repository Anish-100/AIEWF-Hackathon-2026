from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> float:
    return time.time()


ClaimStatus = Literal["detected", "researching", "verified", "flagged"]
Verdict = Literal["true", "false", "unverifiable", "dubious"]
Source = Literal["kb", "memory", "web"]
ContradictionKind = Literal["same_speaker", "cross_speaker"]


@dataclass
class Claim:
    session_id: str
    speaker_id: str
    clip_ts: float
    raw_text: str
    subject: str = ""
    predicate: str = ""
    value: Any = None
    unit: Optional[str] = None
    embedding: list[float] = field(default_factory=list)
    status: ClaimStatus = "detected"
    verdict: Optional[Verdict] = None
    confidence: Optional[float] = None
    source: Optional[Source] = None
    explanation: Optional[str] = None
    id: str = field(default_factory=_uuid)
    detected_at: float = field(default_factory=_now)
    resolved_at: Optional[float] = None
    time_to_verdict_ms: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class VerifiedFact:
    claim_key: str
    subject: str
    canonical_value: Any
    verdict: Verdict
    source: Source
    explanation: str
    unit: Optional[str] = None
    embedding: list[float] = field(default_factory=list)
    id: str = field(default_factory=_uuid)
    first_seen_ts: float = field(default_factory=_now)
    times_seen: int = 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("embedding", None)
        return d


@dataclass
class Contradiction:
    subject: str
    kind: ContradictionKind
    speaker_a_id: str
    speaker_b_id: str
    claim_a_id: str
    claim_b_id: str
    value_a: Any
    value_b: Any
    ts_a: float
    ts_b: float
    explanation: str
    id: str = field(default_factory=_uuid)
    detected_at: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionMetrics:
    checkworthy_seen: int = 0
    checked: int = 0
    memory_hits: int = 0
    memory_misses: int = 0
    contradictions: int = 0
    _ttv_total_ms: int = 0
    _ttv_count: int = 0

    @property
    def coverage(self) -> float:
        if self.checkworthy_seen == 0:
            return 0.0
        return self.checked / self.checkworthy_seen

    @property
    def mean_time_to_verdict_ms(self) -> float:
        if self._ttv_count == 0:
            return 0.0
        return self._ttv_total_ms / self._ttv_count

    def record_verdict(self, time_to_verdict_ms: int) -> None:
        self.checked += 1
        self._ttv_total_ms += time_to_verdict_ms
        self._ttv_count += 1

    def to_dict(self) -> dict:
        return {
            "checkworthy_seen": self.checkworthy_seen,
            "checked": self.checked,
            "coverage": self.coverage,
            "mean_time_to_verdict_ms": self.mean_time_to_verdict_ms,
            "memory_hits": self.memory_hits,
            "memory_misses": self.memory_misses,
            "contradictions": self.contradictions,
        }
