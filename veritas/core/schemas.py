import time
import uuid
from typing import Optional
from pydantic import BaseModel, Field


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    clip_ts: float
    raw_text: str
    subject: str = ""
    predicate: str = ""
    value: Optional[str] = None
    unit: Optional[str] = None
    embedding: list[float] = Field(default_factory=list)
    status: str = "detected"  # detected | researching | verified | flagged
    verdict: Optional[str] = None  # true | false | unverifiable | dubious
    confidence: Optional[float] = None
    source: Optional[str] = None  # kb | memory | web
    explanation: Optional[str] = None
    detected_at: float = Field(default_factory=time.time)
    resolved_at: Optional[float] = None
    time_to_verdict_ms: Optional[int] = None


class VerifiedFact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_key: str
    subject: str
    canonical_value: str
    unit: Optional[str] = None
    verdict: str
    source: str  # kb | web
    explanation: str
    embedding: list[float] = Field(default_factory=list)
    first_seen_ts: float = Field(default_factory=time.time)
    times_seen: int = 1


class Contradiction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str
    claim_a_id: str
    value_a: str
    ts_a: float
    claim_b_id: str
    value_b: str
    ts_b: float
    explanation: str


class SessionMetrics(BaseModel):
    checkworthy_seen: int = 0
    checked: int = 0
    coverage: float = 0.0
    mean_time_to_verdict_ms: float = 0.0
    memory_hits: int = 0
    memory_misses: int = 0
    contradictions: int = 0
