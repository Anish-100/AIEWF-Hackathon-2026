"""Persistent verified-fact store (stub). Phase 3 implements SQLite + sqlite-vec
with a numpy fallback; survives process restart."""
from __future__ import annotations

from typing import Iterable

from core.schemas import VerifiedFact


class Memory:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def put(self, fact: VerifiedFact) -> None:
        raise NotImplementedError("Phase 3")

    def get(self, fact_id: str) -> VerifiedFact | None:
        raise NotImplementedError("Phase 3")

    def vector_search(self, embedding: list[float], top_k: int = 5) -> list[tuple[VerifiedFact, float]]:
        raise NotImplementedError("Phase 3")

    def facts_by_subject(self, subject_embedding: list[float], threshold: float) -> Iterable[VerifiedFact]:
        raise NotImplementedError("Phase 3")

    def touch(self, fact_id: str) -> None:
        raise NotImplementedError("Phase 3")

    def size(self) -> int:
        raise NotImplementedError("Phase 3")

    def close(self) -> None:
        return None
