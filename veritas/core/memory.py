"""Stub: persistent fact store. Real impl uses SQLite + sqlite-vec."""
from .schemas import VerifiedFact


_store: dict[str, VerifiedFact] = {}


def put(fact: VerifiedFact) -> None:
    _store[fact.id] = fact


def get(fact_id: str) -> VerifiedFact | None:
    return _store.get(fact_id)


def vector_search(embedding: list[float], top_k: int = 5) -> list[tuple[VerifiedFact, float]]:
    return []


def facts_by_subject(subject: str) -> list[VerifiedFact]:
    return []


def touch(fact: VerifiedFact) -> None:
    fact.times_seen += 1


def clear() -> None:
    _store.clear()
