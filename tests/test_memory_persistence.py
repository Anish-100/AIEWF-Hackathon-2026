"""Persistence test — the moat.

Write N VerifiedFacts. Close the store. Reopen from the same on-disk path in
a brand-new Memory instance. Assert every fact is recoverable AND vector
search still returns them. If this fails, the continual-learning claim is fake.
"""
from __future__ import annotations

import numpy as np

from core.memory import Memory
from core.schemas import VerifiedFact


def _make_facts(n: int, dim: int = 64, seed: int = 0) -> list[VerifiedFact]:
    rng = np.random.default_rng(seed)
    facts: list[VerifiedFact] = []
    for i in range(n):
        emb = rng.standard_normal(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        facts.append(
            VerifiedFact(
                claim_key=f"k{i:03d}",
                subject=f"subject_{i}",
                canonical_value=float(i),
                unit="%",
                verdict="true",
                source="kb",
                explanation=f"explanation {i}",
                embedding=emb.tolist(),
            )
        )
    return facts


def test_memory_survives_restart(tmp_path):
    db_path = str(tmp_path / "veritas_test.db")
    facts = _make_facts(20)

    m1 = Memory(db_path)
    for f in facts:
        m1.put(f)
    assert m1.size() == 20
    m1.close()

    # Brand-new instance — same on-disk path.
    m2 = Memory(db_path)
    assert m2.size() == 20, "facts disappeared after restart"

    # Every id round-trips.
    for f in facts:
        got = m2.get(f.id)
        assert got is not None, f"missing {f.id}"
        assert got.subject == f.subject
        assert got.canonical_value == f.canonical_value
        assert got.unit == f.unit
        assert got.verdict == f.verdict

    # Vector search still works — query with the first fact's own embedding,
    # it must be the top hit.
    top = m2.vector_search(facts[0].embedding, top_k=3)
    assert top, "vector_search returned nothing after restart"
    assert top[0][0].id == facts[0].id, f"top hit was {top[0][0].id}, not {facts[0].id}"
    assert top[0][1] > 0.99, f"self-similarity too low: {top[0][1]:.3f}"

    m2.close()


def test_sessions_survive_restart(tmp_path):
    db_path = str(tmp_path / "veritas_sess.db")
    m1 = Memory(db_path)
    m1.start_session("s1", topic="t", n_speakers=2)
    m1.log_utterance("s1", "alice", "hello world", 0.1)
    m1.log_utterance("s1", "bob", "goodbye world", 0.2)
    m1.end_session("s1")
    m1.close()

    m2 = Memory(db_path)
    sessions = m2.list_sessions()
    ids = [s["id"] for s in sessions]
    assert "s1" in ids
    transcript = m2.get_session_transcript("s1")
    assert len(transcript) == 2
    assert transcript[0]["speaker_id"] == "alice"
    assert transcript[1]["speaker_id"] == "bob"
    m2.close()
