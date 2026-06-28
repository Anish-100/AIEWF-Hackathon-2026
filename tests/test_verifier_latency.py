"""Hot-path latency budget.

The verifier's network hop is the embedding call (Gemini), which we cache.
Once a subject is embedded, the actual verifier work is `memory.vector_search`
plus a value comparison — both pure-CPU. We assert p50 of `vector_search`
stays under 100 ms over 100 queries against a seeded memory.

We use random in-process embeddings here so the test runs offline.
"""
from __future__ import annotations

import statistics
import time

import numpy as np

from core.memory import Memory
from core.schemas import VerifiedFact


def _seed(mem: Memory, n: int, dim: int = 64) -> list[list[float]]:
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    for i, emb in enumerate(embs):
        mem.put(
            VerifiedFact(
                claim_key=f"k{i:04d}",
                subject=f"subject_{i}",
                canonical_value=float(i),
                unit="%",
                verdict="true",
                source="kb",
                explanation="seed",
                embedding=emb.tolist(),
            )
        )
    return [e.tolist() for e in embs]


def test_vector_search_p50_under_100ms(tmp_path):
    mem = Memory(str(tmp_path / "veritas_latency.db"))
    seeded = _seed(mem, n=500)

    # 100 query embeddings chosen at random from the seeded set (worst case for
    # cosine — we always have a near-1.0 best match somewhere).
    rng = np.random.default_rng(7)
    query_idxs = rng.integers(0, len(seeded), size=100)

    latencies_ms: list[float] = []
    for i in query_idxs:
        t0 = time.perf_counter_ns()
        results = mem.vector_search(seeded[int(i)], top_k=5)
        dt = (time.perf_counter_ns() - t0) / 1_000_000
        latencies_ms.append(dt)
        assert results, "search returned no candidates"
        # The same-vector query should self-match as top hit.
        assert results[0][1] > 0.99, f"top score {results[0][1]:.3f} suspiciously low"

    p50 = statistics.median(latencies_ms)
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    print(f"\nvector_search: p50={p50:.2f}ms  p95={p95:.2f}ms  n={len(latencies_ms)}")
    assert p50 < 100.0, f"p50 latency {p50:.2f}ms exceeds 100ms budget"

    mem.close()
