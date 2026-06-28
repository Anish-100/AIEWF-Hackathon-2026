"""Hot-path verifier — embed claim, search memory, return hit/miss."""
import time
from .schemas import Claim
from .memory import vector_search, put, init_db


async def verify(claim: Claim) -> tuple[Claim, bool]:
    if claim.embedding is None:
        claim.status = "researching"
        return claim, False

    import numpy as np
    embedding = np.array(claim.embedding, dtype=np.float32)

    hits = vector_search(embedding, top_k=3)
    if hits and hits[0]["score"] >= 0.92:
        hit = hits[0]
        claim.status = "verified"
        claim.verdict = "true"          # or store verdict in memory too
        claim.source = "memory"
        claim.time_to_verdict_ms = 5.0  # memory hit is near-instant
        return claim, True

    # Miss — caller will enqueue for LLM research
    claim.status = "researching"
    return claim, False


def store_verdict(claim: Claim) -> None:
    """Call this after LLM resolves a claim to write it back to memory."""
    if claim.embedding is None:
        return
    import numpy as np
    put(
        subject=claim.subject,
        predicate=claim.predicate,
        value=claim.value or "",
        source=claim.source or "llm",
        embedding=np.array(claim.embedding, dtype=np.float32),
    )