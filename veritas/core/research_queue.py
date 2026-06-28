"""Stub: async queue that resolves MISS claims via Interactions API."""
import asyncio
from .schemas import Claim

_queue: asyncio.Queue = asyncio.Queue()


async def enqueue(claim: Claim) -> None:
    await _queue.put(claim)


async def worker(push_fn) -> None:
    """Background worker — resolves misses and calls push_fn(event_dict)."""
    from adapters.gemini_interactions import research_claim
    from core import memory
    from core.schemas import VerifiedFact
    import time

    while True:
        claim: Claim = await _queue.get()
        try:
            result = await research_claim(claim.raw_text)
            fact = VerifiedFact(
                claim_key=f"{claim.subject}:{claim.predicate}",
                subject=claim.subject,
                canonical_value=str(result.get("canonical_value") or claim.value or ""),
                unit=claim.unit,
                verdict=result["verdict"],
                source=result["source"],
                explanation=result["explanation"],
                embedding=claim.embedding,
            )
            memory.put(fact)
            claim.verdict = result["verdict"]
            claim.source = result["source"]
            claim.explanation = result["explanation"]
            claim.status = "verified"
            claim.resolved_at = time.time()
            if claim.detected_at:
                claim.time_to_verdict_ms = int((claim.resolved_at - claim.detected_at) * 1000)
            await push_fn({"type": "claim_update", "claim": claim.model_dump()})
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Research worker error: %s", exc)
        finally:
            _queue.task_done()
