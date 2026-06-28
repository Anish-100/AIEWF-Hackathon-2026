"""Stub orchestrator — wires transcript → detect → verify → contradiction → push."""
import asyncio
import logging
import time
import uuid

log = logging.getLogger(__name__)

_session_id = str(uuid.uuid4())
_session_claims: list = []


async def process_sentence(sentence: str, clip_ts: float, push_fn) -> None:
    """Entry point: one finalized sentence from STT."""
    from adapters.gemini_flash import detect_claim
    from adapters.gemini_embed import embed
    from core.schemas import Claim
    from core import verifier, contradiction, metrics, research_queue

    result = await detect_claim(sentence)
    if not result or not result.get("is_checkworthy"):
        return

    metrics.record_checkworthy()

    claim = Claim(
        session_id=_session_id,
        clip_ts=clip_ts,
        raw_text=sentence,
        subject=result.get("subject", ""),
        predicate=result.get("predicate", ""),
        value=result.get("value"),
        unit=result.get("unit"),
    )
    claim.embedding = await embed(f"{claim.subject} {claim.predicate} {claim.value}")

    await push_fn({"type": "claim_detected", "claim": claim.model_dump()})

    claim, is_hit = await verifier.verify(claim)

    if is_hit:
        metrics.record_verdict(claim.time_to_verdict_ms or 0, claim.source or "memory")
        await push_fn({"type": "claim_update", "claim": claim.model_dump()})
    else:
        await research_queue.enqueue(claim)

    contradiction_event = await contradiction.check_contradiction(claim, _session_claims)
    if contradiction_event:
        metrics.record_contradiction()
        await push_fn({"type": "contradiction", "contradiction": contradiction_event.model_dump()})

    _session_claims.append(claim)
    await push_fn({"type": "metrics", "metrics": metrics.snapshot().model_dump()})


def reset_session() -> None:
    global _session_id, _session_claims
    _session_id = str(uuid.uuid4())
    _session_claims = []
    metrics.reset()
