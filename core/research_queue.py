"""Async research worker queue.

The verifier's MISS path enqueues a Claim here. A single background worker
drains the queue one item at a time (so we don't burst the Interactions API),
hands the claim to `adapters/antigravity.research`, and on success:

  1. Writes a new `VerifiedFact` into persistent memory so future identical
     claims become instant hits.
  2. Calls the per-claim `on_resolved` callback so the orchestrator can
     update the in-flight Claim and republish it (UI flips ⌛ → ✓/✗).

If `USE_ANTIGRAVITY=false` the queue is never started; enqueue() is a no-op.
This keeps the demo path free of any network research when we want to cut it.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Awaitable, Callable, Optional

import config
from adapters import antigravity, gemini_embed
from core import memory
from core.schemas import Claim, VerifiedFact

log = logging.getLogger(__name__)

# Callback the orchestrator registers per enqueued claim. It receives the
# now-resolved Claim object (or the original Claim if research failed and we
# couldn't tag a verdict).
OnResolved = Callable[[Claim, Optional[antigravity.ResearchResult]], Awaitable[None]]


_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None
# Per-session environment id so the agent's open browser tabs / scratch
# files can persist across related claims. Reset when the orchestrator
# starts a new session.
_environment_id: Optional[str] = None


def _claim_key(subject: str) -> str:
    return hashlib.sha1(subject.strip().lower().encode("utf-8")).hexdigest()[:16]


async def start() -> None:
    """Idempotent — call once at orchestrator startup."""
    global _queue, _worker_task, _environment_id
    if _worker_task is not None and not _worker_task.done():
        return
    _queue = asyncio.Queue()
    _environment_id = None
    _worker_task = asyncio.create_task(_worker(), name="research-worker")
    log.info("research_queue: worker started")


async def stop() -> None:
    global _worker_task
    if _worker_task is None:
        return
    _worker_task.cancel()
    try:
        await _worker_task
    except (asyncio.CancelledError, Exception):
        pass
    _worker_task = None
    log.info("research_queue: worker stopped")


def enqueue(claim: Claim, on_resolved: OnResolved) -> bool:
    """Returns True if the claim was queued (USE_ANTIGRAVITY=true and the
    queue is up), False otherwise."""
    if not config.USE_ANTIGRAVITY:
        return False
    if _queue is None:
        log.warning("research_queue: enqueue called before start; dropping")
        return False
    _queue.put_nowait((claim, on_resolved))
    log.info("research_queue: enqueued claim id=%s subject=%r (qsize=%d)",
             claim.id, claim.subject, _queue.qsize())
    return True


async def _worker() -> None:
    global _environment_id
    assert _queue is not None
    while True:
        claim, on_resolved = await _queue.get()
        try:
            await _resolve_one(claim, on_resolved)
        except Exception:
            log.exception("research_queue: unexpected error resolving claim %s", claim.id)
        finally:
            _queue.task_done()


async def _resolve_one(claim: Claim, on_resolved: OnResolved) -> None:
    """Single claim's full resolution: research → fact write → callback."""
    global _environment_id
    started_ns = time.perf_counter_ns()
    try:
        result = await antigravity.research(claim, environment_id=_environment_id)
    except Exception:
        log.exception("research_queue: antigravity.research failed for %s", claim.id)
        # Mark the claim back as 'researching' → orchestrator may leave it
        # there or downgrade to 'unverifiable'.
        try:
            await on_resolved(claim, None)
        except Exception:
            log.exception("research_queue: on_resolved (failure) callback raised")
        return

    # Persist the environment for the next claim in this session.
    if result.environment_id:
        _environment_id = result.environment_id

    # Translate the agent's verdict back into our claim/status vocabulary.
    if result.verdict == "true":
        claim.verdict = "true"
        claim.status = "verified"
    elif result.verdict == "false":
        claim.verdict = "false"
        claim.status = "flagged"
    elif result.verdict == "dubious":
        claim.verdict = "dubious"
        claim.status = "flagged"
    else:
        claim.verdict = "unverifiable"
        claim.status = "verified"
    claim.source = "web"
    claim.explanation = result.explanation
    claim.resolved_at = time.time()
    claim.time_to_verdict_ms = int((time.perf_counter_ns() - started_ns) / 1_000_000)

    # Write the resolved fact back so the SAME claim is an instant memory hit
    # next time around. This is the "continual learning" payoff — every web
    # miss permanently grows our local KB.
    if claim.subject and claim.embedding:
        try:
            fact = VerifiedFact(
                claim_key=_claim_key(claim.subject),
                subject=claim.subject,
                canonical_value=result.canonical_value,
                unit=result.unit,
                verdict=claim.verdict,
                source="web",
                explanation=result.explanation,
                embedding=claim.embedding,
                first_seen_ts=time.time(),
                times_seen=1,
            )
            memory.get_memory().put(
                fact,
                source_session_id=claim.session_id,
                source_speaker=claim.speaker_id,
                extracted_at=time.time(),
                supporting_quote=claim.raw_text,
            )
            log.info(
                "research_queue: wrote fact for subject=%r → memory size now %d",
                claim.subject, memory.get_memory().size(),
            )
        except Exception:
            log.exception("research_queue: memory.put failed for %s", claim.id)

    try:
        await on_resolved(claim, result)
    except Exception:
        log.exception("research_queue: on_resolved callback raised")
