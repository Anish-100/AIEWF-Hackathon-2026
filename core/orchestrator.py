"""Pipeline orchestrator (Phase 1).

Wires LiveKit audio frames → per-speaker Gemini Live transcription →
publishes finalized transcripts onto the in-process event bus as `claim`
events (so the existing UI renders them as cards). Phases 2-4 will upgrade
the pipeline (detect → verify → contradiction → metrics) without changing
this entrypoint.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

import config
from adapters import gemini_flash, gemini_live, livekit_audio
from core import contradiction, event_bus, memory, metrics, research_queue, verifier
from core.schemas import Claim

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self) -> None:
        self._sessions: dict[str, gemini_live.LiveSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        # Each orchestrator run is one session. The id is durable so the
        # end-of-session distiller can replay this conversation's transcript.
        self.session_id: str = "veritas-" + uuid.uuid4().hex[:12]
        # Wall-clock anchor for clip_ts. Set when the orchestrator actually
        # starts (in run()), then used by every claim/transcript event so that
        # timestamps are *session-relative* — they do NOT reset when a
        # LiveSession reconnects after a mute or stall.
        self.session_started_at: float = 0.0
        self._speakers_seen: set[str] = set()
        self._contradiction = contradiction.ContradictionChecker()

    async def run(self) -> None:
        self.session_started_at = time.time()
        log.info("orchestrator: starting session_id=%s", self.session_id)
        try:
            memory.get_memory().start_session(self.session_id, topic="", n_speakers=0)
        except Exception:
            log.exception("orchestrator: failed to record session start (continuing)")
        # Reset metrics for this run and push a baseline (0 across the board)
        # plus the current memory size so the UI shows the cold→warm starting
        # point even before any speaker says anything.
        metrics.reset()
        metrics.publish()
        try:
            metrics.publish_memory_size(memory.get_memory().size())
        except Exception:
            log.exception("orchestrator: failed to publish initial memory size")
        # Start the async research worker (no-op unless USE_ANTIGRAVITY=true).
        if config.USE_ANTIGRAVITY:
            await research_queue.start()
        try:
            await livekit_audio.run_room_subscriber(
                on_audio_frame=self._handle_audio,
                on_speaker_event=self._handle_speaker_event,
                stop=self._stop,
            )
        finally:
            if config.USE_ANTIGRAVITY:
                await research_queue.stop()
            await self._close_all_sessions()
            try:
                memory.get_memory().end_session(self.session_id)
            except Exception:
                log.exception("orchestrator: failed to record session end")
            log.info("orchestrator: stopped session_id=%s", self.session_id)

    async def stop(self) -> None:
        self._stop.set()

    # --- LiveKit -> Gemini ----

    async def _handle_audio(self, speaker_id: str, pcm: bytes, sample_rate: int, _channels: int) -> None:
        sess = await self._get_or_open_session(speaker_id, sample_rate)
        await sess.feed_audio(pcm)

    async def _handle_speaker_event(self, kind: str, speaker_id: str) -> None:
        event_bus.publish({
            "type": "speaker",
            "speaker_id": speaker_id,
            "kind": kind,
        })
        # Close the session on `left` (participant gone) or `muted` (audio paused
        # → Gemini Live will stall anyway). The next audio frame after rejoin /
        # unmute opens a fresh session via _get_or_open_session.
        if kind in ("left", "muted"):
            await self._close_session(speaker_id)
            log.info("orchestrator: closed session for %s due to %s", speaker_id, kind)

    async def _get_or_open_session(self, speaker_id: str, sample_rate: int) -> gemini_live.LiveSession:
        async with self._sessions_lock:
            sess = self._sessions.get(speaker_id)
            if sess is not None and sess.dead.is_set():
                log.warning("orchestrator: existing LiveSession for %s is dead — replacing", speaker_id)
                # Drop ref first so a concurrent caller doesn't reuse the dead one;
                # do the close outside the lock to avoid deadlock with feed_audio.
                self._sessions.pop(speaker_id, None)
                _dead_sess = sess
                sess = None
                asyncio.create_task(_dead_sess.aclose())
            if sess is None:
                sess = gemini_live.LiveSession(
                    speaker_id=speaker_id,
                    on_segment=self._handle_segment,
                    sample_rate=sample_rate,
                )
                await sess.start()
                self._sessions[speaker_id] = sess
                log.info("orchestrator: opened LiveSession for %s", speaker_id)
            return sess

    async def _close_session(self, speaker_id: str) -> None:
        async with self._sessions_lock:
            sess = self._sessions.pop(speaker_id, None)
        if sess:
            await sess.aclose()

    async def _close_all_sessions(self) -> None:
        async with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            try:
                await s.aclose()
            except Exception:
                log.exception("error closing session")

    # --- Gemini -> event bus ----

    def _now_ts(self) -> float:
        """Session-relative wall clock. Stable across LiveSession reconnects."""
        if self.session_started_at == 0.0:
            return 0.0
        return round(time.time() - self.session_started_at, 2)

    async def _handle_segment(self, segment: dict) -> None:
        text = (segment.get("text") or "").strip()
        if not text:
            return
        if not segment.get("is_final"):
            event_bus.publish({
                "type": "partial",
                "speaker_id": segment["speaker_id"],
                "text": text,
                "ts": self._now_ts(),
            })
            return

        # Process check-worthiness + verification off the recv path so we don't
        # block the next incoming transcription chunk.
        asyncio.create_task(self._process_finalized(segment, text))

    async def _process_finalized(self, segment: dict, text: str) -> None:
        speaker_id = segment["speaker_id"]
        clip_ts = self._now_ts()

        # Log every finalized utterance to the durable transcript regardless of
        # claim-worthiness — the end-of-session distiller replays this later
        # and may find facts the per-utterance Flash call missed.
        try:
            memory.get_memory().log_utterance(self.session_id, speaker_id, text, clip_ts)
        except Exception:
            log.exception("orchestrator: log_utterance failed")
        if speaker_id not in self._speakers_seen:
            self._speakers_seen.add(speaker_id)

        # Always surface the finalized transcript to the UI so the user can see
        # that their voice was heard, even when Flash drops it as non-claim.
        event_bus.publish({
            "type": "transcript",
            "speaker_id": speaker_id,
            "text": text,
            "clip_ts": clip_ts,
        })

        try:
            detection = await gemini_flash.detect(text)
        except Exception:
            log.exception("Flash.detect crashed for text=%r", text)
            detection = {"is_checkworthy": False}

        if not detection.get("is_checkworthy"):
            log.info("orchestrator: dropped non-claim from %s: %r", speaker_id, text)
            return

        # Flash said it's a fact-checkable claim → it counts toward coverage.
        metrics.note_checkworthy()
        metrics.publish()

        claim = Claim(
            session_id=self.session_id,
            speaker_id=speaker_id,
            clip_ts=clip_ts,
            raw_text=text,
            subject=str(detection.get("subject") or "").strip(),
            predicate=str(detection.get("predicate") or "").strip(),
            value=detection.get("value") or None,
            unit=str(detection.get("unit") or "").strip() or None,
            status="detected",
        )

        # Push the freshly-detected claim immediately so the UI shows a
        # "researching..." card while verification runs.
        event_bus.publish({"type": "claim", "claim": self._claim_dict(claim)})

        try:
            claim, hit = await verifier.verify(claim)
        except Exception:
            log.exception("verifier crashed for claim=%r", claim.id)
            return

        # Push the verified update (same id → UI replaces the in-place card).
        event_bus.publish({"type": "claim", "claim": self._claim_dict(claim)})
        log.info(
            "orchestrator: claim from %s text=%r → status=%s verdict=%s source=%s",
            speaker_id, text, claim.status, claim.verdict, claim.source,
        )

        # Verifier finished → record hit/miss and time-to-verdict.
        metrics.note_verdict(hit=hit, time_to_verdict_ms=claim.time_to_verdict_ms)
        metrics.publish()
        try:
            metrics.publish_memory_size(memory.get_memory().size())
        except Exception:
            pass

        # MISS path: enqueue async research. The card sits as ⌛ RESEARCHING
        # in the UI until research_queue resolves it via _on_research_resolved.
        if not hit:
            research_queue.enqueue(claim, self._on_research_resolved)

        # Intra-session contradiction scan (same-speaker / cross-speaker).
        try:
            conflict = self._contradiction.record_and_check(claim)
        except Exception:
            log.exception("contradiction check crashed for claim=%r", claim.id)
            conflict = None
        if conflict is not None:
            metrics.note_contradiction()
            metrics.publish()
            event_bus.publish({"type": "contradiction", "contradiction": {
                "id": conflict.id,
                "subject": conflict.subject,
                "kind": conflict.kind,
                "speaker_a_id": conflict.speaker_a_id,
                "speaker_b_id": conflict.speaker_b_id,
                "claim_a_id": conflict.claim_a_id,
                "claim_b_id": conflict.claim_b_id,
                "value_a": conflict.value_a,
                "value_b": conflict.value_b,
                "ts_a": conflict.ts_a,
                "ts_b": conflict.ts_b,
                "explanation": conflict.explanation,
            }})

    async def _on_research_resolved(self, claim: Claim, result) -> None:
        """Called by research_queue when Antigravity finishes a claim.
        Republishes the claim (UI flips ⌛ → ✓/✗), updates metrics,
        and refreshes the on-screen memory size."""
        event_bus.publish({"type": "claim", "claim": self._claim_dict(claim)})
        log.info(
            "orchestrator: research resolved for %s → status=%s verdict=%s source=%s",
            claim.id, claim.status, claim.verdict, claim.source,
        )
        # If the research succeeded, count it as a verdict for the metrics.
        if result is not None:
            metrics.note_verdict(hit=True, time_to_verdict_ms=claim.time_to_verdict_ms)
            metrics.publish()
        try:
            metrics.publish_memory_size(memory.get_memory().size())
        except Exception:
            pass

    @staticmethod
    def _claim_dict(claim: Claim) -> dict:
        return {
            "id": claim.id,
            "session_id": claim.session_id,
            "speaker_id": claim.speaker_id,
            "clip_ts": claim.clip_ts,
            "raw_text": claim.raw_text,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "value": claim.value,
            "unit": claim.unit,
            "status": claim.status,
            "verdict": claim.verdict,
            "source": claim.source,
            "confidence": claim.confidence,
            "explanation": claim.explanation,
            "time_to_verdict_ms": claim.time_to_verdict_ms,
        }


_singleton: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    global _singleton
    if _singleton is None:
        _singleton = Orchestrator()
    return _singleton


async def run() -> None:
    await get_orchestrator().run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    asyncio.run(run())
