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
from collections import deque
from typing import Optional

import config
from adapters import gemini_flash, gemini_live, livekit_audio
from core import contradiction, end_of_session, event_bus, memory, metrics, research_queue, verifier
from core.schemas import Claim

# How long the room can be empty before we declare the conversation over and
# kick off the end-of-session distiller.
_AUTOEND_AFTER_SECONDS = 30.0
# How many prior finalized utterances per speaker we send to Flash as context
# alongside the current sentence. Lets the detector resolve pronouns ("it",
# "their") and relative dates ("last month") to canonical subjects. Small
# number — token cost is per-detect-call.
_HISTORY_PER_SPEAKER = 2

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
        self._active_speakers: set[str] = set()
        # Per-speaker rolling history of recent finalized utterances. Used as
        # prior context for Flash claim detection so pronouns/relative dates
        # in the current sentence can be resolved against what was said just
        # before. Bounded to _HISTORY_PER_SPEAKER entries per speaker.
        self._speaker_history: dict[str, deque[str]] = {}
        self._autoend_task: Optional[asyncio.Task] = None
        self._distilled: bool = False
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
            self._cancel_autoend()
            if config.USE_ANTIGRAVITY:
                await research_queue.stop()
            await self._close_all_sessions()
            # Make sure we distill at least once per run, even if the auto-end
            # timer didn't fire (server shut down before the empty-room window
            # elapsed). Safe: _distill_now is idempotent via self._distilled.
            await self._distill_now(reason="orchestrator-stopped")
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
        # Active-speakers bookkeeping for auto-end. We track "joined" / "left"
        # only; muted/unmuted leaves the participant in the room.
        if kind == "joined":
            self._active_speakers.add(speaker_id)
            self._cancel_autoend()  # someone showed up → don't end yet
        elif kind == "left":
            self._active_speakers.discard(speaker_id)
            if not self._active_speakers:
                self._schedule_autoend()
        # Close the session ONLY on `left` (participant gone for good). On
        # `muted` we deliberately leave the LiveSession open and idle —
        # closing on every mute caused multi-speaker transcription to drop
        # mid-utterance when one speaker toggled their mic. If a long mute
        # actually wedges the Gemini stream, `_stall_watchdog` will mark the
        # session dead after `_STALL_AFTER_SECONDS` and the next audio frame
        # reopens a fresh session via `_get_or_open_session`. `unmuted` /
        # `joined` are no-ops here for the same reason — audio resumes,
        # session continues.
        if kind == "left":
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

    # --- session lifecycle: auto-end + distill -----------------------------

    def _schedule_autoend(self) -> None:
        """Start (or restart) the autoend countdown. If nobody rejoins within
        `_AUTOEND_AFTER_SECONDS`, mark the session ended and kick off the
        end-of-session distiller in the background."""
        self._cancel_autoend()
        self._autoend_task = asyncio.create_task(self._autoend_after_delay(), name="autoend")
        log.info("orchestrator: room empty — scheduled auto-end in %ds", int(_AUTOEND_AFTER_SECONDS))

    def _cancel_autoend(self) -> None:
        t = self._autoend_task
        if t and not t.done():
            t.cancel()
        self._autoend_task = None

    async def _autoend_after_delay(self) -> None:
        try:
            await asyncio.sleep(_AUTOEND_AFTER_SECONDS)
        except asyncio.CancelledError:
            return
        if self._active_speakers:
            return  # someone rejoined just in time
        log.info("orchestrator: room still empty after %ds — distilling", int(_AUTOEND_AFTER_SECONDS))
        await self._distill_now(reason="auto-end")

    async def _distill_now(self, *, reason: str) -> None:
        if self._distilled:
            return
        self._distilled = True
        event_bus.publish({"type": "session_ended", "session_id": self.session_id, "reason": reason})
        try:
            n = await end_of_session.distill_session(self.session_id)
            log.info("orchestrator: distill wrote %d facts", n)
            event_bus.publish({"type": "distilled", "session_id": self.session_id, "new_facts": n})
            try:
                metrics.publish_memory_size(memory.get_memory().size())
            except Exception:
                pass
        except Exception:
            log.exception("orchestrator: distill failed")

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
        source_text = (segment.get("source_text") or text).strip()
        if not text:
            return
        if not segment.get("is_final"):
            event_bus.publish({
                "type": "partial",
                "speaker_id": segment["speaker_id"],
                "text": text,
                "source_text": source_text,
                "ts": self._now_ts(),
            })
            return

        # Process check-worthiness + verification off the recv path so we don't
        # block the next incoming transcription chunk.
        asyncio.create_task(self._process_finalized(segment, text, source_text))

    async def _process_finalized(self, segment: dict, text: str, source_text: str) -> None:
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
            "source_text": source_text,
            "clip_ts": clip_ts,
        })

        # Snapshot prior context BEFORE appending the current sentence so we
        # don't feed the sentence to itself as context.
        prior_context = list(self._speaker_history.get(speaker_id, ()))
        history = self._speaker_history.setdefault(
            speaker_id, deque(maxlen=_HISTORY_PER_SPEAKER),
        )
        history.append(text)

        try:
            detections = await gemini_flash.detect(text, prior_context=prior_context)
        except Exception:
            log.exception("Flash.detect crashed for text=%r", text)
            detections = []

        if not detections:
            log.info("orchestrator: dropped non-claim from %s: %r", speaker_id, text)
            return

        log.info(
            "orchestrator: Flash found %d claim(s) in %r from %s",
            len(detections), text, speaker_id,
        )
        for detection in detections:
            await self._process_one_claim(detection, speaker_id=speaker_id, clip_ts=clip_ts, raw_text=text)

    async def _process_one_claim(self, detection: dict, *, speaker_id: str, clip_ts: float, raw_text: str) -> None:
        # Flash said it's a fact-checkable claim → it counts toward coverage.
        metrics.note_checkworthy()
        metrics.publish()

        claim = Claim(
            session_id=self.session_id,
            speaker_id=speaker_id,
            clip_ts=clip_ts,
            raw_text=raw_text,
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
            speaker_id, raw_text, claim.status, claim.verdict, claim.source,
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
