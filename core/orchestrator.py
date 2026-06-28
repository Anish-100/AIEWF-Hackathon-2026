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
import uuid
from typing import Optional

import config
from adapters import gemini_live, livekit_audio
from core import event_bus

log = logging.getLogger(__name__)

SESSION_ID = "live"


class Orchestrator:
    def __init__(self) -> None:
        self._sessions: dict[str, gemini_live.LiveSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._stop = asyncio.Event()

    async def run(self) -> None:
        log.info("orchestrator: starting")
        try:
            await livekit_audio.run_room_subscriber(
                on_audio_frame=self._handle_audio,
                on_speaker_event=self._handle_speaker_event,
                stop=self._stop,
            )
        finally:
            await self._close_all_sessions()
            log.info("orchestrator: stopped")

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

    async def _handle_segment(self, segment: dict) -> None:
        text = (segment.get("text") or "").strip()
        if not text:
            return
        if segment.get("is_final"):
            event_bus.publish({
                "type": "claim",
                "claim": {
                    "id": uuid.uuid4().hex,
                    "session_id": SESSION_ID,
                    "speaker_id": segment["speaker_id"],
                    "clip_ts": round(segment.get("ts", 0.0), 2),
                    "raw_text": text,
                    "subject": "",
                    "predicate": "",
                    "value": None,
                    "unit": None,
                    "status": "detected",
                    "verdict": None,
                    "source": None,
                },
            })
        else:
            event_bus.publish({
                "type": "partial",
                "speaker_id": segment["speaker_id"],
                "text": text,
                "ts": round(segment.get("ts", 0.0), 2),
            })


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
