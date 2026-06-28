"""Gemini Live transcription adapter (stub).

Phase 1 implementation: one `LiveSession` per speaker. `feed_audio(pcm)` in,
`async for segment in session.segments()` yields `{text, is_final, ts}`.
"""
from __future__ import annotations

from typing import AsyncIterator


class LiveSession:
    def __init__(self, speaker_id: str) -> None:
        self.speaker_id = speaker_id

    async def feed_audio(self, pcm: bytes) -> None:
        raise NotImplementedError("Phase 1: implement against Gemini Live API.")

    async def segments(self) -> AsyncIterator[dict]:
        raise NotImplementedError("Phase 1: implement against Gemini Live API.")
        yield  # pragma: no cover - keeps async generator typing happy

    async def aclose(self) -> None:
        return None
