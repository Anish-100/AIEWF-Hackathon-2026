"""Gemini Live transcription adapter.

One `LiveSession` per LiveKit participant. Audio in via `feed_audio()`;
finalized sentences are pushed out via the `on_segment` async callback.

Implementation notes
--------------------
- We use the `gemini-3.5-live-translate-preview` model purely for its
  streaming `input_transcription` field. The audio response is ignored.
- Sentence finalization: we accumulate `input_transcription.text` chunks
  into a buffer and emit on either (a) a `turn_complete` signal from the
  server, or (b) sentence-ending punctuation followed by whitespace inside
  the buffer.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from google import genai
from google.genai import types

import config

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(.+?[\.\!\?])(\s+|$)", re.DOTALL)

OnSegment = Callable[[dict], Awaitable[None]]
# segment: {"speaker_id": str, "text": str, "is_final": bool, "ts": float}


@dataclass
class _PendingAudio:
    data: bytes


class LiveSession:
    def __init__(
        self,
        speaker_id: str,
        on_segment: OnSegment,
        *,
        model: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY must be set")
        self.speaker_id = speaker_id
        self.on_segment = on_segment
        self.model = model or config.GEMINI_LIVE_MODEL
        self.sample_rate = sample_rate
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._send_queue: asyncio.Queue[Optional[_PendingAudio]] = asyncio.Queue(maxsize=1024)
        self._buffer: str = ""
        self._started_at: float = 0.0
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._started_at = time.time()
        live_cfg = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )
        log.info("LiveSession[%s] connecting model=%s", self.speaker_id, self.model)
        # `async with` would close the session at function exit; we want the session
        # to live for as long as the speaker is in the room, so manage lifecycle ourselves.
        self._session_cm = self._client.aio.live.connect(model=self.model, config=live_cfg)
        self._session = await self._session_cm.__aenter__()
        log.info("LiveSession[%s] connected", self.speaker_id)
        self._tasks.append(asyncio.create_task(self._send_loop(), name=f"live-send-{self.speaker_id}"))
        self._tasks.append(asyncio.create_task(self._recv_loop(), name=f"live-recv-{self.speaker_id}"))

    async def feed_audio(self, pcm: bytes) -> None:
        if self._stop.is_set():
            return
        try:
            self._send_queue.put_nowait(_PendingAudio(pcm))
        except asyncio.QueueFull:
            log.warning("LiveSession[%s] audio queue full; dropping chunk", self.speaker_id)

    async def aclose(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        # Sentinel to unblock send loop.
        try:
            self._send_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self._session_cm.__aexit__(None, None, None)
        except Exception:
            log.exception("LiveSession[%s] error during close", self.speaker_id)
        # Flush any half-built buffer as a final segment.
        if self._buffer.strip():
            await self._emit(self._buffer.strip(), is_final=True)
            self._buffer = ""
        log.info("LiveSession[%s] closed", self.speaker_id)

    # --- internal ----

    async def _send_loop(self) -> None:
        mime = f"audio/pcm;rate={self.sample_rate}"
        while not self._stop.is_set():
            item = await self._send_queue.get()
            if item is None:
                return
            try:
                await self._session.send_realtime_input(
                    audio=types.Blob(data=item.data, mime_type=mime)
                )
            except Exception:
                log.exception("LiveSession[%s] send_realtime_input failed", self.speaker_id)
                return

    async def _recv_loop(self) -> None:
        try:
            async for response in self._session.receive():
                sc = getattr(response, "server_content", None)
                if sc is None:
                    continue
                in_tr = getattr(sc, "input_transcription", None)
                if in_tr and getattr(in_tr, "text", None):
                    await self._ingest_text(in_tr.text)
                if getattr(sc, "turn_complete", False):
                    if self._buffer.strip():
                        await self._emit(self._buffer.strip(), is_final=True)
                        self._buffer = ""
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("LiveSession[%s] recv loop crashed", self.speaker_id)

    async def _ingest_text(self, chunk: str) -> None:
        self._buffer += chunk
        # Emit any complete sentences in the buffer.
        while True:
            m = _SENTENCE_SPLIT.match(self._buffer)
            if not m:
                break
            sentence = m.group(1).strip()
            self._buffer = self._buffer[m.end():]
            if sentence:
                await self._emit(sentence, is_final=True)
        # Push partial (non-final) so the UI shows live typing too.
        if self._buffer.strip():
            await self._emit(self._buffer.strip(), is_final=False)

    async def _emit(self, text: str, *, is_final: bool) -> None:
        try:
            await self.on_segment({
                "speaker_id": self.speaker_id,
                "text": text,
                "is_final": is_final,
                "ts": time.time() - self._started_at,
            })
        except Exception:
            log.exception("LiveSession[%s] on_segment callback raised", self.speaker_id)
