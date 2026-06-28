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
# Sentence buffer flush: how long after the last transcript chunk before we
# emit the buffer as a finalized sentence. 1.0s — anything over a 1-second
# pause is treated as a sentence boundary. Snappier feel but a mid-sentence
# pause longer than 1s WILL split the thought into two cards; raise toward
# 1.5s if that becomes an issue on stage.
_SILENCE_FLUSH_SECONDS = 1.0
# Session-level stall: send active but server has gone fully silent for this
# long → declare dead and reconnect. Generous (15s) so a quiet pause between
# speakers doesn't churn the session.
_STALL_AFTER_SECONDS = 15.0
# Session-level monologue: send active, server IS replying, but every reply is
# `model_turn` (model speaking back) with no transcription text or turn
# boundary for this long → mark dead and reconnect. The hard stall watchdog
# can't catch this because recv IS happening. Only fires while audio is
# actively being sent, so muted/idle sessions are never killed. Generous (20s)
# so a brief drift back into transcription self-recovers without churn.
_MONOLOGUE_AFTER_SECONDS = 20.0
# Translate-mode fallback: how long we wait for an English translation
# (`output_transcription`) before giving up and showing the source-language
# text on its own. Without this, source text arriving with no matching
# translation sits in `_source_buffer` and never gets emitted to the UI.
# Slightly longer than _SILENCE_FLUSH_SECONDS so the normal English flush
# wins when translation is in-flight; only fires when the translation
# genuinely never comes.
_SOURCE_FALLBACK_SECONDS = 2.0

OnSegment = Callable[[dict], Awaitable[None]]
# segment: {
#   "speaker_id": str,
#   "text": str,           # target language (English by default) — used downstream
#   "source_text": str,    # raw input transcription (source language); same as text when no translation
#   "is_final": bool,
#   "ts": float,
# }


_LIVE_TRANSLATE_MODEL = "gemini-3.5-live-translate-preview"


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
        # _buffer is the *target* (English) transcription — what drives sentence
        # boundary detection and what downstream Flash/verifier consume.
        # _source_buffer is the parallel source-language transcription, shown
        # in the UI alongside. When translation is off, they're identical.
        self._buffer: str = ""
        self._source_buffer: str = ""
        self._translate = config.USE_LIVE_TRANSLATE
        # If translation is on, force the translate model regardless of env.
        if self._translate:
            self.model = _LIVE_TRANSLATE_MODEL
        self._last_text_at: float = 0.0
        # When the LAST source-language transcription chunk arrived. Used by
        # the silence-flush loop to emit source-only when translation is
        # missing (see _SOURCE_FALLBACK_SECONDS).
        self._last_source_text_at: float = 0.0
        self._last_send_at: float = 0.0
        self._last_recv_at: float = 0.0
        # Distinct from _last_recv_at: only ticks when the server sends actual
        # transcription text (input or output) or a turn boundary. Pure
        # `model_turn` traffic does NOT update this — it's the signal that
        # drives the monologue watchdog.
        self._last_useful_recv_at: float = 0.0
        self._started_at: float = 0.0
        self._stop = asyncio.Event()
        # Set when the session is no longer usable (stall, server-side close, or
        # crash). Orchestrator polls this in _get_or_open_session and replaces
        # dead sessions with fresh ones on the next audio frame.
        self.dead = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._started_at = time.time()
        # Why this config (do not strip the system_instruction again):
        #
        # The Live API is fundamentally turn-taking. With response_modalities=AUDIO
        # and no instruction, the model transcribes you THEN replies in audio.
        # Every reply closes the bidi stream (turn_complete → server closes), so
        # we'd reconnect on every utterance and the user sees the model "talk back"
        # in our recv log. system_instruction tells the model to stay silent — it
        # mostly complies, the rare model_turn we ignore.
        #
        # We do NOT set turn_coverage=TURN_INCLUDES_ALL_INPUT — that made Gemini
        # buffer transcription forever (`input_transcription` is only flushed at
        # turn boundaries; if all input is one perpetual turn, nothing flushes).
        # VAD tuning so stutters / soft starts / mid-utterance pauses don't get
        # dropped or split. HIGH start sensitivity catches quieter speech
        # onsets; LOW end sensitivity waits longer through pauses before
        # declaring the utterance over; prefix padding keeps the first ~200ms
        # so we don't clip the start of words.
        realtime_cfg = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=200,
                silence_duration_ms=1500,
            ),
        )
        live_cfg_kwargs: dict = dict(
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=realtime_cfg,
            system_instruction=types.Content(
                parts=[types.Part(text=(
                    "You are a passive transcription service. Transcribe everything "
                    "the speaker says verbatim, including stutters, filler words "
                    "(um, uh, like), repetitions, and partial words. Do not clean up, "
                    "summarise, or paraphrase. You MUST NOT respond, comment, or "
                    "produce any audio output. Stay completely silent regardless of "
                    "what the user says or asks. Even if the user addresses you "
                    "directly or asks you to disconnect, produce no output."
                ))]
            ),
        )
        if self._translate:
            # Live Translate mode: source language auto-detected, translated to
            # target_language_code. `input_transcription` carries the source-
            # language text; `output_transcription` carries the translation.
            live_cfg_kwargs["translation_config"] = types.TranslationConfig(
                target_language_code=config.TRANSLATE_TARGET_LANG,
            )
        live_cfg = types.LiveConnectConfig(**live_cfg_kwargs)
        log.info("LiveSession[%s] connecting model=%s", self.speaker_id, self.model)
        # `async with` would close the session at function exit; we want the session
        # to live for as long as the speaker is in the room, so manage lifecycle ourselves.
        self._session_cm = self._client.aio.live.connect(model=self.model, config=live_cfg)
        self._session = await self._session_cm.__aenter__()
        log.info("LiveSession[%s] connected", self.speaker_id)
        now = time.time()
        self._last_recv_at = now            # arm stall watchdog
        self._last_useful_recv_at = now     # arm monologue watchdog
        self._tasks.append(asyncio.create_task(self._send_loop(), name=f"live-send-{self.speaker_id}"))
        self._tasks.append(asyncio.create_task(self._recv_loop(), name=f"live-recv-{self.speaker_id}"))
        self._tasks.append(asyncio.create_task(self._silence_flush_loop(), name=f"live-flush-{self.speaker_id}"))
        self._tasks.append(asyncio.create_task(self._stall_watchdog(), name=f"live-watchdog-{self.speaker_id}"))

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
        sent_chunks = 0
        sent_bytes = 0
        last_log = time.time()
        while not self._stop.is_set():
            item = await self._send_queue.get()
            if item is None:
                return
            try:
                await self._session.send_realtime_input(
                    audio=types.Blob(data=item.data, mime_type=mime)
                )
                self._last_send_at = time.time()
                sent_chunks += 1
                sent_bytes += len(item.data)
                # Periodic heartbeat so we know audio is flowing TO Gemini even
                # when nothing is coming back.
                now = time.time()
                if now - last_log >= 2.0:
                    log.info(
                        "LiveSession[%s] send heartbeat: %d chunks / %d bytes in last %.1fs (queue=%d)",
                        self.speaker_id, sent_chunks, sent_bytes, now - last_log,
                        self._send_queue.qsize(),
                    )
                    sent_chunks = 0
                    sent_bytes = 0
                    last_log = now
            except Exception:
                log.exception("LiveSession[%s] send_realtime_input failed", self.speaker_id)
                return

    async def _recv_loop(self) -> None:
        msg_count = 0
        try:
            async for response in self._session.receive():
                msg_count += 1
                self._last_recv_at = time.time()
                sc = getattr(response, "server_content", None)
                # Surface EVERY server message at debug level so we can see what
                # arrives (or stops arriving) when the user goes silent.
                summary = []
                if sc is None:
                    summary.append("no_server_content")
                else:
                    in_tr = getattr(sc, "input_transcription", None)
                    out_tr = getattr(sc, "output_transcription", None)
                    if in_tr and getattr(in_tr, "text", None):
                        summary.append(f"in_tr={in_tr.text!r}")
                    if out_tr and getattr(out_tr, "text", None):
                        summary.append(f"out_tr={out_tr.text!r}")
                    if getattr(sc, "interrupted", False):
                        summary.append("interrupted")
                    if getattr(sc, "turn_complete", False):
                        summary.append("turn_complete")
                    if getattr(sc, "generation_complete", False):
                        summary.append("generation_complete")
                    mt = getattr(sc, "model_turn", None)
                    if mt is not None:
                        summary.append("model_turn")
                # Always log every server message we receive.
                log.info("LiveSession[%s] recv #%d: %s", self.speaker_id, msg_count, " ".join(summary) or "empty")

                if sc is None:
                    continue
                in_tr = getattr(sc, "input_transcription", None)
                out_tr = getattr(sc, "output_transcription", None)
                # Tick the "useful recv" clock for the monologue watchdog: any
                # transcription text (input OR output) or a turn boundary
                # counts. `model_turn`-only chatter does NOT.
                if (in_tr and getattr(in_tr, "text", None)) \
                        or (out_tr and getattr(out_tr, "text", None)) \
                        or getattr(sc, "turn_complete", False):
                    self._last_useful_recv_at = time.time()
                if self._translate:
                    # In translate mode the target language (English) drives
                    # sentence detection. We accumulate the source-language
                    # chunks in parallel for display.
                    if in_tr and getattr(in_tr, "text", None):
                        self._source_buffer += in_tr.text
                        self._last_source_text_at = time.time()
                    if out_tr and getattr(out_tr, "text", None):
                        await self._ingest_text(out_tr.text)
                else:
                    if in_tr and getattr(in_tr, "text", None):
                        await self._ingest_text(in_tr.text)
                if getattr(sc, "turn_complete", False):
                    if self._buffer.strip():
                        await self._emit(self._buffer.strip(), is_final=True)
                        self._buffer = ""
                        self._source_buffer = ""
                        self._last_source_text_at = 0.0
        except asyncio.CancelledError:
            log.info("LiveSession[%s] recv loop cancelled after %d messages", self.speaker_id, msg_count)
            raise
        except Exception:
            log.exception("LiveSession[%s] recv loop crashed after %d messages", self.speaker_id, msg_count)
            self.dead.set()
        else:
            log.warning("LiveSession[%s] recv loop ended cleanly after %d messages — server closed the stream", self.speaker_id, msg_count)
            self.dead.set()

    async def _ingest_text(self, chunk: str) -> None:
        self._buffer += chunk
        self._last_text_at = time.time()
        # Emit any complete sentences in the buffer.
        while True:
            m = _SENTENCE_SPLIT.match(self._buffer)
            if not m:
                break
            sentence = m.group(1).strip()
            self._buffer = self._buffer[m.end():]
            if sentence:
                source_snapshot = self._source_buffer.strip()
                # The source buffer accumulates across whole turns; for now we
                # snapshot the entire source on each English sentence emit, then
                # clear. Imperfect when source/target sentence boundaries don't
                # align, but good enough for display.
                self._source_buffer = ""
                await self._emit(sentence, is_final=True, source_text=source_snapshot)
        # Push partial (non-final) so the UI shows live typing too.
        if self._buffer.strip():
            await self._emit(
                self._buffer.strip(), is_final=False,
                source_text=self._source_buffer.strip(),
            )

    async def _stall_watchdog(self) -> None:
        """Watch for two recoverable failure modes:

        1. **Silence stall** — send loop is actively pushing audio but the
           recv loop hasn't produced ANY server message in
           `_STALL_AFTER_SECONDS`.
        2. **Model monologue** — send loop is pushing audio, recv IS happening,
           but every message is `model_turn` and no transcription text or
           turn boundary has flushed in `_MONOLOGUE_AFTER_SECONDS`. Live's
           turn protocol suppresses `input_transcription` while the model
           holds a turn, so we never see another finalized sentence — the
           speaker effectively goes silent on the UI.

        Both paths gate on `send_age < 2.0` so they only fire while audio is
        ACTIVELY being sent. Muted / idle sessions are never killed — the
        orchestrator no longer closes on mute either, so a session can sit
        idle indefinitely and resume on unmute.
        """
        try:
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                if self.dead.is_set():
                    return
                now = time.time()
                if self._last_send_at == 0 or self._last_recv_at == 0:
                    continue
                send_age = now - self._last_send_at
                if send_age >= 2.0:
                    continue  # not actively sending — silence is expected
                # 1. Hard silence stall.
                if (now - self._last_recv_at) > _STALL_AFTER_SECONDS:
                    log.warning(
                        "LiveSession[%s] STALL detected: last_send %.1fs ago, last_recv %.1fs ago — marking dead for reconnect",
                        self.speaker_id, send_age, now - self._last_recv_at,
                    )
                    self.dead.set()
                    return
                # 2. Monologue: recv is alive but useless.
                if self._last_useful_recv_at > 0 \
                        and (now - self._last_useful_recv_at) > _MONOLOGUE_AFTER_SECONDS:
                    log.warning(
                        "LiveSession[%s] MONOLOGUE detected: last_send %.1fs ago, last useful recv %.1fs ago "
                        "(model_turn-only stream, no transcription) — marking dead for reconnect",
                        self.speaker_id, send_age, now - self._last_useful_recv_at,
                    )
                    self.dead.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("LiveSession[%s] stall watchdog crashed", self.speaker_id)

    async def _silence_flush_loop(self) -> None:
        """Two flush paths, both polled every 0.3s:

        1. **English silence flush.** If the English buffer has un-punctuated
           text and no new chunks arrived in `_SILENCE_FLUSH_SECONDS`, emit
           it. This is the main path — handles Gemini dropping trailing
           punctuation and short utterances.
        2. **Source-only fallback (translate mode).** If only source-language
           text arrived (no `output_transcription` translation came back)
           and it's been quiet for `_SOURCE_FALLBACK_SECONDS`, emit the
           source text on its own. Without this path, source text gets
           trapped in `_source_buffer` forever when translation flakes.
        """
        try:
            while not self._stop.is_set():
                await asyncio.sleep(0.3)
                now = time.time()
                # Path 1: English buffer is the primary trigger.
                if self._buffer.strip() and self._last_text_at > 0 \
                        and (now - self._last_text_at) >= _SILENCE_FLUSH_SECONDS:
                    sentence = self._buffer.strip()
                    source_snapshot = self._source_buffer.strip()
                    self._buffer = ""
                    self._source_buffer = ""
                    self._last_text_at = 0.0
                    self._last_source_text_at = 0.0
                    await self._emit(sentence, is_final=True, source_text=source_snapshot)
                    continue
                # Path 2: translate mode, source arrived but translation didn't.
                # Only fires when English buffer is empty so we never compete
                # with Path 1.
                if self._translate and self._source_buffer.strip() \
                        and not self._buffer.strip() \
                        and self._last_source_text_at > 0 \
                        and (now - self._last_source_text_at) >= _SOURCE_FALLBACK_SECONDS:
                    source = self._source_buffer.strip()
                    self._source_buffer = ""
                    self._last_source_text_at = 0.0
                    # Use source as both displayed text and source_text so the
                    # UI shows the captured speech even without a translation.
                    # Downstream Flash detection will run on it; if the source
                    # language is English this is fine, otherwise the detector
                    # may not find claims — accepted trade-off vs lost text.
                    log.info(
                        "LiveSession[%s] source-only fallback emit (no translation): %r",
                        self.speaker_id, source,
                    )
                    await self._emit(source, is_final=True, source_text=source)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("LiveSession[%s] silence flush loop crashed", self.speaker_id)

    async def _emit(self, text: str, *, is_final: bool, source_text: str = "") -> None:
        try:
            await self.on_segment({
                "speaker_id": self.speaker_id,
                "text": text,
                "source_text": source_text or text,
                "is_final": is_final,
                "ts": time.time() - self._started_at,
            })
        except Exception:
            log.exception("LiveSession[%s] on_segment callback raised", self.speaker_id)
