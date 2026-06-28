"""LiveKit Agent: subscribes to audio track, streams to Gemini Live, prints transcripts."""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

import config
from google import genai
from google.genai import types as genai_types
from livekit import agents, rtc
from livekit.agents import AgentServer, AutoSubscribe, JobContext

log = logging.getLogger(__name__)
server = AgentServer()

AUDIO_MIME = "audio/pcm;rate=16000"
RESAMPLE_RATE = 16000
SILENCE_RMS = 200


async def _transcribe_track(track: rtc.AudioTrack, session_start: float):
    client = genai.Client(api_key=config.GEMINI_API_KEY)

    # native-audio models only support AUDIO response modality; we still get
    # transcription text via input_audio_transcription and just ignore the audio bytes.
    is_native_audio = "native-audio" in config.GEMINI_LIVE_MODEL
    modalities = ["AUDIO"] if is_native_audio else ["TEXT"]
    live_cfg = genai_types.LiveConnectConfig(
        response_modalities=modalities,
        input_audio_transcription=genai_types.AudioTranscriptionConfig(),
    )

    log.info("Connecting to Gemini Live model=%s modalities=%s", config.GEMINI_LIVE_MODEL, modalities)
    try:
        async with client.aio.live.connect(model=config.GEMINI_LIVE_MODEL, config=live_cfg) as session:
            log.info("Gemini Live session OPEN")
            audio_stream = rtc.AudioStream(track, sample_rate=RESAMPLE_RATE, num_channels=1)

            async def send_audio():
                sent = 0
                async for ev in audio_stream:
                    pcm = bytes(ev.frame.data)
                    rms = float(np.sqrt(np.mean(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) ** 2)))
                    if rms < SILENCE_RMS:
                        continue
                    await session.send_realtime_input(
                        audio=genai_types.Blob(data=pcm, mime_type=AUDIO_MIME)
                    )
                    sent += 1
                    if sent % 100 == 0:
                        log.info("Sent %d active audio frames", sent)
                log.info("Audio stream ended (%d frames)", sent)

            async def recv_transcripts():
                async for msg in session.receive():
                    sc = msg.server_content
                    if not sc:
                        continue

                    text: str | None = None

                    # native-audio / TEXT-modality models: response in model_turn
                    if sc.model_turn:
                        for part in sc.model_turn.parts or []:
                            if getattr(part, "text", None):
                                text = (text or "") + part.text

                    # non-native models with input_audio_transcription enabled
                    if not text and sc.input_transcription and sc.input_transcription.text:
                        text = sc.input_transcription.text

                    if text:
                        text = text.strip()
                    if text:
                        ts = time.time() - session_start
                        print(f"\n[transcript @ {ts:.1f}s] {text}\n", flush=True)
                        log.info("TRANSCRIPT: %s", text)
                        try:
                            from core.orchestrator import process_sentence
                            await process_sentence(text, clip_ts=ts, push_fn=lambda _: asyncio.sleep(0))
                        except Exception:
                            pass

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_transcripts())
            done, pending = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_EXCEPTION
            )
            for t in done:
                if not t.cancelled() and t.exception():
                    log.error("Task error: %s", t.exception())
            for t in pending:
                t.cancel()

    except Exception as exc:
        log.error("Gemini Live error: %s", exc)


AGENT_NAME = "veritas-listener"


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    log.info("Agent joined room: %s", ctx.room.name)

    session_start = time.time()
    tasks: dict[str, asyncio.Task] = {}

    def on_track_subscribed(track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        log.info("Audio track subscribed from %s — starting transcription", participant.identity)
        task = asyncio.ensure_future(_transcribe_track(track, session_start))
        tasks[track.sid] = task

    def on_track_unsubscribed(track, publication, participant):
        task = tasks.pop(track.sid, None)
        if task:
            task.cancel()

    ctx.room.on("track_subscribed", on_track_subscribed)
    ctx.room.on("track_unsubscribed", on_track_unsubscribed)

    for participant in ctx.room.remote_participants.values():
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                on_track_subscribed(pub.track, pub, participant)

    log.info("Waiting for audio tracks…")
    await asyncio.Event().wait()


if __name__ == "__main__":
    os.environ.setdefault("LIVEKIT_URL", config.LIVEKIT_URL)
    os.environ.setdefault("LIVEKIT_API_KEY", config.LIVEKIT_API_KEY)
    os.environ.setdefault("LIVEKIT_API_SECRET", config.LIVEKIT_API_SECRET)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    agents.cli.run_app(server)
