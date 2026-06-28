"""LiveKit Agent: subscribes to clip-player audio track, pipes frames to Gemini Live for STT."""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from google import genai
from google.genai import types as genai_types
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext

log = logging.getLogger(__name__)
server = AgentServer()

AUDIO_MIME = "audio/pcm;rate=16000"
RESAMPLE_RATE = 16000


async def _transcribe_track(track: rtc.AudioTrack, push_fn):
    from core.orchestrator import process_sentence

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    live_config = genai_types.LiveConnectConfig(
        input_audio_transcription=genai_types.AudioTranscriptionConfig(),
    )

    clip_start = time.time()
    print(f">>> Connecting to Gemini Live model={config.GEMINI_LIVE_MODEL}")

    try:
        async with client.aio.live.connect(model=config.GEMINI_LIVE_MODEL, config=live_config) as session:
            print(">>> Gemini Live session OPEN")
            audio_stream = rtc.AudioStream(track, sample_rate=RESAMPLE_RATE, num_channels=1)

            async def send_audio():
                count = 0
                async for frame_event in audio_stream:
                    await session.send_realtime_input(
                        audio=genai_types.Blob(data=bytes(frame_event.frame.data), mime_type=AUDIO_MIME)
                    )
                    count += 1
                    if count % 50 == 0:
                        print(f">>> Sent {count} frames")
                print(f">>> Audio stream ended after {count} frames")

            async def recv_transcripts():
                buffer = []
                async for msg in session.receive():
                    sc = msg.server_content
                    if not sc:
                        continue
                    t = sc.input_transcription
                    if not t or not t.text:
                        continue
                    buffer.append(t.text)
                    if t.finished:
                        text = "".join(buffer).strip()
                        buffer = []
                        if text:
                            clip_ts = time.time() - clip_start
                            print(f"[{clip_ts:.1f}s] {text}")
                            await process_sentence(text, clip_ts=clip_ts, push_fn=push_fn)

            send_task = asyncio.create_task(send_audio())
            recv_task = asyncio.create_task(recv_transcripts())
            done, pending = await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_EXCEPTION)
            for t in done:
                if t.exception():
                    print(f">>> Task failed: {t.exception()}")
                    log.error("Task error: %s", t.exception())
            for t in pending:
                t.cancel()

    except Exception as exc:
        print(f">>> Gemini Live FAILED: {exc}")
        log.error("Gemini Live error: %s", exc)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
    print(f">>> Agent joined room: {ctx.room.name}")
    print(f">>> Remote participants: {list(ctx.room.remote_participants.keys())}")

    async def push_fn(event: dict):
        import aiohttp
        try:
            async with aiohttp.ClientSession() as s:
                await s.post(
                    f"{config.VERITAS_SERVER_URL}/internal/events",
                    json=event,
                    timeout=aiohttp.ClientTimeout(total=2),
                )
        except Exception as exc:
            log.debug("Push failed: %s", exc)

    tasks: dict[str, asyncio.Task] = {}

    def on_track_subscribed(track, publication, participant):
        print(f">>> Track subscribed: kind={track.kind} from {participant.identity}")
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        task = asyncio.ensure_future(_transcribe_track(track, push_fn))
        task.add_done_callback(lambda t: print(f">>> Transcription task ended: {'error: ' + str(t.exception()) if not t.cancelled() and t.exception() else 'ok'}"))
        tasks[publication.sid] = task

    def on_track_unsubscribed(track, publication, participant):
        task = tasks.pop(publication.sid, None)
        if task:
            task.cancel()

    ctx.room.on("track_subscribed", on_track_subscribed)
    ctx.room.on("track_unsubscribed", on_track_unsubscribed)

    for participant in ctx.room.remote_participants.values():
        print(f">>> Existing participant: {participant.identity}")
        for pub in participant.track_publications.values():
            print(f">>> Existing pub: sid={pub.sid} track={pub.track} kind={pub.kind}")
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                on_track_subscribed(pub.track, pub, participant)

    print(">>> Waiting for events…")
    await asyncio.Event().wait()


if __name__ == "__main__":
    os.environ.setdefault("LIVEKIT_URL", config.LIVEKIT_URL)
    os.environ.setdefault("LIVEKIT_API_KEY", config.LIVEKIT_API_KEY)
    os.environ.setdefault("LIVEKIT_API_SECRET", config.LIVEKIT_API_SECRET)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    agents.cli.run_app(server)
