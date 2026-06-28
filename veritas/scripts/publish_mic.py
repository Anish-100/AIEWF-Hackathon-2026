"""Capture default system microphone via ffmpeg and publish as a LiveKit audio track."""
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from livekit import api, rtc
from livekit.api import AccessToken, VideoGrants

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 60
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000   # 960 samples
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * CHANNELS * 2   # 16-bit

AGENT_NAME = "veritas-listener"   # must match @server.rtc_session(agent_name=...) in livekit_audio.py


def _ffmpeg_mic() -> subprocess.Popen:
    if sys.platform == "darwin":
        src = ["-f", "avfoundation", "-i", ":0"]
    elif sys.platform == "win32":
        src = ["-f", "dshow", "-i", "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{default}"]
    else:
        src = ["-f", "pulse", "-i", "default"]

    return subprocess.Popen(
        ["ffmpeg", "-v", "quiet", *src,
         "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS), "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


async def publish_mic(room_name: str = config.LIVEKIT_ROOM_NAME) -> None:
    token = (
        AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity("mic-publisher")
        .with_name("Microphone")
        .with_grants(VideoGrants(
            room=room_name,
            room_join=True,
            can_publish=True,
            can_subscribe=False,
        ))
        .to_jwt()
    )

    room = rtc.Room()
    log.info("Connecting to room '%s' at %s …", room_name, config.LIVEKIT_URL)
    await room.connect(config.LIVEKIT_URL, token)
    log.info("Connected.")

    lkapi = api.LiveKitAPI(
        config.LIVEKIT_URL, config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET
    )
    try:
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=room_name)
        )
        log.info("Dispatched agent '%s' to room (dispatch id=%s)", AGENT_NAME, dispatch.id)
    except Exception as exc:
        log.error("Agent dispatch failed: %s", exc)
    finally:
        await lkapi.aclose()

    source = rtc.AudioSource(SAMPLE_RATE, CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    log.info("Publishing mic — speak now (Ctrl+C to stop).")

    proc = _ffmpeg_mic()
    loop = asyncio.get_event_loop()
    try:
        while True:
            raw = await loop.run_in_executor(None, proc.stdout.read, BYTES_PER_CHUNK)
            if not raw:
                break
            samples = len(raw) // (CHANNELS * 2)
            await source.capture_frame(rtc.AudioFrame(
                data=raw,
                sample_rate=SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=samples,
            ))
    except asyncio.CancelledError:
        pass
    finally:
        proc.terminate()

    log.info("Stopped. Disconnecting…")
    await room.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(publish_mic())
    except KeyboardInterrupt:
        pass
