"""Publish any audio file (WAV/MP3/MP4/M4A/etc.) into a LiveKit room as an audio track."""
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_S = 0.06  # 60ms frames
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION_S)
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * CHANNELS * 2  # 16-bit = 2 bytes


def _ffmpeg_decode(path: str) -> subprocess.Popen:
    """Decode any audio file to raw 16kHz mono s16le PCM via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", path,
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


async def publish_clip(path: str, room_name: str = config.LIVEKIT_ROOM_NAME) -> None:
    token = (
        AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity("clip-player")
        .with_name("Demo Clip")
        .with_grants(VideoGrants(room=room_name, room_join=True, can_publish=True, can_subscribe=False))
    )

    room = rtc.Room()
    log.info("Connecting to LiveKit room '%s'…", room_name)
    await room.connect(config.LIVEKIT_URL, token.to_jwt())
    log.info("Connected.")

    source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("demo-clip", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, opts)
    log.info("Streaming %s…", path)

    proc = _ffmpeg_decode(path)
    try:
        loop = asyncio.get_event_loop()
        while True:
            raw = await loop.run_in_executor(None, proc.stdout.read, BYTES_PER_CHUNK)
            if not raw:
                break
            actual_samples = len(raw) // (CHANNELS * 2)
            frame = rtc.AudioFrame(
                data=raw,
                sample_rate=SAMPLE_RATE,
                num_channels=CHANNELS,
                samples_per_channel=actual_samples,
            )
            await source.capture_frame(frame)
            await asyncio.sleep(CHUNK_DURATION_S)
    finally:
        proc.terminate()

    log.info("Clip finished. Disconnecting…")
    await asyncio.sleep(1)
    await room.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "kb/demo_clip.wav"
    if not Path(path).exists():
        print(f"ERROR: File not found: {path}")
        print("Usage: python scripts/play_clip.py <clip.mp3|clip.mp4|clip.wav|...>")
        sys.exit(1)
    asyncio.run(publish_clip(path))
