"""Capture audio from the system microphone via ffmpeg and publish it into a LiveKit room."""
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


def _ffmpeg_mic() -> subprocess.Popen:
    """Capture default microphone as raw 16kHz mono s16le PCM via ffmpeg."""
    if sys.platform == "darwin":
        # macOS: AVFoundation; ":0" = default audio input device
        src_args = ["-f", "avfoundation", "-i", ":0"]
    elif sys.platform == "win32":
        src_args = ["-f", "dshow", "-i", "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{default}"]
    else:
        # Linux: try PulseAudio first, fall back to ALSA
        src_args = ["-f", "pulse", "-i", "default"]

    cmd = [
        "ffmpeg", "-v", "quiet",
        *src_args,
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


async def publish_microphone(room_name: str = config.LIVEKIT_ROOM_NAME) -> None:
    token = (
        AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity("mic-publisher")
        .with_name("Microphone")
        .with_grants(VideoGrants(room=room_name, room_join=True, can_publish=True, can_subscribe=False))
    )

    room = rtc.Room()
    log.info("Connecting to LiveKit room '%s'…", room_name)
    await room.connect(config.LIVEKIT_URL, token.to_jwt())
    log.info("Connected. Capturing from microphone — Ctrl+C to stop.")

    source = rtc.AudioSource(sample_rate=SAMPLE_RATE, num_channels=CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track("microphone", source)
    opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, opts)

    proc = _ffmpeg_mic()
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
    except asyncio.CancelledError:
        pass
    finally:
        proc.terminate()

    log.info("Microphone stopped. Disconnecting…")
    await room.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(publish_microphone())
    except KeyboardInterrupt:
        pass
