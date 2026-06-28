"""Publish a local .wav file into a LiveKit room as a participant track.

Doubles as our dev harness for Phase 1 and the demo-day fallback if a live
human mic flakes out. Run twice (once per identity) to simulate a debate.

    python -m scripts.play_clip --identity speaker_a --file clips/a.wav
    python -m scripts.play_clip --identity speaker_b --file clips/b.wav

The file is expected to be PCM16 mono at any sample rate; we resample to
16 kHz to match the Gemini Live input format on the subscriber side.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
from livekit import api, rtc

# Allow running as a module from the repo root: `python -m scripts.play_clip ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("play_clip")

FRAME_MS = 20  # WebRTC standard
SAMPLE_RATE = 16000
NUM_CHANNELS = 1


def _mint_publisher_token(room_name: str, identity: str) -> str:
    if not (config.LIVEKIT_API_KEY and config.LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET must be set")
    return (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_subscribe=False,
                can_publish_data=False,
            )
        )
    ).to_jwt()


def _load_wav_pcm16(path: Path) -> tuple[np.ndarray, int, int]:
    """Return (samples_int16, src_sample_rate, src_channels)."""
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sample_width = w.getsampwidth()
        src_rate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sample_width != 2:
        raise RuntimeError(f"only 16-bit PCM wav supported, got {sample_width*8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16)
    return samples, src_rate, n_channels


def _to_mono_16k(samples: np.ndarray, src_rate: int, src_channels: int) -> np.ndarray:
    if src_channels > 1:
        samples = samples.reshape(-1, src_channels).mean(axis=1).astype(np.int16)
    if src_rate != SAMPLE_RATE:
        # Linear resample: cheap but acceptable for STT.
        n_out = int(round(len(samples) * SAMPLE_RATE / src_rate))
        xp = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        samples = np.interp(x, xp, samples.astype(np.float32)).astype(np.int16)
    return samples


def _chunk_frames(samples: np.ndarray) -> Iterator[np.ndarray]:
    frame_size = int(SAMPLE_RATE * FRAME_MS / 1000)
    for i in range(0, len(samples) - frame_size + 1, frame_size):
        yield samples[i : i + frame_size]


async def publish(file_path: Path, identity: str, room_name: str) -> None:
    samples, src_rate, src_channels = _load_wav_pcm16(file_path)
    log.info("loaded %s: %d samples @ %d Hz, %d ch", file_path, len(samples), src_rate, src_channels)
    samples = _to_mono_16k(samples, src_rate, src_channels)
    log.info("resampled to %d samples @ %d Hz mono (%.1fs)", len(samples), SAMPLE_RATE, len(samples) / SAMPLE_RATE)

    token = _mint_publisher_token(room_name, identity)
    room = rtc.Room()
    await room.connect(config.LIVEKIT_URL, token)
    log.info("connected to room=%s as %s", room_name, identity)

    source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    track = rtc.LocalAudioTrack.create_audio_track(f"{identity}-clip", source)
    publication = await room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    log.info("published track sid=%s", publication.sid)

    frame_interval = FRAME_MS / 1000.0
    try:
        for chunk in _chunk_frames(samples):
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(chunk),
            )
            await source.capture_frame(frame)
            await asyncio.sleep(frame_interval)
        log.info("clip finished, draining...")
        await asyncio.sleep(1.0)
    finally:
        await room.disconnect()
        log.info("disconnected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, help="LiveKit participant identity to publish as")
    parser.add_argument("--file", required=True, type=Path, help="Path to a 16-bit PCM wav file")
    parser.add_argument("--room", default=None, help="LiveKit room name (defaults to LIVEKIT_ROOM_NAME)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    if not args.file.exists():
        parser.error(f"file not found: {args.file}")
    if not config.LIVEKIT_URL:
        parser.error("LIVEKIT_URL not set in environment")

    room = args.room or config.LIVEKIT_ROOM_NAME
    asyncio.run(publish(args.file, args.identity, room))


if __name__ == "__main__":
    main()
