"""LiveKit audio adapter.

Connects an `rtc.Room` directly (no agent-server framework), subscribes to
every remote participant's audio track, and pushes raw 16 kHz mono PCM frames
to a callback keyed by the publishing participant's identity.

We deliberately use `rtc.Room` rather than the `@server.rtc_session()` worker
pattern: the agent server framework is designed for cloud dispatch and
per-job processes. For an in-process FastAPI demo we just want a persistent
room subscription.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from livekit import api, rtc

import config

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Gemini Live expects 16 kHz mono PCM16
NUM_CHANNELS = 1

OnAudioFrame = Callable[[str, bytes, int, int], Awaitable[None]]
# (speaker_id, pcm_bytes, sample_rate, num_channels)

OnSpeakerEvent = Callable[[str, str], Awaitable[None]]
# (event_kind, speaker_id) where event_kind is "joined" | "left"


def _mint_token(room_name: str, identity: str) -> str:
    if not (config.LIVEKIT_API_KEY and config.LIVEKIT_API_SECRET):
        raise RuntimeError("LIVEKIT_API_KEY / LIVEKIT_API_SECRET must be set")
    token = (
        api.AccessToken(config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room=room_name,
                room_join=True,
                can_subscribe=True,
                can_publish=False,
                can_publish_data=False,
            )
        )
    )
    return token.to_jwt()


async def _pump_audio_stream(
    speaker_id: str,
    track: rtc.RemoteAudioTrack,
    on_audio_frame: OnAudioFrame,
) -> None:
    stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
    log.info("audio stream open for speaker=%s", speaker_id)
    try:
        async for frame_event in stream:
            frame = frame_event.frame
            try:
                await on_audio_frame(speaker_id, bytes(frame.data), SAMPLE_RATE, NUM_CHANNELS)
            except Exception:
                log.exception("on_audio_frame raised for speaker=%s", speaker_id)
    finally:
        await stream.aclose()
        log.info("audio stream closed for speaker=%s", speaker_id)


async def run_room_subscriber(
    on_audio_frame: OnAudioFrame,
    on_speaker_event: Optional[OnSpeakerEvent] = None,
    *,
    room_name: Optional[str] = None,
    identity: str = "veritas-agent",
    stop: Optional[asyncio.Event] = None,
) -> None:
    """Connect to the LiveKit room and pump audio from all remote participants.

    Runs until `stop` is set (or the task is cancelled).
    """
    room_name = room_name or config.LIVEKIT_ROOM_NAME
    if not config.LIVEKIT_URL:
        raise RuntimeError("LIVEKIT_URL must be set")

    token = _mint_token(room_name, identity)
    room = rtc.Room()
    pump_tasks: dict[str, asyncio.Task] = {}
    loop = asyncio.get_event_loop()

    def _schedule(coro):
        loop.create_task(coro)

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        speaker_id = participant.identity
        log.info("track_subscribed: speaker=%s sid=%s", speaker_id, track.sid)
        task = loop.create_task(_pump_audio_stream(speaker_id, track, on_audio_frame))
        pump_tasks[track.sid] = task
        if on_speaker_event:
            _schedule(on_speaker_event("joined", speaker_id))

    @room.on("track_unsubscribed")
    def _on_track_unsubscribed(track: rtc.Track, *_: object) -> None:
        task = pump_tasks.pop(track.sid, None)
        if task:
            task.cancel()

    @room.on("participant_disconnected")
    def _on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
        if on_speaker_event:
            _schedule(on_speaker_event("left", p.identity))

    log.info("connecting to LiveKit room=%s as %s url=%s", room_name, identity, config.LIVEKIT_URL)
    await room.connect(
        config.LIVEKIT_URL,
        token,
        options=rtc.RoomOptions(auto_subscribe=True),
    )
    log.info("connected to room=%s", room_name)

    try:
        if stop is None:
            stop = asyncio.Event()
        await stop.wait()
    finally:
        for t in pump_tasks.values():
            t.cancel()
        await room.disconnect()
        log.info("disconnected from room=%s", room_name)
