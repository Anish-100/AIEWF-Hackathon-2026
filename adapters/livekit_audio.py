"""LiveKit audio adapter (stub).

Phase 1 implementation: join the LiveKit room as `veritas-agent`, subscribe to
every remote participant's audio track, and invoke `on_audio_frame(speaker_id,
pcm_frame)` for each frame. Speaker attribution = participant identity.
"""
from __future__ import annotations

from typing import Awaitable, Callable

OnAudioFrame = Callable[[str, bytes], Awaitable[None]]


async def run_agent(room_name: str, on_audio_frame: OnAudioFrame) -> None:
    raise NotImplementedError("Phase 1: implement against livekit-agents SDK.")
