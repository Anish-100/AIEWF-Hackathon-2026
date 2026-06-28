"""LiveKit Agent: subscribes to room audio, runs STT, feeds orchestrator."""
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)


async def _push_event(event: dict) -> None:
    import aiohttp
    import config
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(f"{config.VERITAS_SERVER_URL}/internal/events", json=event)
    except Exception as exc:
        log.warning("Failed to push event to server: %s", exc)


async def main():
    """STUB — prints fake sentences. Real impl uses LiveKit AgentSession + Gemini Realtime."""
    from core.orchestrator import process_sentence

    async def push_fn(event):
        await _push_event(event)

    log.info("Stub agent running. Sending fake sentences every 3s.")
    i = 0
    while True:
        sentences = [
            "Revenue for the quarter was 85.8 billion dollars.",
            "iPhone revenue came in at 39.3 billion dollars.",
            "Services grew 13.7 percent year over year.",
            "Our installed base of active devices reached 2.35 billion.",
            "We expect next quarter revenue of approximately 89 billion dollars.",
        ]
        await process_sentence(sentences[i % len(sentences)], clip_ts=float(i * 3), push_fn=push_fn)
        i += 1
        await asyncio.sleep(3)


if __name__ == "__main__":
    import config  # noqa — ensure env loaded
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
