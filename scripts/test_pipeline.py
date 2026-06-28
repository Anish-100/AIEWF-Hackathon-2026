# veritas/scripts/test_pipeline.py
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from core.orchestrator import process_sentence, reset_session
from core.memory import init_db

SERVER_URL = "http://localhost:8000/internal/events"

async def push(event: dict):
    """Mirror what the real agent does — POST to the server."""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(SERVER_URL, json=event, timeout=5)
        except Exception as e:
            print(f"[push] {e} — is the server running?")

async def main():
    init_db()
    reset_session()

    sentences = [
        # (text, clip_offset_seconds)

    # Group 1 — numeric contradiction (Tesla)
    ("Tesla delivered 1.8 million vehicles in 2023.", 5.0),
    ("Tesla's annual revenue is 97 billion dollars.", 15.0),
    ("Tesla actually delivered only 1.2 million vehicles in 2023.", 60.0),  # ← contradiction fires

    # Group 2 — numeric contradiction (Meta)
    ("Meta has 3.2 billion monthly active users.", 80.0),
    ("Meta's workforce is about 67000 employees.", 95.0),
    ("Meta reported 2.1 billion monthly active users last quarter.", 130.0),  # ← contradiction fires

    # Group 3 — string contradiction
    ("Elon Musk is the CEO of Tesla.", 150.0),
    ("Elon Musk is the CEO of SpaceX.", 160.0),  # ← may fire depending on embedding similarity

    # Group 4 — warm memory test (repeat from group 1)
    ("Tesla delivered 1.8 million vehicles in 2023.", 200.0),  # ← should be memory hit, fast verdict

    ]

    for text, ts in sentences:
        print(f"\n[{ts}s] → {text}")
        await process_sentence(text, ts, push)
        await asyncio.sleep(1)   # give server time to broadcast

if __name__ == "__main__":
    asyncio.run(main())