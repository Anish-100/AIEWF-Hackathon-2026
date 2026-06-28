"""Replay a recorded session's transcript and extract durable facts.

Usage:
    python -m scripts.distill_session            # distill the latest ended session
    python -m scripts.distill_session SESSION_ID # distill a specific one
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import end_of_session, memory  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args = sys.argv[1:]
    if args:
        session_id = args[0]
        n = asyncio.run(end_of_session.distill_session(session_id))
    else:
        n = asyncio.run(end_of_session.distill_latest())
    print(f"distilled: {n} new facts written")
    print(f"sessions on disk: {len(memory.get_memory().list_sessions())}")


if __name__ == "__main__":
    main()
