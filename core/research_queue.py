"""Async research worker queue (stub, STRETCH). Phase 6."""
from __future__ import annotations

import asyncio

from core.schemas import Claim


_queue: asyncio.Queue[Claim] | None = None


def enqueue(claim: Claim) -> None:
    raise NotImplementedError("Phase 6")


async def worker() -> None:
    raise NotImplementedError("Phase 6")
