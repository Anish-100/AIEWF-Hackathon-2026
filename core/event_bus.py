"""In-process pub/sub for orchestrator → WebSocket fanout.

Each WS connection calls `subscribe()` and reads from its own bounded queue.
`publish(event)` fans the event out to every active subscriber, dropping on
overflow rather than blocking the producer.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

_QUEUE_MAX = 256
_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue[dict[str, Any]]) -> None:
    _subscribers.discard(q)


def publish(event: dict[str, Any]) -> None:
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            log.warning("event_bus: subscriber queue full, dropping event %s", event.get("type"))


def subscriber_count() -> int:
    return len(_subscribers)
