"""Session metrics — coverage + mean time-to-verdict, pushed to the UI.

Owns one `SessionMetrics` per orchestrator run. Helper functions mutate it
and (optionally) emit a `metrics` event on the bus so the UI counters move
in real time.
"""
from __future__ import annotations

import logging
from typing import Optional

from core import event_bus
from core.schemas import SessionMetrics

log = logging.getLogger(__name__)

_current: SessionMetrics = SessionMetrics()


def current() -> SessionMetrics:
    return _current


def reset() -> None:
    global _current
    _current = SessionMetrics()


def note_checkworthy() -> None:
    """A claim was detected as check-worthy by Flash."""
    _current.checkworthy_seen += 1


def note_verdict(*, hit: bool, time_to_verdict_ms: Optional[int]) -> None:
    """Verifier finished. Hit=True → KB/memory match. Hit=False → MISS."""
    if hit:
        _current.memory_hits += 1
    else:
        _current.memory_misses += 1
    if time_to_verdict_ms is not None:
        _current.record_verdict(time_to_verdict_ms)


def note_contradiction() -> None:
    _current.contradictions += 1


def publish() -> None:
    """Push the current metrics snapshot to the UI."""
    event_bus.publish({"type": "metrics", "metrics": _current.to_dict()})


def publish_memory_size(size: int) -> None:
    event_bus.publish({"type": "memory_size", "size": size})
