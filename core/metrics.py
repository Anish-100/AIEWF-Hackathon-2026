"""Session metrics (stub, fleshed out in Phase 5)."""
from __future__ import annotations

from core.schemas import SessionMetrics


_current = SessionMetrics()


def current() -> SessionMetrics:
    return _current


def reset() -> None:
    global _current
    _current = SessionMetrics()
