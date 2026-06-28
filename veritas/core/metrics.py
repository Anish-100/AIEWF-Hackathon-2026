"""Session metrics tracker."""
import threading
from .schemas import SessionMetrics

_lock = threading.Lock()
_metrics = SessionMetrics()
_latencies: list[float] = []


def record_checkworthy() -> None:
    with _lock:
        _metrics.checkworthy_seen += 1


def record_verdict(time_to_verdict_ms: float, source: str) -> None:
    with _lock:
        _metrics.checked += 1
        _latencies.append(time_to_verdict_ms)
        _metrics.mean_time_to_verdict_ms = sum(_latencies) / len(_latencies)
        if source in ("kb", "memory"):
            _metrics.memory_hits += 1
        else:
            _metrics.memory_misses += 1
        if _metrics.checkworthy_seen > 0:
            _metrics.coverage = _metrics.checked / _metrics.checkworthy_seen


def record_contradiction() -> None:
    with _lock:
        _metrics.contradictions += 1


def snapshot() -> SessionMetrics:
    with _lock:
        return _metrics.model_copy()


def reset() -> None:
    global _metrics, _latencies
    with _lock:
        _metrics = SessionMetrics()
        _latencies = []
