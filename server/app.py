"""FastAPI + WebSocket server.

/ws subscribes to the in-process event bus and forwards every event to the
browser as JSON. The orchestrator (LiveKit + Gemini) auto-starts on app
lifespan if `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and
`GEMINI_API_KEY` are all set. If any are missing, the server still serves
the UI; set `FAKE_EVENTS=true` in the environment to drive the UI from a
synthetic event source for local development.

Run:
    uvicorn server.app:app --reload
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
from core import event_bus

log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


def _have_livekit_creds() -> bool:
    return all([config.LIVEKIT_URL, config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET, config.GEMINI_API_KEY])


def _fake_events_enabled() -> bool:
    return os.environ.get("FAKE_EVENTS", "").lower() in {"1", "true", "yes"}


# --- Fake event source (dev only) ---------------------------------------------------

_FAKE_SPEAKERS = ["speaker_a", "speaker_b"]
_FAKE_CLAIMS = [
    ("Q3 2025 revenue", "equals", 18.4, "USD_B"),
    ("US unemployment rate", "equals", 4.1, "%"),
    ("global EV market share", "equals", 12.0, "%"),
    ("company headcount", "equals", 12500, None),
]


async def _fake_event_source(stop: asyncio.Event) -> None:
    i = 0
    while not stop.is_set():
        i += 1
        subj, pred, val, unit = random.choice(_FAKE_CLAIMS)
        event_bus.publish({
            "type": "claim",
            "claim": {
                "id": uuid.uuid4().hex,
                "session_id": "demo",
                "speaker_id": random.choice(_FAKE_SPEAKERS),
                "clip_ts": round(time.time() % 600, 2),
                "raw_text": f"{subj} {pred} {val}{unit or ''}",
                "subject": subj,
                "predicate": pred,
                "value": val,
                "unit": unit,
                "status": random.choice(["detected", "researching", "verified", "flagged"]),
                "verdict": random.choice([None, "true", "false", "unverifiable"]),
                "source": random.choice([None, "kb", "memory"]),
            },
        })
        event_bus.publish({
            "type": "metrics",
            "metrics": {
                "checkworthy_seen": i,
                "checked": max(0, i - 1),
                "coverage": min(1.0, (i - 1) / i) if i else 0.0,
                "mean_time_to_verdict_ms": max(120, 1500 - i * 30),
                "memory_hits": max(0, i - 2),
                "memory_misses": 1 if i else 0,
                "contradictions": i // 5,
            },
        })
        if i % 4 == 0:
            event_bus.publish({
                "type": "contradiction",
                "contradiction": {
                    "id": uuid.uuid4().hex,
                    "subject": "Q3 2025 revenue",
                    "kind": random.choice(["same_speaker", "cross_speaker"]),
                    "speaker_a_id": "speaker_a",
                    "speaker_b_id": "speaker_b",
                    "value_a": 12,
                    "value_b": 18,
                    "ts_a": 134.0,
                    "ts_b": 207.5,
                    "explanation": "Earlier stated 12% at 02:14; now 18% at 03:27.",
                },
            })
        try:
            await asyncio.wait_for(stop.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


# --- Lifespan: choose the event source ----------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    stop = asyncio.Event()
    bg: asyncio.Task | None = None

    if _have_livekit_creds() and not _fake_events_enabled():
        from core.orchestrator import get_orchestrator
        orch = get_orchestrator()
        log.info("lifespan: starting orchestrator (LiveKit + Gemini Live)")
        bg = asyncio.create_task(orch.run(), name="orchestrator")
        app.state.orchestrator = orch
    elif _fake_events_enabled() or not _have_livekit_creds():
        why = "FAKE_EVENTS=true" if _fake_events_enabled() else "missing credentials"
        log.warning("lifespan: running with fake event source (%s)", why)
        bg = asyncio.create_task(_fake_event_source(stop), name="fake-events")

    try:
        yield
    finally:
        log.info("lifespan: shutting down")
        stop.set()
        if app.state.__dict__.get("orchestrator"):
            await app.state.orchestrator.stop()
        if bg:
            bg.cancel()
            with contextlib.suppress(BaseException):
                await bg


app = FastAPI(title="Veritas", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "ok": True,
        "demo_mode": config.DEMO_MODE,
        "livekit_creds": _have_livekit_creds(),
        "fake_events": _fake_events_enabled(),
        "subscribers": event_bus.subscriber_count(),
    }


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    q = event_bus.subscribe()
    try:
        await socket.send_json({
            "type": "hello",
            "demo_mode": config.DEMO_MODE,
            "mode": "live" if (_have_livekit_creds() and not _fake_events_enabled()) else "fake",
        })
        while True:
            event = await q.get()
            await socket.send_json(event)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        log.exception("ws handler crashed")
        try:
            await socket.send_json({"type": "error", "message": str(exc)})
        finally:
            with contextlib.suppress(Exception):
                await socket.close()
    finally:
        event_bus.unsubscribe(q)
