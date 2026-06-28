"""FastAPI server: WebSocket push to browser + /internal/events receiver from agent."""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)
app = FastAPI()

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_ws_clients: list[WebSocket] = []


async def _broadcast(event: dict) -> None:
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.remove(ws)


@app.post("/internal/events")
async def receive_event(request: Request):
    event = await request.json()
    await _broadcast(event)
    return {"ok": True}


async def _fake_events():
    """Phase 0 demo: stream fake claim cards every 5s until real agent connects."""
    import time
    import uuid
    await asyncio.sleep(2)
    i = 0
    verdicts = ["true", "false", "unverifiable"]
    subjects = ["Q3 revenue", "YoY growth", "operating margin", "installed base", "guidance"]
    while True:
        claim = {
            "id": str(uuid.uuid4()),
            "session_id": "demo",
            "clip_ts": i * 5.0,
            "raw_text": f"Stub sentence #{i} with a number like {42 + i}%.",
            "subject": subjects[i % len(subjects)],
            "predicate": "equals",
            "value": str(42 + i),
            "unit": "%",
            "status": "verified",
            "verdict": verdicts[i % len(verdicts)],
            "source": "kb",
            "detected_at": time.time(),
        }
        await _broadcast({"type": "claim_detected", "claim": claim})
        await asyncio.sleep(0.1)
        await _broadcast({"type": "claim_update", "claim": claim})
        await _broadcast({
            "type": "metrics",
            "metrics": {
                "checkworthy_seen": i + 1,
                "checked": i + 1,
                "coverage": round(min(1.0, (i + 1) / 8), 2),
                "mean_time_to_verdict_ms": max(50, 800 - i * 40),
                "memory_hits": max(0, i - 1),
                "memory_misses": min(2, i + 1),
                "contradictions": 0,
            },
        })
        i += 1
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup():
    if os.getenv("FAKE_EVENTS", "true").lower() == "true":
        asyncio.create_task(_fake_events())


if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
