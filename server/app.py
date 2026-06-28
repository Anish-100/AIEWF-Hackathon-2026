"""FastAPI + WebSocket server.

Phase 0: serves the static UI and pushes a fake claim event every 2s so the
end-to-end transport works before any real adapter is wired up.

Run:
    uvicorn server.app:app --reload
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Veritas")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "demo_mode": config.DEMO_MODE}


# --- Fake event source (Phase 0 only) -------------------------------------------------

_FAKE_SPEAKERS = ["speaker_a", "speaker_b"]
_FAKE_CLAIMS = [
    ("Q3 2025 revenue", "equals", 18.4, "USD_B"),
    ("US unemployment rate", "equals", 4.1, "%"),
    ("global EV market share", "equals", 12.0, "%"),
    ("company headcount", "equals", 12500, None),
]


def _fake_claim_event() -> dict:
    subj, pred, val, unit = random.choice(_FAKE_CLAIMS)
    return {
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
    }


def _fake_metrics_event(i: int) -> dict:
    return {
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
    }


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    await socket.send_json({"type": "hello", "demo_mode": config.DEMO_MODE})
    i = 0
    try:
        while True:
            i += 1
            await socket.send_json(_fake_claim_event())
            await socket.send_json(_fake_metrics_event(i))
            if i % 4 == 0:
                await socket.send_json({
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
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        # Don't crash the server on a single socket failure.
        try:
            await socket.send_json({"type": "error", "message": str(exc)})
        finally:
            await socket.close()
