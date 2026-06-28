# Veritas — Real-time fact-checking that learns the domain as it listens

**AI Engineer World's Fair Hackathon 2026 · Cerebral Valley**
Theme: **Continual Learning** · Tracks: **Gemini 3.5** + **LiveKit**

Two speakers join a LiveKit room. A silent agent listens, transcribes both,
checks every factual claim in real time against a persistent local memory,
fires loud banners when speakers contradict each other or themselves, and
*permanently grows its memory* every time it has to research something on the
web. Restart the process and the learnings survive — that's the
continual-learning moat.

---

## The 60-second demo

1. **Cold pass.** Wipe memory. Two speakers each state a fact. Cards land as
   `⌛ RESEARCHING` while Gemini + Google Search look them up (~5–10 s each).
2. **Now warm.** Same speakers replay the same lines. Every card lands as
   `✓ TRUE` (or `✗ FALSE`) in **<1 s** — the system *learned* the answers.
   `coverage` jumps from 0% to 100%, `mean time-to-verdict` collapses.
3. **Contradiction beat.** Alex says 4.1%, then later says 5.0% — orange
   `⚠ SAME-SPEAKER CONTRADICTION` banner. Bob says 180k payrolls when alex
   said 228k — `⚠ CROSS-SPEAKER CONTRADICTION` banner.
4. **Kill the process and restart.** Memory survives on disk. Replay → still
   instant hits.

Two metrics are on screen and move correctly: **coverage ↑** and **mean
time-to-verdict ↓**.

The full stage runbook is in [`scripts/demo_cold_vs_warm.md`](scripts/demo_cold_vs_warm.md).

---

## Architecture

```
 Two LiveKit Meet participants ──► LiveKit room
                                    │
                                    ▼
              Veritas agent (FastAPI + LiveKit Agents, all in one process)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  Gemini Live API           Gemini 2.5 Flash             Gemini Embeddings
  (per-speaker              (claim detection,            (subject vectors
   transcription)            strict-JSON schema)          for memory lookup)
        │                           │                           │
        └─────────► finalized text ─┴─► structured claim ───────┘
                                    │
                                    ▼
                  Verifier ── cosine match against ──► SQLite + numpy
                  (HOT PATH, <100ms p50)               persistent memory
                                    │
                            MISS ───┴─── HIT
                              ▼          ▼
              Gemini Flash + Google      ✓ / ✗ verdict
              Search grounding           (under 1s)
              (writes new fact back
               to memory, source=web)
                                    │
                                    ▼
                  Contradiction checker (same-speaker / cross-speaker)
                                    │
                                    ▼
                 FastAPI WebSocket → pastel-orange dashboard
                 (claim cards, banners, live metrics)
```

### Components

- **LiveKit Agents (Python)** — joins the room as `veritas-agent`, subscribes
  to every remote participant's audio track, attributes audio to the
  participant `identity` (no diarization).
- **Gemini Live API** (`gemini-3.1-flash-live-preview`) — one streaming session
  per speaker. Mute-aware: auto-reconnects on silence-stall.
- **Gemini 2.5 Flash** — strict-JSON claim detection via `response_schema`.
- **Gemini embeddings** + numpy cosine — sub-2ms KB lookups at our scale (26
  KB facts + arbitrary number of session-discovered facts).
- **SQLite** (with optional `sqlite-vec`) — the persistent memory layer. The
  moat. `tests/test_memory_persistence.py` verifies facts survive a process
  restart.
- **Gemini Flash + Google Search grounding** — the "research" path for cache
  misses. Returns a sourced JSON verdict in ~5–10 s, writes the result back
  to memory (`source="web"`), so the same claim becomes an instant hit
  forever after.
- **End-of-session distiller** (`core/end_of_session.py`) — replays the full
  transcript through Flash with conversational context, extracts durable
  facts with provenance (`source_session_id`, `source_speaker`,
  `supporting_quote`), writes them back. Each session permanently grows the
  KB.
- **Contradiction checker** — intra-session, requires canonical subject
  equality AND value conflict; tags `same_speaker` / `cross_speaker`.

### Continual learning, three layers

| Layer | When it learns | Source label |
|---|---|---|
| Curated KB | Once, before demo | `kb` |
| Async research on misses | During the session | `web` |
| End-of-session distiller | After the session | `session` |

All three write into the same `VerifiedFact` rows in SQLite. The verifier
treats them identically — only the `source` chip on each card differs.

---

## Quickstart

```bash
# 1. Set up env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Add credentials → .env.local (use .env.example as a template)
#    Need: GEMINI_API_KEY, LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET

# 3. Seed the curated KB
python -m scripts.reset_memory && python -m scripts.seed_memory

# 4. Run the server (FastAPI + orchestrator in one process)
uvicorn server.app:app --reload
# → dashboard at http://localhost:8000

# 5. Mint two LiveKit tokens with distinct identities
lk token create --identity alex --room veritas-demo --join --valid-for 1h
lk token create --identity bob  --room veritas-demo --join --valid-for 1h

# 6. Join both at https://meet.livekit.io (paste url + token in each tab)
#    Speak labor-market claims. Watch the dashboard.
```

### Useful scripts

```bash
python -m scripts.reset_memory      # wipe memory (COLD)
python -m scripts.seed_memory       # load curated KB (WARM)
python -m scripts.distill_session   # replay latest session → durable facts
python -m scripts.play_clip --identity alex --file clips/a.wav  # publish wav as a speaker
```

### Tests

```bash
pytest tests/ -v
```

12 tests cover memory persistence (the moat), verifier latency (p50 < 100ms),
contradiction detection (same-speaker / cross-speaker / unit-aware / different
months / different units).

---

## Configuration

All in [`.env.local`](.env.example) (which extends `.env.example`):

```
GEMINI_API_KEY=...
GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_EMBED_MODEL=gemini-embedding-2

LIVEKIT_URL=wss://....livekit.cloud
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
LIVEKIT_ROOM_NAME=veritas-demo

SIM_THRESHOLD=0.82
SUBJECT_MATCH_THRESHOLD=0.85
VALUE_TOLERANCE=0.0
MEMORY_DB_PATH=./veritas_memory.db
DEMO_MODE=warm                 # warm = seeded memory; cold = empty
USE_ANTIGRAVITY=false          # true = Gemini + Search runs research on cache misses
```

---

## Repository layout

```
adapters/
  livekit_audio.py        # join room, subscribe to per-speaker audio tracks
  gemini_live.py          # per-speaker streaming transcription
  gemini_flash.py         # strict-JSON claim detection
  gemini_embed.py         # subject embeddings (LRU-cached)
  antigravity.py          # research path: Gemini Flash + Google Search grounding

core/
  schemas.py              # Claim, VerifiedFact, Contradiction, SessionMetrics
  memory.py               # SQLite + numpy persistent fact store + utterances/sessions
  verifier.py             # hot-path cosine match + unit-aware value comparison
  contradiction.py        # intra-session same/cross speaker detector
  metrics.py              # coverage + time-to-verdict, pushed to UI
  event_bus.py            # in-process pub/sub for orchestrator → WebSocket
  research_queue.py       # async worker that resolves cache misses
  end_of_session.py       # post-session transcript → durable facts distiller
  orchestrator.py         # wires it all together

kb/
  demo_topic_facts.yaml   # 26 curated US labor-market facts (illustrative)
  build_kb.py             # yaml → embed → memory

server/
  app.py                  # FastAPI + WebSocket + lifespan
  static/index.html       # pastel-orange dashboard (single file)

scripts/
  reset_memory.py         # wipe memory → COLD
  seed_memory.py          # load curated KB → WARM
  play_clip.py            # publish a wav file as a LiveKit participant
  distill_session.py      # run end_of_session.distill_session(...)
  demo_cold_vs_warm.md    # stage runbook for the demo

tests/                    # 12 tests, all green
  test_memory_persistence.py
  test_verifier_latency.py
  test_contradiction.py

HANDOFF.md                # current build state for fresh sessions
PLAN.md                   # phase-by-phase build plan with cut order
claude.md                 # original detailed spec
```

---

## What was built during the hackathon

Everything in `adapters/`, `core/`, `server/`, `kb/`, `scripts/`, `tests/`,
plus `HANDOFF.md`, `PLAN.md`, `requirements.txt`, `.env.example`,
`config.py` — all new this weekend. The only pre-existing file is
[`claude.md`](claude.md), which is the original product spec.

## Acknowledgements

- **LiveKit** — programmable participant, multi-track audio, mute/unmute events.
- **Google DeepMind / Gemini** — Live API, 2.5 Flash with structured output and Google Search grounding, embeddings.
- **Cerebral Valley** — running the event.
