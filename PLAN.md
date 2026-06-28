# Veritas — Implementation Plan

> Real-time fact-checking that **learns the domain as it listens**. Cold run = slow/misses.
> Warm run (memory pre-populated) = instant verdicts. The cold→warm delta IS the demo.

---

## Current State

Only `gemini-hacker-starter/` boilerplate exists. Zero Veritas code.
Goal: build `veritas/` from scratch, Phases 0–7 in order.

---

## Architecture

```
Demo clip (any audio file — MP3/MP4/WAV/etc.)
  → scripts/play_clip.py  (ffmpeg decode → PCM → LiveKit room as audio track)
    → adapters/livekit_audio.py  (LiveKit Agent)
      → on_track_subscribed → rtc.AudioStream → raw PCM frames
        → client.aio.live.connect(model=GEMINI_LIVE_MODEL)
          → send_realtime_input(audio=Blob(pcm, "audio/pcm;rate=16000"))
          → receive() → sc.input_transcription.text  →  finalized sentence
        → adapters/gemini_flash.py  →  {is_checkworthy, subject, predicate, value, unit}
          → core/verifier.py  (embed → SQLite+sqlite-vec vector search)
            ├─ HIT  → verdict <100ms
            └─ MISS → status="researching" → research_queue.py (async)
                         → adapters/gemini_interactions.py
                             client.interactions.create(
                               model=FLASH, input=claim, tools=[{"type":"google_search"}],
                               background=True
                             )
                         → poll → write VerifiedFact to memory
                         → push update: card flips ⏳→✓/✗, now permanent in memory
          → core/contradiction.py  (new value vs session history + memory)
        → HTTP POST  →  server/app.py  (FastAPI)
          → WebSocket  →  browser  (claim cards, contradiction alert, 2 metrics)
```

---

## Key Technical Decisions

| Concern | Decision |
|---|---|
| STT | Direct Gemini Live API via `client.aio.live.connect()` + `send_realtime_input(audio=Blob(...))` + `sc.input_transcription.text`. Bypasses `AgentSession` entirely — Live models don't support TEXT-only modality via the agents abstraction. |
| Claim detection | `google-genai` Flash via direct API call → strict JSON |
| Embeddings | `google-genai` embedding model |
| Memory | SQLite + `sqlite-vec`; numpy cosine fallback |
| UI comms | Agent → HTTP POST localhost → FastAPI → WebSocket → browser |
| Demo clip | Real earnings call (Apple Q3 FY2025); download with `yt-dlp`, trim with `ffmpeg` |
| Async research | Interactions API: `client.interactions.create(model=FLASH, input=claim, tools=[{"type":"google_search"}], background=True)` → poll → write to memory |

---

## Directory Structure

```
veritas/
├── requirements.txt
├── .env.example
├── config.py
├── adapters/
│   ├── __init__.py
│   ├── livekit_audio.py         # Agent: STT → sentence events → orchestrator
│   ├── gemini_flash.py          # Flash: sentence → structured claim JSON
│   ├── gemini_embed.py          # Embeddings: text → float[]
│   └── gemini_interactions.py   # Interactions API: claim → web research → verdict + source
├── core/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic: Claim, VerifiedFact, Contradiction, SessionMetrics
│   ├── memory.py             # SQLite + sqlite-vec; put/get/vector_search/touch
│   ├── verifier.py           # Hot-path: embed → search → hit/miss
│   ├── contradiction.py      # New claim vs session history + memory
│   ├── metrics.py            # coverage + rolling mean_time_to_verdict
│   └── orchestrator.py       # Wires transcript → detect → verify → contradiction → push
├── kb/
│   ├── demo_clip_facts.yaml  # ~20 curated ground-truth facts for chosen clip
│   └── build_kb.py           # Embed facts → write VerifiedFact rows to SQLite
├── server/
│   ├── app.py                # FastAPI + WebSocket + /internal/events receiver
│   └── static/
│       └── index.html        # Claim cards + contradiction banner + 2 big live metrics
└── scripts/
    ├── play_clip.py          # livekit-rtc: join room, publish audio file as track
    ├── reset_memory.py       # DELETE all rows → COLD
    └── seed_memory.py        # build_kb → WARM
tests/
├── test_memory_persistence.py
├── test_contradiction.py
└── test_verifier_latency.py
```

---

## Demo Clip: Real Earnings Call

**Recommended:** Apple Q3 FY2025 earnings (reported late July 2025).

### Prep (do before Phase 2 coding)
```bash
yt-dlp "https://www.youtube.com/..." -o demo_clip_raw.%(ext)s   # official IR upload
ffmpeg -i demo_clip_raw.* -ss 00:02:00 -t 00:05:00 -ar 16000 -ac 1 veritas/kb/demo_clip.wav
```
Target: 4–6 min CFO prepared remarks (densest verifiable claims).

### KB Facts to curate (example — verify against Apple's official press release)

| Subject | Value | Verdict |
|---|---|---|
| Q3 FY2025 revenue | $85.8B | true |
| iPhone revenue Q3 | $39.3B | true |
| Services revenue Q3 | $23.9B | true |
| YoY revenue growth | 5% | true |
| Installed base active devices | 2.35B | true |
| Q4 guidance revenue | ~$89B | unverifiable (forward-looking) |
| *(any figure speaker rounds differently in two places)* | — | **contradiction** |

**All figures must be verified against Apple's official Q3 FY2025 press release before committing.**

Natural contradictions: look for YoY vs QoQ confusion, or the same metric rounded differently in prepared remarks vs Q&A.

---

## Model IDs — Verify Before Coding

Do NOT invent these. Check https://aistudio.google.com or the Gemini changelog.

- `GEMINI_LIVE_MODEL` — already in starter: `gemini-3.1-flash-audio-eap`
- `GEMINI_FLASH_MODEL` — current text Flash: likely `gemini-2.5-flash` or `gemini-3.5-flash`
- `GEMINI_EMBED_MODEL` — likely `text-embedding-004` or `gemini-embedding-exp-03-07`

---

## Phase Build Order

### Phase 0 — Scaffold (~30 min)
- `requirements.txt`, `.env.example`, `config.py`
- Stub all adapters (return fakes)
- `server/app.py` + `index.html` with fake WebSocket events every 5s
- **Accept:** `python -m server.app` → browser shows UI with fake claim cards

### Phase 1 — Audio Spine (~45 min)
- `scripts/play_clip.py` — `livekit.rtc` join room, ffmpeg-decode any format, publish 16kHz mono PCM frames
- `adapters/livekit_audio.py` — LiveKit Agent (no AgentSession): `on_track_subscribed` → `rtc.AudioStream` → pipe PCM frames to `client.aio.live.connect()` → `sc.input_transcription.text` → print + orchestrator
- **Why no AgentSession:** Live models only output AUDIO modality; bypassing it gives direct control over frame routing and transcription
- **Accept:** `play_clip.py` → agent logs finalized sentences in real-time

### Phase 2 — Detection + Curated KB + Fast Verdicts (~60 min)
- `adapters/gemini_flash.py` — structured claim JSON via Flash
- `kb/demo_clip_facts.yaml` + `kb/build_kb.py` — embed and store 20 ground-truth facts
- `core/verifier.py` — numpy cosine search (fast fallback, no sqlite-vec needed yet)
- `core/orchestrator.py` — pipe sentence → flash → verify → HTTP POST to server
- `server/app.py` — `/internal/events` endpoint + WebSocket broadcast
- `index.html` — claim cards (status, subject, verdict icon, source)
- **Accept:** clip plays → known claims show ✓/✗ in browser within 1.5s

### Phase 3 — Persistent Memory (~60 min) ← THE MOAT
- `core/memory.py` — SQLite + `sqlite-vec`; `put`, `get`, `vector_search`, `facts_by_subject`, `touch`
- Verifier writes resolved facts back to memory
- `scripts/reset_memory.py` + `scripts/seed_memory.py`
- `tests/test_memory_persistence.py` — write facts, subprocess kill, re-open, assert present
- **Accept:** test passes; second clip run shows lower mean_time_to_verdict

### Phase 4 — Contradiction Detection (~45 min) ← DEMO MOMENT
- `core/contradiction.py` — embed subjects (cosine ≥ 0.85), compare numeric values
- Push `{type: "contradiction", ...}` event from orchestrator
- `index.html` — full-width red banner "⚠️ CONTRADICTION — said X at 0:45, now Y at 2:30"
- **Accept:** the targeted contradiction fires a loud alert live

### Phase 5 — Metrics + Cold/Warm (~30 min)
- `core/metrics.py` — thread-safe `SessionMetrics`, pushed after every event
- `index.html` — two big live counters: "Coverage: 67%" and "Avg verdict: 234ms"
- **Accept:** counters move; COLD run shows higher latency than WARM run

### Phase 6 — Interactions API Research Worker (~45 min)
- `adapters/gemini_interactions.py`:
  ```python
  interaction = client.interactions.create(
      model=GEMINI_FLASH_MODEL,
      input=f"Verify this claim and return JSON {{verdict, value, source, explanation}}: {claim.raw_text}",
      tools=[{"type": "google_search"}],
      background=True,
  )
  # poll interaction.id until status == "completed"
  # parse interaction.output_text → VerifiedFact → memory.put()
  ```
- `core/research_queue.py` — asyncio queue; dequeues MISS claims, calls adapter, writes to memory, HTTP POSTs update to UI
- The learning story: MISS during run → Interactions researches → writes to memory → NEXT run of same clip → instant hit, card was ⏳ now permanent ✓/✗
- **Accept:** one unrehearsed claim resolves from ⏳ to ✓/✗ live; identical claim on replay is instant

### Phase 7 — UI Polish (~30 min)
- Color-coded cards, status transitions ⏳→✓/✗, COLD/WARM pill, auto-dismiss contradiction banner
- **Accept:** first-time viewer immediately understands what happened

### Phase 8 — Rehearse + Record
- 5 runs (2 COLD, 2 WARM, 1 COLD again)
- Record 1-min demo video, repo public, all members on submission

---

## Verification Checklist (Definition of Done)

- [ ] `python -m server.app` → browser shows UI
- [ ] `python scripts/play_clip.py` → agent logs sentences
- [ ] `python kb/build_kb.py` → SQLite has 20 VerifiedFact rows with embeddings
- [ ] Clip plays → claim cards appear with ✓/✗ within 1.5s
- [ ] `pytest tests/test_memory_persistence.py` PASSES
- [ ] Second clip run is measurably faster (mean_time_to_verdict drops)
- [ ] Contradiction banner fires on the targeted metric discrepancy
- [ ] Coverage and mean_time_to_verdict move live on screen
- [ ] Demo runs with wifi off (except pre-warmed Gemini calls)
- [ ] 1-min video recorded; repo public; all team members on submission