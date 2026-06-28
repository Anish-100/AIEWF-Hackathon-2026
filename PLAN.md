# Veritas — Build Plan (AIEWF 2026 Hackathon)

## Context

We are building a real-time fact-checking system for a live multi-speaker session (debate, panel, or prepared speech) streamed through a LiveKit room. The system targets the **Gemini 3.5** and **LiveKit** prize tracks, and qualifies under the **Continual Learning** theme.

The repository is currently empty except for `claude.md` (a detailed spec) and an empty `README.md`. The spec assumes a single broadcast clip; this plan adapts it to a **2+ participant live LiveKit room** while keeping the spec's "persistent memory + contradiction detection" moat intact.

**Decisions locked from clarifying questions:**
- **Speaker setup:** two remote LiveKit participants (separate clients). Each participant's LiveKit `identity` gives us free speaker attribution — **no diarization needed**. For demo safety we can also have one client be a script that publishes a pre-recorded audio file as a track.
- **Topic strategy:** pre-pick a topic, curate ~20–30 verified facts into the KB. Demo hot path stays local. Antigravity research is the stretch / cuttable.
- **Agent presence:** silent — subscribes to room audio, renders everything to a web dashboard. No TTS, no chat messages back into the room.

**What makes this win** (per `claude.md` §0, preserved):
1. **Persistent memory that survives restart** — cold run is slow & sparse, warm run is instant. The visible cold→warm delta is the proof of "continual learning."
2. **Cross-speaker and same-speaker contradiction detection.** With two LiveKit participants, the showstopper becomes: "Speaker A said 12% at 02:14, Speaker B just claimed 18%" — live, on screen, with timestamps and source attribution.

Two metrics must be live and must move during the demo:
- `coverage` = check-worthy claims caught / present (↑)
- `mean_time_to_verdict_ms` (↓ as memory fills)

---

## System design

### High-level dataflow

```
 Speaker A client ──┐
                     ├──► LiveKit room ──► Veritas agent (LiveKit Agents SDK, Python)
 Speaker B client ──┘                              │
                                                   ▼
                          ┌────────────────────────────────────────────────┐
                          │  Per-participant audio track subscriber         │
                          │  (one Gemini Live session per speaker)          │
                          └───────────────┬────────────────────────────────┘
                                          ▼ {speaker_id, text, ts}
                          ┌────────────────────────────────────────────────┐
                          │  Sentence segmenter (finalized utterances only) │
                          └───────────────┬────────────────────────────────┘
                                          ▼
                          ┌────────────────────────────────────────────────┐
                          │  Flash detector: strict-JSON check-worthy +     │
                          │  normalized {subject, predicate, value, unit}   │
                          └───────────────┬────────────────────────────────┘
                                          ▼
                          ┌────────────────────────────────────────────────┐
                          │  Verifier (HOT PATH, LOCAL)                     │
                          │   embed claim → vector search KB + memory       │
                          │   ├─ HIT   → verdict <100ms                     │
                          │   └─ MISS  → status="researching" + enqueue     │──┐
                          └───────────────┬────────────────────────────────┘  │ async
                                          ▼                                    ▼
                          ┌──────────────────────────┐         ┌──────────────────────────┐
                          │  Contradiction checker    │         │  Research worker (STRETCH)│
                          │  new vs (session ∪ memory)│         │  Antigravity / Interactions│
                          │  same-speaker + cross-spk │         │  → write VerifiedFact     │
                          └───────────────┬──────────┘         │  → push claim update      │
                                          ▼                     └─────────────┬────────────┘
                          ┌──────────────────────────┐                        │
                          │  Metrics tracker          │◄───────────────────────┘
                          └───────────────┬──────────┘
                                          ▼
                          ┌──────────────────────────┐
                          │  FastAPI + WebSocket → UI │
                          │  claim cards, contradictions, 2 metrics, COLD/WARM badge
                          └──────────────────────────┘
```

### Key adaptations from `claude.md`

| Area | claude.md (single clip) | This plan (2-participant LiveKit) |
|---|---|---|
| Audio source | one published track | one Gemini Live session **per LiveKit participant**; speaker attribution = LiveKit `identity` |
| Claim schema | `Claim` | add `speaker_id: str` (participant identity) |
| Contradiction | within one session | tag as `same_speaker` vs `cross_speaker`; cross-speaker is the loud UI moment |
| Demo robustness | play a clip | `scripts/play_clip.py` joins the room as a synthetic second participant playing a pre-recorded audio file — works as both a dev harness and a stage-safe fallback |

### Repository layout

```
adapters/
  livekit_audio.py     # join room, multi-participant subscriber, yields (speaker_id, pcm_frame)
  gemini_live.py       # per-speaker session: pcm → finalized text segments
  gemini_flash.py      # sentence → strict-JSON {is_checkworthy, subject, predicate, value, unit}
  gemini_embed.py      # text → vector (batched)
  antigravity.py       # claim → researched verdict (STRETCH; stub by default)

core/
  schemas.py           # Claim (with speaker_id), VerifiedFact, Contradiction, SessionMetrics
  memory.py            # SQLite + sqlite-vec store; put/get/vector_search/facts_by_subject/touch
  verifier.py          # hot-path lookup, hit/miss decision
  contradiction.py     # session + memory comparison; same_speaker vs cross_speaker
  research_queue.py    # async miss-resolution worker (STRETCH)
  metrics.py           # coverage + rolling mean_time_to_verdict
  orchestrator.py      # wires transcript → detect → verify → contradiction → emit

kb/
  build_kb.py          # offline: yaml → embed → write
  demo_topic_facts.yaml

server/
  app.py               # FastAPI + WebSocket
  static/index.html    # claim cards, contradiction banner, 2 metrics, COLD/WARM badge

scripts/
  play_clip.py         # publish a recorded audio file into the room as a participant
  reset_memory.py      # COLD
  seed_memory.py       # WARM
  demo_cold_vs_warm.md # runbook

tests/
  test_contradiction.py
  test_memory_persistence.py    # survives process restart
  test_verifier_latency.py
```

### Schemas (delta from `claude.md` §4)

Add to `Claim`:
- `speaker_id: str` — LiveKit participant identity ("speaker_a", "speaker_b", or human-readable).

Add to `Contradiction`:
- `kind: "same_speaker" | "cross_speaker"`
- `speaker_a_id: str`, `speaker_b_id: str`

### External APIs — wrap, do not inline

Every Gemini / LiveKit / Antigravity call lives behind a thin function in `adapters/` so the bleeding-edge signatures churn in **one file**. Verify exact IDs against the docs listed in `claude.md` §1 before writing each adapter; do not invent method names.

---

## Phases

After **Phase 2** we have a demoable fallback. Phases 3–4 are the moat (never cut). Phase 6 (Antigravity) is the designated cut if behind.

### Phase 0 — Scaffold  *(cut: n/a)*

**Write:**
- `requirements.txt`: `livekit-agents`, `google-genai`, `fastapi`, `uvicorn`, `websockets`, `pydantic`, `sqlite-vec`, `numpy`, `pyyaml`, `python-dotenv`.
- `.env.example`: per `claude.md` §7, plus `LIVEKIT_ROOM_NAME=veritas-demo`.
- `config.py`: loads env, exposes model name constants and thresholds (`SIM_THRESHOLD`, `SUBJECT_MATCH_THRESHOLD`, `VALUE_TOLERANCE`).
- `core/schemas.py`: full dataclasses incl. `speaker_id` on `Claim` and `kind` on `Contradiction`.
- Stub every `adapters/*.py` (return canned data) and every `core/*.py` (no-ops) so imports succeed.
- `server/app.py`: FastAPI app with one WebSocket endpoint `/ws` that pushes a fake claim every 2s.
- `server/static/index.html`: minimal page with an event log and the two metric counters.

**Accept:** `uvicorn server.app:app` serves the UI; fake events stream in over WS.

### Phase 1 — Audio spine: multi-participant LiveKit → Gemini Live  *(cut: never)*

**Write:**
- `adapters/livekit_audio.py`: `async def run_agent(room_name, on_audio_frame)` using `livekit.agents` — joins the room as `veritas-agent`, subscribes to every remote participant's audio track, calls `on_audio_frame(speaker_id, pcm_frame)`. Use participant `identity` as `speaker_id`.
- `adapters/gemini_live.py`: `class LiveSession` — opens one Gemini Live WebSocket per speaker. `feed_audio(pcm)` in, `async for segment in session.segments()` out. Yields `{text, is_final, ts}`.
- `scripts/play_clip.py`: standalone script that joins the LiveKit room as a chosen identity and publishes a local `.wav` as an audio track. Doubles as dev harness — run twice with different identities for the "debate" demo.
- `core/orchestrator.py` (v1): wire `livekit_audio` → per-speaker `LiveSession` → push finalized segments to the WS.

**Accept:** Two clients speak/play into the room; finalized sentences from both stream into the UI tagged with the speaker identity.

### Phase 2 — Detection + curated KB + fast verdicts  *(cut: never)*

**Write:**
- `adapters/gemini_flash.py`: `async def detect(sentence) -> dict`. Strict JSON: `{is_checkworthy, subject, predicate, value, unit}`. Canonical, stable subject phrasing. Retry once on JSON parse failure, then drop.
- `adapters/gemini_embed.py`: `async def embed(text) -> list[float]` with in-process LRU cache.
- `kb/demo_topic_facts.yaml`: ~20–30 hand-curated verified facts for the chosen topic.
- `kb/build_kb.py`: read yaml → embed → write `VerifiedFact` rows into the store.
- `core/verifier.py`: `async def verify(claim) -> (claim, hit)` per `claude.md` §5.1. Hit: compare values with `VALUE_TOLERANCE`. Miss: status="researching", enqueue.
- `core/orchestrator.py` (v2): segment → detect → verify → push claim card to UI.

**Accept:** Speak a known check-worthy claim → ✓/✗ card appears within ~1.5s, source="kb".

### Phase 3 — Persistent memory + embeddings  *(cut: never — this is the moat)*

**Write:**
- `core/memory.py`: SQLite at `MEMORY_DB_PATH`. Table `verified_facts(...)` + `sqlite-vec` virtual table for embeddings. `put`, `get`, `vector_search`, `facts_by_subject`, `touch`. Numpy fallback if `sqlite-vec` unavailable.
- Hook verifier to write resolved facts back to memory; `touch` increments `times_seen` on hits.
- `scripts/reset_memory.py`, `scripts/seed_memory.py`.
- `tests/test_memory_persistence.py`: write N facts, close, reopen, assert all present and vector search still returns them. **Blocks Phase 4 if it fails.**
- `tests/test_verifier_latency.py`: pre-seed memory, run 100 claims, assert p50 < 100ms.

**Accept:** Persistence test passes. Second run of same audio shows lower `mean_time_to_verdict_ms`.

### Phase 4 — Contradiction detection (multi-speaker)  *(cut: low — this is the demo moment)*

**Write:**
- `core/contradiction.py`: `check_contradiction(new_claim) -> Contradiction | None`. Subject match via embedding cosine ≥ `SUBJECT_MATCH_THRESHOLD`; value conflict beyond tolerance. Tag `kind`:
  - `same_speaker` if speaker_ids equal (changed story)
  - `cross_speaker` if different (disagreement)
- Scripted demo audio with: 1 same-speaker contradiction, 1 cross-speaker, 1 KB-conflict.
- `tests/test_contradiction.py`: unit tests for `same_subject`, `values_conflict`, all three kinds.

**Accept:** Each scripted contradiction fires a distinct, loud banner in the UI with both speaker IDs, both timestamps, and both values.

### Phase 5 — Metrics + cold/warm runbook  *(cut: low)*

**Write:**
- `core/metrics.py`: `SessionMetrics`, recomputed on every claim event, pushed to WS.
- `server/static/index.html`: two big counters, COLD/WARM badge from `DEMO_MODE`, "memory size: N facts" indicator.
- `scripts/demo_cold_vs_warm.md`: exact stage runbook.

**Accept:** Both metrics move during a run. Cold→warm runbook produces a repeatable delta.

### Phase 6 — Async research via Antigravity / Interactions  *(cut: HIGH — cut first if behind)*

**Write:**
- `adapters/antigravity.py`: `async def research(claim) -> ResearchResult`. Wraps the Interactions API call to the hosted Antigravity agent. Persist environment ID for follow-up calls. Verify exact method names against docs at hack-time.
- `core/research_queue.py`: asyncio queue + worker. On result: build `VerifiedFact`, `memory.put`, update claim (`⏳ → ✓/✗`), push WS update, record latency.
- Wire `verifier.verify` MISS path to enqueue.
- `USE_ANTIGRAVITY=false` short-circuits to demo without it.

**Accept:** One rehearsed off-KB claim resolves live; replaying makes it an instant hit.

### Phase 7 — UI polish for the live catch  *(cut: medium)*

**Visual style (locked):**
- Palette: **pastel orange + white**. Background white, accents pastel orange (`#FFD6A5` / `#FFB07A` range). No Claude/Anthropic-style look.
- Typography: **black text on white** everywhere. No light-on-dark.
- Buttons: **solid pastel fills, no translucency**. No glassmorphism, no backdrop blur. Crisp 1–2px borders.
- Contradiction banner: solid pastel orange background, black text, hard edges.
- Speaker chips: distinct solid pastels per speaker, black text.

**Write:**
- `server/static/index.html` (+ inline `style`, no CSS framework): claim cards with status states (`detected` → `researching` → `verified`/`flagged`), pastel speaker chips, prominent contradiction banner, big metric counters, COLD/WARM badge.
- Minimal one-screen layout. No "dashboard as main feature" (disqualifying per hackathon rules §5).

**Accept:** A first-time viewer instantly understands what happened when a contradiction banner fires.

### Phase 8 — Rehearse + record  *(cut: never)*

**Write:**
- 30–60s prepared audio (or two clips) with: 1 same-speaker contradiction, 1 cross-speaker, 1 KB-conflict, 3+ KB hits, optionally 1 off-KB miss for Antigravity.
- Rehearse end-to-end 5+ times. Pre-warm Gemini sessions.
- Record 1-min submission video.
- Demo runs with WiFi disabled except the two pre-warmed Gemini calls.

**Accept:** Repeatable demo; submission video recorded; repo public; all team members on submission.

---

## Files (critical)

- `core/schemas.py`, `core/memory.py`, `core/verifier.py`, `core/contradiction.py`, `core/metrics.py`, `core/orchestrator.py`
- `adapters/livekit_audio.py`, `adapters/gemini_live.py`, `adapters/gemini_flash.py`, `adapters/gemini_embed.py`, `adapters/antigravity.py`
- `kb/build_kb.py`, `kb/demo_topic_facts.yaml`
- `server/app.py`, `server/static/index.html`
- `scripts/play_clip.py`, `scripts/reset_memory.py`, `scripts/seed_memory.py`, `scripts/demo_cold_vs_warm.md`
- `tests/test_memory_persistence.py`, `tests/test_verifier_latency.py`, `tests/test_contradiction.py`
- `requirements.txt`, `.env.example`, `config.py`

## Verification

End-to-end smoke flow after each phase:

1. `python -m scripts.reset_memory && python -m scripts.seed_memory`
2. `uvicorn server.app:app` in one terminal; LiveKit agent process in another
3. `python -m scripts.play_clip --identity speaker_a --file demo_a.wav` and `--identity speaker_b --file demo_b.wav`
4. Open `http://localhost:8000`

Automated:
- `pytest tests/test_memory_persistence.py` after Phase 3
- `pytest tests/test_verifier_latency.py` (p50 < 100ms) after Phase 3
- `pytest tests/test_contradiction.py` after Phase 4

Demo proof: run the cold→warm runbook and screenshot the two metrics before and after. The visible delta is the continual-learning evidence.

---

## Handoff — context a future session needs

### Hackathon logistics

- **Event:** AI Engineer World's Fair Hackathon 2026 (Cerebral Valley)
- **Venue:** Shack15, Ferry Building, 2nd floor, San Francisco
- **Dates:** Saturday June 27 – Sunday June 28, 2026
- **Submission deadline:** Sunday June 28, **12:00 PM** (Pacific). First-round judging 12:30 PM, finals 2:00 PM, winners 3:15 PM.
- **Demo format:** ~3 min live demo + 1–2 min Q&A. Submission video is **1 minute**, build-only.
- **WiFi at venue:** `SHACK15_Members` / `M3mb3r$4L!f3` — assume saturated; design hot path to not depend on it.

### Tracks we are targeting

- **Theme (required):** **Continual Learning** — "agents/harnesses/frameworks that allow continual learning through memory, user feedback, prompt optimization, self-reflection, toolkit expansion." Our memory layer (Phase 3) is the qualifying mechanism.
- **Prize 1: Best Usage of Gemini 3.5** ($5,000 cash). Need at least one of: Managed Agents/Interactions API (Antigravity), Computer Use in Flash, Gemini Live Translate, Nano Banana / Gemma 4. We use Gemini Live (transcription), Gemini 3.5 Flash (detection), Gemini embeddings, and **Antigravity via Interactions API** (Phase 6 = the multi-feature combo for bonus points).
- **Prize 2: Best Usage of LiveKit** (Keychron Q3 Max keyboards). Multi-participant LiveKit room with a Python LiveKit Agent subscribing to per-participant tracks is the qualifying integration.

### Hard rules (banned / disqualifying)

From hackathon rules §5 — must avoid:
- "Any project where a **dashboard is the main feature**." Our UI exists to *frame the catch*, not to be the product. Keep it minimal one-screen. If a judge calls it a dashboard, we lose.
- Basic RAG apps, Streamlit apps, image analyzers, etc. We are not those.
- **Open source required** (repo public). **New work only** — judges must see what was built this weekend.

### Non-negotiables from `claude.md` §0 (DO NOT break)

1. Persistent memory survives restart (the moat — Phase 3).
2. Cross-session / cross-speaker contradiction detection (Phase 4).
3. **Demo hot path must not touch the network** except the two Gemini calls (Live + Flash), which are pre-warmed.
4. Demo input is **chosen in advance** (we control it). Not live political/news content.
5. **Text/visual output only.** No TTS, no synthesized voice. LiveKit is input-only.

### Decisions already locked (do not re-litigate)

- Speaker setup: two remote LiveKit participants with distinct `identity` strings — no diarization.
- Topic: pre-pick + curated KB (~20–30 facts). Topic-agnostic mode rejected.
- Agent presence: silent (no LiveKit chat/TTS back into room). Web dashboard only.
- UI palette: pastel orange + white, black-on-white text, **no translucent buttons**, no Claude-style look. See Phase 7.
- Memory backend: **local SQLite + `sqlite-vec`** with numpy fallback. MongoDB Atlas explicitly rejected for the hot path (network).
- MiniMax: **not used**. No load-bearing role; forcing it adds risk.
- DigitalOcean: optional deploy target at the very end if time remains. Demo runs locally regardless.

### Cut order if behind

1. **Cut first:** Phase 6 Antigravity research. Demo runs fully without it. Decide by ~9 AM Sunday.
2. **Never cut:** Phase 1 (audio spine), Phase 2 (detection + KB), Phase 3 (persistent memory), Phase 4 (contradiction), Phase 8 (rehearse + record). These ARE the project.

### Repo / environment

- **Working dir:** `/Users/anish/Code/AIEWF-Hackathon-2026`
- **Python venv:** `./venv` (activate with `source venv/bin/activate`)
- **Git:** `main` branch. Last commit at handoff: `e49b344 resetting everything`. Most recent meaningful prior work: `8482322 phase 0 executed. voice to agent` and `7b80e53 final version` (from an earlier reset).
- **Source-of-truth files:**
  - `claude.md` — original detailed spec (kept for reference; section §1 has API doc links, §7 has env vars, §8 has risk register).
  - `PLAN.md` (this file) — adapted plan with multi-speaker LiveKit + locked decisions. **Read this first.**
- **`.gitignore`:** `venv`, `.DS_Store`, `**/.env.local`, `**/__pycache__`. Memory DB (`*.db`) should be added before Phase 3 commits.

### Required env vars (`.env.example`)

```
GEMINI_API_KEY=
GEMINI_LIVE_MODEL=          # verify exact id in Live API docs
GEMINI_FLASH_MODEL=         # verify exact id
GEMINI_EMBED_MODEL=         # verify exact id
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_ROOM_NAME=veritas-demo
ANTIGRAVITY_MODEL=antigravity-preview-05-2026   # verify; STRETCH only
SIM_THRESHOLD=0.82
SUBJECT_MATCH_THRESHOLD=0.85
VALUE_TOLERANCE=0.0
MEMORY_DB_PATH=./veritas_memory.db
DEMO_MODE=warm              # warm = seeded memory; cold = empty
USE_ANTIGRAVITY=false       # flip true only when Phase 6 is reliable
```

### API documentation — verify before writing any adapter

Bleeding-edge SDKs at the time of this hackathon. **Do not invent method names.** Wrap each call in `adapters/` so signature churn touches one file.

- Interactions API (Antigravity): https://ai.google.dev/gemini-api/docs/get-started
- Live API (Live Translate): https://ai.google.dev/gemini-api/docs/live-api/live-translate
- What's new in Gemini 3.5 Flash: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
- Gemma (if ever needed): https://ai.google.dev/gemma/docs/get_started
- Gemini API keys: https://aistudio.google.com/api-keys
- Antigravity download: https://antigravity.google/download
- LiveKit hackathon resources: https://www.livekit.info/aiewf-hackathon-2026
- LiveKit docs are also available via the `livekit-docs` MCP server in this Claude Code session — use `docs_search` / `code_search` / `get_python_agent_example` before writing the LiveKit adapter.

### Demo storyline (what the 3-minute pitch shows)

1. **Cold run:** reset memory, play the prepared audio. Coverage low at first; mean-time-to-verdict high. Most claims sit in `researching`.
2. **Memory fills:** by the second half of the clip, verdicts are instant; coverage climbs.
3. **Same-speaker contradiction:** Speaker A says "12%" early, "18%" late. Banner fires referencing the earlier timestamp.
4. **Cross-speaker contradiction:** Speaker B asserts X; banner shows A's prior counter-claim. This is the visceral moment.
5. **Restart proof:** kill the process, restart, replay — verdicts are instant from frame 1 because memory persisted. This is the continual-learning evidence.

### Two metrics on screen (must move)

- `coverage` ↑ over the run
- `mean_time_to_verdict_ms` ↓ as memory fills

If these aren't visibly moving on stage, the demo fails its own thesis.

### Submission checklist (Sunday before 12 PM)

- [ ] Repo public on GitHub
- [ ] 1-minute build-only demo video uploaded
- [ ] All team members added on the submission form
- [ ] Demo link / instructions in README
- [ ] Cold→warm runbook captured (screenshots in `scripts/demo_cold_vs_warm.md`)

