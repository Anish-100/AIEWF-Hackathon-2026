# Veritas — Build Spec

> Real-time fact-checking system that **learns the domain of a broadcast as it listens**:
> it gets faster and starts catching contradictions the longer it runs, because it builds a
> persistent verified-fact memory instead of re-checking every claim from scratch.

This document is the implementation spec for Claude Code. Build in the phase order given.
Each phase has **acceptance criteria** — do not move on until they pass. Phases are ordered so
that after Phase 2 you always have a working demo; everything after that increases the score
and is individually cuttable under time pressure (cut priority is marked per phase).

---

## 0. North star & non-negotiables

**The thing that makes this win (and not look like an existing free tool):**
1. **Persistent memory that survives a restart.** Cold run = slow, misses, high latency. Warm run (memory pre-populated) = instant hits, low latency. The cold-vs-warm diff is the proof of "continual learning." If this isn't real and measurable, the project fails its own theme. Build it early (Phase 3) and protect it.
2. **Contradiction detection across the session.** "Speaker said 12% at 02:14, now says 18%" — a stateless checker structurally cannot do this. This is the visceral demo moment.

**Two numbers must be live on screen and must move:**
- `coverage` = check-worthy claims caught / check-worthy claims present (↑ over the run)
- `mean_time_to_verdict_ms` (↓ as memory fills)

**Hard constraints:**
- The **demo hot path must not touch the network.** Shared venue wifi will be saturated. All demo-path verification resolves from local memory / curated KB. The only network calls on the hot path are Gemini Live (transcription) and Flash (detection) — pre-warm and tolerate; everything else is local or async.
- The demo clip is **chosen in advance** (product keynote or earnings call — verifiable, low-stakes claims; NOT live political news). The whole design assumes we control the input.
- Output is **text/visual only** (no synthesized voice out). Voice is input-only via LiveKit.

---

## 1. Tech stack (decisions + rationale)

| Concern | Choice | Why |
|---|---|---|
| Audio ingest + transport | **LiveKit Agents (Python)** | Required for LiveKit track. Handles audio plumbing; subscribe to a track, stream frames out. |
| Transcription (audio→text) | **Gemini Live API** (`gemini-3.5-live` family) | Required for Gemini track; flagship feature. Streaming partials. |
| Claim detection + normalization | **Gemini 3.5 Flash** | Fast, in the hot path. Detects "is this check-worthy?" and extracts a structured claim. |
| Semantic matching / embeddings | **Gemini embeddings** | Match a new claim to stored claims (memory hit?) and to contradiction candidates. Pre-compute KB embeddings offline. |
| Persistent memory (PRIMARY) | **Local: SQLite + `sqlite-vec`** (fallback: numpy cosine over an in-memory list) | Hot-path lookups must be local + offline + sub-100ms. Survives restart = the moat. |
| Async research (STRETCH) | **Gemini Antigravity / Interactions API** | The slow "learning edge": resolve unknown claims in the hosted sandbox, write result back to memory. Combining with Live API = Gemini "bonus points." Cuttable. |
| Frontend | **Minimal: FastAPI + a single static HTML/JS page over WebSocket** | The UI surfaces claim cards, contradiction alerts, and the two metrics. Keep it lean — a dashboard is NOT the product (and "dashboard as main feature" is banned). It exists to show the live catch. |

**Optional / not in critical path:*b*
- **MongoDB Atlas** — a legitimate swap for the memory layer (native vector search, earns the MongoDB resource). BUT it's cloud → puts hot-path lookups on the network. Only use it if you also keep a local cache for the demo, or use it purely as the *persistence backend that syncs to a local read cache*. Default to local SQLite for the demo.
- **DigitalOcean** — optional deploy target at the very end if time remains and you want the DO prize. Demo runs locally regardless.
- **MiniMax** — not used. No load-bearing role here; forcing it adds risk. (First-place prize includes its credits regardless of use.)

> **API VERIFICATION REQUIRED.** Exact signatures for Gemini Live, Flash, embeddings, and Antigravity/Interactions, and for LiveKit Agents, are bleeding-edge — DO NOT invent method names. Verify against:
> - Interactions API: https://ai.google.dev/gemini-api/docs/get-started
> - Live API: https://ai.google.dev/gemini-api/docs/live-api/live-translate
> - What's new in Gemini 3.5 Flash: https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5
> - Gemma (if ever needed): https://ai.google.dev/gemma/docs/get_started
> - API keys: https://aistudio.google.com/api-keys
> - Antigravity: https://antigravity.google/download
> - LiveKit: https://www.livekit.info/aiewf-hackathon-2026
> Wrap each external call in a thin adapter (see `adapters/`) so a signature change touches one file.

---

## 2. Architecture

```
 Demo clip (audio/video file)
        │  (played into a LiveKit room as a published track)
        ▼
 ┌─────────────────┐
 │ LiveKit Agent   │  subscribes to the audio track, forwards frames
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Gemini Live API │  streaming transcript (partial + final segments)
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ Flash: detector │  per finalized sentence: check-worthy? → structured claim
 └────────┬────────┘
          ▼
 ┌──────────────────────────────────────────────┐
 │ Verifier (hot path, all LOCAL)               │
 │  embed claim → vector search in memory + KB  │
 │   ├─ HIT  → verdict instantly (<100ms)       │
 │   └─ MISS → status="researching" + enqueue   │──┐ async
 └────────┬─────────────────────────────────────┘  │
          ▼                                          ▼
 ┌─────────────────┐                      ┌─────────────────────┐
 │ Contradiction   │                      │ Research worker      │
 │ checker         │                      │ (Antigravity, STRETCH)│
 │ new vs memory   │                      │ resolve → write back │
 └────────┬────────┘                      │ to memory            │
          │                               └──────────┬──────────┘
          ▼                                          │ updates card
 ┌─────────────────┐                                 │ + memory
 │ Metrics tracker │◄────────────────────────────────┘
 └────────┬────────┘
          ▼
 ┌─────────────────┐
 │ WebSocket → UI  │  claim cards (⏳→✓/✗), contradiction alerts, 2 metrics
 └─────────────────┘
```

The orchestrator wires these as independent workers. **Treat the orchestrator as plumbing, not the pitch** — AutoGen/LangGraph already do orchestration; our story is the persistent measurable improvement.

---

## 3. Repository structure

```
veritas/
  README.md
  PLAN.md                      # this file
  .env.example                 # API keys & config (never commit real keys)
  requirements.txt
  config.py                    # loads env, model names, thresholds

  adapters/                    # thin wrappers around each external API (isolate signature churn)
    livekit_audio.py           # join room, subscribe to audio track, yield frames
    gemini_live.py             # frames → transcript segments (async generator)
    gemini_flash.py            # sentence → {is_checkworthy, normalized_claim}
    gemini_embed.py            # text → vector
    antigravity.py             # claim → researched verdict (STRETCH; stub-able)

  core/
    schemas.py                 # dataclasses / pydantic models (section 4)
    memory.py                  # persistent store: get/put facts, vector search, restart-safe
    verifier.py                # hot-path lookup, hit/miss decision
    contradiction.py           # new-claim vs memory comparison
    research_queue.py          # async miss-resolution worker
    metrics.py                 # coverage + time-to-verdict, rolling
    orchestrator.py            # wires transcript → detect → verify → contradiction → emit

  kb/
    build_kb.py                # offline: curate demo-clip facts → embed → write to store
    demo_clip_facts.yaml       # hand-curated verified facts for the chosen clip

  server/
    app.py                     # FastAPI + WebSocket; serves UI, streams events
    static/index.html          # minimal claim-card UI + metrics + cold/warm toggle

  scripts/
    play_clip.py               # publish the demo clip into a LiveKit room
    reset_memory.py            # wipe memory → COLD state
    seed_memory.py             # load curated KB → WARM state
    demo_cold_vs_warm.md       # runbook for the restart proof

  tests/
    test_contradiction.py
    test_memory_persistence.py # asserts memory survives a process restart
    test_verifier_latency.py
```

---

## 4. Data models (`core/schemas.py`)

```python
# Claim: one detected statement in the stream
Claim:
    id: str                    # uuid
    session_id: str
    clip_ts: float             # seconds into the clip
    raw_text: str              # the sentence as transcribed
    subject: str               # normalized entity, e.g. "Q3 2025 revenue"
    predicate: str             # e.g. "equals" / "grew by"
    value: str | float | None  # e.g. 12 ; unit captured separately
    unit: str | None           # e.g. "%", "USD_millions"
    embedding: list[float]
    status: str                # "detected" | "researching" | "verified" | "flagged"
    verdict: str | None        # "true" | "false" | "unverifiable" | "dubious"
    confidence: float | None
    source: str | None         # "kb" | "memory" | "web"
    explanation: str | None
    detected_at: float         # wall clock
    resolved_at: float | None
    time_to_verdict_ms: int | None

# VerifiedFact: the PERSISTENT memory unit (survives restart)
VerifiedFact:
    id: str
    claim_key: str             # canonical key from (subject, predicate)
    subject: str
    canonical_value: str | float
    unit: str | None
    verdict: str
    source: str                # "kb" | "web"
    explanation: str
    embedding: list[float]
    first_seen_ts: float
    times_seen: int

# Contradiction
Contradiction:
    id: str
    subject: str
    claim_a_id: str; value_a; ts_a
    claim_b_id: str; value_b; ts_b
    explanation: str

# SessionMetrics (recomputed, pushed to UI)
SessionMetrics:
    checkworthy_seen: int
    checked: int               # got a verdict (hit or resolved)
    coverage: float            # checked / checkworthy_seen
    mean_time_to_verdict_ms: float
    memory_hits: int
    memory_misses: int
    contradictions: int
```

Normalization is what makes matching and contradiction detection work. **Flash must return a structured claim**, not just a boolean. Prompt it to output strict JSON: `{is_checkworthy, subject, predicate, value, unit}`. `subject` should be canonical and stable ("Q3 2025 revenue", not "their revenue last quarter") so the same fact across the clip maps to the same key.

---

## 5. Key logic (pseudocode for the tricky parts)

### 5.1 Hot-path verify (`core/verifier.py`)
```
def verify(claim):
    claim.embedding = embed(claim.raw_text or claim.subject)
    candidates = memory.vector_search(claim.embedding, top_k=5)
    best = best_match(candidates, claim)        # cosine >= SIM_THRESHOLD AND subject match
    if best:
        claim.verdict   = verdict_from(best, claim)   # compare claim.value to best.canonical_value
        claim.source    = best.source
        claim.confidence= best.score
        claim.status    = "verified"
        memory.touch(best)                       # times_seen += 1
        record_latency(claim)                    # this is a HIT → low latency
        return claim, hit=True
    else:
        claim.status = "researching"             # MISS
        research_queue.enqueue(claim)            # async; resolves later
        return claim, hit=False
```
`verdict_from`: if the claim's value matches the canonical value → `true`; if it conflicts → `false` (and this is *also* a contradiction signal vs ground truth); if no comparable value → `unverifiable`.

### 5.2 Contradiction check (`core/contradiction.py`)
```
def check_contradiction(new_claim):
    # search BOTH this session's prior claims AND persistent memory
    prior = session_claims_by_subject(new_claim.subject) + memory.facts_by_subject(new_claim.subject)
    for p in prior:
        if same_subject(p, new_claim) and values_conflict(p.value, new_claim.value, p.unit, new_claim.unit):
            emit Contradiction(subject=new_claim.subject,
                               value_a=p.value, ts_a=p.clip_ts,
                               value_b=new_claim.value, ts_b=new_claim.clip_ts,
                               explanation=f"Earlier stated {p.value}{p.unit} at {fmt(p.ts_a)}, now {new_claim.value}{new_claim.unit}")
```
`same_subject` = embedding cosine over `subject` ≥ threshold (handles paraphrase). `values_conflict` = normalized numeric/string mismatch beyond tolerance. **This is the demo moment — make the emitted alert loud in the UI.**

### 5.3 Async research worker (`core/research_queue.py`, STRETCH)
```
async worker():
    while True:
        claim = await queue.get()
        result = await antigravity.research(claim)   # browses web in hosted sandbox
        fact = VerifiedFact(from result)
        memory.put(fact)                              # NOW future identical claims are instant
        claim.update(verdict=result.verdict, source="web", status="verified")
        record_latency(claim)                         # high latency (this was the learning event)
        push_update(claim)                            # UI: ⏳ → ✓/✗
```
This worker is the *only* place the open web is touched, and it's off the hot path. If cut, misses simply stay `"unverifiable"` and the demo runs entirely off curated KB + memory.

### 5.4 Cold / warm (the proof)
- `scripts/reset_memory.py` → empties the persistent store (COLD).
- `scripts/seed_memory.py` → loads curated KB + any facts learned in a prior run (WARM).
- `tests/test_memory_persistence.py` → write facts, kill process, reopen store, assert facts present. **If this test doesn't pass, the continual-learning claim is fake.**

---

## 6. Build phases

> After **Phase 2** you have a demoable system. Phases 3–4 are the moat (highest score, do not skip lightly). Phases 5–7 make it legible and live. Phase 6 (Antigravity) is the designated cut if behind.

### Phase 0 — Scaffold  (cut priority: n/a)
- Repo structure, `requirements.txt`, `.env.example`, `config.py`.
- Stub every adapter so the app imports and runs with fakes.
- **Accept:** `python -m server.app` serves the empty UI; fake transcript events appear in the UI over WebSocket.

### Phase 1 — Audio spine  (cut priority: never)
- `adapters/livekit_audio.py`: join room, subscribe to audio track, yield frames.
- `scripts/play_clip.py`: publish the demo clip file into the room.
- `adapters/gemini_live.py`: frames → streaming transcript segments.
- **Accept:** play the clip → finalized sentences stream into the UI in near-real-time. Nothing clever yet.

### Phase 2 — Detection + curated KB + fast verdicts  (cut priority: never)
- `adapters/gemini_flash.py`: sentence → strict-JSON structured claim (section 4).
- `kb/demo_clip_facts.yaml` + `kb/build_kb.py`: curate ~15–30 verified facts for the chosen clip, embed, load into the store.
- `core/verifier.py` hot path against the KB.
- **Accept:** clip plays → check-worthy claims detected → known claims show ✓/✗ cards within ~1.5s. This is your fallback demo. Protect it.

### Phase 3 — Persistent memory + embeddings  (cut priority: never — this is the moat)
- `core/memory.py`: SQLite + `sqlite-vec` (fallback numpy). `put`, `get`, `vector_search`, `facts_by_subject`, `touch`. Writes to disk.
- Verifier writes resolved facts back to memory; repeat claims become hits.
- **Accept:** `tests/test_memory_persistence.py` passes (survives restart). Second play of the clip is measurably faster than first (mean_time_to_verdict drops).

### Phase 4 — Contradiction detection  (cut priority: low — this is the demo moment)
- `core/contradiction.py` per section 5.2.
- Scripted clip contains an internal contradiction (e.g., 12% early, 18% late).
- **Accept:** the contradiction fires a distinct, loud UI alert referencing the earlier timestamp, live.

### Phase 5 — Metrics + cold/warm runbook  (cut priority: low)
- `core/metrics.py`: coverage + rolling mean_time_to_verdict; push on every event.
- `scripts/reset_memory.py`, `scripts/seed_memory.py`, `scripts/demo_cold_vs_warm.md`.
- **Accept:** UI shows both metrics moving; you can run a clean COLD→WARM comparison on command.

### Phase 6 — Async research via Antigravity  (cut priority: HIGH — cut this first if behind; decide by ~9am Sun)
- `adapters/antigravity.py` + `core/research_queue.py` per section 5.3.
- Misses resolve in the background and write to memory ("⏳ → ✓, now permanent").
- **Accept:** one rehearsed claim, unknown at start, gets researched live, resolves, and is instant on a replay. If this can't be made reliable, leave it stubbed — system still fully demos.

### Phase 7 — UI polish for the live catch  (cut priority: medium)
- Claim cards with status transitions; prominent contradiction banner; the two metrics as big counters; a COLD/WARM indicator.
- Keep it minimal and legible — not a "dashboard." It frames the catch; it is not the product.
- **Accept:** a first-time viewer instantly understands what just happened when a contradiction fires.

### Phase 8 — Rehearse + record  (cut priority: never)
- Run the clip 5+ times. Pre-cache the demo path so no live web call is needed on stage.
- Record the 1-min build-only demo video. Repo public. All members on the submission.
- **Accept:** the demo runs end-to-end with wifi disabled (except the two unavoidable Gemini calls, which you've pre-warmed/tolerated).

---

## 7. Config (`.env.example`)

```
GEMINI_API_KEY=
GEMINI_LIVE_MODEL=        # verify exact id in the Live API docs
GEMINI_FLASH_MODEL=       # verify exact id
GEMINI_EMBED_MODEL=       # verify exact id
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
ANTIGRAVITY_MODEL=antigravity-preview-05-2026   # verify; STRETCH only
SIM_THRESHOLD=0.82        # tune: cosine for "same fact"
SUBJECT_MATCH_THRESHOLD=0.85
VALUE_TOLERANCE=0.0       # exact for %/$ unless you want fuzzy
MEMORY_DB_PATH=./veritas_memory.db
DEMO_MODE=warm            # warm = seeded memory; cold = empty
USE_ANTIGRAVITY=false     # flip true only when Phase 6 is reliable
```

---

## 8. Risk register & cut order

1. **Cut first if behind:** Antigravity research (Phase 6). System demos fully without it.
2. **Never cut:** audio spine (1), detection+KB (2), persistent memory (3). These ARE the project.
3. **Latency:** if Flash detection is slow, batch on finalized sentences only (not partials), and cache embeddings for repeat subjects.
4. **Wifi failure on stage:** demo path is local; pre-record a backup video (Phase 8) as the ultimate fallback.
5. **"Isn't this a stateless free fact-checker?"** → it is, for the first 30 seconds; then show cold-vs-warm + the contradiction catch, which a stateless tool cannot do.
6. **"Is it learning or just a longer context window?"** → it persists to disk and survives a restart; show the same clip cold vs warm.

---

## 9. Definition of done (MVP)

- [ ] Clip plays → live transcript → check-worthy claims detected.
- [ ] Known claims verified from memory/KB in <1.5s, shown as cards.
- [ ] Memory persists to disk and survives a process restart (test passes).
- [ ] Second run of the clip is measurably faster than the first.
- [ ] An internal contradiction fires a loud, timestamped live alert.
- [ ] Coverage and mean-time-to-verdict are live on screen and move correctly.
- [ ] Whole demo runs with the network off except the two pre-warmed Gemini calls.
- [ ] Repo public; 1-min build-only video recorded; all members on submission.