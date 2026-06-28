# Veritas — Session Handoff

> **Read order for a fresh session:** this file → `PLAN.md` → `claude.md` (original detailed spec).
>
> This file is the "you are here" snapshot. Update it at the end of every session.

## Where we are right now

- **Phase 0 (Scaffold):** ✅ done and verified.
- **Phase 1 (Audio spine):** 🚧 in progress.
  - `core/event_bus.py` ✅ written
  - `adapters/livekit_audio.py` ✅ written (uses `rtc.Room` directly, not the worker framework)
  - `adapters/gemini_live.py` ⏳ next
  - `core/orchestrator.py` v1 ⏳ next
  - `scripts/play_clip.py` ⏳ next
  - `server/app.py` refactor (subscribe to event bus, gate fake events behind env) ⏳ next
- **Phases 2–8:** not started; see `PLAN.md` §Phases.

## What is built and verified

Phase 0 acceptance was confirmed on Sat 2026-06-27:
- `uvicorn server.app:app` boots cleanly
- `GET /` returns the pastel-orange UI (9.1 KB)
- `GET /healthz` → `{"ok":true,"demo_mode":"warm"}`
- `WS /ws` delivers `hello`, `claim`, `metrics` events every ~2s (also `contradiction` every 8s)

UI style is locked: **pastel orange + white, black text on white, no translucent buttons, no Claude-style look.** Do not redesign without permission.

## Key architectural decisions (locked — do not re-litigate)

1. **Speaker setup:** two remote LiveKit participants joining with distinct `identity` strings. No diarization. `speaker_id` = participant identity.
2. **Topic strategy:** pre-pick + curated KB (~20–30 facts). Topic-agnostic was explicitly rejected.
3. **Agent presence:** silent — subscribes to room audio, never publishes back. UI is the only surface.
4. **Memory backend:** local SQLite + `sqlite-vec` (Phase 3). Numpy fallback. **MongoDB Atlas explicitly rejected for the hot path** (network).
5. **Demo hot path must not touch the network** except the two Gemini calls (Live + Flash), pre-warmed.
6. **MiniMax: not used.** Forcing it adds risk with no load-bearing role.
7. **Antigravity research (Phase 6)** is the designated cut if behind. Decide by ~9 AM Sun.

## API signatures we have verified (do not invent your own)

### LiveKit (Python `livekit==1.1.12`, `livekit-agents==1.6.4`)

```python
from livekit import api, rtc

# Mint a token
token = (
    api.AccessToken(api_key, api_secret)
    .with_identity("veritas-agent")
    .with_grants(api.VideoGrants(room="veritas-demo", room_join=True, can_subscribe=True))
).to_jwt()

# Connect
room = rtc.Room()
@room.on("track_subscribed")
def _(track, publication, participant):
    if track.kind == rtc.TrackKind.KIND_AUDIO:
        asyncio.create_task(_pump(participant.identity, track))

await room.connect(url, token, options=rtc.RoomOptions(auto_subscribe=True))

# Pump frames
stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
async for ev in stream:
    frame_bytes = bytes(ev.frame.data)
```

For publishing audio from a file (used by `scripts/play_clip.py`):
```python
source = rtc.AudioSource(sample_rate=SR, num_channels=NC)
track = rtc.LocalAudioTrack.create_audio_track("mic", source)
await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
# Then loop: await source.capture_frame(rtc.AudioFrame(...))
```

### Gemini Live (`google-genai==2.10.0`)

```python
from google import genai
from google.genai import types

client = genai.Client()  # picks up GEMINI_API_KEY
model = "gemini-3.5-live-translate-preview"
config = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)
async with client.aio.live.connect(model=model, config=config) as session:
    # send 16 kHz mono PCM16 chunks:
    await session.send_realtime_input(audio=types.Blob(data=pcm_chunk, mime_type="audio/pcm;rate=16000"))
    # recv:
    async for response in session.receive():
        if response.server_content and response.server_content.input_transcription:
            text = response.server_content.input_transcription.text
```

For our use case we only consume `input_transcription`. The `response_modalities=["AUDIO"]` is required by the live-translate model; we drop the audio.

### Translation/transcription model choice

We use `gemini-3.5-live-translate-preview` even though we don't need translation — it gives us streaming `input_transcription`. If a non-translate live model surfaces with the same transcription field, switch via `GEMINI_LIVE_MODEL` env without touching adapter code.

## Environment

- Working dir: `/Users/anish/Code/AIEWF-Hackathon-2026`
- Python 3.14 in `./venv` (`source venv/bin/activate`)
- Installed deps include: `fastapi 0.138`, `uvicorn 0.49`, `google-genai 2.10`, `livekit 1.1.12`, `livekit-agents 1.6.4`, `livekit-api 1.1.1`, `sqlite-vec 0.1.9`, `numpy 2.5`, `pydantic 2.13`, `pyyaml 6.0`, `websockets 16.0`, `python-dotenv 1.2`
- Env vars: see `.env.example`. Real keys go in `.env.local` (gitignored).
- Git: `main` branch. `e49b344 resetting everything` is the last pre-build commit. **No commits made by this build yet.**

## Hackathon clock

- **Submission deadline: Sunday 2026-06-28, 12:00 PM Pacific.**
- First-round judging 12:30 PM. Stage finals 2:00 PM. Winners 3:15 PM.
- Demo format: 3 min live + 1–2 min Q&A. Submission video: **1 minute, build-only**.

## When you sit down, do this first

1. `source venv/bin/activate`
2. `python -c "import config; print(config.LIVEKIT_URL, bool(config.GEMINI_API_KEY))"` — make sure creds are populated in `.env.local`.
3. Skim `PLAN.md` §Phases for current phase's "Write" list.
4. Skim this file's "Where we are right now" to see what's already done.
5. Resume from the first ⏳ item.

## Cut order if behind

1. **Cut first:** Phase 6 (Antigravity research).
2. **Never cut:** Phase 1 (audio spine), Phase 2 (detection + KB), Phase 3 (persistent memory), Phase 4 (contradiction), Phase 8 (rehearse + record).

## Demo storyline (the 3-minute pitch)

1. **Cold run:** reset memory, play prepared audio. Coverage low; mean-time-to-verdict high. Most claims sit in `researching`.
2. **Memory fills:** by second half of clip, verdicts are instant; coverage climbs.
3. **Same-speaker contradiction:** speaker A says "12%" early, "18%" late. Banner fires with earlier timestamp.
4. **Cross-speaker contradiction:** speaker B asserts X; banner shows A's counter-claim. *The* visceral moment.
5. **Restart proof:** kill process, restart, replay — verdicts instant from frame 1. This is the continual-learning evidence.

## Two metrics on screen (must move)

- `coverage` ↑ over the run
- `mean_time_to_verdict_ms` ↓ as memory fills

If they don't visibly move on stage, the demo fails its own thesis.
