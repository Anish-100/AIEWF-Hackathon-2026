# Cold → Warm demo runbook

The proof that "this is continual learning, not just a longer context window."

We run the **same** prepared speech twice. First time the memory is empty (COLD)
— most claims are MISSes, `coverage` is low, `mean_time_to_verdict_ms` is high.
Second time the memory is seeded (WARM) — claims are instant hits.

The two on-screen metrics *must* move visibly:

- **Coverage** ↑ (more of the spoken claims get a verdict)
- **Mean time-to-verdict** ↓ (warm hits are sub-100 ms vs cold misses sitting in `researching`)

---

## Before the demo (5 minutes)

1. **Plug in.** Power, ethernet if available, mic if using one.
2. **Activate the venv** and confirm creds:
   ```bash
   cd /Users/anish/Code/AIEWF-Hackathon-2026
   source venv/bin/activate
   python -c "import config; assert all([config.GEMINI_API_KEY, config.LIVEKIT_URL, config.LIVEKIT_API_KEY, config.LIVEKIT_API_SECRET]); print('env ok')"
   ```
3. **Mint two LiveKit tokens** (one per debater). Keep both pasted somewhere fast to grab:
   ```bash
   lk token create --url wss://aiewf-2026-8kx5874s.livekit.cloud \
     --api-key APIPdTcJmrzRmRG --api-secret RXGfmvwKRb81x2ldfbo41CyerHadErxaSUOHhUq0UEOB \
     --identity alex --room veritas-demo --join --valid-for 2h
   lk token create --url wss://aiewf-2026-8kx5874s.livekit.cloud \
     --api-key APIPdTcJmrzRmRG --api-secret RXGfmvwKRb81x2ldfbo41CyerHadErxaSUOHhUq0UEOB \
     --identity bob   --room veritas-demo --join --valid-for 2h
   ```
4. **Open two LiveKit Meet tabs** (https://meet.livekit.io), one per token. **Do not mute** — and confirm the lock icon is OFF (E2EE would block our agent).
5. **Open the dashboard** in a third tab: `http://localhost:8000`. Big screen if possible.

## The cold pass (~60s)

1. **Reset memory.** Empty DB → COLD.
   ```bash
   python -m scripts.reset_memory
   ```
2. **Start the server** in another terminal.
   ```bash
   uvicorn server.app:app --reload
   ```
   In the log you should see `lifespan: starting orchestrator (LiveKit + Gemini Live)` and `connected to room=veritas-demo`.
3. **Screenshot the metrics panel.** Coverage: 0% · TTV: – ms · Memory: 0.
4. **Speak the script** (slowly, one sentence at a time, ~2 s pause between):
   - **alex**: "US unemployment in May 2025 was four point one percent."
   - **alex**: "Nonfarm payrolls grew by two hundred twenty eight thousand in March 2025."
   - **bob**: "Healthcare added fifty four thousand jobs in March 2025."
   - **bob**: "Labor force participation in June 2025 was sixty two point three percent."
5. Watch the cards land. With cold memory most of these go to **⌛ RESEARCHING** (no KB to match against). The metrics show `coverage` climbing slightly but `mean_time_to_verdict_ms` will be the embedding round-trip plus zero KB hits.
6. **Screenshot.** Cold-state metrics snapshot.

## The warm pass (~60s)

1. **Seed memory** without restarting the server (the orchestrator re-reads memory on every claim):
   ```bash
   python -m scripts.seed_memory
   # → "seeded 26 facts (WARM)"
   ```
2. The dashboard's **memory badge** should jump to `memory: 26` within a couple seconds (next claim publishes a fresh size). If you want the badge instant: restart the server (`Ctrl-C`, then `uvicorn server.app:app --reload`).
3. **Speak the same script again** — same sentences, same speakers, same order.
4. This time every sentence lands as **✓ TRUE** within ~500 ms. `coverage` climbs to near 100%. `mean_time_to_verdict_ms` drops dramatically — cold misses inflated the average; warm hits are sub-second.
5. **Screenshot.** Warm-state metrics snapshot.

## The contradiction beat (the visceral moment, ~15s)

After the warm pass, add two more lines:
- **alex**: "Actually, US unemployment in May 2025 was five point zero percent."
   → ✗ FALSE card **and** orange `⚠ SAME-SPEAKER CONTRADICTION` banner (alex 4.1 → 5.0).
- **bob**: "Nonfarm payrolls only grew by one hundred eighty thousand in March 2025."
   → ✗ FALSE card **and** orange `⚠ CROSS-SPEAKER CONTRADICTION` banner (alex's 228k vs bob's 180k).

`contradictions` counter ticks to 2.

## What to say while showing the screens

> "Here's the same one-minute speech, twice. The first time, memory is empty — coverage 0%, lots of question-marks. We don't kill the process; we just seed memory with our curated KB and replay. Now coverage hits 100% in seconds, and mean time-to-verdict drops from N hundred to under 500ms — because the system has *learned* the domain. And because we now have multiple claims on the same subject in the same session, contradictions surface automatically — both same-speaker ("you said 4.1, now 5.0") and cross-speaker ("alex said 228, bob said 180")."

## If something goes wrong on stage

| Failure | Recovery |
|---|---|
| Card stuck on ⌛ RESEARCHING | Don't panic — Phase 6 (Antigravity) is a stretch and probably not wired. Cold pass is *supposed* to look like this. |
| WiFi flakes mid-pass | Re-run `play_clip.py` with the prerecorded audio file — it publishes the same content into the same room. |
| Banner doesn't fire | Check log for `contradiction (kind):` — if absent, the subject similarity dropped below 0.85; rephrase to match the prior subject more closely. |
| Server crashed | Restart with `uvicorn server.app:app` — memory and seeded facts persist. |
| Mic acting weird | Mute & unmute in Meet; our stall watchdog reopens the Gemini session on the next audio frame (~5s). |

## Definition of "demo succeeded"

- [ ] Cold metrics screenshot shows ~0% coverage / high TTV
- [ ] Warm metrics screenshot shows ~100% coverage / <1 s TTV
- [ ] At least one ✓ TRUE card appeared in the warm pass within 1.5 s
- [ ] At least one ✗ FALSE card appeared
- [ ] Both contradiction banners fired (same-speaker AND cross-speaker)
- [ ] Memory badge incremented visibly between cold and warm
