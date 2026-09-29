# Streaming over WebSocket

Lesson 31 of `AGENTS.md`. Wire format, turn detection, bounded queue and the uncertainty policy for live turns.


### 31. Streaming: One Speech State Per Turn, Receiver Never Blocks
- Wire format: `struct.pack("<II", channel, sequence)` (8-byte header) + int16 LE
  PCM 16k mono. **Validate `len(data) >= 8` before `unpack_from`** and count
  malformed frames; channel 0 = mic, non-mic packets are skipped WITHOUT
  touching the speech state (a speaker channel must not open/close a mic turn).
- `streaming/turn_detector.py` (stdlib + numpy only) owns the speech state:
  adaptive noise floor (tracks down fast / up slow, clamped), hysteresis
  (start ratio 2.5 / end ratio 1.6 — end must be lower or the detector never
  closes), `start_confirm_frames`, `hangover_ms` close, pre/post-roll padding,
  `min_voiced_ms` drop, `max_turn_sec` force split, and `flush()` for
  disconnect.
- `min_voiced_ms` must count **voiced frames**, not the padded span — pre/post
  roll inflates the span so every click would pass the gate.
- **A single bounded worker queue** (`queue_size`, drop-oldest) plus one serial
  ASR worker task. Running the decode inline blocks packet reception and mixes
  turns; drop-oldest is better than unbounded growth. Counters in `stats()`.
- Timestamps derive from the detector's **sample counters**, never wall clock,
  so back-pressure cannot shift turn times.
- On disconnect: flush the detector tail, `queue.join()` (bounded timeout),
  then a sentinel + worker shutdown — otherwise the last turn is silently lost.
- Live turns have no diarization, so they follow the same policy:
  `speaker=None`, `source="unknown"`, `uncertain=True`,
  `reason="live_turn_unattributed"`. Never emit a placeholder name.
- `asr_mcp/streaming/__init__.py` imports `handle_ws_stream` **lazily** (PEP-562
  `__getattr__`) so `asr_mcp.streaming.turn_detector` is importable without
  FastAPI. Same trick in `asr_mcp/speaker/__init__.py` for `uncertainty.py`
  (torch-free) — do NOT restore eager `from ... import ...` re-exports there.

### Channel-aware live attribution (see [live-client-design.md](live-client-design.md))
- The server now also accepts **turn frames** from the client (magic `LVT1`,
  24-byte header carrying `channel` + `start_sample` + `n_samples`), so the
  client can do its own endpointing and live/offline boundaries match. The
  magic discriminates turn frames from the legacy 8-byte raw-PCM header — no
  config flag, and old clients keep working.
- **Never voiceprint-match the mic channel.** The default input device is the
  local user, who is in their own voiceprints; matching returns a confident
  wrong answer whenever they have more than one profile. The channel *is* the
  evidence → `LOCAL_SPEAKER_LABEL`, `speaker_source="input_device"`.
- Speaker-channel turns are 1–3s, the low end for ECAPA, so
  `live_attribution.min_match_confidence` is 0.60 — deliberately higher than the
  file path's 0.35. Separate config section so live tuning cannot move file
  behaviour.
- `WS /asr/ws/stream` was **completely unauthenticated**. `_websocket_user()`
  now mirrors `get_current_user` (session → `X-API-Key`/`?token=` →
  `is_valid_api_key` → `DEFAULT_USER`) and closes 1008 *before* `accept()` so
  the client sees a 403 rather than a socket that dies immediately. Starlette's
  `SessionMiddleware` does populate `scope["session"]` for websockets — verify
  in its source before changing `_websocket_user`, do not guess a cookie path.
- `language` is threaded into the live `transcribe_audio_sync` call (lesson 28).

### Turn coalescing and the pre-ASR speech gate

Both exist because a live turn arrives too small to be useful, and the failure
was visible as `UNKNOWN: Thank you` in the transcript.

- **Never concatenate audio with `a or b`** when one side is a numpy array —
  it raises `ValueError: truth value of an array ... is ambiguous`. This bit
  `TurnCoalescer._turn_payload` on the first run.
- **Coalesce, don't just lengthen the hangover.** Natural speech has 300–800 ms
  pauses mid-sentence and `hangover_ms` is 320 ms, so *every* one of them cut a
  turn. Raising the hangover trades latency for the same result; holding a
  closed turn for `streaming.merge_gap_sec` (1.0 s) and merging the next one
  into it costs the same latency only for pauses shorter than that, and leaves
  real conversational pauses submitting immediately.
- **The coalescer needs a deadline (`poll()`), not just a next-turn trigger.**
  Releasing only on the *next* turn means a session that ends after one sentence
  sends nothing until shutdown. `poll()` is called every main-loop iteration
  (client) and every received packet (server) and releases the held turn once
  `merge_gap_sec` of wall time has passed.
- **Release order is observable.** The server's `detector.flush()` tail must go
  through `coalescer.submit()`, not straight to `_submit()` — releasing the tail
  first and the held turn second enqueued them backwards (2.08 s before 0.90 s)
  and the client printed transcripts out of order.
- **One coalescer per channel.** Merging across channels would hand one person's
  audio to the other. Turn frames from the client are already coalesced and must
  not pass through a second server-side coalescer.
- **The gate is fail-open.** `turn_has_speech()` returns "transcribe" when the
  VAD is missing, disabled or throws. Refusing to transcribe real speech is a
  worse failure than transcribing a noise fragment.
- **`probe_speech` is not `run_vad_onnx`.** The file path's VAD applies a 250 ms
  minimum-duration filter, which discards exactly the 0.3–1 s replies that matter
  live. The gate scores every 512-sample frame and reports `(mean_prob,
  speech_ratio)` so either can gate.
- **Resolve `state.vad_session` at call time, never pin it** — the idle-TTL
  monitor unloads models between turns (lesson 26).
- **Whisper's `no_speech_prob` does not work — do not build a gate on it.** It
  was implemented (`is_hallucination`, `whisper.no_speech_max` = 0.6) and then
  removed after measuring the real backend in-container (faster-whisper 1.2.1 /
  CTranslate2 4.8.2 / large-v3-turbo): `no_speech_prob` came back **0.0000 for
  every segment** — real speech, white noise at −40 and −30 dBFS, and digital
  silence — at beam sizes 1 and 5. faster-whisper's own `no_speech_threshold`
  (default 0.6) is applied to that same value, so its built-in filter is inert
  too. `avg_logprob` is not a substitute: silence decodes to "Thank you." at
  −0.29, real speech at −0.30. Only `compression_ratio` separated them (speech
  0.89–1.00 vs noise 0.27–0.56) and openai-whisper only uses it as a repetition
  guard (default 2.4), never as a speech test. Dead configuration that reads as
  protection is worse than none: the Silero gate is the single filter, and the
  file path relies on faster-whisper's own Silero `vad_filter`.
- Measured with the shipped thresholds on real audio: speech mean 0.46–0.57
  (pass), white noise 0.08–0.23, 3 kHz hiss 0.14, sniff 0.04, silence 0.04
  (all reject) — a wide margin around `min_speech_prob` 0.30.
- End-to-end through the real handler with the real Whisper and the real
  Silero (7 turn frames: silence, sniff, speech, silence, speech, hiss,
  silence): both speech turns decoded correctly and were labelled `You`, all
  five non-speech turns were skipped with zero decodes (`non_speech_turns: 5`,
  `dropped_turns: 0`), and no "Thank you." appeared. That run is the evidence
  that the gate is the filter — the unit tests alone could not have shown it.
- A rejected turn is reported to the client as `{"type": "empty", "skipped":
  ...}` so the sidecar can distinguish "discarded as noise" from "transcribed to
  nothing". Otherwise the two are indistinguishable in the output.
