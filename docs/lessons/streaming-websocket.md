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
