# Processing: diarization, turns and transcription

How audio becomes a transcript, and where a boundary is decided. Endpoint
reference is in [api.md](api.md); every tunable is
`asr_mcp/config/thresholds.json`; what an `UNKNOWN` label means is in
[uncertain-speakers.md](uncertain-speakers.md).

- [The shape of a request](#the-shape-of-a-request)
- [Diarization pipeline](#diarization-pipeline)
- [Turn preparation](#turn-preparation)
- [Boundary accuracy](#boundary-accuracy)
- [Transcription](#transcription)
- [Voiceprint learning](#voiceprint-learning)
- [The uncertainty policy](#the-uncertainty-policy)
- [Live streaming](#live-streaming)
- [Progress events](#progress-events)
- [Background](#background)

---

## The shape of a request

An upload takes the same route regardless of the mode, and the mode decides only
how much of the tail runs:

```
audio → ffmpeg → wav 16k mono
         ↓
      diarize            ← 13 steps, produces segments + raw VAD sections
         ↓
      prepare turns      ← segments → the timeline the user sees and is attributed on
         ↓
      refine boundaries  ← exact cuts, using the raw VAD the diarizer already ran
         ↓
      learn voiceprints  ← snippets cut from the TURNS, trimmed to whole VAD sections
         ↓
      transcribe         ← skipped for a diarize-only run
         ↓
      attribute          ← ASR items → turns → runs of one speaker
```

`/transcribe/upload`, `/diarize/upload` and `/attribution/upload` share the
diarizer, the turn preparation, the boundary engine and the learner. A run that
never decodes text produces the **same boundary set** as one that does, so
learning from a diarize-only run is as good as learning from a transcription.

## Diarization pipeline

Step numbers match the `# Step N` comments in
`asr_mcp/diarization/pipeline.py::Diarizer.run`, which is the source of truth.

| # | Step | What it does |
|---|---|---|
| 1 | VAD | Silero ONNX → **raw uncollapsed** sections (kept for step 13) |
| 1b | — | merge sections <0.5 s apart for clustering |
| 2 | Energy-dip splitting | `min_segment_dur` 3.0 s, `dip_ratio` 0.35, `min_dip_dur` 0.5 s, `min_split_piece` 2.0 s |
| 3 | Sliding windows | 2.0 s window, 1.2 s stride — **for embeddings only**, never for segment length |
| 4 | Embedding | ECAPA-TDNN512, 192-dim, ONNX CUDA, MD5-keyed LRU cache |
| 5 | Clustering | AgglomerativeClustering, cosine. Threshold path: average linkage at `distance_threshold`. Forced-k path (`num_speakers` given): `n_clusters=k` with `forced_k_linkage` (default `complete`) and **no** greedy merge |
| 5b | Label assignment | every segment gets a label, including the short ones |
| 5c | Overlap detection | `proximity_ratio` 0.08, `min_distance` 0.40 |
| 6 | Build segments | labels → `Speaker N` (1-indexed), same-speaker merge at `max_speaker_gap` 1.0 s, `build_overlap_segments` |
| 7 | Cleanup | split single-speaker from OVERLAP, absorb islands, refine boundaries |
| 8 | Profiling | pitch, energy, spectral, MFCC |
| 8b | Relabel by pitch | highest f0 becomes Speaker 1; cluster centroids injected |
| 9 | Known-speaker matching | multi-feature: 60% embedding + 15% pitch + 10% spectral + 10% MFCC. **Before** ghost elimination so it can populate `alternatives` |
| 10 | Ghost elimination | <10 s total speech → suppressed to UNKNOWN, or to a matched voiceprint alternative; `ghost_max_share` rescues a large share |
| 11 | Minority suppression | `max_utterance` 5.0 s, `min_speaker_dur` 8.0 s; matched voiceprints protected |
| 12 | Second pass | re-identify unknown speakers, gated by `second_pass.min_identity_confidence`, one-to-one claim |
| 13 | Exact boundary refinement | from the raw VAD sections |

Two ordering rules are load-bearing and easy to break:

- **Step 9 before step 10.** Matching must run before ghosts are suppressed, so a
  ghost cluster can be rescued onto a real voiceprint rather than lost.
- **`num_speakers` is a different pipeline, not a hint.** It selects the hard-k
  path with no greedy merge (lesson 32). If a result looks lopsided, read the
  `Cluster balance: ...` line in the log before touching the naming policy.

`Diarizer.run(known_speakers=...)` receives the caller's voiceprints as a dict
and never touches the database itself. The single funnel that selects them is
`asr_router._load_known_speakers(settings, user_id)` — pending profiles are
excluded there, not in the pipeline.

## Turn preparation

`_prepare_turns` (`asr_router.py`) turns segments into the timeline. Every second
of the audio belongs to **exactly one** turn (lesson 17):

1. Merge same-speaker diarized segments into turns at gaps ≤
   `uncertainty.turn_merge_gap_sec`. A `speaker is None` span is never merged.
2. Split turns longer than `MAX_TURN_SEC` (120 s) at their largest internal
   diarized-segment gap.
3. **Exact boundary refinement** — see below.
4. **Fallback for any remaining gap**: spectral change → energy-dip cut → quietest
   frame → gap midpoint. Steps 3 and 4 are the *same engine*; the router's own VAD
   pass is a fallback used only by `/attribution`, where no diarizer ran.
5. The first turn starts at 0.0 and the last ends at `audio_duration`.

Turns are **not** decode units. One whole-file decode is issued and the returned
items are mapped onto the turns afterwards, so audio is never cut at a speaker
boundary.

## Boundary accuracy

Every speaker transition is cut by **one** engine,
`asr_mcp/speaker/boundary.py::cut_between`, shared by the pipeline, the turn
timeline and the fallback chain (lesson 40). It tries in order:

1. **VAD-section ownership** — sections spanning the gap are embedded and
   attributed to the better-matching adjacent speaker (a known DB voiceprint,
   else ≤30 s of that speaker's own turn audio), then `best_split` picks the
   ownership cut, margin-weighted.
2. **Spectral change** — frame-to-frame log-spectral-distance peak.
3. **Energy dip** — the longest dip run.
4. **Quietest frame.**
5. **Gap midpoint.**

Whichever method wins, the two spans either side move to the **same** cut, so no
second of the timeline is unclaimed or double-claimed. The three former
implementations disagreed with each other on the same gap; one engine removes
that by construction.

### Measured

Every internal boundary scored against the **raw Silero VAD sections** — distance
from the cut to the nearest speech edge, lower is better. No ASR involved, so the
boundary decision is isolated.

| material | `acoustic: false` p50/p90/max | `acoustic: true` p50/p90/max |
|---|---|---|
| `0-four-speakers-zh` 56.9 s | 1.397 / 3.970 / 6.738 | **0.000 / 0.550 / 0.550** |
| `1-two-speakers-en` 16.0 s | 0.864 / 1.530 / 1.696 | **0.391 / 0.678 / 0.750** |
| `2-two-speakers-en` 34.0 s | 0.304 / 0.504 / 0.576 | **0.106 / 0.227 / 0.398** |
| `3-two-speakers-en` 54.8 s | 0.806 / 1.488 / 1.616 | **0.550 / 0.634 / 0.690** |
| ZO249 podcast, 57 min, 471 cuts | 0.448 / 1.168 / 3.792 | **0.000 / 0.350 / 1.984** |

Acoustic refinement wins on all five files at every percentile, by the widest
margin on the hardest material. `boundary_refine.acoustic: false` reproduces the
first column and is the A/B switch; `boundary_refine.enabled: false` rolls the
engine back wholesale to gap midpoints.

**Two results recorded because they do not flatter the work:**

- **`spectral_change` never fired.** The `on` and `legacy` arms are byte-identical
  on all five files and `spectral_change` is their only difference. Its two
  thresholds are reasoned defaults, the only evidence is a synthetic pure-tone
  test, and on real material it contributed nothing. It is kept because it is
  cheap and gated, not because it is proven.
- **The A/B arms do not isolate the router.** `acoustic: false` also makes step 7
  a no-op and downgrades step 13, so the arms produce different segment sets
  (ZO249: 616 turns / 7 speakers off vs 472 / 4 on). Per-boundary distance is
  comparable; the turn counts are not evidence of anything.

The `learning` distance thresholds are calibrated on **one file**. See
[voiceprints.md](voiceprints.md#matching-a-distance-reads-differently-than-a-confidence).

## Transcription

`run_transcribe` makes **one** whole-file `transcribe_audio_sync()` call. The
backend decodes in its own windows:

| Backend | Windowing |
|---|---|
| `cohere` | `>3000` mel frames (30 s) with no KV/prefix bridge → `_plan_window_bounds` cuts at the minimum frame-energy point inside a ±100-frame band, never mid-word; tails <300 frames merge backwards. A window still over the encoder limit is split into overlapping 30 s windows at 25%, trimmed **in output space**. A window error is always surfaced in `result["error"]`, even when earlier windows produced text — otherwise a failure looks like silent truncation. |
| `qwen3-asr` | forced aligner for word timings, 4-bit, OOM-backoff chunker |
| `whisper` | faster-whisper large-v3-turbo, int8-quantized; segment timings feed attribution |

`attribute_items` (`speaker/attribution.py`) then maps each returned item onto
the turns and groups consecutive same-speaker items into `TranscribeResult` runs.

`?language=` is threaded router → prompt → decode and never forced to English
(lesson 28). The faster-whisper note: it hard-sets `language_probability=1.0` when
language is forced, so the log line prints `requested=<x> lang=<detected>` to
distinguish forced from detected.

## Voiceprint learning

Learning runs **after** turn preparation and is passed the **turns**, not the raw
segments, so every learned snippet carries the same boundaries the transcript was
attributed on. Each snippet is trimmed **inward** to whole VAD sections
(lesson 41) — turns abut by design, so padding outward would pull in the
neighbour's speech and duplicate it.

A learned-but-unnamed speaker becomes a **pending profile**, which is excluded
from every match until it is named (lesson 39). The full policy — the cohesion
split, the re-join, the caps, one-file-one-contribution — is in
[voiceprints.md](voiceprints.md#how-a-profile-is-created).

## The uncertainty policy

When an identity cannot be established the server **suppresses the name and keeps
the text** rather than force-fitting the nearest or most common speaker.

```json
{ "start": 9.2, "end": 10.6, "text": "Yes.",
  "speaker": null, "speaker_confidence": 0.0, "speaker_source": "unknown",
  "uncertain": true, "attribution_reason": "boundary_crossing" }
```

Every result carries `speaker_confidence`, `speaker_source`
(`known_voiceprint` | `diarization_cluster` | `unknown`) and, when the identity
was withheld, an `attribution_reason`. The `done` event reports `status` =
`ok` | `partial` | `error` plus `uncertain_segments`. **Text is never dropped**,
only the name.

One policy module owns all of it (`speaker/uncertainty.py`).
`uncertainty.enabled: false` is the full legacy rollback; to keep the policy and
restore only absorption use
`{"suppress_ghost_speakers": false, "suppress_minority_speakers": false}`.

Full reference, including every `attribution_reason`: **[uncertain-speakers.md](uncertain-speakers.md)**.

## Live streaming

`WS /api/asr/ws/stream` carries **two** channels as LVT1 turn frames: channel 0
is the mic (the identity evidence — it is never voiceprint-matched, lesson 34) and
channel 1 is tab/system audio via `getDisplayMedia`.

The server runs **neither** detector on LVT1 frames — `handler.py` builds the
`Turn` straight from the frame. The clients run ports of
`streaming/turn_detector.py` and `TurnCoalescer`, drift-guarded against the server
by `tests/test_live_js_protocol.py`.

A live boundary is the client's **claim** and the server's **decision**
(lesson 42):

- Silero (`speech_gate.py`, fail-open) trims each turn's leading and trailing
  frames.
- A `TimelineGuard` rejects a `start_sample` that goes backwards or runs
  implausibly far ahead, and reports the drift instead of silently shifting every
  later boundary on that channel.
- A coalesced turn carries its gap as **silence**, so the declared span equals the
  audio actually sent.

Auth precedes `accept()`; the socket closes 1008 on failure. The browser cannot
set headers, so it uses the session cookie with `?token=` as a fallback.

Transcript items arrive in **decode-completion order** — sort by `start` before
rendering or re-attributing. `Session.turnSplit()` in the end-of-session note is
the one number that explains a bad transcript: `mic 0, speaker 12` means the mic
recording has no speech for the re-attribution to diarize.

## Progress events

Upload routes return `text/event-stream`. Producers yield (`sleep(0.01)`) and run
heavy work off-loop (lesson 19). `stage: "done"` **finishes the job**, so a
progress event must never use it (lesson 25).

The UI needs per-window time interpolation (lesson 23): a progress frame carries a
window index and an estimate, and the client maps that onto a time range rather
than a percentage. Paragraph breaks at pauses snap to a sentence end within 40
characters (lesson 24).

`/api/asr/activity/stream` pushes job start/finish to every subscriber but
**filters to the caller's own job**; another user's job is a 404 on
`/api/asr/active/stream`, and the Cancel button is only offered for a job you
own.

### The gap after diarization

The ASR backend is **lazily** loaded. After `diarization_complete` the router calls
`unload_embedding()` then `ensure_backend_ready()`, and when the model is not
already resident the second call builds a multi-gigabyte CUDA session — a
stretch with no events in it, during which the user watches a completed
diarization bar do nothing. So the router emits a `Loading speech recognition
model` event **before** those calls, in the **diarization** phase so the bar
holds the 100% it just earned instead of collapsing back to a transcription
zero. No progress fraction rides along: there is nothing honest to report while
a session is being constructed, and a fake ramp would be worse than the label.

The diarization pipeline's back half had the same problem one level down. Clustering,
overlap detection, boundary trimming, known-speaker matching and unknown-speaker
consolidation each ran behind a single earlier label, so a bar could sit on
`Clustering speakers` for several minutes with nothing to say. Those steps now
emit their own events inside the 0.6–1.0 band the coarse stages already
occupied, strictly increasing, with `diarization_finished` still the terminal
event at 1.0.

---

## Background

- [docs/lessons/diarization-and-speakers.md](lessons/diarization-and-speakers.md)
  — lessons 1, 3, 4, 11, 29, 32, 33, 40, 43, 46
- [docs/lessons/transcription-turns.md](lessons/transcription-turns.md) —
  lessons 17, 18, 41
- [docs/lessons/asr-decoding.md](lessons/asr-decoding.md) — lessons 7, 8, 12–16
- [docs/lessons/streaming-websocket.md](lessons/streaming-websocket.md) — lesson 31
- [docs/lessons/live-client-design.md](lessons/live-client-design.md) —
  lessons 34–37, 42
- [docs/lessons/api-sse-and-auth.md](lessons/api-sse-and-auth.md) —
  lessons 9, 19, 23–25, 28, 38
- [docs/lessons/gpu-and-onnx-runtime.md](lessons/gpu-and-onnx-runtime.md) —
  lessons 5, 6, 27, 44
- [boundary-measurements.md](boundary-measurements.md) — the measured A/B behind
  the engine, the two results that do not flatter it, and what is still
  uncalibrated