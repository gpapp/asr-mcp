# Frontend, GUI and client output

Lessons 21, 22, 23, 24, 26 of `AGENTS.md`. CSS scoping, refined-segment display, progress interpolation, live session resolution.


### 21. CSS for JS-generated Elements Must Not Be Scoped Under Classes JS Never Adds
- Timeline styles were scoped `.speaker-timeline .bar-seg`, but `renderTimeline()` never added that class → `position:absolute` etc. silently no-op'd: segments stacked (one visible speaker), legend swatches got zero size.
- Rule: add the scoping class in JS OR write selectors against the real elements. Verify by inspecting computed styles, not just rendered HTML.

### 22. Display REFINED Segments, Not Raw Diarization
- `_prepare_turns()` closes inter-turn gaps (VAD voiceprint attribution + midpoint cuts) so turns cover the full timeline. Emit ITS output in `diarization_complete` and set `diarization["segments"]` to it — raw segments leave visible gaps between bands.
- Strip the nested `segments` list before emitting (`{start, end, speaker}` only) to keep SSE payloads small.
- `auto_collect_from_diarization()` must still receive the RAW segments (happens before replacement).

### 23. Progress UI Needs Sub-Turn Time Interpolation
- Window-progress events for one long turn all carried the same `segment_start`/`segment_end` → the caret never moved despite events arriving.
- Interpolate per window in `_make_window_cb`: `win_start = turn.start + span * (i-1)/n`.
- Elements shown under a bar whose container has `overflow:hidden` must live in a sibling wrapper or they are clipped (caret moved into `position:relative; padding-bottom` wrapper).
- UI chrome (SPA tab bar) lives in `templates/_nav.html`, injected by `_render(name, active=...)` via `__NAV__` + `__ACT_*__` markers — edit the snippet, not `app.html`. All four page routes (`/gui`, `/voices`, `/transcriptions`, `/settings`) render `app.html`; tab switching is client-side (pushState + `activateTab`).

### 24. Paragraph Breaks at Pauses Must Snap to Sentence Ends
- `_apply_paragraph_breaks()` (asr_router.py): RMS interior pauses ≥ `PARAGRAPH_PAUSE_SEC` (1.5s), map each to the nearest decoder split-token segment boundary, then snap to `[.!?…]` within **40 chars**.
- If no punctuation is close: break at the boundary anyway and append `.` (after stripping trailing `,;—`) so the break is still a sentence boundary.
- **Why 40 chars**: a 120-char window snapped BACKWARD across the pause to an earlier sentence end, burying the pause mid-paragraph. Keep the snap window tight.


### 26. Never Pin Session Objects — Resolve `state.*_session` Live
- Storing `state.embedding_session` into another object (`set_embedding_session`) leaves a stale/None reference once lazy-load/TTL/phase-unload replaces or clears it → `'NoneType' ... get_inputs` deep in a helper.
- Rule: look up the session at call time (`VoiceprintService._emb_session()` = `self._embedding_session or state.embedding_session`). Grep for direct `state.*_session` field reads inside services before adding pins.
