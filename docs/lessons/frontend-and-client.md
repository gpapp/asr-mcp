# Frontend, GUI and client output

Lessons 21, 22, 23, 24, 26 of `AGENTS.md`. CSS scoping, refined-segment display, progress interpolation, live session resolution.


### 21. CSS for JS-generated Elements Must Not Be Scoped Under Classes JS Never Adds
- Timeline styles were scoped `.speaker-timeline .bar-seg`, but `renderTimeline()` never added that class → `position:absolute` etc. silently no-op'd: segments stacked (one visible speaker), legend swatches got zero size.
- Rule: add the scoping class in JS OR write selectors against the real elements. Verify by inspecting computed styles, not just rendered HTML.

#### 21a. A Pane Composes Shared Classes — It Does Not Declare Its Own
The Live tab arrived with `.live-status` / `.live-meters` / `.live-meter-*` /
`.live-text` / `.live-note` / `.live-downloads`: a private second copy of the
pill, the level meter and the transcript surface that the SPA already had. A
private copy is not a style choice, it is a second implementation that only the
newer one gets fixed.

- **The shared primitives**: `.pill` (status chip), `.note` (inline caution /
  success), `.mini-bar` + `.meter-row` (level meter), `.text-pane` +
  `.pane-short|tall|live` (transcript surface), `.btn` + `.row` /
  `.row-spread`, `.badge`, `.toast`, `.hidden` + `show(id, on)`. Status colours
  are `:root` tokens (`--ok-ink --bad-ink --warn-ink --*-soft`); a second
  literal green means the token and the ad-hoc rule now disagree.
- **Both directions of the lesson-21 trap are defects**: a class the JS adds
  with no rule is a silent no-op, and a rule whose class nothing ever adds is
  dead CSS. `tests/test_ui_design_system.py` pins both, plus "no pane declares
  a private copy of a shared primitive".
- **The load-bearing detail that started all of it**: `.live-meter-fill` was a
  `<span>`, and `width` does not apply to a non-replaced inline box, so every
  width write was dropped and a working microphone read as a dead capture. The
  rule is `.mini-bar > div { display: block }` and the test asserts the fill is
  a `<div>`.
- **Inline styles are how the private copies came back.** `style.display =
  'block'` hardcodes an element's layout mode at the call site; a flex row
  restored as `''` and a div restored as `'block'` are the two bugs waiting.
  Visibility is the `.hidden` class plus `show()`.
- **Breakpoints are max-width only, and there are two: 860px and 560px.** Base
  rules are the desktop layout, so a `min-width` query is a rule violation. The
  560px block is the phone layout: a scrollable tab strip (the wrapped header
  ate a third of the viewport), a 16px input floor (iOS focus-zoom is
  unrecoverable inside a scrollable pane), 44px tap targets, stacked controls,
  `env(safe-area-inset-*)` (inert without `viewport-fit=cover` in the viewport
  meta) and `prefers-reduced-motion`.
- **Fixed widths in a row are a phone bug.** The meter row had 6.5rem + 9.5rem
  of fixed flex basis — 16rem of a 360px viewport before the bar got anything.
  It is a grid with `grid-template-areas: "name bar label"` so the phone block
  can reflow it to `"name label" "bar bar"`.

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
- `_apply_paragraph_breaks()` (asr_router.py): RMS interior pauses ≥ `PARAGRAPH_PAUSE_SEC` (3.0s), map each to the nearest decoder split-token segment boundary, then snap to `[.!?…]` within **40 chars**.
- If no punctuation is close: break at the boundary anyway and append `.` (after stripping trailing `,;—`) so the break is still a sentence boundary.
- **Why 40 chars**: a 120-char window snapped BACKWARD across the pause to an earlier sentence end, burying the pause mid-paragraph. Keep the snap window tight.


### 26. Never Pin Session Objects — Resolve `state.*_session` Live
- Storing `state.embedding_session` into another object (`set_embedding_session`) leaves a stale/None reference once lazy-load/TTL/phase-unload replaces or clears it → `'NoneType' ... get_inputs` deep in a helper.
- Rule: look up the session at call time (`VoiceprintService._emb_session()` = `self._embedding_session or state.embedding_session`). Grep for direct `state.*_session` field reads inside services before adding pins.
