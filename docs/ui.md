# The web UI

The single-page app in `asr_mcp/templates/app.html`, plus `static/live.js`,
`static/live-worklet.js` and `static/viewer.js`. Endpoint reference is in
[api.md](api.md#pages).

- [Five tabs, one page](#five-tabs-one-page)
- [Design system](#design-system)
- [Transcribe tab](#transcribe-tab)
- [Voiceprints tab](#voiceprints-tab)
- [History tab](#history-tab)
- [Settings tab](#settings-tab)
- [Live tab](#live-tab)
- [Known traps](#known-traps)

---

## Five tabs, one page

`app.html` is every tab. The nav is a separate snippet,
`templates/_nav.html`, inlined at build time and marked with `__NAV__` /
`__ACT_*__` placeholders.

| Tab | Path | Loaded by `activateTab` |
|---|---|---|
| Transcribe | `/gui` | default landing |
| Live | `/live` | `initLive()` — needs a **user gesture**, see below |
| Voiceprints | `/voices` | `refresh()` |
| History | `/transcriptions` | `loadTranscripts()` |
| Settings | `/settings` | `checkStatus()` + `loadAdminUsers()` |

`activateTab` only toggles visibility and dispatches those loaders. It never
re-renders, because a redraw must not close what the user opened.

**The Live tab is initialised on tab entry, not at page load.** An AudioWorklet
requires a secure context (https or localhost) and a `getUserMedia` call that
cannot happen before a user gesture; entering the tab *is* the gesture.

`activateTab` also guards against a job in another pane hijacking the tab:
`/api/asr/activity/stream` is global, so a `consolidate` or `rescan` job started
on the Voiceprints tab reaches the Transcribe pane's `attachActiveJob`. The mode
is checked against `TRANSCRIPTION_MODES = ['transcribe', 'diarize']` **before**
any tab switch.

## Design system

All CSS is inline in `app.html` inside a single `<style>` block. There is no CSS
file and no build step; the tests strip comments before matching.

**A pane composes shared classes; it never declares a private copy.** The
primitives are:

`.pill` · `.note` · `.mini-bar` + `.meter-row` · `.text-pane` +
`.pane-short|tall|live` · `.btn` + `.row` / `.row-spread` · `.badge` ·
`.toast` · `.hidden` + `show(id, on)`

Status colours are `:root` tokens (`--ok-ink --bad-ink --warn-ink --*-soft
--pane-h`) and must not be re-hardcoded. The Live tab shipped with
`.live-status` / `.live-meters` / `.live-meter-*` / `.live-text` /
`.live-downloads` — a private second copy of the pill, the level meter and the
transcript surface. That is the failure this rule prevents.

**Exactly two max-width breakpoints: 860px and 560px.** Base rules are the
desktop layout, so a `min-width` query is a rule violation. The 560px block is the
phone layout — scrollable tab strip, 16px input floor for iOS, 44px tap targets,
stacked controls, safe-area insets, `prefers-reduced-motion`. The
`<meta name="viewport">` carries `viewport-fit=cover` or the safe-area padding is
inert.

**No inline styles.** Visibility is the `.hidden` class plus `show(id, on)`.
There are no `style.display` writes and no `style="..."` in markup.

`tests/test_ui_design_system.py` pins all of this in both directions: no orphan
class either way, no private primitive, no inline style, token-only colours,
exactly two breakpoints.

## Transcribe tab

Upload a recording, watch progress, read the transcript. Progress needs
per-window time interpolation (lesson 23) — see
[processing.md](processing.md#progress-events).

`viewer.js` renders blocks and owns the speaker palette. Its most important line
is a refusal:

```js
// The server suppresses speaker identity (speaker === null) instead of
// guessing a nearby speaker. Never invent a name for those spans.
const UNKNOWN_SPEAKER = 'UNKNOWN';
```

Every suppressed span shares one colour bucket, so `UNKNOWN` reads as *deliberate*
rather than as a rendering bug.

The Windows client's `.txt` format is
`profiles banner + [HH:MM:SS] Speaker (NN%): paragraphs`. A format change means
updating the `mem-mcp process-transcription` skill, step 2.

## Voiceprints tab

Two sub-tabs: **Known speakers** and **Unnamed**. They were one list with the
unnamed profiles in a card above the named ones, which made the named list
unreadable and asked the user to identify a voice from a row of counts.

`.pill-tab` is **one** primitive for both strips — the header nav (anchors) and
the in-pane sub-tabs (buttons). An anchor-scoped `.tabs a` rule is what a button
sub-tab would silently miss, so that rule was replaced.

An unnamed profile is a fold (`togglePending` → `loadSnippetBody` →
`snippetListHtml`), exactly like a named one: same endpoint, same player, same
remove button. Its audio is the only evidence of who the voice is, so it must be
playable **in place**. The action row differs on purpose — naming, merging and
discarding are identity decisions, and only an unnamed profile has a "this is
___" input.

**Every list is sorted by name** (`byName` with `numeric: true`), because these
labels are generated and a lexicographic order puts `Speaker 10` between
`Speaker 1` and `Speaker 2`. This applies to the known list, the unnamed list and
the flagged card's speaker groups, which is why `byName` accepts a string as well
as a row.

`refreshPending` writes the count into the tab label — a hidden card with no
affordance is how the profiles went unnoticed. Naming or merging switches to the
Known tab, because the profile just moved there.

The backend for all of this is in [voiceprints.md](voiceprints.md).

**`Clean Up Snippets` is the one library-wide button that deletes audio**, so it
is the only red one on this pane: every other destructive button here is scoped
to a single profile and shows which snippet it drops. It runs
`/integrity/cleanup` **twice** — `dry_run: true` first, `confirm` with the counts
the plan returned, then `dry_run: false` — and leaves the three steps with a
wrong answer off (`duplicates`, `dead_snippets`, `stale_totals`): a button
labelled "clean up snippets" must not be the thing that merges two colleague
profiles, purges rows, or rewrites a number. It is a JSON round trip rather than
an SSE job, so the status line's stage text is the only progress; the embedding
model has to come up (`ensure_gpu`) before any of it happens.

## History tab

Saved transcripts, scoped to the signed-in user. `?save=false` on an upload skips
the save; an SSE disconnect before `done` saves anyway, so the run appears here.

## Settings tab

Status, the Windows client token card, the session, and — for the configured
admin — the **Accounts** card.

The Accounts card is hidden by default (`class="card hidden" id="adminCard"`) and
revealed by `loadAdminUsers()`. **On any failure the card stays hidden**: a 403
or an unreachable server means the card simply does not exist for that account,
which is the correct behaviour and not an error to display.

Each user row shows the id, email and joined sources; flags; data counts; and
Disable/Enable when the account is manageable, plus a "also delete recordings"
checkbox next to Delete when it is deletable. Deleting asks to confirm, and the
server refuses two accounts outright: the one you are signed in as, and
`TRANSCRIBE_ADMIN_USER` itself.

`POST /shutdown` sits in this card and is gated on `require_admin`, because
stopping the process cancels other users' jobs.

## Live tab

`static/live.js` is a third live client alongside
`asr-client/live_client.py`. It re-implements the wire protocol and the
endpointing in JS, so both are drift-guarded by
`tests/test_live_js_protocol.py`: constants vs. `protocol.py`, `DETECTOR_DEFAULTS`
vs. `turn_detector.config()`, packed frames vs. `unpack_turn`, node-vs-server
detector boundary equality, and `buildTranscript` vs.
`transcribe_client.build_transcript`.

**Capture.** `getUserMedia` (mono, AEC/NS) → `AudioWorkletNode`
(`live-worklet.js`) that resamples to 16 kHz, levels the input and posts
1024-sample int16 blocks. Channel 1 is `getDisplayMedia({audio, video: true})`
tab/system audio — Chrome only returns tab audio when video is requested, and a
failed display-media call degrades to mic-only with a visible note.

**Input levelling.** Both live clients apply the same boost-only AGC in the
capture chain, *before* the int16 conversion, so the frames on the wire AND the
WAV re-uploaded for re-attribution carry the same level: `TARGET_RMS 0.125`,
`MIN_ENV 0.0008`, `MAX_GAIN 16.0`, log-domain attack/release, per-block peak
ceiling. `desired = TARGET_RMS / rms` — dividing by the *already-gained* level
settles at `sqrt(TARGET/rms)`, which looks like convergence and is not (lesson
37). The `--no-agc` flag exists on the Python client.

**Stop.** `MSG_FLUSH`, then wait for the server's `stats` frame (the drain signal)
before building the transcript. A failed drain must not strand the tab: the
transport buttons are restored in a `finally`, and a socket that dropped *after*
transcripts arrived still gets downloads and the History save.

**Downloads** are generated client-side: `.txt`, `.asr.json`, and per-channel
`.wav`. Re-attribution goes to `POST /api/asr/attribution/upload` (no re-decode);
`POST /api/asr/live/save` persists text to History — text only, audio goes
through the capped upload route.

**`Session.turnSplit()`** appears in the end-of-session note, the sidecar and the
`.txt` header, and it is the one number that explains a bad transcript.
`mic 0, speaker 12` means the mic recording has no speech for the re-attribution
to diarize, so every item collapses onto one speakerless turn and the whole
session returns UNKNOWN with no reason shown.

Transcript items arrive in **decode-completion order** — sort by `start` before
rendering or re-attributing.

## Known traps

These are not hypothetical; each one shipped and was caught by a test written
afterwards. `tests/test_ui_design_system.py` and
`tests/test_voiceprint_panes.py` pin them.

**A name in an `onclick` is `jsStr(name)`, never `escHtml(name)`.** `escHtml`
escapes `& < >` and deliberately leaves quotes alone, and the browser
HTML-decodes the attribute *before* the JS is parsed, so
`onclick="f('${escHtml(name)}')"` is a `SyntaxError` on an apostrophe and the
button silently does nothing. A learned profile takes its name from a source
**filename**, so the apostrophe arrives without anybody typing it.

**Markup inside a JS template literal is not validated by anything.**
`onclick="f(${jsStr(n)})""` — a doubled closing quote, which is what a careless
search-and-replace of `'${escHtml(x)}'` into `${jsStr(x)}` produces — silently
kills the button. `node --check` parses past it, because the markup is inside a
template literal, and the design-system tests did not catch it either.
`test_no_html_attribute_is_closed_twice` scans for `""` and `">>` outside
`<style>`, `<script>` and line comments.

**An address typed by a stranger is written back with `textContent`, never
`innerHTML`** — the registration-pending message on the login page is the one
place this happens.

**A redraw must not close what the user opened.** `refresh()` replaces both
lists' `innerHTML`, so it captures the open fold ids (`captureOpenFolds` —
`body-<name>` / `pbody-<name>`) and reopens them afterwards (`restoreOpenFolds`,
reusing `loadSnippetBody` + `knownActionsHtml`). Reviewing means open a profile,
remove three snippets, look again; collapsing on every removal made that
impossible.

**`snippetListHtml` + `loadSnippetBody` are the single renderer/loader pair.** The
test pins one definition, one call in the loader, and one call from each fold.

---

## Background

- [docs/lessons/frontend-and-client.md](lessons/frontend-and-client.md) —
  lessons 21–24, 26, 36
- [docs/lessons/live-client-design.md](lessons/live-client-design.md) —
  lessons 34–37, 42
- [docs/lessons/diarization-and-speakers.md](lessons/diarization-and-speakers.md)
  — lesson 39 (an unnamed profile's audio must be playable in place)