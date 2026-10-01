# Boundary refinement & new-voice learning — improvement plan

**Status:** **P0–P4 implemented** (2026-10-01), in `c01b4fd`. Every phase shipped behind a
config flag with a documented rollback. The P0 A/B measurement that the plan was written to
enable is in **section 2** — it resolved the open question rather than leaving it open. **25 of
27 steps are done; the step-by-step status, the two that are not, and why they matter are in
section 3.** What did *not* get proven is stated in section 2 as well: `spectral_change` never
fired on real material, and the `learning` distance thresholds remain calibrated on one file.
**Origin:** review request — *"review how boundaries are refined, consider finding better methods. also review new voice creation … to ensure boundaries are found. create todo, ensure it works"* and *"for live transcriptions as well as pre-recorded"*.

**Scope:** how speaker/turn boundaries are decided, and how a new voice is learned, on
both the pre-recorded path (`/diarize`, `/transcribe/upload`, `/attribution/upload`) and the
live paths (`WS /ws/stream`, plus the shutdown re-attribution both live clients run).

**Method of the review:** three read-only code inventories (offline boundaries, live paths,
new-voice learning) plus my own verification against the code and against the ZO249
(57-minute, 2-speaker + inserted clips) run. Every claim below cites a file and line or a
measurement; nothing here is a preference.

---

## 1. Findings

### F1 — There are two independent boundary sets, and the transcript does not use the one that trains the voices

| | Set A — diarization segments | Set B — turns |
|---|---|---|
| built by | `Diarizer.run` (`pipeline.py:46`, steps 1–13) | `_prepare_turns` (`asr_router.py:628`), recomputed from set A |
| what the user sees | no | **yes** — `display_segments` is overwritten with set B (`asr_router.py:1243-1262`) |
| what ASR text is attributed on | no | **yes** — `attribute_items` (`speaker/attribution.py:156`) |
| what voiceprint snippets are cut from | **yes** — `_auto_collect` (`asr_router.py:1153`, `:1638`) | no |

Silero VAD also runs **twice** over the whole file with no shared result: `pipeline._run_vad`
(`pipeline.py:268`) and `asr_router._raw_vad_sections` (`asr_router.py:406`). The two sets
are then merged with *different* gap thresholds (1.5 / 1.0 / 0.5 s for set A vs 1.0 s for
set B) and refined by *different parameters* (step 13 uses `margin=0.5` and drops sections
< 0.1 s; the router's version has no lookaround and drops < 0.3 s).

**Why this matters more than any single bad cut:** a learned embedding becomes a *reference
for cutting future boundaries* — `pipeline.py:546-551` (`_get_speaker_ref`) and
`asr_router.py:479-489` (`_speaker_refs`) both prefer `known_speakers[name]["embedding"]`.
So the loop is: mid-word snippet → mis-shaped reference → worse cuts next run. This is the
structural root cause behind "new voice creation … ensure boundaries are found".

### F2 — Only ~3 of ~10 boundary decisions use acoustic evidence

The three that do (all offline, all embedding-based):

1. `speaker/audio.py:74 refine_speaker_boundaries` (step 7) — ±`search_sec=1.2` around
   `left["end"]`, 0.5 s/0.2 s sub-windows, each CMN+ECAPA; the new boundary is the
   **midpoint between the last left-owned and first right-owned sub-window centre**
   (`:243`), accepted only if it moves > 0.05 s and leaves both sides ≥ 0.3 s — otherwise
   segments < 0.3 s are **deleted** (`:247-253`).
2. `pipeline.py:590 _refine_turn_boundaries_exact` (step 13) — raw VAD sections overlapping
   `[gap_start-0.5, gap_end+0.5]` with duration ≥ 0.1 s, each embedded (batch 16, cached),
   ownership by cosine to the left/right reference weighted `|sa-sb|+1e-3`, brute-force
   min-cost cut (`:744-760`); `cut` = start of the first right-owned section, or `gap_end`
   when the gap belongs entirely to the left speaker; both turns share the one cut.
   Fallback `_gap_energy_cut` (`:783`) = centre of the single quietest 20 ms frame.
3. `asr_router.py:541 _refine_boundaries_with_vad` (on set B) — the same algorithm, with
   `_speaker_refs` preferring a known voiceprint (`:479-489`) and falling back to ≤ 30 s of
   that speaker's own turn audio (`:491-509`). Residual gaps are closed by `_gap_boundary`
   (`:349`) = longest sub-threshold run ≥ 0.12 s, ties → nearest gap centre, else the single
   quietest frame, else midpoint.

Everything else is a heuristic: RMS minima, VAD-section edges, arithmetic midpoints
(`pipeline.py:235`, `overlap.py:131`, `vad.py:90`, `asr_router.py:361/367/796`), and
nearest-window labels (`pipeline.py:473`, up to ±1.0 s). `_split_long_turn`
(`asr_router.py:105`) cuts a > 120 s turn at the largest internal *segment* gap and never
inspects audio at all (measured dead on ZO249: 0 of 462 turns exceed 120 s).

**There is no zero-crossing, pitch-contour discontinuity, spectral-change/novelty, or
forced-alignment snapping anywhere in the codebase.** The closest thing to signal-based
snapping is the 20 ms / 32 ms frame grid.

Two concrete harms:
- The three gap-cut implementations **disagree with each other** on the same gap —
  `_gap_energy_cut` picks the quietest single frame while `_gap_boundary` searches for a
  dip run, and `refine_speaker_boundaries` picks a sub-window midpoint.
- `split_at_energy_dips` **deletes real audio**: pieces shorter than `min_split_piece=2.0`
  are dropped (`vad.py:105`). Its threshold is `median(segment energies) * 0.35`, i.e.
  relative to each segment's own median, so a uniformly quiet passage is never split.

### F3 — Live boundaries are never refined, and carry three concrete defects

The only live endpointing is an energy-threshold crossing plus fixed pre/post-roll on a
32 ms grid: start = onset minus ≤ 160 ms, end = 160 ms after the last above-threshold frame.
Both shipped clients (browser `static/live.js`, Windows `asr-client/live_client.py`) run
bit-identical ports of the server's `streaming/turn_detector.py`, and `speech_gate.probe_speech`
(Silero, every 512-sample frame) only accepts or rejects a whole turn — it can never move
an edge. **No live boundary is ever refined with acoustic evidence.**

- **(a) Coalesced turns span audio that is not contiguous.** `turn_detector.py:483-498`,
  `live.js:375-378` and `live_client.py:413-417` all build the merged payload as plain
  `head + tail` concatenation with **no padding for the gap**, while `_rebuild_turn` sets
  `start = pending.start_sample, end = turn.end_sample`. The ASR therefore decodes a
  *discontinuous* turn, and every downstream attribution uses a span up to `merge_gap_sec`
  = 1.0 s longer than the audio that produced it.
- **(b) A client-declared `start_sample` is trusted verbatim.** `protocol.unpack_turn`
  validates only length, magic, version and `n_samples*2 == len(payload)`; `handler._turn_from_frame`
  (`:105-120`) copies the value through. One dropped worklet block, or a `getDisplayMedia`
  that starts late, silently shifts every later boundary on that channel.
- **(c) The session tail is duplicated into the WAV.** `live.js:943-951` and
  `live_client.py:1589-1596` write the flushed turn's PCM *again* after every block was
  already written, so the recording is up to one turn too long with the tail duplicated.
  This over-reports `audio_duration_sec` / `micSeconds()` and the duplicate region becomes
  extra diarization segments — which can spawn spurious pending profiles.
- **Secondary:** `merge_gap_sec` = 1.0 s absorbs genuine 0.3–0.8 s inter-person pauses, so
  two speakers become one turn with one identity; `max_turn_sec` force-splits at a hard
  29.98 s with no acoustic reasoning.
- **Also:** `attribution.attribute_live_turn`'s gates are identity/acceptance only, and
  `max_boundary_cross_sec` = 0.3 s means a genuine disagreement below that is never even
  reported.
- **Unvalidated:** `live_attribution.min_match_confidence = 0.60` was set from a
  *false-positive* measurement only (commit `51497d7`). No genuine non-matcher population
  has ever been measured, so the threshold is not yet defensible in either direction.
- **Gap:** voiceprints are loaded once at connect (`handler.py:137-148`) and never
  refreshed, and `voiceprint_router` / `transcript_router` have **no** re-attribution route
  — so a saved History transcript is never re-mapped after a profile is confirmed.

### F4 — New-voice learning has correctness bugs beyond boundary quality

- **The extend lookup keys on the source *filename stem*** (`service.py:855`:
  `if prefix in cand and (not label or label in cand)`), so the same colleague in a
  **second recording** mints a **second** pending profile — and the second `confirm` is then
  *refused* because the name is taken. The docstring at `:834-836` claiming a colleague
  "accumulates into ONE profile" is only true for the same recording; both tests that look
  like they cover it pass the same `source_id`. Nothing anywhere compares pending profiles
  to each other.
- **Label substring collision:** `"Speaker_1"` is a substring of `"Speaker_10"`, so
  cluster `Speaker 1`'s audio can be appended to `Speaker 10`'s profile.
- **Cluster-label reuse:** re-running the same file with a different `num_speakers` or
  threshold renumbers the clusters, so the person who was `Speaker 5` may now be
  `Speaker 3` — and any cluster that lands on `Speaker 5` is appended to the *wrong*
  profile with **no distance check**.
- **The `" #2"` collision loop checks the DB only** (`:862`), not snippet directories.
- **`_split_cluster_by_cohesion` fails open on every error path** (knob off, < 2 segments,
  no embedding session, sklearn missing, embedding throws, < 2 usable vectors, clustering
  throws, **< 2 groups**) — and a blend whose voices sit within 0.32 is *undetectable by
  construction*. No purity statistic is retained anywhere.
- **No ceiling on the learn path:** `AUTO_COLLECT_MAX_SEGMENT_SEC` / `MAX_TOTAL_SEC` are
  collect-only, while `_auto_refine` is duration-weighted (`:1132-1139`), so one 200 s
  snippet can dominate a profile containing four 5 s snippets.
- **`load_audio_segment` decodes the ENTIRE file per call** (`utils.py:107` →
  `load_audio(wav_path)`), synchronously inside `async def` request bodies. ZO249's 89-snippet
  cluster meant 89 whole-file decodes plus 89 embedding calls on the event loop (lesson 19).
- Un-embeddable ≥ 3 s segments are silently dropped, and the cohesion split runs *before*
  the `LEARN_MIN_*` floor, so sub-threshold clusters still pay for the clustering.

**Measured proof this is not hypothetical:** the default ZO249 run learned
`Pending 2026-09-30 19:43 Speaker_6 zo249` — 89 snippets / 606.13 s. Ranking six of that
cluster's spans individually against all 35 registered profiles gave

```
s6_0  19.5s  Alexander Radyukin 0.533  Nico Zeissig 0.573       margin 0.040
s6_1 547.6s  Alexander Radyukin 0.567  Marius Babrauskas 0.589 margin 0.022
s6_2 1142.5s Alexander Radyukin 0.589  Marius Babrauskas 0.611 margin 0.022
s6_3 1997.1s Gergely Papp      0.386  Alexander Radyukin 0.508 margin 0.122
s6_4 2737.3s Gergely Papp      0.531  Alexander Radyukin 0.533 margin 0.002
s6_5 3389.2s Gergely Papp      0.516  Alexander Radyukin 0.645 margin 0.130
within-group 0.175-0.291, between-group 0.368-0.448
```

That cluster was the guest in the first half **and** Gergely himself in the second half.
Naming it would have asserted one name for two people, and — because both refinement sites
consume the profile — corrupted the turn boundaries of every later run. Note the guest's
nearest registered profiles are all real colleagues, so the wrong name would have been
*confidently* wrong. Also note `pitch_std` is 40–59 Hz for **every** cluster including
Gergely's own (57.3), so the f0 tracker produces octave garbage on this material and pitch
cannot serve as a purity signal.

### F5 — Two open UI bugs of mine

- `app.html:2067` references an undefined `names` inside `openMergeDialog` (which defines
  `named` and `all`). The `ReferenceError` is thrown *before*
  `dialog.classList.remove('hidden')`, so **the generic merge dialog never opens**. The
  one-click "Merge into X" candidate path still works.
- `app.html:2075-2088 mergeSpeakers` ignores the response `error` field, so a refused merge
  toasts success.

### F6 — Dead configuration

Verified by grep over `asr_mcp/` excluding `thresholds.json`:

| key | references | note |
|---|---|---|
| `boundary_refine` (whole section) | **0** | only the JSON block and one log string; `_gap_boundary`'s 0.35/0.12 are hardcoded signature defaults |
| `diarization.close_match_threshold` | **0** | |
| `matching.embed_only_accept_threshold` | 1 | declaration only |
| `second_pass.known_speaker_margin_bias` | 2 | declaration only |
| `streaming.noise_floor_ratio` | 1 | read by nothing, mirrored in 4 files |
| `streaming.end_confirm_frames` | 1 | read by nothing, mirrored in 4 files |

### F7 — Measured baseline (ZO249, default path, 3439 s)

- 462 turns. Duration p50 **5.58 s**, p90 15.96 s, max **43.70 s**. **0 turns > 120 s.**
  **105 turns (22.7 %) shorter than 3.0 s**, totalling 195.0 s.
- All 461 turn pairs **abut** (gap ≤ 1 ms) — this is the *designed* invariant of
  `_prepare_turns` (lesson 17), because both refiners set `left["end"] = right["start"] = cut`.
  It is **not** evidence that refinement is a no-op.
- Set A: 480 segments, 244 real gaps (max 16.16 s), 235/479 pairs abutting; of the gaps
  > 1.0 s, 164 same-speaker and 36 different-speaker; 62 different-speaker gaps survive
  after step 13.
- Outcome quality is already good: 290 results, 41 696 chars, 7.15× realtime,
  Gergely Papp 2475.8 s / 128 runs at conf 0.528–0.763, the guest 685.1 s left as
  `Speaker 6`, UNKNOWN only 5.8 % of text. Independent ECAPA ranking confirmed the host is
  #1 of 35 with a 0.23 margin, and that the guest does not match him.
- The 18 ghost clusters (52.5 s) are the **inserted clips** from many different speakers —
  exactly the case the policy is designed for: each got its own cluster, fell under the 10 s
  ghost rule, lost its identity and **kept its words**.

---

## 2. The measurement that phase P0 was for — RESULT

The caveat that opened this document has been **resolved**. See
[docs/lessons/diarization-and-speakers.md](lessons/diarization-and-speakers.md) lesson 40
for the full argument; the numbers:

Every internal turn boundary scored against the **raw Silero VAD sections** (independent
evidence of where speech actually starts and stops), as distance to the nearest VAD edge in
seconds — lower is better. No ASR involved, so the boundary decision is isolated.

| clip | `acoustic: false` p50 / p90 / max | `acoustic: true` p50 / p90 / max |
|---|---|---|
| `0-four-speakers-zh` 56.9 s | 1.397 / 3.970 / 6.738 | **0.000 / 0.550 / 0.550** |
| `1-two-speakers-en` 16.0 s | 0.864 / 1.530 / 1.696 | **0.391 / 0.678 / 0.750** |
| `2-two-speakers-en` 34.0 s | 0.304 / 0.504 / 0.576 | **0.106 / 0.227 / 0.398** |
| `3-two-speakers-en` 54.8 s | 0.806 / 1.488 / 1.616 | **0.550 / 0.634 / 0.690** |
| **ZO249** 3439 s, 471 boundaries | 0.448 / 1.168 / 3.792 | **0.000 / 0.350 / 1.984** |

**Acoustic refinement wins on all five files, at every percentile**, by a wide margin, and
the margin is largest on the hardest material. So the existing engine was *not* a no-op —
my original inference, which I retracted, was wrong in the other direction too.

Three results that do **not** flatter the work, recorded because they matter:

- **`spectral_change` never fired.** The `on` and `legacy` arms are byte-identical on all
  five files, and `spectral_change` is the only difference between them. Its two thresholds
  are uncalibrated reasoned defaults, the only evidence for it is a synthetic pure-tone
  test, and on real material it contributed **nothing**. It is kept because it is cheap and
  gated, not because it is proven.
- **The arms do not isolate the router.** `acoustic: false` also makes the pipeline's step 7
  (embedding-only) a no-op and downgrades step 13, so the arms produce different segment sets
  — 616 turns / 7 speakers off versus 472 / 4 on for ZO249. The per-boundary distance is
  comparable; the turn counts are not evidence of anything.
- **`inside_pct` had to be discarded as a primary metric**, and the first version of the
  scorer had it wrong (it counted a boundary sitting exactly *at* a VAD onset as a
  cut-through, producing a nonsense "0.000 s median beside 65 % inside"). Distance carries the
  conclusion; `inside_pct` does not.

Every threshold in `learning` (`max_profile_intra_dist`, `min_inter_dist`,
`extend_max_dist`, all 0.32) still comes from **one file**. It is set to refuse rather than
assert, so a wrong constant loses a profile instead of inventing one, but it is not
calibrated and lesson 33's offline sweep is the way to change that.

---

## 2b. What the caveat had said (kept for the record)


I could **not** measure whether the existing acoustic refinement helps, and no claim in this
document should pretend otherwise.

- "All 461 turns abut" is a designed invariant, not a no-op signal. I initially inferred a
  no-op from it and **retracted that inference**.
- A rough set-A→set-B boundary-shift comparison (296 different-speaker set-A boundaries vs
  the nearest set-B boundary: p50 0.000 s, p90 0.438 s, max 3.232 s, 241/296 unchanged) is
  **confounded**: set B merges 480 segments into 462 turns, so absorbed set-A boundaries have
  no counterpart and inflate the apparent shift.
- The `exact_boundary_refinement_done: 152/152 transitions refined` line appeared in the
  k=2 and default runs, but the per-boundary lines were **rotated away by container restarts**.

So the first phase makes refinement measurable rather than continuing to argue about it.

---

## 3. Phases

Each phase ships behind a config flag with a documented rollback, and is verified on ZO249
**and** on the short ground-truth clips — a long file alone never surfaced these bugs (lesson 33).

### Step status

| # | Step | Status |
|---|------|--------|
| 1 | `boundary_refine` made real | **done** — `acoustic`, `spectral_novelty`, `lookaround_sec`, `min_gap_sec`, `min_section_sec`, `dip_ratio`, `min_dip_sec`, 4 spectral keys |
| 2 | Per-boundary `method` + `shift_sec` logging | **done** — DEBUG per boundary, `BoundaryStats.summary()` at INFO from all four call sites |
| 3 | A/B on ZO249 + 4 clips | **done** — section 2, with three recorded negatives |
| 4 | P0 tests | **done** — `tests/test_boundary_engine.py`, 45 tests, frozen pre-P1 arithmetic as the oracle |
| 5–9 | P1 one engine, one VAD pass, no deleted audio | **done** — `speaker/boundary.py`; `Diarizer.run` returns `raw_vad_sections`; three thin callers |
| 10–17 | P2 learn from set B | **done** — with the step-11 correction above |
| 18–20, 23 | P3 live defects | **done** — silence padding, `TimelineGuard`, WAV tail fixed, 22 tests |
| 21 | P3 server-side turn-edge trim | **done** — `speech_gate.trim_turn_edges`, trims before the gate |
| **22** | **Re-measure `min_match_confidence` against a real non-matcher population** | **NOT DONE** |
| 24–26 | P4 UI bugs, docs, dead config | **done** — lessons 40/41/42; 5 dead keys removed |
| **27** | **Rollout notes** | **PARTIAL** — the P0 A/B numbers are in section 2; the per-phase before/after attribution counts are **NOT recorded** |

**Two gaps, both honest rather than forgotten:**

- **Step 22.** `min_match_confidence` is still `0.60`, unchanged, and `min_match_margin: 0.10`
  is **still unvalidated**: a single-profile menu has no runner-up, so the margin gate has
  never been tested against a real false positive. The `0.60` value rests on a single measured
  false positive (a 5 s turn of an unregistered speaker named at conf 0.539), not on a
  measured population. Until it is done, live per-turn naming should be treated as a bonus and
  the shutdown re-diarization as authoritative — which is what the `thresholds.json` comment
  already says.
- **Step 27.** No fixture speaker-count table exists for this work, so the effect of P2 on the
  short ground-truth clips is **unmeasured**. That matters because of one deliberate
  consequence: learning now reads *turns* rather than segments, and turns are coarser, so a
  cluster whose segments all sit within `turn_merge_gap_sec` arrives as a single span →
  `unverified:too_few_segments` → nothing learned, where previously it was. That is fail-closed
  working as designed (a blend must not be learned), but it can *reduce* learning on some
  material and no run has quantified by how much. Rollback: `learning.fail_closed: false`.

### P0 — Make boundary refinement measurable (prerequisite)

1. Make `thresholds.json → boundary_refine` **real**: `acoustic: on|off`, `lookaround_sec`,
   `min_section_sec`, `dip_ratio`, `min_dip_sec`. Today nothing reads it.
2. Log per boundary: `method` (`vad_embedding` / `spectral_change` / `energy_dip` /
   `quietest_frame` / `midpoint`), the nominal and final boundary, and `shift_sec`. Add a
   per-run summary histogram.
3. Run the A/B on ZO249 and the four sherpa-onnx clips: method histogram, |shift|
   distribution, and the change in `boundary_crossing` / `low_span_turn_overlap` counts.
   Record the numbers in this file; every later phase is judged against them.
4. Tests: the config is actually read; each method is reachable; `acoustic: off` reproduces
   the pre-P1 numbers exactly (regression lock).

### P1 — One boundary engine, one VAD pass

5. New `asr_mcp/speaker/boundary.py` exposing
   `cut_between(start, end, refs, audio, cfg) -> (cut, method)`, owning the **single**
   fallback chain: VAD+embedding ownership → **new** spectral-novelty / pitch-contour
   discontinuity → longest energy dip → quietest frame → midpoint. It also owns the single
   ownership + min-cost cut that is currently duplicated three times.
6. Rewrite `pipeline._refine_turn_boundaries_exact`,
   `asr_router._refine_boundaries_with_vad` and `speaker/audio.refine_speaker_boundaries` as
   thin callers. This removes the current disagreement between them by construction.
7. Have `Diarizer.run` return `self._raw_vad_sections` so `asr_router` reuses the existing
   Silero pass instead of running the model a second time over the whole file.
8. Make `split_at_energy_dips` stop deleting audio: pieces shorter than `min_split_piece` are
   attached to the neighbour rather than dropped.
9. Tests: method selection, shift bounds, **coverage of every second of the timeline**
   (lesson 17), nothing deleted, and the three callers agreeing on the same gap.

### P2 — Learn from the boundaries the transcript uses

10. Move `_auto_collect` **after** `_prepare_turns` in both `run_transcribe` and
    `_attribute_items_against_audio`, and cut snippets from **set B**.
11. Trim each snippet to the VAD sections actually inside it so a boundary cannot cut a
    phoneme. **Inward only** — turns abut by design (lesson 17), so padding *outward* to a
    section edge would pull in the neighbour's speech and duplicate it. An edge that lands
    inside a speech section is left where it is; only leading and trailing silence is
    removed. *(This corrects the plan's original wording, which asked for outward padding.)*
12. Extend pending profiles by **embedding distance** to existing pending profiles; the
    source-stem name match becomes the fallback, not the primary key.
13. Fix the `Speaker_1` ⊂ `Speaker_10` substring collision, and make the `" #2"` collision
    loop check snippet directories as well as the DB.
14. Add per-snippet and per-profile caps to the learn path, and cap the length weight in
    `_auto_refine` so one long snippet cannot dominate.
15. Retain a `purity` statistic (max intra / min inter distance) on every learned profile and
    surface it on the pending card. **Fail closed**: a blend, or a split that cannot be
    verified, means "do not learn", not "learn a blend".
16. Batch the audio: one `load_audio` per request with in-memory slicing, and run the
    learner in the executor (lesson 19).
17. Tests: snippet `start_sec` / `end_sec` **asserted** (never asserted anywhere today), one
    decode per request, no `Speaker_1`/`Speaker_10` collision, cross-recording extension
    merges instead of minting a second profile, caps enforced, purity retained, and
    `learn_new=False` behaviour unchanged.

### P3 — Live: fix the defects, then refine

18. Coalescer: pad with the real gap samples, or declare `end` from the samples actually
    sent — the decoded audio and the declared span must agree.
19. Validate `start_sample` monotonicity in `unpack_turn` / `_turn_from_frame`; flag a
    channel as drifted and report it to the client instead of silently shifting boundaries.
20. Stop re-writing the flushed tail into the WAV in `live.js` and `live_client.py`.
21. **Server-side turn-edge refinement:** the VAD session is already loaded for the speech
    gate, so trimming each turn's leading and trailing frames by Silero frame probability
    adds no model and no measurable latency. Because shutdown re-attribution re-runs the
    offline pipeline, the same `boundary.py` then governs the final `.txt` — one fix serves
    both paths.
22. Re-measure `min_match_confidence` (0.60) against a real non-matcher population before
    changing it, and record the genuine/false-positive margin distributions.
    **NOT DONE — the only substantive gap in this plan.**
23. Tests: coalescer padding makes declared span == samples actually sent; a
    `start_sample` discontinuity is detected and reported; the tail is not duplicated;
    JS / Python / server detectors remain bit-equal; a live session's refined boundaries
    survive shutdown re-attribution.

### P4 — UI, docs, rollout

24. Fix `app.html:2067` (`names` → the real variable) and make `mergeSpeakers` surface the
    response `error`; show purity and split parts on the pending card.
25. Documentation: AGENTS.md lessons **40** (one boundary engine) and **41** (learn from the
    boundaries you display); `docs/lessons/diarization-and-speakers.md`,
    `docs/lessons/streaming-websocket.md`, `docs/uncertain-speakers.md`,
    `docs/lessons/live-client-design.md`, `README.md`, `asr-client/README.md`, and the
    `thresholds.json` provenance comments (including the ZO249 numbers from F4).
26. Remove or wire up the dead configuration listed in F6.
27. Rollout notes: record the A/B numbers from P0 step 3 here, and the before/after
    attribution counts for every phase. *(A/B numbers: done, section 2. Before/after
    attribution counts: **not recorded** — see the step-status table.)*

---

## 4. Explicitly out of scope

**Reducing OVERLAP with a voiceprint is not achievable in this codebase.**
`diarization/overlap.py` never references `known_speakers`, and `detect_overlaps` runs at
step 5c — before matching at step 9. A named profile can only reduce the *damage* afterwards
(step 7 splits single-speaker from OVERLAP; ghost/minority suppression and the second pass can
rescue text trapped in a false overlap). Claiming otherwise would be a false promise, which is
why it is written down here rather than quietly attempted.

---

## 5. Housekeeping — closed

Both items that were outstanding when this plan was written are done:

- The ZO249-phase work (`uncertainty.naming_blocked_reason` + the second-pass naming gate +
  `_split_cluster_by_cohesion` + the `learning` config section) shipped with the rest of P0–P4.
- The `memories` API token was restored to
  `eaa702985a2819e20fc7a118135e7cbe9474efdb02c51ab5f96e943058ea841b` /
  `2026-09-28 07:40:08.825719`; the test token is gone and the `default` and `tester` rows were
  never touched.

## 6. What is still worth doing

Not commitments — the honest residue, in the order it would pay off:

1. **Step 22: measure the live match gates against a real non-matcher population.** The single
   highest-value remaining item, because a false name is worse than no name and the current
   value is justified by one anecdote. Needs several *distinct unregistered* speakers, not
   corrupted audio — the earlier calibration failed precisely by measuring the wrong population.
2. **Step 27: the per-phase attribution counts on the ground-truth clips**, which would quantify
   the `unverified:too_few_segments` consequence described in section 3.
3. **Calibrate `spectral_change` or remove it.** It never fired on any real material tested, its
   thresholds are reasoned defaults, and its only evidence is a synthetic pure tone. Either move
   `spectral_min_ratio` / `spectral_min_abs` against real gaps or drop the method and keep the
   four-step chain.
4. **The four `learning` distance thresholds are all the same uncalibrated `0.32`**, taken from
   one file. They are all set to *refuse* rather than assert, so they fail safe, but a second
   file with known speaker counts would do more than another test.