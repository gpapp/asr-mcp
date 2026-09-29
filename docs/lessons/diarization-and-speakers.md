# Diarization, speaker identity and the uncertainty policy

Lessons 1, 2, 3, 4, 11, 29, 30, 32, 33 of `AGENTS.md`. Speaker labels, ghost/minority suppression, attribution, and clustering quality.


### 1. Speaker Labels: ALWAYS "Speaker N" (1-indexed)
- Every code path that creates or assigns speaker labels MUST use `f"Speaker {n}"` with 1-indexed integers.
- NEVER use `"SPEAKER_00"`, `"SPEAKER_01"`, or any zero-indexed underscore format.
- **Why**: The renumbering logic in `pipeline.py` parses labels with `int(name.split()[-1])`. Non-"Speaker N" labels cause `ValueError: invalid literal for int() with base 10`.
- Affected files: `profiling.py` (`relabel_by_pitch`), `segment_ops.py` (`absorb_islands`), `clustering.py` (`match_known_speakers_full`), `pipeline.py` (renumbering block).

### 2. Wire Up Loaded Models — Don't Just Load Them
- If an ONNX session is loaded into `ModelState`, every code path that uses it MUST pass the session object through.
- **Why**: Silero VAD was loaded but `_run_vad()` never passed `state.vad_session` to `run_vad_onnx()`, silently falling back to energy-based VAD for weeks.
- **Rule**: After loading a model in `model_loader.py`, grep for all call sites and verify the session is actually used.

### 3. Diarization Segments Must Be Speech-Length, Not Window-Length
- VAD + energy-dip splitting produces the base segments. Sliding windows are ONLY for embedding extraction, NOT for defining segment boundaries.
- After embedding/clustering, `collapse_same_speaker_segments(max_gap=0.5)` merges same-speaker windows.
- **Why**: 2.0s windows with 1.2s stride create overlapping segments that don't match natural speech.
- Current params: window_sec=3.0, stride_sec=2.5, collapse max_gap=0.5s, absorb_islands gap 0.5s.

### 4. Merge Nearby VAD Regions Before Splitting
- `merge_vad_sections(self._raw_vad_sections, max_gap_sec=0.5)` (`pipeline.py:83`) merges VAD regions separated by <0.5s silence. The helper's own default is 0.1s, so the pipeline must pass 0.5 explicitly.
- **Why**: Silero VAD produces many short regions during brief pauses (breaths, filler sounds) within a single speaker's turn.


### 11. When Modifying Pipeline Order, Update ALL Downstream Code
- The 13-step pipeline has strict ordering: profiling → merge_similar → relabel → renumber centroids → boundary refine → ghost → match.
- After any reorder, check that centroid key formats (integer vs string "Speaker N"), segment label formats, and lookup methods all still match.
- **Why**: Mismatched centroid keys caused silent failures where `match_known_speakers_full` received empty clusters.


### 29. Auto-Collect: Cumulative Total vs Per-Chunk Cap
- `auto_collect_from_diarization()` (voiceprint/service.py) pre-loops each segment while `current_total < AUTO_COLLECT_MAX_SEGMENT_SEC` (300s) — but `current_total` is the speaker's CUMULATIVE DB total (target cap is `AUTO_COLLECT_MAX_TOTAL_SEC` = 1200s). Any speaker already ≥300s (23 of 35 here) never entered the loop → silently collected zero snippets forever. The per-chunk length is already bounded inside by `chunk_end = min(seg_start + MAX_SEGMENT_SEC, seg_end)`.
- Rule: gate the OUTER loop on `MAX_TOTAL_SEC` only; keep `MAX_SEGMENT_SEC` as an inner chunk bound. Cap skips now log `Auto-collect: %s at snippet cap`, and upload routers log `Auto-collected %d snippets from %d segments` so silent skips are visible.
- Auto-collect runs unconditionally after diarization (NOT gated on `save=false`) and only for speakers with a registered voiceprint for the requesting `user_id` (generic `Speaker N` labels are skipped by design).


### 30. Uncertain Speakers: Suppress the IDENTITY, Keep the Text
- **Policy**: when a speaker identity cannot be established, emit `speaker=None`
  (+ `uncertain=True`, `attribution_reason=...`, `speaker_source="unknown"`).
  Never remap the span to the nearest / most common / temporally closest
  speaker — a confident-looking wrong label is worse than an honest blank.
  The default **retains the text** (`uncertainty.retain_uncertain_text`); only
  drop content if a product requirement demands it.
- **Single source of truth**: `speaker/uncertainty.py` (dependency-free: no
  torch/ORT, so it is unit-testable). Every module goes through it —
  `diarization/segment_ops.py` (ghost/minority), `speaker/attribution.py`
  (span→turn), `voiceprint/service.py` (auto-collect), `streaming/handler.py`
  (live turns), `api/asr_router.py` (result assembly). The attribution *logic*
  also lives outside `asr_router` on purpose: that module imports torch (via
  `speaker.vad`), so keeping `attribute_span` in a torch-free module is what
  makes it testable.
- `suppress_uncertain_speaker()` returns a **copy** with `original_speaker`
  kept for debugging; it never mutates the input segment.
- `is_uncertain_label()` is for **auto-collection/label hygiene** (rejects
  generic `Speaker N`); `has_speaker_identity()` is for **merging/attribution**
  (accepts `Speaker N` — the diarizer legitimately emits them). Using the wrong
  one suppresses every ordinary result or merges unknown spans together.
- Ghost/minority speakers are **suppressed, not reassigned**
  (`suppress_ghost_speakers` / `suppress_minority_speakers`; pass `suppress=False`
  to restore the legacy nearest-speaker behaviour). Suppression happens AFTER
  pitch relabel + `Speaker N` renumbering, so `int(name.split()[-1])` parsing
  (lesson 1) can never see a `None` label.
- `_collapse_confident_only()` replaces `collapse_same_speaker_segments` after
  suppression — the latter would glue adjacent UNKNOWN spans into one block.
- Uncertain spans are **never merged** with each other: `_merge_into_turns` and
  `_merge_consecutive_same_speaker_results` require a non-`None` speaker. Two
  unrelated "yes" answers are not one turn.
- Merge gaps dropped 3.0s → `uncertainty.turn_merge_gap_sec` (1.0s) and
  `result_merge_gap_sec` (1.0s); wide bridges were a major source of cross-turn
  attribution.
- **Boundary crossing** (`speaker/attribution.py::attribute_span`): a **short**
  span (≤ `uncertainty.max_boundary_cross_span_sec`, 3.0s) whose midpoint turn
  holds less than `max_boundary_cross_sec` (0.3s) AND less than
  `1 - max_boundary_cross_ratio` (0.34) of the span is mostly *other*
  speakers' audio. Midpoint attribution cannot split it → `speaker=None`,
  reason `boundary_crossing`. Sub-threshold bleed (≤0.2s) stays attributed.
  This is the "Yes."-straddling-the-cut case the policy targets.
- **Long spans are NOT boundary crossings** (`low_span_turn_overlap`): coarse
  backends emit one segment per ~30s decode window, so *any* span longer than a
  turn is mostly outside it — the crossing test degenerates and blanked whole
  files. Long spans are attributed only when the turn actually holds ≥
  `min_speaker_confidence` (0.35) of the span; otherwise the identity is
  dropped with reason `low_span_turn_overlap`. Verified on a real 60s run: the
  two coarse spans went from one merged `boundary_crossing` block to two
  truthful `low_span_turn_overlap` results.
- **Uncertain items are never merged into one run** (`attribute_items`): runs
  are keyed by `(speaker, reason, turn_index)`, so two unknown spans from
  different turns stay separate. Keying by `(speaker, reason)` alone produced
  one giant UNKNOWN block that hid the real segmentation.
- **Ghost share guard** (`ghost_max_share`, 0.25 in
  `diarization/segment_ops.py::eliminate_ghost_speakers`): the 10s absolute
  ghost rule is a knife edge on short clips — on a 60s file a 9.7s speaker was a
  "ghost" while a 10.9s one survived. A speaker holding >25% of the recording's
  speech is a participant, not a ghost, and is never suppressed.
- `policy_enabled()` (`uncertainty.enabled: false`) is the documented rollback
  and must bypass **all** suppression, including `attribute_span` — not just the
  confidence checks in `uncertainty.py`. Verified: with the policy off a 60s
  single-speaker file returns one result labelled `Speaker 7`, `status: ok`.
- **Caveat seen in production**: that same 60s file clustered into 20 clusters
  (capped to 15) for a single speaker, so the policy honestly reports most of it
  as UNKNOWN while the legacy absorb step hid the failure behind one name. If a
  recording is routinely blanked this way, fix the clustering
  (`thresholds.json` distance/greedy-merge values) instead of re-enabling the
  proximity guess.
- Every `TranscribeResult` carries `speaker_confidence`, `speaker_source`
  (`known_voiceprint` | `diarization_cluster` | `unknown`), `uncertain`,
  `attribution_reason`. The `done` SSE event adds `status`
  (`ok` | `partial` | `error`) and `uncertain_segments`; the GUI, the JSON
  API and the Windows client must all render `None` as `UNKNOWN` and must not
  present an empty metadata-only `.txt` as success.
- **Auto-collect gate**: `eligible_for_auto_collect()` requires a real named
  voiceprint + `speaker_confidence` ≥ 0.35 + ≥0.5s. UNKNOWN/OVERLAP/generic
  spans can never poison a voiceprint.
- `pre_segmented=True` (streaming live turns) tells whisper to skip its own
  Silero VAD and `condition_on_previous_text` — the server already did the
  segmentation, and a second VAD drops short answers.
- **Every rename must carry its evidence.** `apply_identity()`
  (`speaker/uncertainty.py`) stamps `speaker_confidence` / `speaker_source` /
  `speaker_margin` / `speaker_match_dist` onto the segments it renames, and
  `_merge_into_turns` / `_split_long_turn` propagate those keys onto turns.
  Without them `attribute_span` finds no confidence and falls back to the
  geometric overlap ratio — reporting a 0.39-confidence voiceprint match as
  `1.0`. A segment renamed to a real name without `speaker_confidence` is a bug.
- **The second pass may not rename on a weak match** (`second_pass
  .min_identity_confidence`, default 0.5, linear in the combined distance:
  `1 - combined/0.5`). A 52-minute, 3-person podcast with 35 voiceprints in
  the menu had both of its dominant clusters renamed to the same person at
  combined distances 0.303/0.324 (confidence 0.39/0.35) — the best of a bad
  lot, presented as certainty. Below the gate the cluster stays `Speaker N`.
  This is separate from, and additional to, the one-to-one `claimed_by`
  constraint in `collapse_unknown_speakers_second_pass`.
- **Honest output can still be wrong** — a generic `Speaker N` is only better
  than a false name. If a routine recording comes back mostly UNKNOWN, check
  the `_cluster_embeddings` balance diagnostic in the log and fix the
  CLUSTERING (lesson 32), not the naming policy. On the long podcast the
  clustering was fine; on **short** ground-truth clips the same diagnostic
  shows 9–13 clusters for 2 speakers, and ghost cleanup then eats a real
  participant (lesson 33).


### 32. `num_speakers` Is a DIFFERENT Clustering Path — Never Assume It Is Just k
- `_cluster_embeddings` branches on `num_speakers`:
  - **`num_speakers` given** → hard `n_clusters=k`, and `greedy_merge_clusters`
    is **SKIPPED** (plain centroids instead of merged ones).
  - **`num_speakers` omitted (the default)** → threshold agglomerative →
    `cap_clusters(15)` → `greedy_merge_clusters(0.25)`.
- The forced-k branch used **average linkage**, which chains: it merges a few
  far-apart windows early and then drags everything into the first cluster.
  Measured on zo230's 1364 real ECAPA windows (2070s of speech), k=3:
  - average → `182 / 1180 / 2` windows = **86.5%**, 1924s in ONE cluster
  - complete → `762 / 244 / 358` = **55.9%**, 1223s / 216s / 630s
  - single → `1362 / 1 / 1` = 99.9% (catastrophic — hence complete, not single)
  - default path (no `num_speakers`) → `712 / 442 / 151` = **52.2 / 32.4 / 11.1%**
    with all other clusters ≤20 windows, and that split is **identical for every
    `distance_threshold` 0.30–0.60 and for both average and complete** —
    `greedy_merge_clusters` dominates, so `distance_threshold` is NOT the lever.
- Fixed: the forced-k branch now uses `linkage` from
  `diarization.forced_k_linkage` (default **complete**, validated against
  `complete|average|single`, falling back to complete on anything else).
- End-to-end on zo230 after the fix (`POST /api/asr/diarize/upload`,
  3132s, `memories` with 35 voiceprints incl. Gergely Papp):
  - `num_speakers=3` → **68.3% Gergely Papp / 31.2% Speaker 3 / 0.5% OVERLAP**,
    3 speakers, 0 uncertain (was 99.3% Gergely Papp / 0.7% everything else).
  - default (no `num_speakers`) → 61.6% / 32.6% + 5 small fragments, 15 uncertain
    segments (1.5%). The dominant/secondary split is the same either way.
- **Do not "fix" a lopsided split by editing the naming policy.** Check
  `Cluster balance: N windows -> K clusters (largest X%)` in the log first, and
  remember that lopsidedness is a property of WHICH branch ran. An earlier
  session attributed a 99.3%-one-identity podcast to ECAPA quality when the
  cause was its own `num_speakers=3` test parameter; the default path was fine
  **on that 52-minute file**. Lesson 33 shows the default path still fails on
  short clips — measure on both before calling a threshold good.
- The pathology is **not reproducible from synthetic embeddings** (a
  three-group fixture collapses identically under every linkage). It depends on
  real embedding geometry, so `tests/test_clustering_linkage.py` tests the
  plumbing (which linkage is used, the fallback, config isolation) and the
  numbers above are recorded here rather than asserted in a test.
- Material note: zo230's per-window spectral flatness is p50 **0.227** (voiced
  speech is ~0.01–0.1) and no window is below RMS 0.067 — a noisy/roomy or
  music-bedding source, which is why ECAPA separation there is weak in the
  first place. Adding a silence filter + per-window L2 renormalisation to
  `speaker/embedding.py` did NOT help (0.0707 → 0.0707) and was reverted,
  because it would also have invalidated all 35 stored voiceprint embeddings.

### 33. Ground-Truth Fixtures: Short Clips Over-Cluster, and Ghost Cleanup Then Eats a Real Speaker
Lesson 32 concluded the default path "was fine". **It is not — it was only
measured on a 52-minute file.** Ground-truth clips show the opposite failure on
short audio, and it is a *different* failure from the forced-k one.

**Fixtures** (public, no auth, mono 16 kHz; fetched from the sherpa-onnx
`speaker-segmentation-models` release):
```
https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/
  0-four-speakers-zh.wav   56.9s   4 speakers
  1-two-speakers-en.wav    16.0s   2 speakers
  2-two-speakers-en.wav    34.0s   2 speakers
  3-two-speakers-en.wav    54.8s   2 speakers
```
Run them as a user with **no voiceprints** (a static `X-API-Key` maps to
`default`), otherwise the 35 `memories` voiceprints confound the measurement.

**Measured, default path, no `num_speakers`, BEFORE the `merge_threshold`
fix below (dt 0.35 / mt 0.25):**
| clip | truth | found |
|---|---|---|
| `1-two-speakers-en` | 2 | 2 speakers, 2 uncertain |
| `2-two-speakers-en` | 2 | **1** + 19.6s UNKNOWN (10 of 12 segments) |
| `3-two-speakers-en` | 2 | **1** + 36.2s UNKNOWN (14 of 17 segments) |
| `0-four-speakers-zh` | 4 | **1** + 29.5s UNKNOWN (12 of 13 segments) |

- Mechanism: `Cluster balance: 17 windows -> 9 clusters` for a **two-person**
  clip. The clusterer fragments each real speaker into ~7 pieces; every piece
  is then under `eliminate_ghost_speakers`' 10s absolute rule, so all of them
  are suppressed, and `ghost_max_share` (0.25) rescues only the single largest
  cluster. In a 2-person conversation the *other* person therefore always loses.
- The separation was correct before the damage: on `3-two-speakers-en` the
  transcript gives `Speaker 8` the answers ("Yeah, sure…", "Sure. One popular
  example is GPT-3") and UNKNOWN the questions. The right turns were found and
  then discarded as ghosts.
- **The uncertainty policy's core promise held**: 324 / 486 characters of text
  retained on the UNKNOWN spans, zero content lost. The failure is a
  *usability* one (wrong identity withheld), not a data-loss one.
- Forced-k linkage is a **no-op at this size** — `complete` and `average`
  produced byte-identical output on all four clips (verified by flipping
  `forced_k_linkage` in the container's config copy and confirming via
  `get_config()` that the override loaded). At n=17 the dendrogram barely
  depends on linkage. Lesson 32's `complete` default still stands on the
  1364-window podcast evidence.

**Threshold sweep — the lever is `merge_threshold`, and the podcast cannot
detect the bug.** zo230 replayed offline from cached per-window embeddings
(`cap_clusters` + `greedy_merge_clusters` via the `tests/conftest.load_module`
trick, no GPU needed):
| `distance_threshold` | `merge_threshold` | 2-speaker clips | 4-speaker | zo230 podcast |
|---|---|---|---|---|
| 0.35 (old) | 0.25 (old) | 100%, 100% | 100% | 52.2 / 32.4 / 11.1 |
| 0.55 | 0.35 | 100%, **50%**, **50%** | **57%** | 52.2 / 32.4 / 11.1 |

- zo230 is **identical to two decimals across `distance_threshold` 0.35–0.65 ×
  `merge_threshold` 0.25–0.35** (0.45 → 54.6/32.4/11.1; 0.55 collapses to
  87.8%). So `distance_threshold` is not the lever, and a podcast-only test
  suite can never surface this class of bug.
- A 50/50 split on a 2-person clip is the target: `big=NN%` in the table is
  the largest cluster's share of speech, so 100% means one speaker swallowed.
- **Applied**: `diarization.distance_threshold` 0.35 → **0.55** and
  `merge_threshold` 0.25 → **0.35** (thresholds.json). Re-verified end to end
  in the real pipeline, default path, `known_speakers={}`:
  | clip | before (0.35/0.25) | after (0.55/0.35) |
  |---|---|---|
  | 1-two-speakers-en (16s) | 2 speakers, 2 uncertain | 1 speaker 7.2s + 2.5s UNKNOWN |
  | 2-two-speakers-en (34s) | **1** + 19.6s UNKNOWN (10/12 segs) | **2 speakers 9.8s / 9.8s** + 7.6s UNKNOWN |
  | 3-two-speakers-en (55s) | **1** + 36.2s UNKNOWN (14/17 segs) | **2 speakers 25.8s / 25.7s, 0 uncertain** |
  | 0-four-speakers-zh (57s) | **1** + 29.5s UNKNOWN (12/13 segs) | 2 speakers 10.0s / 7.5s + 14.5s UNKNOWN |
  The forced-k path is untouched by these keys and produced byte-identical
  output before and after. zo230 (3132s) end to end: 62.5% / 33.2% with 1.6%
  uncertain, versus 61.6% / 32.6% with 1.5% before — **no long-file
  regression**. The 16s clip is a wash (it has only 9.7s of speech, one turn
  each), and the 4-speaker clip still finds 2 of 4.
- No invented names appeared in any run (the fixtures run as a user with no
  voiceprints).

**Generalisable rules**
- Any diarization quality claim must be checked on a **short** clip as well as
  a long one. A 52-minute file has 1364 windows; a 34s file has 17. The
  short-file regime is where the thresholds were wrong.
- `ghost_max_share` protects exactly ONE cluster. It is a mitigation for
  lesson 30's knife edge, not a general fix for fragmentation. After the
  `merge_threshold` fix the 34s and 55s clips split correctly, so the residual
  failures are fragmentation that survives a *correct* split — not the merge
  threshold.
- For clustering work, **cache the per-window embeddings once**
  (`/tmp/.../zo230_emb.npz`) and sweep offline. Each GPU embedding pass is
  ~40s, but a threshold sweep is then seconds and needs no GPU.
- Drive `Diarizer.run()` / `_transcribe_file()` **directly in-container** for
  diagnostics: it bypasses auth, cannot mutate the DB, and `known_speakers={}`
  removes the voiceprint confound. Pass **numpy** to `_transcribe_file` —
  a torch tensor yields an empty transcript that looks like a product bug.
- Side finding: whisper fails `cuda/int8_float16` *and* `int8` on a 4GB card
  when `ensure_ready()` has already taken the embedding arena, then silently
  runs on CPU. Unrelated to the policy, still open.
