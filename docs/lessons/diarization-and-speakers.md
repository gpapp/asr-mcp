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

## 39. A learned-but-unnamed speaker is a PENDING profile

### The gap

"Starting from the client does not learn new speakers." Two independent causes,
both confirmed in code:

1. `VoiceprintService.auto_collect_from_diarization` ended with
   ```python
   if is_spurious_speaker_name(speaker_name):
       continue
   if not self._db.get(speaker_name, user_id=user_id):
       continue
   ```
   Auto-collect only ever **added snippets to an already-registered**
   voiceprint. It never created one. A colleague who had never been registered
   could therefore never be learned — a chicken-and-egg the user cannot break
   from the client. (It was also not caused by the client's `save=false`;
   auto-collect runs regardless.)
2. The live WebSocket path did **no** auto-collect at all
   (`grep -c auto_collect asr_mcp/streaming/handler.py` = 0). Its call sites
   were only the three upload/path endpoints.

### Why "just create the voiceprint" is the wrong fix

The obvious fix — let a generic `Speaker N` cluster become a real voiceprint —
re-introduces exactly the failure lesson 30 exists to prevent. A cluster has no
identity; giving it a name means the next recording's nearest-neighbour vote can
be won by a cluster that was never a person, and the false attribution comes
back wearing a confident label. That is the `Gergely Papp`-at-99.3% bug from
lesson 30, in a new place.

### The shape that works: pending, then named

An unknown speaker becomes a **pending profile** — snippets on disk, a name that
says what it is, and a `pending` flag that keeps it out of every match until a
human names it.

- `voiceprints.pending BOOLEAN` + an additive `ALTER TABLE` migration in
  `init_db`, mirroring the existing snippets migration.
- `VoiceprintDB.save(..., pending=None)` where **`None` preserves the stored
  flag**. A naive `pending=False` default let any rebuild silently *promote* a
  pending profile — caught by a test. Promotion must be explicit.
- `list_all(include_pending=False)` and `search()` exclude them;
  `_load_known_speakers` — the single funnel for file diarization, live
  attribution and re-attribution — passes `include_pending=False`. That one
  line is what makes the guarantee hold everywhere.
- `set_pending(name, True)` is called **after** `_auto_refine`, because
  `_auto_refine` creates the row and a new row defaults to non-pending. Getting
  this order wrong made a learned profile matchable from its first instant.
- `is_pending_profile(name, voiceprint=None)` checks the `Pending ` **name
  prefix** as well as the flag: the flag lives on the voiceprint row, so a
  profile learned while the embedding session was unavailable (snippets but no
  row) would otherwise be invisible.
- `confirm_pending` refuses a name already used by a row *or* a snippet
  directory, so confirming cannot clobber a real colleague's profile.

### The cluster label belongs in the profile name

`pending_profile_name()` puts the sanitised cluster label in the name:
`Pending 2026-09-30 12:45 Speaker_5 talk.wav`. Without it, two speakers in one
recording produced the *same* name and the second cluster's snippets were
folded into the first person's profile. Two consequences followed:

- the extend-rather-than-mint lookup must compare the **sanitised** label
  (`Speaker_5`, not `Speaker 5`), or every re-run mints a `#2` profile;
- `LEARN_MIN_SPEECH_SEC = 10.0` and `LEARN_MIN_SEGMENTS = 2` keep a cough or a
  single bark from creating a profile at all.

### Learning has its own eligibility test

`uncertainty.eligible_for_learning(segment)` is deliberately **not**
`is_uncertain_label()` — that also rejects generic `Speaker N`, which is the one
label class learning exists to handle. It rejects: no name, `UNKNOWN`, `OVERLAP`,
an explicit `uncertain` flag, and an explicit `speaker_confidence` below the
threshold. Absence of confidence is treated as "no evidence of doubt"; only an
explicitly low value rejects.

The learning step is strictly additive: `learn_new=False` reproduces the old
behaviour exactly, and registered-speaker collection is untouched.

### The user has to be told, every time

A pending profile is invisible by design — that person reads `UNKNOWN` in the
transcript and on every future run until it is named. Silence makes the feature
look broken, so:

- the Voiceprints tab has an **Unnamed speakers** card with a name field and a
  Save action (`GET /voiceprint/pending`, `POST /voiceprint/pending/{name}/confirm`);
- `diarization_complete` (SSE) and `AttributionResponse` (live re-attribution)
  both carry `pending_profiles`;
- both clients print what was learned and say explicitly that it is **EXCLUDED**
  from matching until named, and where to act.

### Where learning can and cannot happen

The WebSocket path itself cannot learn: it only sees short endpointed turns and
never has the whole recording. Learning therefore happens at the two points that
do — `transcribe/upload` and the client's shutdown re-attribution, both of which
have the full audio.

### Extending must be idempotent, and a rename must move the file paths

Both of these were found by the real-model end-to-end run, not by the unit
tests, and both are the kind of bug that looks like "the feature is flaky"
rather than "the code is wrong".

**Re-processing the same recording duplicated every snippet.** The extend path
re-ran `add_snippet_from_segment` for each segment, which is keyed on
`{hash(source_audio)}_{format_time_short(start_sec)}_{duration}` — so a second
pass produced *the same filenames again* and a second row for each. The profile
filled with duplicates and the embedding was computed over doubly-counted
audio. The unit test that "the same recording extends one pending profile"
passed throughout, because it asserted the number of **profiles** and never the
number of snippets. `_learn_unknown_speaker` now skips a segment whose
`hash_start` key is already present, and a pass that adds nothing still reports
the profile (with `snippets: 0`) so a still-unnamed speaker stays visible.

**Renaming copied `file_path` verbatim.** `_rename_snippet_dir` moves the
directory with `Path.rename()` and then re-inserts each snippet row — passing
the old `file_path` straight through, which is the obvious thing to write and
is wrong. The row keeps pointing at a directory that no longer exists, so every
later load of that snippet fails with `No such file or directory` and the
profile cannot be re-refined. The path is now re-pointed at the new directory.

The end-to-end check that caught both asserts the property a user actually
cares about: after `confirm_pending`, every `file_path` in the database must be
a file that exists on disk. Counting rows is not evidence.

### 39a. A learner is often someone you already have — offer the merge, not the rename

`confirm_pending` refuses a name that is already taken. That is correct — it
means confirming can never overwrite a colleague's existing profile — but it is
the *wrong answer* in the most likely case, which is that the pending profile
holds 25 seconds of a person who has been registered for months. The user is
then stuck between three bad options: name them something new (two profiles for
one person, and the uncertainty policy will split their speech between two
names forever), merge them (impossible from the UI, because the merge dialog
builds its lists from `speakersData`, which filters out pending profiles), or
delete the profile and throw away the audio that was just collected.

So `pending_candidates` scores a pending profile's embedding against every
*registered* profile and returns them ranked, with the
`live_attribution.min_match_confidence` / `min_match_margin` gates applied as an
advisory `likely` flag. The UI turns each likely candidate into a one-click
"Merge into <name>", and `merge_pending_into` reuses `merge_speakers` so the
snippets move, the pending profile ceases to exist, and the target is
re-refined over the larger corpus. The merge dialog now also lists pending
profiles as merge *sources* (a pending profile is never offered as a target —
merging one unnamed profile into another just produces a second unnamed
profile).

Three rules this shape depends on:

- **Never a pending profile as a candidate.** A pending profile has no
  identity to lend; letting it win a nearest-neighbour vote is exactly what the
  uncertainty policy forbids.
- **The `likely` flag is advisory only.** Nothing merges without an explicit
  POST. A profile-vs-profile score is a cleaner comparison than a 3-second live
  turn, which is why the live gates are conservative here — but the two scales
  are not the same and the number must not be read as a live-turn confidence.
- **`merge_pending_into` re-checks `is_pending_profile`.** Without it, a named
  profile could be passed as the source and a colleague's audio moved into
  someone else's profile.

## 40. ONE boundary engine decides every cut

Until now the pipeline placed a boundary in **three different ways** and the
turn timeline in a **fourth**, and they did not agree with each other:

| Site | Rule it used |
|---|---|
| `pipeline._refine_turn_boundaries_exact` (step 13) | VAD+embedding → **quietest single 20 ms frame** → gap midpoint |
| `asr_router._refine_boundaries_with_vad` (turn timeline) | VAD+embedding → **longest energy dip run** → quietest frame → midpoint |
| `speaker/audio.refine_speaker_boundaries` (step 7) | sub-window embedding → **midpoint between two sub-window centres** |
| `asr_router._gap_boundary` (residual gaps) | longest energy dip run, thresholded on the gap **peak** |

The second and fourth disagree by construction: one looks for the *quietest
frame*, the other for the *longest dip*. On the same gap they returned different
cuts, so which one ran decided the answer.

There is also no **signal-based snapping** anywhere: no zero-crossing, no
pitch-contour discontinuity, no spectral-change detection, no forced alignment.
The only signals used to place a cut were frame-RMS minima, Silero section
edges, window-grid midpoints and arithmetic midpoints.

`asr_mcp/speaker/boundary.py` now owns the single chain:

```
vad_embedding  ->  spectral_change  ->  energy_dip  ->  quietest_frame  ->  midpoint
```

`cut_between(start, end, refs, audio, cfg) -> (cut, method)` returns the cut
**and the method that produced it**, and `refine_gap(left, right, …)` applies it
to both spans so they share one cut (lesson 17). The three former callers are
thin wrappers. The engine is numpy-only at import time — the embedding is
injected through `refs` — so it is unit-testable without the ML stack, and
`tests/test_boundary_engine.py` contains **frozen verbatim copies of the old
arithmetic** so "did this change the numbers?" is answerable by a test rather
than by a run.

Two rules the design depends on:

- **The two flags are not the same question.** `boundary_refine.acoustic: false`
  disables methods 1 and 2 (the deliberately *worse* arm used for the A/B);
  `spectral_novelty: false` alone reproduces the pre-P1 chain exactly and is the
  **regression lock**. "Reproduce today's numbers" and "disable acoustics" cannot
  both be one flag, because today's behaviour already used method 1.
- **A refused cut keeps the nominal boundary.** `refine_gap` will not move a cut
  that would leave either side below `min_side_sec`, or that moves less than
  `min_shift_sec`. The old step-7 code *deleted* the short segment instead;
  `split_at_energy_dips` dropped pieces below `min_split_piece` outright. Real
  audio is no longer discarded anywhere in the boundary path.

### What the measurement says (and what it does not)

`/tmp/opencode/ab/ab_measure.py` scores every internal turn boundary against the
**raw Silero VAD sections** — independent evidence of where speech actually
starts and stops. A boundary's *distance to the nearest VAD edge* is the metric;
lower is better. No ASR is involved, so this isolates the boundary decision.

Distance in seconds to the nearest VAD edge — **lower is better**.

| clip | `acoustic: false` (energy only) p50 / p90 / max | `acoustic: true` p50 / p90 / max |
|---|---|---|
| `0-four-speakers-zh` 56.9 s | 1.397 / 3.970 / 6.738 | **0.000 / 0.550 / 0.550** |
| `1-two-speakers-en` 16.0 s | 0.864 / 1.530 / 1.696 | **0.391 / 0.678 / 0.750** |
| `2-two-speakers-en` 34.0 s | 0.304 / 0.504 / 0.576 | **0.106 / 0.227 / 0.398** |
| `3-two-speakers-en` 54.8 s | 0.806 / 1.488 / 1.616 | **0.550 / 0.634 / 0.690** |
| **ZO249** 3439 s (471 boundaries) | 0.448 / 1.168 / 3.792 | **0.000 / 0.350 / 1.984** |

`acoustic: true` is better on **all five files**, on every percentile, by a wide
margin — and the improvement is largest exactly where the material is hardest
(four-speaker Chinese, and the 57-minute podcast). The `on` and `legacy` arms are
**byte-identical on all five files**; see the note on `spectral_change` below.

**The acoustic path is what makes boundaries land on real speech edges, and by a
large margin.** With `acoustic: false` (energy-only) the median boundary sits
0.4–1.4 s away from the nearest VAD edge; with it on, the median is 0.00–0.55 s
and on the podcast exactly 0.000 s — the `first_right_start` rule puts the cut on
a real speech onset.

**`spectral_change` contributed nothing measurable.** It is the *only* difference
between the `on` and `legacy` arms, and the two are byte-identical on all four
ground-truth fixtures. Its two thresholds (`spectral_min_ratio` 1.5,
`spectral_min_abs` 0.15) are **UNCALIBRATED** reasoned defaults, the only
evidence for the method is a synthetic pure-tone test, and the honest statement
is: a new capability, present and tested, that did not fire on this material. It
is kept because it is cheap and gated, not because it is proven.

Two methodological notes, because both were got wrong first:

- A boundary sitting **at** a VAD section start is the *best* placement, not a
  cut-through. The first version of the scorer tested `t < section.end` and so
  counted the ideal `first_right_start` cut as bad — which is how a 0.000 s
  median appeared next to a 65 % "inside speech" rate. Inside now means
  *strictly* inside, with a 50 ms edge tolerance.
- `inside_pct` alone is **confounded**: a boundary dumped in the middle of a
  long pause scores 0 % and is also wrong. Distance is the primary metric.
  (`acoustic: false` scores 0 % inside on three of four fixtures while placing
  boundaries up to 6.7 s from any speech edge — the clearest possible
  demonstration that the secondary metric cannot carry the conclusion.)

### One caveat about what the arms isolate

`boundary_refine.acoustic: false` does **not** only disable the router's
refinement. It also makes the pipeline's own step 7 (`refine_speaker_boundaries`,
which is embedding-only) a no-op and downgrades step 13 to the energy chain, so
the arms produce **different segment sets** — 616 turns with acoustics off
versus 472 with them on for ZO249, and 7 speakers versus 4. The distance metric
is still comparable per boundary, but this is not a clean isolation of the
router's decision, and the differing turn counts should not be read as "acoustic
refinement produces fewer turns".

## 41. Learn a voiceprint from the boundaries the TRANSCRIPT uses

The learning path cut its snippets from the **diarization segments** (set A)
while the transcript is attributed against the **turns** (set B) — two
independently computed boundary sets, produced by a second full-file Silero pass
and different merge gaps. Worse, two of the three placers in set A are
*midpoints* (`pipeline.py` final overlap resolution, `speaker/audio.py` step 7),
so **mid-word cuts were the normal case, not the exception.**

That closes a bad loop: a learned embedding is used as a *reference for cutting
future boundaries* (`pipeline._get_speaker_ref`, `asr_router._speaker_refs`
prefer `known_speakers[name]["embedding"]`). A mid-word snippet becomes a
mis-shaped reference, which then cuts the next run's boundaries worse.

Now:

- `Diarizer.run` returns its raw VAD sections, so `_prepare_turns` reuses that
  pass instead of running Silero a second time over the same audio.
- **The learner runs after `_prepare_turns` and is handed the turns**
  (`learn_spans = turns or list(segments)`), so snippets and transcript share
  one boundary set. Ordering is the fix; there is no second code path.
- Each snippet is **trimmed to the VAD sections inside it**, so a boundary
  cannot cut a phoneme. Trimming is **inward-only**: turns abut by design, so
  padding outward would pull in the neighbour's speech and duplicate it. Edges
  that land inside a speech section are left alone; only leading/trailing
  silence is removed.
- One `AudioSegmentSource` per request decodes the file **once**. The old
  `load_audio_segment` decoded the *entire* file per segment — 89 whole-file
  decodes for one ZO249 cluster, synchronously in the event loop.
- The learner runs in an executor (`auto_collect_from_diarization_async`).

### Learning now fails closed, and extends by embedding

`_split_cluster_by_cohesion` used to return the cluster **whole** on every
failure path — no session, `<2` segments, sklearn missing, clustering threw,
`<2` groups. A blend closer than the threshold was therefore undetectable, and
"the check failed" silently became "this is one person". It now returns
`(parts, status)` with status `ok` / `opted_out` / `unverified:<why>` and
**refuses to learn** when the split cannot be verified. `learning.fail_closed:
false` restores the old behaviour.

That matters because the failure is real and was measured. On ZO249 a cluster
that the pipeline called `Speaker 6` was learned as one pending profile of 89
snippets / 606 s — and it contained **two people**: the guest in the first half
and the host in the second. Ranking the individual spans against all 35
registered profiles gave *Alexander Radyukin* at 0.53–0.59 for the guest's spans
and *Gergely Papp* at 0.39–0.53 for the host's, with a within-group spread of
0.175–0.291 and a between-group spread of 0.368–0.448. Naming that profile would
have asserted one name for two people **and corrupted the turn boundaries of
every later run**, because both refinement engines consume the profile. Note
also that `pitch_std` is 40–59 Hz for *every* cluster on this material, so pitch
cannot serve as a purity signal.

`_find_pending_to_extend` now extends a pending profile by **embedding distance**,
with the old source-filename-stem name match demoted to a fallback. The old key
meant the same colleague in a *second* recording minted a *second* unnamed
profile, and the second "Save name" was then **refused** as a duplicate. Also
fixed: `Speaker_1` is a substring of `Speaker_10` (now token-exact), the `" #2"`
collision loop only checked the DB (now snippet directories too), the learn path
had **no** duration cap while `_auto_refine` is duration-weighted (now capped at
`max_refine_weight_ratio × median`, and per-snippet / per-profile ceilings), and
a per-profile `purity` statistic is stored (`voiceprints.purity`, additive
migration) and surfaced on the pending card.

**Every one of these distance thresholds is the same uncalibrated 0.32**, taken
from the single ZO249 measurement above. Each gate is set to **refuse** rather
than assert, so a wrong constant loses a profile instead of inventing one — but
0.32 has been validated on one file and should be swept properly (lesson 33:
cache the embeddings and sweep offline) before it is trusted as calibrated.

A deliberate consequence of using turns: a cluster whose segments all fall within
`turn_merge_gap_sec` arrives as a single span, so it is reported
`unverified:too_few_segments` and nothing is learned. Previously it would have
been learned from many short segments.
