# Voiceprints

How a speaker's voice profile is created, kept honest, and reviewed. Endpoint
reference is in [api.md](api.md#voiceprints-apivoiceprint); settings are in
[INSTALLATION.md](../INSTALLATION.md#server-settings-reference).

- [How a profile is created](#how-a-profile-is-created)
- [Pending profiles](#pending-profiles)
- [One file, one contribution](#one-file-one-contribution)
- [The per-file snippet budget](#the-per-file-snippet-budget)
- [Integrity audit and rebuild](#integrity-audit-and-automatic-rebuild)
- [Consolidation](#consolidation-finding-snippets-that-are-not-this-speaker)
- [Matching: distance vs confidence](#matching-a-distance-reads-differently-than-a-confidence)

---

## How a profile is created

A speaker with no voiceprint yet is learned automatically — but as a **pending
profile**, never as a usable voiceprint.

After a file upload, or a live session's re-attribution, an unknown speaker
(≥10 s of speech across ≥2 segments) is saved as
`Pending <date> Speaker_5 <source>` with its snippets. The client prints what it
learned.

Three properties make this safe:

**Snippets are cut from the boundaries the transcript uses.** Learning runs
*after* turn preparation, and is passed the **turns**, not the raw diarization
segments — so every learned snippet carries the same boundaries the transcript was
attributed on. Each snippet is then trimmed inward to whole VAD sections, so a
boundary can never land mid-word. A mid-word snippet would otherwise become the
*reference* for cutting the next run's boundaries.

**One recording, one contribution.** See below.

**A cluster is not a person.** Before learning, `_split_cluster_by_cohesion`
embeds each span individually and re-clusters them at
`learning.max_intra_cluster_dist`. Two voices in one cluster then yield two
profiles instead of one poisoned profile.

### The cohesion split re-joins; it does not refuse

This is the counterintuitive part. If the groups produced by the split sit closer
together than `learning.min_inter_dist`, that split is an **artefact** and the
groups are **merged back into one part** — which also brings back any dropped
one-span groups.

It used to return "unverified: groups too close" and learn nothing. That
discarded the *largest and richest* clusters: on `2026-10-07_17-00-32.mp4`
fourteen clusters were refused at 0.175–0.281 apart, while `Speaker 1` and
`Speaker 9` — learned from the same run — had intra-group **means** of 0.179 and
0.170. The refused "groups" were closer together than a learned single speaker is
spread internally. Too close is evidence of **one voice**.

**There is no maximum-based purity gate, on any group**, re-joined or not. A
maximum over n(n−1)/2 pairs grows with the span count, so it rises with the
richness of the cluster — which is why gating on it refused every cluster of
`2026-09-04 12-08-59.mp4` (22 spans, mean 0.21, max 0.379). A re-joined group
briefly had such a gate (`max_profile_intra_max` 0.40), justified by a synthetic
two-voice blend (mean 0.252, max 0.42); real data killed it, because `Speaker 9` —
a speaker the user had **already learned** — is 20 spans, mean 0.255, max **0.463**.

The merge window and the blend window are **disjoint**: the re-join fires only
when every group's nearest neighbour is under `min_inter_dist`, and ZO249 — the
real blend this exists for — had its groups 0.368 apart, so it splits into two
profiles. Undoing the merge cannot smuggle a blend past the split.

Consequence: `min_inter_dist` is a **merge trigger, not a refusal bar**. One number
decides learn-one-or-learn-two. A split is only ever *refused* for
`too_few_segments`, `too_few_embeddings`, `no_embedding_session`, `no_sklearn`,
or `clustering_failed` — all fail-closed via `learning.fail_closed`.

> When writing a test fixture for the re-join, use a **wide-spread single voice**
> (`_one_voice(jitter=0.68)`), not a two-voice fixture. Two clean voices cannot
> reach the branch: average linkage splits on the *mean* cross-pair distance while
> the check compares *centroids*, and for two clean voices those coincide — so the
> window is empty and the branch is unreachable without monkeypatching a
> threshold. A real cluster is never clean.

---

## Pending profiles

A pending profile is **excluded from every voiceprint match**. An unnamed profile
has no identity, so it can never win a nearest-neighbour vote; that person's
speech therefore still reads `UNKNOWN` (text kept) until you name them.

**Naming is the only action that makes one recognisable.** In the web UI:
Voiceprints → Unnamed → type a name → *Save name*. Confirming refuses a name
already in use, so it cannot overwrite a real colleague's profile.

Re-running the same recording — **or a different one** — extends an existing
pending profile instead of minting a second one. The match is by **embedding
distance** to the other pending profiles (`learning.extend_max_dist`), with the
filename stem only as a fallback. Before extending, the new audio is compared
against the profile's own embedding and refused if it spreads wider than
`learning.max_profile_intra_dist`.

> Keying on the filename stem instead was a real bug: the same colleague in a
> second recording minted a second pending profile, and the second confirm was
> then *refused* because the name was taken.

### "Is this someone I already added?"

Every pending profile is scored against your registered speakers. When one looks
like somebody you already have, the card offers **Merge into …** instead of
making you invent a second name for the same person — two profiles for one person
splits their speech between two names in every transcript.

**Nothing is ever merged automatically.** The flag is a suggestion; merging is
your decision.

### "Who is this?" — the audio has to be playable

An unnamed profile folds out into its own snippet list with the same play button,
filename, source and *Remove* control a named speaker has. The audio is the only
evidence of who the voice is, so it must be playable **in place**, not summarised
as a row of counts.

---

## One file, one contribution

Re-processing a recording must **replace** that recording's take, not stack a
second copy of the same room and channel onto a profile — which is how a profile
ends up fitted to a single recording.

A recording is identified by its **filename hash** (`generate_segment_hash`, md5
of the source stem — the same hash snippet filenames are built from). A reprocess
therefore matches across a different directory, extension, upload name, and every
moved span or duration.

- **Upload path** — `add_snippet(replace_source=True)` deletes the same speaker's
  rows from the same source **before** writing the new file. Required, because
  the filename derives from stem+span+duration, so a re-run lands on the same
  path.
- **Reprocess path** — `auto_collect_from_diarization` and `_learn_unknown_speaker`
  **collect first**, then `replace_source_contribution`, then a single
  `_auto_refine`. Collecting first is what makes it safe: if every span was
  rejected, the previous take is still there.
- **Scope** — one speaker, one user. A file may feed several profiles, and only a
  profile this run rebuilt is rewritten. A row with `source_audio IS NULL` is
  replaced too when its filename carries the recording hash
  (`snippet_source_identity`); only a filename naming no hash is left alone.
  Clips from other recordings are always kept.
- `SnippetDB.find_duplicate` is **speaker-scoped** and *deletes* what it finds, so
  an unscoped lookup would let one profile's collection remove another's audio.

The learn cap and `AUTO_COLLECT_MAX_TOTAL_SEC` are judged on the
**post-replacement** total, so a profile sitting at the cap because of this
recording can refresh it.

Opt out per-request with `?replace_source=false` on
`POST /api/voiceprint/speakers/{name}/upload`.

---

## The per-file snippet budget

One recording may contribute **at most `AUTO_COLLECT_MAX_PER_SOURCE` (10)
snippets to one profile**, and **every snippet is at least
`AUTO_COLLECT_MIN_DURATION` (5s)** long. Both rules apply to every profile —
registered and pending alike, because an unnamed profile is a profile the user
will eventually name.

### Why a per-file cap at all

The failure this prevents is not a reprocess; the one-file-one-contribution rule
already handles that. It is a single **long** recording: a fifty-minute meeting
attributed to one colleague for forty of those minutes yields forty clips from
one file, and the profile then describes that room, that microphone and that
afternoon rather than the person. The cumulative `AUTO_COLLECT_MAX_TOTAL_SEC`
does not stop it, because it counts seconds across all files and the profile
never reaches the cap from one recording of ordinary length.

It is independent of the cumulative cap: a speaker heard on ten recordings may
hold ten full per-file budgets, and one long recording may not eat the whole
1200s.

### How the ten are chosen

**Best fit first.** Candidate spans are ranked by the attribution's own verdict —
`speaker_confidence`, with duration as the tiebreak — and the budget is spent in
that order. This is free: it reads fields the turn already carries, and costs no
embedding pass. A confident 12s turn is collected before a marginal 40s one
that would otherwise consume half the ceiling.

A generic `Speaker N` cluster carries no confidence at all
(`learning_refusal_reason` does not require one). Those rank at **1.0**, not
0.0 — ranking them at the bottom would starve the learn path of the budget it
has to spend. With no verdict recorded the ranking falls back to duration, which
is the same thing the cleanup ranks on.

**The floor is measured on the stored clip, after the VAD trim.** A 5.2s turn
whose head is silence is a 4.6s snippet once its edges are snapped onto speech
sections, and it spends a slot for audio that is mostly silence. The trim happens
after the eligibility check, so a span can clear the floor as a turn and fail it
as a clip.

A **trailing sliver** left behind by `AUTO_COLLECT_MAX_SEGMENT_SEC` (300s) or by
`learning.max_snippet_sec` is dropped rather than stored: extending the chunk
would break the per-snippet cap.

When spans beyond the budget are dropped, the run's diagnostics say so
(`over_per_file_cap`, `too_short_after_trim`), so "3 of 27 spans" is
distinguishable from "there was almost no speech".

### Cleaning up an existing library

A reprocess applies the budget to the file it re-runs — which means the policy
only ever reached the recording you happened to re-process next. The `trim_snippets`
step of `POST /api/voiceprint/integrity/cleanup` applies it backwards over the
whole library:

| Rule | Detail |
|---|---|
| **≤ 10 snippets per file** | Ranked by stored duration, longest first, then earliest start, then lowest id — total and deterministic, so a re-run makes the same decisions. |
| **≥ 5s per snippet** | Decided **first and independently of rank**: a sub-floor snippet never earns a slot, so it cannot crowd out a good clip by taking one. |
| **Same file across names** | One recording, identified by the filename hash — a different directory, extension or upload name is still one file and one budget. |
| **No recorded source ⇒ read the filename** | A `NULL` `source_audio` is no longer automatic grounds for exemption: `snippet_source_identity` recovers the hash `add_snippet` wrote into the filename, so the row is grouped with the rows that kept their path and trimmed with them. Only a row that names **no recording at all** — no stored source *and* no hash in the filename (`Speaker_1.wav` dropped in by an operator) — is exempt, because no per-file budget can be counted against a file nobody can name. `<hash>_manual.flac` hashes the audio, not a recording, so it is not an identity either. |

It is **dry-run by default**, like the rest of that endpoint, and it re-refines
each touched speaker **once** over the final set — not once per removed row.

Note what the ranking is **not**: "newest wins" would mean ranking on the row id,
which records when a row was written rather than how much of the speaker it
holds. A re-processed file never reaches the trim anyway, because its previous
take was already replaced.

---

## Integrity audit and automatic rebuild

`asr_mcp/voiceprint/audit.py` is **read-only** and reports 8 findings.
`voiceprint/rebuild.py` + `cleanup_integrity` are the two halves that act. A
finding with **one** correct answer is repaired automatically; anything needing a
human decision is only reported.

### Automatic (on audit failure)

Runs in the server lifespan — the documented exception to "no model load at
startup":

1. `run_startup_audit` returns reports; `vp_rebuild.enqueue_from_reports(...)`
   writes `<data_dir>/voiceprint_rebuild_queue.json`.
2. The TTL monitor's 60 s tick drains the queue, but only when it is non-empty
   **and** no job is running. It calls `state.ensure_diarize_ready()` (VAD +
   ECAPA only — **never** the ASR backend), then `run_pending` in the executor.
3. `rebuild_profile_from_scratch(speaker, user_id)`: purge dead snippet rows →
   drop the derived `voiceprints` row → `_auto_refine` → re-apply `pending` and
   `purity`.

**Why dropping the row is safe**: the snippets and their FLAC audio are the
source of truth and are **never** touched. Only the derived embedding row is
discarded and recomputed. Idempotent; cannot lose material.

**Purge first, rebuild second.** `_purge_dead_snippet_rows` deletes **rows only**
— never a file that resolves — for rows whose audio is gone. Measured on the live
store: the first boot re-embedded 219 snippets (~25 min of GPU) and fixed nothing,
because 73 rows pointed at `.wav` files left over from a migration. Purge needs no
GPU, so it runs even when the models are absent (then the entry is `skipped` and
kept). A row whose file merely fails to **load** is not dead and is kept.

`pending` and `purity` are read **before** the delete and re-applied after,
because `save(pending=None)` preserves the flag but a deleted row destroys it —
without the re-apply a learned-but-unnamed profile would silently become
matchable.

### What is and is not a target

| Automatic repair (`TARGET_FINDINGS`) | Reported only |
|---|---|
| `snippets_without_profile` | `near_duplicate_profiles` |
| `profiles_without_snippets` | `name_collisions` |
| `missing_snippet_files` | `duplicate_snippet_content` |
| | `orphan_snippet_files` |
| | `stale_speech_totals` |

Everything in the right column is a **user decision** — merging profiles is not
automatic.

### Convergence

`MAX_ATTEMPTS = 3`, then the entry is dropped with an error naming
`POST /api/voiceprint/integrity/cleanup`. A `skipped` result is **not** an attempt.
A crash mid-rebuild is self-healing: the next audit reports
`snippets_without_profile`, which is itself a rebuild target.

### Manual

`GET /api/voiceprint/integrity` — read-only report.
`POST /api/voiceprint/integrity/cleanup` — fixes duplicate-content rows and dead
rows, enforces the per-file snippet budget, then re-refines each affected speaker
**once**. **Dry run unless `dry_run: false`.**

---

## Consolidation: finding snippets that are not this speaker

A profile is the weighted average of its snippets' embeddings. A snippet that
belongs to somebody else — a bad diarization turn, a crossed channel, a
mis-attributed span — drags that average toward itself, quietly making the
profile worse for every later recording.

`consolidate_speaker` answers *"is every snippet of this profile actually the same
person?"*. It **flags** the outliers and recomputes from the **kept** snippets.
**It deletes nothing.** Removal is a separate, explicit, dry-run-by-default step —
because a flag is an accusation the user must be able to check.

### One embedding pass, shared with refine

`_embed_snippets` loads and embeds every usable snippet **once** and returns
`{rows, vectors, durations, waveforms, missing}`. Both `_auto_refine` and the
consolidation pass consume it. Do not reintroduce a direct `extract_embedding`
call in either — two implementations drift, and each re-embed of a 219-snippet
profile is ~25 min of GPU.

### The detector

Pure module-level `score_outliers(vectors, cfg=None)` — no DB, no session, so it
is unit-testable with plain vectors:

- **Iterative reweighting** (`outlier_mad_rounds`, default 2): each round
  recomputes the centre from the **kept** vectors only. One centre over
  everything is dragged toward the outliers and hides them.
- **Accusation needs both tests**: distance to the centre must exceed
  `median + k·MAD·1.4826` (`outlier_mad_k`) **and** an absolute floor
  (`outlier_floor` = **0.35**). The statistical half adapts to a tight profile;
  the floor stops a legitimately wide-but-continuous profile being carved up.
  The floor is deliberately **not** the uncalibrated `0.32` the learning block
  reuses.
- **Guardrails**: `n < outlier_min_keep` (3) accuses nobody;
  `outlier_max_fraction` (0.2) is a **whole-pass** budget enforced worst-first,
  floored at 1 (so a 5-snippet profile is not silenced by `floor(0.2*5) == 0`).
  A capped or spared snippet still gets a reason string, and one that ends up
  flagged never keeps a "capped"/"min_keep" reason.
  `outlier_mad_rounds: 0` = report-only.

Under-flagging is recoverable (re-run with a lower floor). Over-flagging destroys
good audio on one bad run. The defaults are deliberately conservative.

> **Why not reuse `_split_cluster_by_cohesion`?** It flags a whole *group*, which
> is right when learning a diarization cluster and wrong for review, and it has
> hard `unverified:*` exits that would abort a library sweep.

### Marks are durable evidence, not a boolean

`snippets.review_flag` / `review_score` / `review_reason` (additive migration) plus
`set_review` / `clear_review` / `list_flagged` (user-scoped, ranked by descending
score). A bare flag would make every item an unverifiable accusation.

Re-running consolidation **clears marks it no longer makes**, and
`keep_flagged_snippet` clears one — a kept snippet must not keep a stale
accusation.

### Flow

1. Consolidate (per speaker, or whole library) — progress streamed.
2. `GET /flagged` — read-only, ranked by distance.
3. Per item: **Keep** or **Remove**.
4. `POST /flagged/remove` — **one** destructive call, `dry_run=True` by default.

Removal never calls `delete_snippet` (it re-refines on every call → N flagged
snippets = N full re-embeds). `remove_flagged_snippets` collects doomed ids first,
deletes rows, then does **one** `_auto_refine` per touched speaker.

If **every** snippet is flagged, nothing is saved — fitting an unapproved set is
worse than leaving the old profile.

### Hazards designed around

- `_auto_refine` returns **silently** when the embedding session is `None`, and
  the TTL monitor unloads it whenever no job is running. A sweep that does not
  register a job can have ECAPA pulled out mid-pass and report success having
  done nothing. `POST /consolidate/stream` therefore registers a job, and
  `_embed_snippets` re-checks the session **before every embed**, returning
  `status: "skipped"` — not a false `clean`.
- A **zero-norm** embedding means "no usable window", not "maximally far away". It
  is dropped and counted in `missing` → `unusable_snippets`, so a run that lost
  half its corpus cannot read as clean.
- A whole-library run takes real GPU time (~25 min per 219 snippets on a GTX
  1650). Progress is streamed, so it is safe to start and walk away.

---

## Matching: a distance reads differently than a confidence

Two thresholds that both live near 0.32–0.35 do **different jobs**, and the band
between them is deliberate:

| Gate | Reads | Meaning |
|---|---|---|
| **Naming** (`matching.accept_threshold` = 0.35) | a distance | "Is this the same person?" → assign the name |
| **Learning** (`learning.*` = 0.32) | a confidence | "Is this one coherent voice?" → build a profile |

A match in the 0.325–0.35 band therefore **gets the speaker's name and is refused
for learning, on purpose**. Read
[docs/lessons/diarization-and-speakers.md](lessons/diarization-and-speakers.md)
(lesson 43) before changing either threshold. Both come from limited material —
the learning distances from one 2-host podcast with inserted clips (ZO249), and
lesson 33's offline sweep is the way to change them.

---

## Background

- [docs/lessons/diarization-and-speakers.md](lessons/diarization-and-speakers.md)
  — lessons 29, 39, 43, 45, 46
- [docs/lessons/transcription-turns.md](lessons/transcription-turns.md) —
  lessons 17, 18, 41
- [docs/uncertain-speakers.md](uncertain-speakers.md) — why a speaker stays
  unnamed