# Uncertain Speakers

Reference for the speaker-uncertainty policy: how a result is marked
uncertain, every `attribution_reason`, the gates that decide when a cluster may
be named after a registered person, and the known limitation on short
recordings. See `README.md` for the short version and `AGENTS.md` for the
developer rules.

When a speaker identity cannot be established, the server **suppresses the
identity and keeps the text** instead of force-fitting the segment to the
nearest or most common speaker:

```json
{ "start": 9.2, "end": 10.6, "text": "Yes.",
  "speaker": null, "speaker_confidence": 0.0, "speaker_source": "unknown",
  "uncertain": true, "attribution_reason": "boundary_crossing" }
```

- `speaker_source`: `known_voiceprint` | `diarization_cluster` | `unknown`.
- `attribution_reason`: `boundary_crossing` (short span straddling a turn
  boundary), `low_span_turn_overlap` (span too long for any single turn to own),
  `ghost_speaker`, `minority_speaker`, `turn_uncertain`, `turn_label_uncertain`,
  `no_turns`, `no_diarization`, `live_turn_unattributed`, …
- The `done` SSE event reports `status` = `ok` | `partial` | `error` and
  `uncertain_segments` (count of segments with `speaker: null`).
- The web UI and the Windows client render `null` as `UNKNOWN (<reason>)`;
  the client also prints a warning and can write a `.json` sidecar
  (`WRITE_JSON=1` or `--json`).
- Thresholds live in `asr_mcp/config/thresholds.json` under `uncertainty`
  (minimum match confidence/margin, boundary-crossing limits, merge gaps,
  `retain_uncertain_text`) and `streaming` (turn-detector energy/hangover).
- Set `"uncertainty": {"enabled": false}` to restore the legacy
  nearest-speaker behaviour (plain midpoint attribution, ghosts/minorities
  absorbed into the nearest speaker). To keep the policy but restore absorption:
  `"suppress_ghost_speakers": false, "suppress_minority_speakers": false`.
- A recording whose clustering is unreliable (noisy audio can fragment one
  speaker into 15 clusters) is honestly reported as mostly `UNKNOWN` rather than
  being forced onto one name. Fix the diarization clustering thresholds for such
  material instead of re-enabling the proximity guess.
- Naming a cluster after a registered person is gated twice: the second pass
  may only rename a cluster to a voiceprint whose match confidence reaches
  `second_pass.min_identity_confidence` (default 0.5; confidence is
  `1 - combined_distance/0.5`), and a single voiceprint may not be claimed by
  two clusters that sound different from each other. Below the gate the cluster
  stays `Speaker N` — with a large voiceprint menu the best of a bad lot is not
  an identification. Every rename also stamps `speaker_confidence`,
  `speaker_source`, `speaker_margin` and `speaker_match_dist` onto the
  segments, so a low-confidence match is reported as low confidence instead of
  the attribution fallback (1.0).
- Passing `num_speakers` selects a *different* clustering path (hard k, no
  greedy merge) and used to use average linkage, which collapsed a real
  3-person podcast into one cluster (86.5% of windows, 1924s of 2070s in a
  single identity) where complete linkage gives 55.9%. The forced-k path now
  uses `diarization.forced_k_linkage` (default `complete`). The default path
  (no `num_speakers`) is unchanged. If a result looks lopsided, check the
  `Cluster balance: ...` line in the server log before touching the naming
  policy.
- **Known limitation — short recordings over-cluster.** On ground-truth
  two-person clips (16–55s, from the sherpa-onnx
  `speaker-segmentation-models` release) the default path produces 9–13
  clusters for 2 speakers; the short-cluster fragments then fall under the
  10s ghost rule and most of one speaker's turns come back `UNKNOWN`. Text is
  always retained — this withholds the *name*, it does not drop content. A
  52-minute podcast is unaffected (the same thresholds give an identical
  52.2/32.4/11.1 split across `distance_threshold` 0.35–0.65), which is why
  long-file testing alone never surfaces it. See AGENTS.md lesson 33.

## Why a colleague did not become a voiceprint

Every run of `/api/asr/transcribe/upload` also *learns*: audio from a named
speaker is added to their profile, and an unidentified cluster becomes a
**pending** profile you are asked to name. That happens silently unless
something was refused, so the `diarization_complete` SSE event carries a
`learning` block, and the web UI renders it as a note under the transcript:

```json
"learning": {
  "snippets_added": 12,
  "profiles": [{"name": "Pending Speaker 8", "cluster": "Speaker 8",
                "speech_sec": 96.4, "existing": false}],
  "not_collected": [{"speaker": "Test Person One", "reason": "at_snippet_cap"}],
  "not_learned": [{"cluster": "Speaker 2", "reason": "unverified:impure_group"}],
  "refused_total": 88,
  "refused_reasons": {"low_margin": 12, "overlap_label": 76}
}
```

- `not_collected` is a decision you made: `at_snippet_cap` means the speaker is
  at `AUTO_COLLECT_MAX_TOTAL_SEC` (default 1200s) and their profile will not
  improve until you delete snippets or raise the cap. A voiceprint that stops
  getting better is otherwise indistinguishable from one that is finished.
- `not_learned` is a cluster that failed closed — the cohesion check could not
  prove it was one person, so nothing was learned rather than learning a blend
  (see `learning.fail_closed`).
- `refused_reasons` counts spans that carried a name and were not stored.
  The reason vocabulary is fixed and lives in
  `asr_mcp/speaker/uncertainty.py` (`REASON_*`, with wording in
  `LEARNING_REFUSAL_TEXT`):

  | reason | meaning |
  |---|---|
  | `no_identity` / `unknown_label` | no speaker was assigned to the span |
  | `overlap_label` | overlapping speech, no single owner |
  | `generic_label` | an unnamed cluster (see `profiles` — this is what learning is for) |
  | `explicitly_uncertain` | the pipeline asserted an identity and then suppressed it |
  | `low_confidence` | the voiceprint match was too weak |
  | `low_margin` | two speakers matched almost equally well |
  | `too_short` | too little audio to judge |

  A span is only counted here when it is a *near-miss*: a generic cluster that
  was learned is not a refusal, and a span refused for being generic is reported
  under the gate that actually stopped it. The per-span records (up to 25, with
  the runner-up candidates and the measured confidence/margin/distance) are in
  the service result and the server log, not in the SSE event.
- `low_confidence` does **not** contradict a name in the transcript. Naming and
  learning read the same match in different units: naming accepts a voiceprint
  distance up to `matching.accept_threshold` (0.35), learning requires a
  confidence of at least `uncertainty.min_speaker_confidence` (0.35), and
  confidence is derived as `1 - distance / confidence_max_distance` (0.5). A
  match in the 0.325–0.35 distance band is therefore shown under the
  speaker's name and its audio is refused for learning — the band is kept
  deliberately, because naming asserts an identity for display while learning
  decides what becomes a permanent reference, and the second is the more
  expensive mistake. Both numbers are uncalibrated; see the `_comment` on
  `uncertainty` in `asr_mcp/config/thresholds.json`.
- Nothing in this block changes the policy. It is a report: every gate above
  still refuses, and the transcript keeps the words either way.
