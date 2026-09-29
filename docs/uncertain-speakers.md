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
