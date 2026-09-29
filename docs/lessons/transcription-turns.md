# Transcription turns and exact boundaries

Lessons 17, 18 of `AGENTS.md`. Full-coverage invariant and VAD-based boundary refinement.


### 17. Every Second Must Be Covered by Exactly One Transcription Turn
- Inter-turn gaps (e.g. 22.6→28.0s) where speech exists are silently UNTRANSCRIBED — no error, no empty row, just missing text.
- `_prepare_turns` must close every gap >1ms between consecutive turns AND extend edges to 0/duration. Full coverage is an invariant; keep it through any refactor.
- Fallback chain per gap (cheap → expensive): VAD-section voiceprint attribution → energy-dip centre (`_gap_boundary`) → midpoint.

### 18. Exact Turn Boundaries: Uncollapsed VAD Sections + Voiceprint Attribution
- Use RAW VAD (`run_vad_onnx` over the full file — no `_merge_nearby_speech`, no `split_at_energy_dips`) so each speech region keeps its true edges.
- Embed every section spanning the gap (`extract_embedding`); reference per speaker = known DB voiceprint if the turn label matches (try `name.strip("[]")`), else ≤30s of that speaker's own diarized turn audio. Both refs required — otherwise fall back to energy.
- Ownership split via `_best_split(owners, weights)` with weights = embedding-distance margin (confident matches dominate noise). Same-speaker boundaries (from long-turn splits) are all-left → left absorbs the gap.
- Apply ONE shared cut per gap (`left.end == right.start`), clamped into `[gap_start, gap_end]` so turns stay contiguous and can never overlap or chain-react. Cut = exact start of the first right-owned section, or `gap_end` when all-left.
- Log `Boundary refinement N: ... cut at X.XXs` — grep this to confirm the refinement is active after a rebuild.
