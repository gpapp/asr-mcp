"""
Overlap detection and segment construction for speaker diarization.

Detects windows where multiple speakers are simultaneously active (overlap)
and builds annotated segments for downstream ASR alignment.

Two criteria identify overlapping windows:
  A) Top-2 centroid distances are close — the embedding is torn between two speakers.
  B) Even the nearest centroid is far — the mixed embedding doesn't cleanly match anyone.
"""

import numpy as np
from asr_mcp.config import get, is_debug


def detect_overlaps(
    raw_embeddings: np.ndarray,
    cluster_centroids: dict[int, np.ndarray],
    embeddable_indices: list[int],
    all_segments_meta: list[dict],
    proximity_ratio: float = 0.08,
    min_distance: float = 0.40,
) -> list[int]:
    """Tag overlapping speech windows in all_segments_meta.

    Each embedding window is compared against every cluster centroid.
    Windows whose top-2 centroid distances satisfy criteria A or B are
    flagged as *overlap windows*.

    Returns list of meta indices flagged as overlap. Side-effects
    *is_overlap*, *overlap_speakers_raw*, and *overlap_clarity* keys
    onto every element of *all_segments_meta*.
    """
    n_speakers = len(cluster_centroids)
    if len(raw_embeddings) == 0 or n_speakers < 2:
        for meta in all_segments_meta:
            meta["is_overlap"] = False
        return []

    spk_ids = sorted(cluster_centroids.keys())
    centroid_stack = np.stack([cluster_centroids[s] for s in spk_ids])

    # Cosine distances:  [N_windows, N_speakers]
    dists = 1.0 - (raw_embeddings @ centroid_stack.T)
    sorted_idx = np.argsort(dists, axis=1)
    sorted_dists = np.sort(dists, axis=1)

    closest_d = sorted_dists[:, 0]
    top2_gap = sorted_dists[:, 1] - closest_d

    # Criterion A:  top-2 contenders are very close (gap < ratio * primary + offset)
    close_contenders = top2_gap < (closest_d * proximity_ratio + 0.015)

    # Criterion B:  even the best match is poor → mixed embedding
    far_from_all = closest_d > min_distance

    overlap_mask = close_contenders | far_from_all
    overlap_meta_indices = []

    for k, meta_idx in enumerate(embeddable_indices):
        if k >= len(overlap_mask):
            break
        meta = all_segments_meta[meta_idx]
        ov = bool(overlap_mask[k])
        meta["is_overlap"] = ov
        if ov:
            spk_a = int(spk_ids[sorted_idx[k, 0]])
            spk_b = int(spk_ids[sorted_idx[k, 1]])
            meta["overlap_speakers_raw"] = (spk_a, spk_b)
            meta["overlap_clarity"] = float(top2_gap[k] / max(closest_d[k], 1e-6))
            overlap_meta_indices.append(meta_idx)
        else:
            meta["overlap_speakers_raw"] = None

    # Ensure non-embeddable (short) windows have a default is_overlap flag
    embeddable_set = set(embeddable_indices)
    for i, meta in enumerate(all_segments_meta):
        if i not in embeddable_set and "is_overlap" not in meta:
            meta["is_overlap"] = False

    if is_debug() and len(overlap_meta_indices) > 0:
        print(f"OVERLAP: {len(overlap_meta_indices)}/{len(embeddable_indices)} windows flagged as overlap")

    return overlap_meta_indices


def build_overlap_segments(
    all_segments_meta: list[dict],
    max_gap: float = 1.5,
    min_overlap_dur: float = 0.3,
) -> list[dict]:
    """Merge contiguous windows into speaker segments with overlap awareness.

    Adjacent windows sharing the same *merge key* (same speaker label, or same
    sorted overlap-speaker pair) are collapsed into a single segment.  Overlap
    windows produce ``{"speaker": "OVERLAP", "speakers": [...]}`` segments.

    **Quality gate** — a merged OVERLAP segment whose constituent windows have
    mean *overlap_clarity* > ``max_overlap_clarity`` is treated as a false
    positive (the embedding gap between top-2 centroids is too large to indicate
    genuine overlap).  The segment is converted back to a single-speaker label
    using the dominant (first-listed) overlap speaker.

    When two differently-labelled windows overlap in time (from the sliding
    window stride), the boundary is split at the midpoint — same as the legacy
    behaviour, except that overlap vs non-overlap type mismatches are treated
    as separate-merge-key edges and *not* merged.
    """
    if not all_segments_meta:
        return []

    sorted_meta = sorted(all_segments_meta, key=lambda x: x["start"])
    merged = []
    current = None

    for seg in sorted_meta:
        window = _to_window(seg)
        if current is None:
            current = window
            continue

        same = _merge_key(current) == _merge_key(window)

        if same and seg["start"] <= current["end"] + max_gap:
            current["end"] = max(current["end"], seg["end"])
            current["_clarity_sum"] += window["_clarity"]
            current["_clarity_count"] += 1
            continue

        if seg["start"] < current["end"]:
            mid = (seg["start"] + current["end"]) / 2.0
            current["end"] = mid
            window["start"] = mid

        merged.append(current)
        current = window

    if current is not None:
        merged.append(current)

    # ── Quality gate: discard weak OVERLAP segments ─────────────────────
    max_overlap_clarity = get("overlap", "max_clarity", 0.08)
    result = []
    for seg in merged:
        if seg["speaker"] == "OVERLAP":
            dur = seg["end"] - seg["start"]
            if dur < min_overlap_dur:
                # Too short — convert to the dominant overlap speaker
                seg["speaker"] = seg.get("speakers", ["Speaker 1"])[0]
                seg.pop("speakers", None)
                seg.pop("is_overlap", None)
            else:
                mean_clarity = seg["_clarity_sum"] / max(1, seg["_clarity_count"])
                if mean_clarity > max_overlap_clarity:
                    # Weak evidence: revert to single-speaker label
                    if is_debug():
                        print(f"OVERLAP REJECT: clarity={mean_clarity:.3f} > {max_overlap_clarity}, "
                              f"dur={dur:.1f}s, {seg.get('speakers', [])}")
                    seg["speaker"] = seg.get("speakers", ["Speaker 1"])[0]
                    seg.pop("speakers", None)
                    seg.pop("is_overlap", None)
        # Clean up internal tracking fields
        seg.pop("_clarity_sum", None)
        seg.pop("_clarity_count", None)
        result.append(seg)

    return result


# ── internal helpers ──────────────────────────────────────────────────────


def _merge_key(seg: dict) -> tuple:
    """Return a hashable merge key for a window/segment dict."""
    if seg.get("is_overlap", False):
        speakers = seg.get("overlap_speakers", [])
        return ("OVERLAP", tuple(sorted(speakers)))
    return ("SINGLE", seg.get("speaker", ""))


def _to_window(meta: dict) -> dict:
    """Convert a window metadata dict into a mutable segment dict."""
    w = {
        "start": meta["start"],
        "end": meta["end"],
        "speaker_raw": meta.get("speaker_raw"),
        "speaker": meta.get("speaker", "UNKNOWN"),
        "is_overlap": meta.get("is_overlap", False),
        "_clarity": meta.get("overlap_clarity", 0.0),
        "_clarity_sum": meta.get("overlap_clarity", 0.0),
        "_clarity_count": 1,
    }
    if meta.get("is_overlap", False):
        overs = meta.get("overlap_speakers", [])
        w["speaker"] = "OVERLAP"
        w["speakers"] = sorted(overs) if isinstance(overs, (list, tuple)) else [str(overs)]
    return w
