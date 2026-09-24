import logging
from typing import Optional

logger = logging.getLogger("asr_mcp.diarization.segment_ops")


def collapse_same_speaker_segments(segments: list, max_gap: float = 0.5) -> list:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x["start"])
    result = [segments[0].copy()]
    for seg in segments[1:]:
        prev = result[-1]
        if (seg.get("speaker") == prev.get("speaker")
                and seg.get("start", 0) - prev.get("end", 0) <= max_gap):
            prev["end"] = max(prev.get("end", 0), seg.get("end", 0))
            if "text" in seg and "text" in prev:
                prev["text"] = prev["text"] + " " + seg["text"]
        else:
            result.append(seg.copy())
    return result


def absorb_islands(segments: list, min_island_dur: float = 1.0) -> list:
    """Absorb short segments sandwiched between the same speaker on both sides.

    OVERLAP segments are never absorbed — they represent genuine multi-speaker
    activity even when brief.
    """
    changed = True
    while changed:
        changed = False
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            if seg.get("speaker") == "OVERLAP":
                continue
            dur = seg.get("end", 0) - seg.get("start", 0)
            prev_spk = segments[i - 1].get("speaker")
            next_spk = segments[i + 1].get("speaker")
            if dur < min_island_dur and prev_spk == next_spk and seg.get("speaker") != prev_spk:
                seg["speaker"] = prev_spk
                changed = True
        if changed:
            segments = collapse_same_speaker_segments(segments, max_gap=1.0)
    return segments


def absorb_minority_speakers(
    segments: list,
    max_utterance_sec: float = 5.0,
    min_speaker_dur: float = 8.0,
    protected_speakers: set = None,
) -> list:
    """Reassign all utterances of minority speakers to the temporally nearest main speaker.

    A speaker is "main" if they have at least one utterance of
    max_utterance_sec or longer (per-utterance criterion), OR a total
    duration of min_speaker_dur or more. Everyone else is a minority
    speaker and their utterances are absorbed into the nearest main speaker.

    Args:
        protected_speakers: Set of speaker names that are NEVER absorbed,
            regardless of duration (e.g. matched known voiceprints).
    """
    if len(segments) < 2:
        return segments

    protected_speakers = protected_speakers or set()

    speaker_totals = {}
    speaker_longest = {}
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        dur = seg.get("end", 0) - seg.get("start", 0)
        speaker_totals[spk] = speaker_totals.get(spk, 0) + dur
        speaker_longest[spk] = max(speaker_longest.get(spk, 0), dur)

    main = {
        spk for spk in speaker_totals
        if spk in protected_speakers
        or speaker_longest.get(spk, 0) >= max_utterance_sec
        or speaker_totals.get(spk, 0) >= min_speaker_dur
    }
    if not main or len(main) == len(speaker_totals):
        return segments

    minority = set(speaker_totals) - main

    result = []
    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "UNKNOWN")
        if spk in minority:
            best_alt = None
            best_dist = float("inf")
            seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2.0
            for m_spk in main:
                dist = _nearest_midpoint_distance(segments, seg_mid, m_spk)
                if dist < best_dist:
                    best_dist = dist
                    best_alt = m_spk
            if best_alt:
                seg = {**seg, "speaker": best_alt}
                logger.debug("Absorbed minority %s -> %s at %.1f (%.1fs utter)",
                             spk, best_alt, seg.get("start", 0),
                             seg.get("end", 0) - seg.get("start", 0))
        result.append(seg)

    return collapse_same_speaker_segments(result)


def eliminate_ghost_speakers(
    segments: list,
    profiles: Optional[dict] = None,
    ghost_threshold_sec: float = 10.0,
    min_duration: Optional[float] = None,
) -> list:
    """Remove ghost speakers (< ghost_threshold_sec total speech).

    Ghost segments are reassigned in priority order:
    1. Best alternative matched known speaker from seg["alternatives"] that is NOT a ghost.
    2. Temporally nearest non-ghost speaker (actual audio midpoint distance).

    After reassignment, ghost entries are removed from profiles.
    """
    if min_duration is not None:
        ghost_threshold_sec = min_duration
    if not segments:
        return []

    speaker_durations: dict = {}
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        dur = seg.get("end", 0) - seg.get("start", 0)
        speaker_durations[spk] = speaker_durations.get(spk, 0) + dur

    ghost_speakers = {spk for spk, dur in speaker_durations.items() if dur < ghost_threshold_sec}

    if not ghost_speakers:
        return segments

    non_ghost = {spk for spk in speaker_durations if spk not in ghost_speakers}

    result = []
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk in ghost_speakers:
            best_alt = None

            # Priority 1: alternatives from known speaker matching
            for alt in seg.get("alternatives", []):
                alt_spk = alt.get("speaker")
                if alt_spk and alt_spk in non_ghost:
                    best_alt = alt_spk
                    break

            # Priority 2: temporally nearest non-ghost speaker
            if best_alt is None and non_ghost:
                seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2.0
                best_dist = float("inf")
                for other in segments:
                    if other is seg or other.get("speaker") in ghost_speakers:
                        continue
                    d = abs((other.get("start", 0) + other.get("end", 0)) / 2.0 - seg_mid)
                    if d < best_dist:
                        best_dist = d
                        best_alt = other.get("speaker")

            if best_alt:
                seg = dict(seg)
                seg["speaker"] = best_alt
                logger.debug("Ghost %s -> %s at %.1f", spk, best_alt, seg.get("start", 0))
            else:
                seg = dict(seg)
                seg["speaker"] = "Speaker 1"
        result.append(seg)

    # Remove ghost entries from profiles
    if profiles:
        for ghost in ghost_speakers:
            profiles.pop(ghost, None)

    return collapse_same_speaker_segments(result)


def _nearest_midpoint_distance(segments: list, target_mid: float, speaker: str) -> float:
    """Return the minimum audio midpoint distance from target_mid to any segment of speaker."""
    best = float("inf")
    for seg in segments:
        if seg.get("speaker") == speaker:
            mid = (seg.get("start", 0) + seg.get("end", 0)) / 2.0
            d = abs(mid - target_mid)
            if d < best:
                best = d
    return best


# Kept for backwards compatibility; new code should use _nearest_midpoint_distance
def _find_nearest_temporal(segments: list, target_idx: int, speaker: str) -> Optional[int]:
    for offset in range(1, len(segments)):
        for idx in [target_idx + offset, target_idx - offset]:
            if 0 <= idx < len(segments):
                if segments[idx].get("speaker") == speaker:
                    return idx
    return None


def merge_profiles(profiles: dict, target: str, source: str):
    if source not in profiles or target not in profiles:
        return
    t = profiles[target]
    s = profiles[source]
    t_dur = t.get("total_speech_sec", 0)
    s_dur = s.get("total_speech_sec", 0)
    total = t_dur + s_dur
    if total <= 0:
        return
    for field in ["pitch_hz", "pitch_std", "energy_rms", "spectral_centroid", "spectral_rolloff"]:
        t_val = t.get(field, 0)
        s_val = s.get(field, 0)
        t[field] = (t_val * t_dur + s_val * s_dur) / total
    t["total_speech_sec"] = total
