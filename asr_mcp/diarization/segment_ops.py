import logging
from typing import Optional

logger = logging.getLogger("asr_mcp.diarization.segment_ops")


def collapse_same_speaker_segments(segments: list, max_gap: float = 0.5) -> list:
    if not segments:
        return []
    result = [segments[0].copy()]
    for seg in segments[1:]:
        if (seg.get("speaker") == result[-1].get("speaker")
                and seg.get("start", 0) - result[-1].get("end", 0) <= max_gap):
            result[-1]["end"] = seg.get("end", result[-1].get("end"))
            if "text" in seg and "text" in result[-1]:
                result[-1]["text"] = result[-1]["text"] + " " + seg["text"]
        else:
            result.append(seg.copy())
    return result


def absorb_islands(segments: list, min_island_dur: float = 1.0) -> list:
    if len(segments) < 3:
        return segments
    result = [segments[0].copy()]
    for i in range(1, len(segments)):
        curr = segments[i]
        if i < len(segments) - 1:
            prev_speaker = result[-1].get("speaker")
            next_speaker = segments[i + 1].get("speaker")
            curr_speaker = curr.get("speaker")
            curr_dur = curr.get("end", 0) - curr.get("start", 0)

            if (curr_speaker != prev_speaker
                    and curr_speaker != next_speaker
                    and prev_speaker == next_speaker
                    and curr_dur < min_island_dur):
                curr["speaker"] = prev_speaker
                logger.debug("Absorbed island at %.1f-%.1f (%.1fs) into %s",
                             curr.get("start", 0), curr.get("end", 0), curr_dur, prev_speaker)

        if (curr.get("speaker") == result[-1].get("speaker")
                and curr.get("start", 0) - result[-1].get("end", 0) <= 0.5):
            result[-1]["end"] = curr.get("end", result[-1].get("end"))
        else:
            result.append(curr.copy())

    return result


def eliminate_ghost_speakers(
    segments: list,
    profiles: Optional[dict] = None,
    ghost_threshold_sec: float = 5.0,
) -> list:
    if not segments:
        return []

    speaker_durations = {}
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        dur = seg.get("end", 0) - seg.get("start", 0)
        speaker_durations[spk] = speaker_durations.get(spk, 0) + dur

    ghost_speakers = {spk for spk, dur in speaker_durations.items() if dur < ghost_threshold_sec}

    if not ghost_speakers:
        return segments

    result = []
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        if spk in ghost_speakers:
            best_alt = None
            best_dist = float("inf")
            for other_spk in speaker_durations:
                if other_spk not in ghost_speakers and other_spk != spk:
                    idx = segments.index(seg)
                    nearest = _find_nearest_temporal(segments, idx, other_spk)
                    if nearest is not None:
                        dist = abs(nearest - idx)
                        if dist < best_dist:
                            best_dist = dist
                            best_alt = other_spk
            if best_alt:
                seg["speaker"] = best_alt
                logger.debug("Ghost %s -> %s at %.1f", spk, best_alt, seg.get("start", 0))
            else:
                if result and result[-1].get("speaker") != spk:
                    seg["speaker"] = result[-1].get("speaker", "Speaker 1")
                else:
                    seg["speaker"] = "Speaker 1"
        result.append(seg)

    return collapse_same_speaker_segments(result)


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
