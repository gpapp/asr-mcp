import asyncio
import bisect
import json
import logging
import re
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from asr_mcp.api.schemas import (
    DiarizeRequest, DiarizeResponse, DiarizeResult,
    TranscribeResponse, TranscribeResult,
)
from asr_mcp.api.security import verify_api_key, get_current_user, validate_upload_filename
from asr_mcp.api.auth import get_session_user
from asr_mcp.config.settings import Settings, get_settings
from asr_mcp.core import job_state
from asr_mcp.speaker.vad import split_at_energy_dips

logger = logging.getLogger("asr_mcp.api.asr_router")
router = APIRouter(prefix="/asr", tags=["ASR"])

MAX_TURN_SEC = 120.0
PARAGRAPH_PAUSE_SEC = 3.0
_SENT_END_RE = re.compile(r'[.!?…]["”’)\]]?(?=\s|$)')


class JobCancelled(BaseException):
    """Raised from progress callbacks to abort a cancelled job.

    Subclasses BaseException so backend/diarizer ``except Exception`` blocks
    do not swallow it; only run_transcribe's explicit handler catches it.
    """


def _merge_into_turns(segments, max_gap_sec=3.0):
    """Merge consecutive same-speaker segments into full speaker turns.

    A turn is a continuous span attributed to one speaker.  Segments from the
    same speaker separated by <= *max_gap_sec* are folded into a single turn
    so the transcriber receives coherent, single-speaker audio.
    """
    if not segments:
        return []
    turns = []
    cur = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "speaker": segments[0].get("speaker", "UNKNOWN"),
        "segments": [segments[0]],
    }
    for seg in segments[1:]:
        same_speaker = seg.get("speaker") == cur["speaker"]
        gap = seg["start"] - cur["end"]
        if same_speaker and gap <= max_gap_sec:
            cur["end"] = seg["end"]
            cur["segments"].append(seg)
        else:
            turns.append(cur)
            cur = {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker", "UNKNOWN"),
                "segments": [seg],
            }
    turns.append(cur)
    return turns


def _split_long_turn(turn):
    """Split a turn longer than MAX_TURN_SEC at the largest internal gap."""
    if turn["end"] - turn["start"] <= MAX_TURN_SEC:
        return [turn]
    segs = turn["segments"]
    if len(segs) <= 1:
        return [turn]
    best_gap, best_idx = 0, 1
    for i in range(1, len(segs)):
        gap = segs[i]["start"] - segs[i - 1]["end"]
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    t1 = {
        "start": turn["start"],
        "end": segs[best_idx - 1]["end"],
        "speaker": turn["speaker"],
        "segments": segs[:best_idx],
    }
    t2 = {
        "start": segs[best_idx]["start"],
        "end": turn["end"],
        "speaker": turn["speaker"],
        "segments": segs[best_idx:],
    }
    out = []
    out.extend(_split_long_turn(t1))
    out.extend(_split_long_turn(t2))
    return out


def _find_interior_pauses(audio, sample_rate, min_pause_sec=PARAGRAPH_PAUSE_SEC,
                          frame_ms=30.0):
    """Silence runs >= min_pause_sec strictly inside the audio (edges skipped)."""
    import numpy as np
    if audio is None or len(audio) < int(0.5 * sample_rate):
        return []
    frame = max(1, int(sample_rate * frame_ms / 1000))
    n = len(audio) // frame
    if n < 4:
        return []
    rms = np.sqrt(np.mean(
        audio[:n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1) + 1e-12)
    thr = max(float(np.percentile(rms, 25)) * 0.3, float(rms.max()) * 0.02, 1e-5)
    quiet = rms < thr
    dur = len(audio) / sample_rate
    frame_sec = frame_ms / 1000.0
    pauses = []
    i = 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        s, e = i * frame_sec, j * frame_sec
        if e - s >= min_pause_sec and s > 0.25 and e < dur - 0.25:
            pauses.append((s, e))
        i = j
    return pauses


def _snap_sentence_end(text, pos, window=40):
    """Index just past the sentence-ending punctuation nearest to *pos*.

    Returns None when no [.!?…] occurs within *window* chars on either side.
    A tight window keeps the break near the actual pause; callers fall back to
    the raw boundary (plus a completed sentence) when nothing is close.
    """
    lo = max(0, pos - window)
    hi = min(len(text), pos + window)
    region = text[lo:hi]
    best = None
    for m in _SENT_END_RE.finditer(region):
        end = lo + m.end()
        d = abs(end - pos)
        if best is None or d < best[0]:
            best = (d, end)
    return best[1] if best else None


def _apply_paragraph_breaks(text, segments, turn_audio, sample_rate):
    """Insert \\n\\n at long interior pauses, snapped to sentence boundaries.

    Segment timestamps come from the decoder's split tokens; pause times come
    from RMS silence runs in the turn audio. Each pause maps to the nearest
    segment boundary, then snaps to the closest [.!?…] within 40 chars so the
    break never lands mid-sentence. If no punctuation exists nearby the break
    is taken at the boundary and a '.' is appended to complete the sentence.
    """
    if not text or not segments or len(segments) < 2 or turn_audio is None:
        return text

    rebuilt = " ".join((s.get("text") or "") for s in segments)
    if abs(len(rebuilt) - len(text)) > max(16, len(text) // 20):
        return text

    pauses = _find_interior_pauses(turn_audio, sample_rate)
    if not pauses:
        return rebuilt

    bounds = []
    pos = 0
    for seg in segments:
        pos += len(seg.get("text") or "")
        bounds.append((pos, float(seg.get("end", 0.0))))
        pos += 1

    marks = {}
    for ps, pe in pauses:
        mid = (ps + pe) / 2.0
        best_i, best_d = None, None
        for i in range(len(bounds) - 1):
            d = abs(bounds[i][1] - mid)
            if best_d is None or d < best_d:
                best_i, best_d = i, d
        if best_i is None:
            continue
        boundary = bounds[best_i][0]
        snapped = _snap_sentence_end(rebuilt, boundary)
        if snapped is not None:
            marks.setdefault(snapped, False)
        else:
            marks.setdefault(boundary, True)

    parts = []
    prev = 0
    for idx in sorted(marks):
        if idx <= prev or idx - prev < 25 or len(rebuilt) - idx < 25:
            continue
        chunk = rebuilt[prev:idx].strip()
        if not chunk:
            continue
        if marks[idx]:
            chunk = chunk.rstrip(",;:—–- ")
            if chunk and chunk[-1] not in ".!?…":
                chunk += "."
        parts.append(chunk)
        prev = idx
    tail = rebuilt[prev:].strip()
    if tail:
        parts.append(tail)
    return "\n\n".join(p for p in parts if p)


def _speaker_for_span(start, end, turns, starts):
    """Diarization speaker for a time span: midpoint lookup in sorted turns."""
    if not turns:
        return None
    mid = (start + end) / 2.0
    i = bisect.bisect_right(starts, mid) - 1
    if i < 0:
        return turns[0].get("speaker")
    if i >= len(turns):
        return turns[-1].get("speaker")
    turn = turns[i]
    if float(turn["start"]) <= mid <= float(turn["end"]):
        return turn.get("speaker")
    if i + 1 < len(turns):
        nxt = turns[i + 1]
        if mid - float(turn["end"]) <= float(nxt["start"]) - mid:
            return turn.get("speaker")
        return nxt.get("speaker")
    return turn.get("speaker")


def _transcribe_file(audio_np, turns, sample_rate=16000, progress_cb=None):
    """Transcribe the whole file in one backend call; attribute output post-hoc.

    The backend decodes in its own windows and returns timestamped items on
    the file timeline. Each item's midpoint maps onto the diarized turns to
    recover its speaker; consecutive same-speaker items become one
    TranscribeResult run, so the audio is never cut at speaker boundaries.
    """
    from asr_mcp.core.transcriber import transcribe_audio_sync

    dur = len(audio_np) / sample_rate
    turns = list(turns or [])
    logger.info("Transcribing file: %.1fs (%d diarization turns for attribution)",
                dur, len(turns))
    tr = transcribe_audio_sync(audio=audio_np, progress_cb=progress_cb)
    logger.info("File result: text=%d chars, segments=%d, inference=%.2fs, error=%s",
                len(tr.get("text") or ""), len(tr.get("segments") or []),
                tr.get("inference_time_sec", 0), tr.get("error"))

    error = tr.get("error")
    text_all = (tr.get("text") or "").strip()
    if not text_all and not tr.get("segments") and not error:
        return []
    if not turns:
        turns = [{"start": 0.0, "end": dur, "speaker": None}]
    starts = [float(t["start"]) for t in turns]

    items = sorted(
        (s for s in (tr.get("segments") or []) if (s.get("text") or "").strip()),
        key=lambda s: float(s.get("start") or 0.0),
    )
    runs = []
    for it in items:
        s = float(it.get("start") or 0.0)
        e = float(it.get("end") or s)
        spk = _speaker_for_span(s, e, turns, starts)
        if runs and runs[-1]["speaker"] == spk:
            runs[-1]["items"].append(it)
        else:
            runs.append({"speaker": spk, "items": [it]})

    if not runs:
        return [TranscribeResult(
            text=text_all,
            start=0.0,
            end=dur,
            speaker=_speaker_for_span(0.0, dur, turns, starts),
            audio_duration_sec=tr.get("audio_duration_sec", 0),
            inference_time_sec=tr.get("inference_time_sec", 0),
            tokens_generated=tr.get("tokens_generated", 0),
            error=error,
        )]

    logger.info("Attributed %d items into %d speaker runs", len(items), len(runs))
    results = []
    for ri, run in enumerate(runs):
        segs = run["items"]
        start = min(float(s.get("start") or 0.0) for s in segs)
        end = max(float(s.get("end") or 0.0) for s in segs)
        text = " ".join((s.get("text") or "") for s in segs)
        run_audio = audio_np[int(start * sample_rate):int(end * sample_rate)]
        rel = [
            {
                "start": float(s.get("start") or 0.0) - start,
                "end": float(s.get("end") or 0.0) - start,
                "text": s.get("text") or "",
            }
            for s in segs
        ]
        text = _apply_paragraph_breaks(
            text, rel, run_audio if len(run_audio) else None, sample_rate,
        ).strip()
        results.append(TranscribeResult(
            text=text,
            segments=segs,
            start=start,
            end=end,
            speaker=run["speaker"],
            audio_duration_sec=tr.get("audio_duration_sec", 0),
            inference_time_sec=tr.get("inference_time_sec", 0) if ri == 0 else 0.0,
            tokens_generated=tr.get("tokens_generated", 0) if ri == 0 else 0,
            error=error if ri == len(runs) - 1 else None,
        ))
    return results


def _gap_boundary(audio, gap_start_sec, gap_end_sec, sample_rate=16000,
                  frame_ms=20.0, dip_ratio=0.35, min_dip_sec=0.12):
    """Pick the cut point inside a gap: the centre of its quietest pause.

    Frame RMS energies are computed over the gap; the longest run below
    *dip_ratio* of the peak (same heuristic as split_at_energy_dips) is the
    real inter-speaker pause and the cut goes to its centre, so the boundary
    never lands mid-word. Falls back to the single quietest frame, then to
    the midpoint, when no dip is found (continuous speech or no audio).
    """
    import numpy as np

    mid = (gap_start_sec + gap_end_sec) / 2.0
    if audio is None or len(audio) == 0:
        return mid
    s = max(0, int(gap_start_sec * sample_rate))
    e = min(len(audio), int(gap_end_sec * sample_rate))
    frame_len = max(1, int(frame_ms / 1000 * sample_rate))
    if e - s < 2 * frame_len:
        return mid
    chunk = audio[s:e].astype(np.float32)
    energies = [
        float(np.sqrt(np.mean(chunk[i:i + frame_len] ** 2)))
        for i in range(0, len(chunk) - frame_len + 1, frame_len)
    ]
    if not energies:
        return mid
    max_e = max(energies)
    if max_e < 1e-8:
        return mid
    thresh = max_e * dip_ratio
    runs = []
    i = 0
    while i < len(energies):
        if energies[i] < thresh:
            j = i
            while j < len(energies) and energies[j] < thresh:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    min_frames = max(1, int(round(min_dip_sec * 1000 / frame_ms)))
    eligible = [r for r in runs if r[1] - r[0] >= min_frames]
    if eligible:
        centre_idx = len(energies) / 2.0
        best = max(
            eligible,
            key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2.0 - centre_idx)),
        )
        cut_frame = (best[0] + best[1]) // 2
    else:
        cut_frame = min(range(len(energies)), key=lambda k: energies[k])
    cut_sample = s + cut_frame * frame_len + frame_len // 2
    return min(max(cut_sample / sample_rate, gap_start_sec), gap_end_sec)


def _raw_vad_sections(audio, sample_rate):
    """Run uncollapsed VAD over the whole file (no merge / dip splitting)."""
    import numpy as np
    import torch
    from asr_mcp.config import get_config
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.vad import run_vad_chunked, run_vad_onnx

    try:
        cfg = get_config() or {}
    except Exception:
        cfg = {}
    vcfg = cfg.get("vad", {}) if isinstance(cfg, dict) else {}
    threshold = vcfg.get("default_threshold", 0.5)
    min_ms = int(vcfg.get("min_speech_duration_ms", 250))
    wf = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    if state.vad_session is not None:
        return run_vad_onnx(
            wf, state.vad_session,
            sample_rate=sample_rate, threshold=threshold,
            min_speech_duration_ms=min_ms,
            merge_close=False,
        )
    return run_vad_chunked(
        wf, sample_rate=sample_rate,
        threshold=threshold, min_speech_duration_ms=min_ms,
    )


def _embed_section(audio, start_sec, end_sec, sample_rate):
    """Voiceprint embedding for one VAD section, or None if unusable."""
    import numpy as np
    import torch
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.embedding import extract_embedding

    if state.embedding_session is None:
        return None
    start_sample = int(start_sec * sample_rate)
    end_sample = min(len(audio), int(end_sec * sample_rate))
    chunk = audio[start_sample:end_sample]
    if len(chunk) < int(0.3 * sample_rate):
        return None
    try:
        return extract_embedding(
            torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)),
            sample_rate, state.embedding_session,
        )
    except Exception as e:
        logger.warning("Section embedding failed: %s", e)
        return None


def _speaker_refs(turns, audio, sample_rate, known_speakers=None):
    """One reference embedding per distinct turn speaker.

    Known DB voiceprints win when the turn label matches; otherwise up to 30s
    of that speaker's own diarized turn audio is embedded as the reference.
    """
    import numpy as np
    import torch
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.embedding import extract_embedding

    refs = {}
    seen = []
    for t in turns:
        name = t.get("speaker")
        if name and name not in seen:
            seen.append(name)

    for name in seen:
        emb = None
        if known_speakers:
            for key in (name, str(name).strip("[]")):
                entry = known_speakers.get(key)
                if entry:
                    e = entry.get("embedding")
                    if e is not None:
                        e = np.asarray(e, dtype=np.float32).ravel()
                        nrm = float(np.linalg.norm(e))
                        if nrm > 1e-8:
                            emb = e / nrm
                    break
        if emb is None and audio is not None and state.embedding_session is not None:
            for t in turns:
                if t.get("speaker") != name:
                    continue
                s = int(t["start"] * sample_rate)
                e = min(len(audio), s + int(30 * sample_rate))
                if e - s < int(0.3 * sample_rate):
                    continue
                try:
                    emb = extract_embedding(
                        torch.from_numpy(
                            np.ascontiguousarray(audio[s:e], dtype=np.float32)
                        ),
                        sample_rate, state.embedding_session,
                    )
                except Exception as ex:
                    logger.warning("Speaker ref embed failed for %s: %s", name, ex)
                    emb = None
                if emb is not None:
                    break
        if emb is not None:
            refs[name] = np.asarray(emb, dtype=np.float32)
    return refs


def _best_split(owners, weights):
    """Best cut index k (before section k) minimising weighted disagreements.

    owners[i] = 0 for the left turn's speaker, 1 for the right turn's.
    Cost of cut k: right-owned sections left of k plus left-owned sections
    right of k, weighted by embedding-distance margin (confident mistakes
    cost more). Same-speaker boundaries (all one side) collapse to k = n.
    """
    n = len(owners)
    if n == 0:
        return 0
    best_k, best_cost = 0, float("inf")
    for k in range(n + 1):
        cost = 0.0
        for i in range(k):
            if owners[i] == 1:
                cost += weights[i]
        for i in range(k, n):
            if owners[i] == 0:
                cost += weights[i]
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_k = k
    return best_k


def _refine_boundaries_with_vad(turns, audio, sample_rate, known_speakers=None):
    """Exact turn boundaries from uncollapsed VAD sections + voiceprints.

    For every gap between consecutive turns: collect the raw VAD sections
    spanning it, embed each one, attribute it to the better-matching adjacent
    turn speaker (known voiceprint or reference embedding of that speaker's
    own turn audio), find the ownership split, then extend both turns to a
    single shared cut at the exact start of the first right-owned section
    (or the gap end when the gap belongs entirely to the left speaker).
    Full timeline coverage is preserved; gaps without usable sections or
    references are left for the energy-dip fallback in _prepare_turns.
    """
    import numpy as np

    if not turns or audio is None or len(audio) == 0 or len(turns) < 2:
        return turns

    gaps = [
        (i, turns[i]["end"], turns[i + 1]["start"])
        for i in range(len(turns) - 1)
        if turns[i + 1]["start"] - turns[i]["end"] > 1e-3
    ]
    if not gaps:
        return turns

    try:
        sections = _raw_vad_sections(audio, sample_rate)
    except Exception as e:
        logger.warning("Uncollapsed VAD failed during boundary refinement: %s", e)
        return turns
    if not sections:
        return turns

    try:
        refs = _speaker_refs(turns, audio, sample_rate, known_speakers)
    except Exception as e:
        logger.warning("Speaker reference embeddings failed: %s", e)
        return turns
    if not refs:
        return turns

    sections = sorted(sections, key=lambda s: s["start"])
    refined = 0
    for i, gap_start, gap_end in gaps:
        left = turns[i]
        right = turns[i + 1]
        ref_l = refs.get(left.get("speaker"))
        ref_r = refs.get(right.get("speaker"))
        if ref_l is None or ref_r is None:
            continue
        gap_secs = [
            s for s in sections
            if s["end"] > gap_start and s["start"] < gap_end
        ]
        owners, weights, valid = [], [], []
        for s in gap_secs:
            emb = _embed_section(audio, s["start"], s["end"], sample_rate)
            if emb is None:
                continue
            sa = float(np.dot(emb, ref_l))
            sb = float(np.dot(emb, ref_r))
            owners.append(0 if sa >= sb else 1)
            weights.append(abs(sa - sb) + 1e-3)
            valid.append(s)
        if not valid:
            continue
        k = _best_split(owners, weights)
        if k < len(valid):
            cut = float(valid[k]["start"])
            cut = min(max(cut, gap_start), gap_end)
        else:
            cut = gap_end
        n_left = sum(1 for o in owners if o == 0)
        logger.info(
            "Boundary refinement %d: %.2f-%.2fs, %d VAD sections "
            "(%d left / %d right), cut at %.2fs",
            i, gap_start, gap_end, len(valid), n_left, len(valid) - n_left, cut,
        )
        left["end"] = float(cut)
        right["start"] = float(cut)
        refined += 1
    if refined:
        logger.info("Refined %d/%d turn boundaries via VAD voiceprint attribution",
                    refined, len(gaps))
    return turns


def _prepare_turns(segments, audio_duration_sec=None, audio=None,
                   sample_rate=16000, known_speakers=None):
    """Merge diarized segments into turns, split long ones, close gaps.

    Boundaries are first refined exactly: every gap is stepped through using
    uncollapsed VAD sections attributed to the adjacent speaker by voiceprint
    similarity (_refine_boundaries_with_vad), so each turn is extended to the
    exact length of its attributed speech. Any gap still open (no VAD
    sections, no references, VAD failure) falls back to the energy-dip cut
    (_gap_boundary). Every second of the timeline ends up covered by exactly
    one transcription turn. Edges are extended to the file start/end as well.
    """
    turns = _merge_into_turns(segments)
    split = []
    for t in turns:
        split.extend(_split_long_turn(t))
    try:
        split = _refine_boundaries_with_vad(
            split, audio, sample_rate, known_speakers,
        )
    except Exception as e:
        logger.warning("VAD voiceprint boundary refinement failed: %s", e)
    for i in range(len(split) - 1):
        gap_start = split[i]["end"]
        gap_end = split[i + 1]["start"]
        if gap_end - gap_start > 1e-3:
            cut = _gap_boundary(audio, gap_start, gap_end, sample_rate)
            split[i]["end"] = cut
            split[i + 1]["start"] = cut
    if split and audio_duration_sec is not None:
        if split[0]["start"] > 1e-3:
            split[0]["start"] = 0.0
        if audio_duration_sec - split[-1]["end"] > 1e-3:
            split[-1]["end"] = float(audio_duration_sec)
    return split


def _merge_consecutive_same_speaker_results(results, max_gap_sec=3.0):
    """Merge consecutive same-speaker TranscribeResult entries into coherent turns."""
    if not results:
        return []
    merged = []
    for r in results:
        if not merged:
            merged.append(r)
            continue
        prev = merged[-1]
        prev_spk = getattr(prev, "speaker", None) if hasattr(prev, "speaker") else prev.get("speaker")
        curr_spk = getattr(r, "speaker", None) if hasattr(r, "speaker") else r.get("speaker")
        prev_end = getattr(prev, "end", 0.0) if hasattr(prev, "end") else prev.get("end", 0.0)
        curr_start = getattr(r, "start", 0.0) if hasattr(r, "start") else r.get("start", 0.0)

        if prev_spk == curr_spk and (curr_start - prev_end) <= max_gap_sec:
            prev_text = (getattr(prev, "text", "") if hasattr(prev, "text") else prev.get("text", "")) or ""
            curr_text = (getattr(r, "text", "") if hasattr(r, "text") else r.get("text", "")) or ""

            if prev_text and curr_text:
                if prev_text.endswith(("\n", "\n\n")):
                    combined_text = prev_text + curr_text
                elif prev_text[-1] in ".!?…" and (curr_start - prev_end) >= 2.0:
                    combined_text = prev_text + "\n\n" + curr_text
                else:
                    combined_text = prev_text + " " + curr_text
            else:
                combined_text = prev_text or curr_text

            prev_segs = getattr(prev, "segments", []) if hasattr(prev, "segments") else prev.get("segments", [])
            curr_segs = getattr(r, "segments", []) if hasattr(r, "segments") else r.get("segments", [])
            combined_segs = list(prev_segs or []) + list(curr_segs or [])
            curr_end = getattr(r, "end", 0.0) if hasattr(r, "end") else r.get("end", 0.0)

            if hasattr(prev, "end"):
                prev.end = max(prev.end, curr_end)
                prev.text = combined_text
                prev.segments = combined_segs
                if hasattr(r, "tokens_generated") and hasattr(prev, "tokens_generated"):
                    prev.tokens_generated += r.tokens_generated
                if hasattr(r, "inference_time_sec") and hasattr(prev, "inference_time_sec"):
                    prev.inference_time_sec += r.inference_time_sec
            else:
                prev["end"] = max(prev.get("end", 0.0), curr_end)
                prev["text"] = combined_text
                prev["segments"] = combined_segs
                prev["tokens_generated"] = prev.get("tokens_generated", 0) + (r.get("tokens_generated", 0) if isinstance(r, dict) else 0)
                prev["inference_time_sec"] = prev.get("inference_time_sec", 0.0) + (r.get("inference_time_sec", 0.0) if isinstance(r, dict) else 0.0)
        else:
            merged.append(r)
    return merged


def _transcribe_diarized(audio_np, segments, sample_rate=16000, known_speakers=None):
    turns = _prepare_turns(
        segments,
        audio_duration_sec=len(audio_np) / sample_rate,
        audio=audio_np,
        sample_rate=sample_rate,
        known_speakers=known_speakers,
    )
    _switch_to_backend_phase()
    all_results = _transcribe_file(audio_np, turns or segments, sample_rate)
    return _merge_consecutive_same_speaker_results(all_results)


def _switch_to_backend_phase():
    """Phase switch: diarization/turn-prep done (uses VAD+embedding) -> transcribe turns (ASR backend).

    Unloads the embedding session (frees its GPU arena) then loads the ASR
    backend, so diarization and transcription never hold VRAM at the same time.
    """
    from asr_mcp.core.model_state import state
    state.unload_embedding()
    state.ensure_backend_ready()


def _result_to_dict(r):
    if hasattr(r, 'model_dump'):
        return r.model_dump()
    if hasattr(r, '__dict__'):
        d = {}
        for k, v in r.__dict__.items():
            if hasattr(v, 'model_dump'):
                d[k] = v.model_dump()
            elif isinstance(v, list):
                d[k] = [item.model_dump() if hasattr(item, 'model_dump') else item for item in v]
            else:
                d[k] = v
        return d
    return r


def _safe_json(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    return obj


async def _sse_put(queue: asyncio.Queue, evt) -> None:
    """Put an SSE event, then yield long enough for it to reach the socket.

    The producer does heavy sync CPU work (ONNX, clustering, transcription)
    inside an async task. queue.put() never suspends, and a single
    asyncio.sleep(0) only lets the consumer grab the item — the
    BaseHTTPMiddleware body pump and uvicorn transport need additional
    loop cycles to flush the bytes before the producer blocks again.
    A short real sleep gives them that window.
    """
    await queue.put(evt)
    job_state.publish(evt)
    await asyncio.sleep(0.01)


def _load_known_speakers(settings, user_id: str) -> dict:
    """Load all stored voiceprints from DB for speaker matching."""
    try:
        from asr_mcp.db.manager import DatabaseManager, VoiceprintDB
        from asr_mcp.voiceprint.service import is_spurious_speaker_name
        db = DatabaseManager(settings.db_path)
        vp_db = VoiceprintDB(db)
        all_vps = vp_db.list_all(user_id=user_id)
        valid_vps = {}
        for name, vp in all_vps.items():
            if is_spurious_speaker_name(name):
                try:
                    vp_db.delete(name, user_id=user_id)
                    logger.info("Purged spurious voiceprint %r from DB", name)
                except Exception:
                    pass
            else:
                valid_vps[name] = vp
        if valid_vps:
            logger.info("Loaded %d known voiceprints for user %s", len(valid_vps), user_id)
        return valid_vps
    except Exception as e:
        logger.warning("Failed to load known voiceprints: %s", e)
        return {}


@router.post("/diarize", response_model=DiarizeResponse)
async def diarize_endpoint(
    req: DiarizeRequest,
    settings: Settings = Depends(get_settings),
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.core.model_state import state
    from asr_mcp.diarization.pipeline import Diarizer

    state.ensure_diarize_ready()
    if not state.ready_for_diarize:
        return DiarizeResponse(segments=[], total_time_sec=0, error="Models not loaded. CUDA GPU required.")
    diarizer = Diarizer(state, settings)
    known_speakers = req.known_speakers or _load_known_speakers(settings, user_id)
    result = await diarizer.run(
        audio_path=req.wav_path,
        num_speakers=req.num_speakers,
        diarization_threshold=req.diarization_threshold,
        vad_threshold=req.vad_threshold,
        known_speakers=known_speakers,
    )

    if "error" in result:
        return DiarizeResponse(segments=[], total_time_sec=0, error=result["error"])

    segments = result.get("segments", [])

    try:
        from asr_mcp.db.manager import DatabaseManager
        from asr_mcp.voiceprint.service import VoiceprintService
        db = DatabaseManager(settings.db_path)
        vp_service = VoiceprintService(settings.data_dir, db)
        vp_service.set_voices_dir(settings.voices_dir)
        collected = vp_service.auto_collect_from_diarization(
            audio_path=req.wav_path, segments=segments, user_id=user_id,
        )
        if collected:
            logger.info("Auto-collected %d snippets for user %s", len(collected), user_id)
    except Exception as e:
        logger.warning("Auto-collect failed: %s", e)

    return DiarizeResponse(
        segments=[DiarizeResult(start=s["start"], end=s["end"], speaker=s["speaker"]) for s in segments],
        total_time_sec=result.get("total_time_sec", 0),
        total_speakers=result.get("total_speakers", 0),
        audio_duration_sec=result.get("audio_duration_sec", 0),
    )


@router.post("/diarize/upload")
async def diarize_upload(
    file: UploadFile = File(...),
    num_speakers: int = None,
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    busy = job_state.get_running()
    if busy is not None:
        content = {"detail": "A transcription is already in progress"}
        m = busy.meta()
        m["can_cancel"] = (
            busy.user_id == user_id and busy.mode == "transcribe" and busy.status == "running"
        )
        content["job"] = m
        return JSONResponse(status_code=409, content=content)

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return JSONResponse(status_code=413, detail="File too large (max 200MB)")

    validate_upload_filename(file.filename)

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    from asr_mcp.core.model_state import state
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import convert_to_wav

    settings = get_settings()

    try:
        wav_path = convert_to_wav(str(tmp_path), tmp_dir)
    except Exception as e:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=400, content={"detail": f"Audio conversion failed: {e}"})

    if not state.ready_for_diarize:
        state.ensure_diarize_ready()
    if not state.ready_for_diarize:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=503, content={"detail": "Models not loaded. CUDA GPU required."})

    queue = asyncio.Queue()
    job = job_state.start_job(mode="diarize", filename=file.filename, user_id=user_id)

    from asr_mcp.core.model_state import log_gpu_memory
    log_gpu_memory("diarize start")

    async def run_diarize():
        try:
            diarizer = Diarizer(state, settings)
            known_speakers = _load_known_speakers(settings, user_id)

            async def progress_cb(evt):
                evt.setdefault("phase", "diarization")
                await _sse_put(queue, evt)

            result = await diarizer.run(
                audio_path=str(wav_path),
                num_speakers=num_speakers,
                known_speakers=known_speakers or None,
                progress_callback=progress_cb,
            )

            segments = result.get("segments", [])
            try:
                from asr_mcp.db.manager import DatabaseManager
                from asr_mcp.voiceprint.service import VoiceprintService
                db = DatabaseManager(settings.db_path)
                vp_service = VoiceprintService(settings.data_dir, db)
                vp_service.set_voices_dir(settings.voices_dir)
                vp_service.auto_collect_from_diarization(
                    audio_path=str(wav_path), segments=segments, user_id=user_id,
                    source_id=file.filename,
                )
            except Exception as e:
                logger.warning("Auto-collect failed: %s", e)

            await _sse_put(queue, {"stage": "done", "progress": 1.0, "result": result})
        except Exception as e:
            logger.error("Diarize failed: %s", e)
            await _sse_put(queue, {"stage": "error", "error": str(e)})
        finally:
            job_state.ensure_finished(job)
            from asr_mcp.core.model_state import log_gpu_memory as _log_gpu
            _log_gpu("diarize end")
            await queue.put(None)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    asyncio.create_task(run_diarize())

    async def event_stream():
        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield f"data: {json.dumps(_safe_json(evt))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/transcribe")
async def transcribe_endpoint(
    req: DiarizeRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    _: str = Depends(verify_api_key),
):
    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import load_audio
    import numpy as np

    state.ensure_diarize_ready()
    if not state.ready_for_diarize:
        return TranscribeResponse(
            results=[TranscribeResult(error="Models not loaded. CUDA GPU required.")],
            total_time_sec=0,
        )

    diarizer = Diarizer(state, settings)
    user_id = get_session_user(request) or "default"
    known_speakers = req.known_speakers or _load_known_speakers(settings, user_id)
    diarization = await diarizer.run(
        audio_path=req.wav_path, num_speakers=req.num_speakers,
        known_speakers=known_speakers,
    )

    waveform, sr = load_audio(req.wav_path)
    audio_np = waveform.numpy().squeeze().astype(np.float32)

    segments = diarization.get("segments", [])
    if not segments:
        _switch_to_backend_phase()
        result = transcribe_audio_sync(audio=audio_np)
        return TranscribeResponse(
            results=[TranscribeResult(**result)],
            total_time_sec=result.get("inference_time_sec", 0),
        )

    results = _transcribe_diarized(audio_np, segments, sr, known_speakers=known_speakers)

    total_time = sum(r.inference_time_sec for r in results)
    return TranscribeResponse(results=results, total_time_sec=total_time)


@router.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    num_speakers: int = None,
    save: bool = Query(True),
    settings: Settings = Depends(get_settings),
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    busy = job_state.get_running()
    if busy is not None:
        content = {"detail": "A transcription is already in progress"}
        m = busy.meta()
        m["can_cancel"] = (
            busy.user_id == user_id and busy.mode == "transcribe" and busy.status == "running"
        )
        content["job"] = m
        return JSONResponse(status_code=409, content=content)

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"detail": "File too large (max 200MB)"})

    validate_upload_filename(file.filename)

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    from asr_mcp.voiceprint.utils import convert_to_wav
    wav_path = convert_to_wav(str(tmp_path), tmp_dir)

    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import load_audio
    import numpy as np

    if not state.ready_for_diarize:
        state.ensure_diarize_ready()
    if not state.ready_for_diarize:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=503, content={"detail": "Models not loaded. CUDA GPU required."})

    queue = asyncio.Queue()
    job = job_state.start_job(mode="transcribe", filename=file.filename, user_id=user_id)
    client_disconnected = False

    from asr_mcp.core.model_state import log_gpu_memory
    log_gpu_memory("transcribe start")

    async def run_transcribe():
        job_t0 = time.monotonic()
        try:
            diarizer = Diarizer(state, settings)
            known_speakers = _load_known_speakers(settings, user_id)

            async def progress_cb(evt):
                if job.cancel_requested:
                    raise JobCancelled()
                evt.setdefault("phase", "diarization")
                await _sse_put(queue, evt)

            await _sse_put(queue, {"stage": "Running diarization", "progress": 0.0, "phase": "diarization"})
            diarization = await diarizer.run(
                audio_path=str(wav_path), num_speakers=num_speakers,
                known_speakers=known_speakers or None,
                progress_callback=progress_cb,
            )

            waveform, sr = load_audio(str(wav_path))
            audio_np = waveform.numpy().squeeze().astype(np.float32)

            segments = diarization.get("segments", [])
            audio_dur = diarization.get("audio_duration_sec", 0.0)

            def _done_event(result):
                processing = round(time.monotonic() - job_t0, 1)
                speedup = round(audio_dur / processing, 2) if processing > 0 and audio_dur else None
                logger.info(
                    "Transcription finished: %.1fs audio in %.1fs (%s)",
                    audio_dur or 0.0, processing,
                    f"{speedup:g}x realtime" if speedup else "n/a",
                )
                payload = {
                    "stage": "done",
                    "progress": 1.0,
                    "result": result,
                    "processing_time_sec": processing,
                }
                if speedup:
                    payload["speedup"] = speedup
                return payload

            def _persist_transcript(reason):
                try:
                    import hashlib as _hl
                    file_hash = _hl.sha256(content).hexdigest()[:16]
                    from asr_mcp.db.manager import DatabaseManager, TranscriptDB
                    db = DatabaseManager(settings.db_path)
                    tdb = TranscriptDB(db)
                    tdb.save(
                        audio_filename=file.filename,
                        result=diarization,
                        user_id=user_id,
                        file_hash=file_hash,
                        total_speakers=diarization.get("total_speakers", 0),
                        audio_duration_sec=diarization.get("audio_duration_sec", 0.0),
                        processing_time_sec=diarization.get("total_time_sec", 0.0),
                    )
                    if reason:
                        logger.info("Transcript preserved server-side (%s): %s", reason, file.filename)
                except Exception as e:
                    logger.warning("Failed to save transcription: %s", e)

            try:
                from asr_mcp.db.manager import DatabaseManager
                from asr_mcp.voiceprint.service import VoiceprintService
                db = DatabaseManager(settings.db_path)
                vp_service = VoiceprintService(settings.data_dir, db)
                vp_service.set_voices_dir(settings.voices_dir)
                vp_service.auto_collect_from_diarization(
                    audio_path=str(wav_path), segments=segments, user_id=user_id,
                    source_id=file.filename,
                )
            except Exception as e:
                logger.warning("Auto-collect failed: %s", e)

            turns = []
            if segments:
                try:
                    turns = _prepare_turns(
                        segments,
                        audio_duration_sec=audio_dur or None,
                        audio=audio_np,
                        sample_rate=sr or 16000,
                        known_speakers=known_speakers or None,
                    )
                except Exception as e:
                    logger.warning("Boundary refinement failed: %s", e)
                    turns = list(segments)
            display_segments = (
                [{"start": t["start"], "end": t["end"], "speaker": t["speaker"]} for t in turns]
                if turns else segments
            )
            diarization["segments"] = display_segments

            await _sse_put(queue, {
                "stage": "diarization_complete",
                "progress": 1.0,
                "phase": "diarization",
                "segments": display_segments,
                "audio_duration_sec": audio_dur,
                "total_speakers": diarization.get("total_speakers", 0),
            })

            state.unload_embedding()
            state.ensure_backend_ready()
            if job.cancel_requested:
                raise JobCancelled()

            if not segments:
                await _sse_put(queue, {"stage": "Transcribing audio", "progress": 0.0, "phase": "transcription"})
                result = transcribe_audio_sync(audio=audio_np)
                diarization["results"] = [_result_to_dict(TranscribeResult(**result))]
                if client_disconnected:
                    _persist_transcript("client disconnected")
                await _sse_put(queue, _done_event(diarization))
                return

            attr_turns = turns or segments
            dur = audio_dur or (len(audio_np) / (sr or 16000))
            starts = [float(t["start"]) for t in attr_turns]
            loop = asyncio.get_running_loop()
            transcribe_t0 = time.monotonic()

            def _timing(processed):
                now = time.monotonic()
                payload = {"elapsed_sec": round(now - job_t0, 1)}
                total = audio_dur or 0.0
                if processed > 0 and total > 0:
                    trans_elapsed = now - transcribe_t0
                    if trans_elapsed > 0:
                        payload["predicted_remaining_sec"] = round(
                            max(total - processed, 0.0) * trans_elapsed / processed, 1
                        )
                return payload

            state.touch()
            await _sse_put(queue, {
                "stage": "Transcribing audio",
                "progress": 0.0,
                "phase": "transcription",
                "segment_index": 0,
                "total_segments": 1,
                "segment_speaker": _speaker_for_span(0.0, 0.0, attr_turns, starts) or "",
                "segment_start": 0.0,
                "segment_end": 0.0,
                **_timing(0.0),
            })

            def _make_window_cb():
                def _emit_window(evt):
                    queue.put_nowait(evt)
                    job_state.publish(evt)

                def cb(i, n):
                    if job.cancel_requested:
                        raise JobCancelled()
                    frac = i / max(n, 1)
                    win_start = dur * ((i - 1) / max(n, 1))
                    win_end = dur * (i / max(n, 1))
                    loop.call_soon_threadsafe(_emit_window, {
                        "stage": f"Transcribing — window {i}/{n}",
                        "progress": frac,
                        "phase": "transcription",
                        "segment_index": i - 1,
                        "total_segments": n,
                        "segment_speaker": _speaker_for_span(win_start, win_end, attr_turns, starts) or "",
                        "segment_start": round(win_start, 2),
                        "segment_end": round(win_end, 2),
                        "turn_start": round(win_start, 2),
                        "turn_end": round(win_end, 2),
                        "window": i,
                        "total_windows": n,
                        **_timing(dur * frac),
                    })
                return cb

            results = await loop.run_in_executor(
                None, _transcribe_file, audio_np, attr_turns, 16000, _make_window_cb(),
            )
            if job.cancel_requested:
                raise JobCancelled()
            results = _merge_consecutive_same_speaker_results(results)
            diarization["results"] = [_result_to_dict(r) for r in results]
            diarization["total_time_sec"] = sum(
                (r.inference_time_sec if hasattr(r, 'inference_time_sec') else 0) for r in results
            )

            if save:
                _persist_transcript(None)
            elif client_disconnected:
                _persist_transcript("client disconnected")

            await _sse_put(queue, _done_event(diarization))
        except JobCancelled:
            logger.info("Transcription cancelled by user: %s", file.filename)
            await _sse_put(queue, {
                "stage": "cancelled",
                "progress": 0.0,
                "message": "Cancelled by user",
            })
        except Exception as e:
            logger.error("Transcribe failed: %s", e)
            await _sse_put(queue, {"stage": "error", "error": str(e)})
        finally:
            job_state.ensure_finished(job)
            from asr_mcp.core.model_state import log_gpu_memory as _log_gpu
            _log_gpu("transcribe end")
            await queue.put(None)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    asyncio.create_task(run_transcribe())

    async def event_stream():
        nonlocal client_disconnected
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(_safe_json(evt))}\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            client_disconnected = True
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/active")
async def active_job(
    request: Request,
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    job = job_state.get_running()
    if job is None:
        return {"active": False}
    meta = job.meta()
    meta["can_cancel"] = (
        job.user_id == user_id and job.mode == "transcribe" and job.status == "running"
    )
    return {"active": True, "job": meta}


@router.get("/active/stream")
async def active_job_stream(
    request: Request,
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    job = job_state.get_running()
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "No active job"})

    owner = job.user_id == user_id
    snap, sub = job_state.attach(job)

    def _scrub(evt):
        if not owner and isinstance(evt, dict) and "result" in evt:
            return {k: v for k, v in evt.items() if k != "result"}
        return evt

    async def event_stream():
        for evt in snap:
            yield f"data: {json.dumps(_safe_json(_scrub(evt)))}\n\n"
        if sub is None:
            return
        while True:
            evt = await sub.get()
            if evt is None:
                break
            yield f"data: {json.dumps(_safe_json(_scrub(evt)))}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.post("/active/cancel")
async def active_job_cancel(
    request: Request,
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    job = job_state.get_for_user(user_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "No active job"})
    if job.status != "running":
        return JSONResponse(status_code=409, content={"detail": f"Job is already {job.status}"})
    if job.mode != "transcribe":
        return JSONResponse(status_code=409, content={"detail": "Cancel is only supported for transcriptions"})
    job.cancel_requested = True
    logger.info("Cancel requested for job %s (%s)", job.id, job.filename)
    return {"status": "cancelling", "job": job.meta()}


@router.post("/stream")
async def stream_placeholder():
    from starlette.responses import JSONResponse as StarletteJSONResponse
    return StarletteJSONResponse(
        status_code=501,
        content={"detail": "Streaming is available via WebSocket at /api/asr/ws/stream"},
    )


@router.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    from asr_mcp.streaming.handler import handle_ws_stream
    await handle_ws_stream(websocket)
