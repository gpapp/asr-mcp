import asyncio
import json
import logging
import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from asr_mcp.api.schemas import (
    DiarizeRequest, DiarizeResponse, DiarizeResult,
    TranscribeResponse, TranscribeResult,
)
from asr_mcp.api.security import verify_api_key, get_current_user
from asr_mcp.api.auth import get_session_user
from asr_mcp.config.settings import Settings, get_settings
from asr_mcp.speaker.vad import split_at_energy_dips

logger = logging.getLogger("asr_mcp.api.asr_router")
router = APIRouter(prefix="/asr", tags=["ASR"])

MIN_CHUNK_SAMPLES = 1600
MAX_TURN_SEC = 30.0
PARAGRAPH_PAUSE_SEC = 1.5
_SENT_END_RE = re.compile(r'[.!?…]["”’)\]]?(?=\s|$)')


def _merge_into_turns(segments, max_gap_sec=1.5):
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


def _transcribe_turn(audio_np, turn, sample_rate=16000, progress_cb=None):
    """Transcribe a single speaker turn as one coherent audio chunk."""
    from asr_mcp.core.transcriber import transcribe_audio_sync

    start_sample = int(turn["start"] * sample_rate)
    end_sample = int(turn["end"] * sample_rate)
    turn_audio = audio_np[start_sample:end_sample]

    if len(turn_audio) < MIN_CHUNK_SAMPLES:
        return []

    dur = (end_sample - start_sample) / sample_rate
    logger.info("Transcribing turn: %.1f-%.1fs (%.1fs, %s)",
                turn["start"], turn["end"], dur, turn["speaker"])

    tr = transcribe_audio_sync(audio=turn_audio, progress_cb=progress_cb)

    logger.info("Turn result: text=%d chars, tokens=%d, inference=%.2fs, error=%s",
                len(tr.get("text", "")), tr.get("tokens_generated", 0),
                tr.get("inference_time_sec", 0), tr.get("error"))

    text = (tr.get("text") or "").strip()
    segments = tr.get("segments")
    error = tr.get("error")
    if not text and not segments and not error:
        return []

    text = _apply_paragraph_breaks(text, segments, turn_audio, sample_rate)

    return [TranscribeResult(
        text=text,
        segments=segments,
        start=turn["start"],
        end=turn["end"],
        speaker=turn["speaker"],
        audio_duration_sec=tr.get("audio_duration_sec", 0),
        inference_time_sec=tr.get("inference_time_sec", 0),
        tokens_generated=tr.get("tokens_generated", 0),
        error=error,
    )]


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
        )
    return run_vad_chunked(
        wf, sample_rate=sample_rate,
        threshold=threshold, min_speech_duration_ms=min_ms,
    )


def _embed_section(audio, start_sample, end_sample, sample_rate):
    """Voiceprint embedding for one VAD section, or None if unusable."""
    import numpy as np
    import torch
    from asr_mcp.core.model_state import state
    from asr_mcp.speaker.embedding import extract_embedding

    if state.embedding_session is None:
        return None
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
            if s["end"] > int(gap_start * sample_rate)
            and s["start"] < int(gap_end * sample_rate)
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
            cut = valid[k]["start"] / sample_rate
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


def _transcribe_diarized(audio_np, segments, sample_rate=16000, known_speakers=None):
    turns = _prepare_turns(
        segments,
        audio_duration_sec=len(audio_np) / sample_rate,
        audio=audio_np,
        sample_rate=sample_rate,
        known_speakers=known_speakers,
    )
    all_results = []
    for turn in turns:
        all_results.extend(_transcribe_turn(audio_np, turn, sample_rate))
    return all_results


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
    await asyncio.sleep(0.01)


def _load_known_speakers(settings, user_id: str) -> dict:
    """Load all stored voiceprints from DB for speaker matching."""
    try:
        from asr_mcp.db.manager import DatabaseManager, VoiceprintDB
        db = DatabaseManager(settings.db_path)
        vp_db = VoiceprintDB(db)
        all_vps = vp_db.list_all(user_id=user_id)
        if all_vps:
            logger.info("Loaded %d known voiceprints for user %s", len(all_vps), user_id)
        return all_vps
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

    state.ensure_ready()
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
        vp_service.set_embedding_session(state.embedding_session)
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
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return JSONResponse(status_code=413, detail="File too large (max 200MB)")

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

    queue = asyncio.Queue()

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
                vp_service.set_embedding_session(state.embedding_session)
                vp_service.auto_collect_from_diarization(
                    audio_path=str(wav_path), segments=segments, user_id=user_id,
                )
            except Exception as e:
                logger.warning("Auto-collect failed: %s", e)

            await _sse_put(queue, {"stage": "done", "progress": 1.0, "result": result})
        except Exception as e:
            logger.error("Diarize failed: %s", e)
            await _sse_put(queue, {"stage": "error", "error": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_diarize())

    async def event_stream():
        import shutil
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(_safe_json(evt))}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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
    from asr_mcp.core.transcriber import transcribe_audio_sync, _compute_mel_spectrogram_fast
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import load_audio
    import numpy as np

    state.ensure_ready()
    if not state.is_ready:
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
        mel = _compute_mel_spectrogram_fast(audio_np)
        result = transcribe_audio_sync(mel_spectrogram=mel)
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
    settings: Settings = Depends(get_settings),
    user_id: str = Depends(get_current_user),
    _: str = Depends(verify_api_key),
):
    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"detail": "File too large (max 200MB)"})

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(content)

    from asr_mcp.voiceprint.utils import convert_to_wav
    wav_path = convert_to_wav(str(tmp_path), tmp_dir)

    from asr_mcp.core.model_state import state
    from asr_mcp.core.transcriber import transcribe_audio_sync, _compute_mel_spectrogram_fast
    from asr_mcp.diarization.pipeline import Diarizer
    from asr_mcp.voiceprint.utils import load_audio
    import numpy as np

    if not state.is_ready:
        state.ensure_ready()
    if not state.is_ready:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return JSONResponse(status_code=503, content={"detail": "Models not loaded. CUDA GPU required."})

    queue = asyncio.Queue()

    async def run_transcribe():
        try:
            diarizer = Diarizer(state, settings)
            known_speakers = _load_known_speakers(settings, user_id)

            async def progress_cb(evt):
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

            try:
                from asr_mcp.db.manager import DatabaseManager
                from asr_mcp.voiceprint.service import VoiceprintService
                db = DatabaseManager(settings.db_path)
                vp_service = VoiceprintService(settings.data_dir, db)
                vp_service.set_voices_dir(settings.voices_dir)
                vp_service.set_embedding_session(state.embedding_session)
                vp_service.auto_collect_from_diarization(
                    audio_path=str(wav_path), segments=segments, user_id=user_id,
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

            if not segments:
                await _sse_put(queue, {"stage": "Transcribing audio", "progress": 0.0, "phase": "transcription"})
                mel = _compute_mel_spectrogram_fast(audio_np)
                result = transcribe_audio_sync(mel_spectrogram=mel)
                diarization["results"] = [_result_to_dict(TranscribeResult(**result))]
                await _sse_put(queue, {"stage": "done", "progress": 1.0, "result": diarization})
                return

            total_turns = len(turns)
            loop = asyncio.get_running_loop()
            results = []
            for turn_idx, turn in enumerate(turns):
                p = turn_idx / max(total_turns, 1)
                await _sse_put(queue, {
                    "stage": f"Transcribing turn {turn_idx+1}/{total_turns} ({turn['speaker']})",
                    "progress": p,
                    "phase": "transcription",
                    "segment_index": turn_idx,
                    "total_segments": total_turns,
                    "segment_speaker": turn["speaker"],
                    "segment_start": turn["start"],
                    "segment_end": turn["end"],
                })

                def _make_window_cb(turn_idx=turn_idx, turn=turn):
                    def cb(i, n):
                        frac = (turn_idx + i / max(n, 1)) / max(total_turns, 1)
                        span = turn["end"] - turn["start"]
                        win_start = turn["start"] + span * ((i - 1) / max(n, 1))
                        win_end = turn["start"] + span * (i / max(n, 1))
                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "stage": f"Transcribing turn {turn_idx+1}/{total_turns} — window {i}/{n} ({turn['speaker']})",
                            "progress": frac,
                            "phase": "transcription",
                            "segment_index": turn_idx,
                            "total_segments": total_turns,
                            "segment_speaker": turn["speaker"],
                            "segment_start": round(win_start, 2),
                            "segment_end": round(win_end, 2),
                            "turn_start": turn["start"],
                            "turn_end": turn["end"],
                            "window": i,
                            "total_windows": n,
                        })
                    return cb

                turn_results = await loop.run_in_executor(
                    None,
                    _transcribe_turn, audio_np, turn, 16000, _make_window_cb(),
                )
                results.extend(turn_results)

            diarization["results"] = [_result_to_dict(r) for r in results]
            diarization["total_time_sec"] = sum(
                (r.inference_time_sec if hasattr(r, 'inference_time_sec') else 0) for r in results
            )

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
            except Exception as e:
                logger.warning("Failed to save transcription: %s", e)

            await _sse_put(queue, {"stage": "done", "progress": 1.0, "result": diarization})
        except Exception as e:
            logger.error("Transcribe failed: %s", e)
            await _sse_put(queue, {"stage": "error", "error": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_transcribe())

    async def event_stream():
        import shutil
        try:
            while True:
                evt = await queue.get()
                if evt is None:
                    break
                yield f"data: {json.dumps(_safe_json(evt))}\n\n"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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
