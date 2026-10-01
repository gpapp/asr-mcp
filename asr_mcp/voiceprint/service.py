import asyncio
import datetime
import functools
import hashlib
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from asr_mcp.db.manager import DatabaseManager, SnippetDB, VoiceprintDB, DEFAULT_USER
from asr_mcp.speaker.uncertainty import (
    eligible_for_auto_collect, eligible_for_learning, is_generic_speaker,
)
from asr_mcp.speaker.embedding import (
    extract_embedding, batch_embed_files, compute_pitch, compute_energy,
)
from asr_mcp.voiceprint.utils import (
    AudioSegmentSource, load_audio, load_audio_segment,
    generate_segment_hash, format_time_short,
)

logger = logging.getLogger("asr_mcp.voiceprint.service")

SAMPLE_RATE = 16000
AUTO_COLLECT_MIN_DURATION = 3.0
AUTO_COLLECT_MAX_TOTAL_SEC = 1200.0
AUTO_COLLECT_MAX_SEGMENT_SEC = 300.0
AUTO_COLLECT_MIN_SPEAKER_SEGMENTS = 2
MIN_SNIPPET_DURATION = 1.5

# Learning an unknown speaker. A pending profile is created from a generic
# "Speaker N" cluster, collects snippets, and is EXCLUDED from voiceprint
# matching until the user names it -- so learning a colleague can never produce
# a confident misattribution (lesson 30).
PENDING_PREFIX = "Pending"
LEARN_MIN_SPEECH_SEC = 10.0
LEARN_MIN_SEGMENTS = 2


def _learning_cfg() -> dict:
    """`learning` section of thresholds.json, with the shipped defaults."""
    defaults = {
        "split_clusters": True,
        "max_intra_cluster_dist": 0.32,
        # P2 (fail closed).  False restores the pre-P2 behaviour of learning the
        # whole cluster whenever the cohesion check could not be completed.
        "fail_closed": True,
        "max_profile_intra_dist": 0.32,
        "min_inter_dist": 0.32,
        "extend_max_dist": 0.32,
        "max_snippet_sec": 30.0,
        "max_profile_total_sec": 900.0,
        "max_refine_weight_ratio": 4.0,
    }
    try:
        from asr_mcp.config import get_config

        raw = get_config() or {}
        section = raw.get("learning") or {}
        if isinstance(section, dict):
            defaults.update(section)
    except Exception:
        pass
    return defaults


def _cosine_distance(a, b) -> float:
    """1 - cosine similarity of two vectors (0 = identical direction)."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if not np.isfinite(na) or not np.isfinite(nb) or na <= 1e-8 or nb <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(va, vb) / (na * nb))


def _centroid(vectors) -> Optional[np.ndarray]:
    if not vectors:
        return None
    arr = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return mean / norm


def _max_pairwise_distance(vectors) -> Optional[float]:
    """Largest within-group cosine distance, or None for < 2 vectors."""
    if not vectors or len(vectors) < 2:
        return None
    arr = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
    sim = np.clip(arr @ arr.T, -1.0, 1.0)
    size = sim.shape[0]
    iu = np.triu_indices(size, k=1)
    return float(np.max(1.0 - sim[iu]))


def _name_tokens(name: str) -> list[str]:
    """Lowercase alphanumeric tokens of a profile name / cluster label.

    Exact TOKEN matching is what keeps ``Speaker 1`` from being treated as a
    substring of ``Speaker 10``: the label tokens are ``["speaker", "1"]`` and
    ``["speaker", "10"]``, which are not the same.
    """
    return re.findall(r"[a-z0-9]+", str(name or "").lower())


def _tokens_contain(haystack: list[str], needle: list[str]) -> bool:
    """Is ``needle`` a contiguous token sub-list of ``haystack``?"""
    if not needle or len(needle) > len(haystack):
        return False
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return True
    return False


def trim_span_to_vad(start_sec: float, end_sec: float, sections,
                     eps: float = 1e-3):
    """Snap a snippet's edges to the VAD speech sections inside it.

    Why: a learned snippet is embedded verbatim, and that embedding is later
    used as a **reference for cutting boundaries**
    (``diarization/pipeline.py::_get_speaker_ref``,
    ``api/asr_router.py::_speaker_refs``).  An edge that lands inside a speech
    section keeps a fragment of a phoneme in the training audio.

    The operation is deliberately **inward only**.  Two of the three places that
    place a boundary do it arithmetically (a midpoint, or a cut clamped into a
    gap the neighbouring turn also owns), and turns ABUT by design
    (lesson 17), so padding outward would pull the neighbouring speaker's
    speech into this snippet and duplicate it there — trading a cut phoneme for
    a contaminated sample.  So each edge is moved at most onto the outermost
    section that lies inside the span, and never past it.

    Returns ``(start, end)``; unchanged when there is no usable section.
    """
    start = float(start_sec)
    end = float(end_sec)
    if not sections or end <= start:
        return start, end
    ordered = sorted(
        (s for s in sections if s.get("end", 0) > s.get("start", 0)),
        key=lambda s: float(s["start"]),
    )
    overlapping = [
        s for s in ordered
        if float(s["end"]) > start + eps and float(s["start"]) < end - eps
    ]
    if not overlapping:
        return start, end
    new_start = max(start, float(overlapping[0]["start"]))
    new_end = min(end, float(overlapping[-1]["end"]))
    if new_end - new_start < MIN_SNIPPET_DURATION:
        # Trimming must not produce an unusable snippet: keep the untrimmed
        # span and let the existing duration floor decide.
        return start, end
    return new_start, new_end


def is_pending_profile(name: str, voiceprint: dict | None = None) -> bool:
    """Is this an auto-learned profile still waiting for a name?

    Checks the stored ``pending`` flag AND the name prefix. The prefix check
    matters because the flag lives on the voiceprints row, and that row is only
    written by ``_auto_refine`` -- so if embedding was unavailable when a
    speaker was learned, its snippets exist with no row and would otherwise be
    invisible. A directory named "Pending ..." is a pending profile by
    construction, and both signals say "not matchable" anyway.
    """
    if str(name).startswith(PENDING_PREFIX + " "):
        return True
    return bool((voiceprint or {}).get("pending"))


def pending_profile_name(source_id: str, cluster: str = "") -> str:
    """A unique, human-readable placeholder name for an unnamed profile.

    The CLUSTER label is part of the name on purpose. Without it, two different
    speakers in the same recording produce the same name (same minute, same
    source stem) and the second cluster silently extends the first one's
    profile -- merging two people. Including "Speaker 5" keeps one profile per
    cluster per recording, which is also what makes the extend-rather-than-
    create rule below safe.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [PENDING_PREFIX, stamp]
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", cluster or "").strip("_")
    if label:
        parts.append(label)
    stem = _origin_prefix(source_id)
    if stem:
        parts.append(stem)
    return " ".join(parts)[:120]


def _origin_prefix(audio_path: str) -> str:
    """Filesystem-safe prefix from the source audio filename stem."""
    stem = Path(audio_path).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return (stem or "audio")[:64]


def is_spurious_speaker_name(name: str) -> bool:
    """Check if a speaker name is generic or auto-generated.

    Delegates to the shared uncertainty policy (asr_mcp.speaker.uncertainty) so
    this and the auto-collection gate can never disagree on what counts as an
    identity.
    """
    from asr_mcp.speaker.uncertainty import is_uncertain_label
    return is_uncertain_label(name)


class VoiceprintService:
    def __init__(self, data_dir: Path, db_manager: DatabaseManager, embedding_session=None):
        self._data_dir = data_dir
        self._db = VoiceprintDB(db_manager)
        self._snippets = SnippetDB(db_manager)
        self._db_manager = db_manager
        self._embedding_session = embedding_session
        self._voices_dir = data_dir / "voices"

    def set_embedding_session(self, session):
        self._embedding_session = session

    def _emb_session(self):
        from asr_mcp.core.model_state import state
        return self._embedding_session or state.embedding_session

    async def initialize(self):
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        count = self._db.count()
        logger.info("VoiceprintService initialized — %d voiceprints", count)

    @property
    def voices_dir(self) -> Path:
        return self._voices_dir

    def set_voices_dir(self, path: Path):
        self._voices_dir = path
        path.mkdir(parents=True, exist_ok=True)

    # ── Snippet Management ───────────────────────────────────────────

    def add_snippet(
        self,
        speaker_name: str,
        audio_data: np.ndarray,
        user_id: str = DEFAULT_USER,
        sample_rate: int = SAMPLE_RATE,
        source_audio: str = None,
        start_sec: float = None,
        end_sec: float = None,
    ) -> dict:
        user_dir = self._voices_dir / user_id
        speaker_dir = user_dir / speaker_name
        speaker_dir.mkdir(parents=True, exist_ok=True)

        duration = len(audio_data) / sample_rate
        if duration < MIN_SNIPPET_DURATION:
            return {"error": f"Snippet too short ({duration:.1f}s < {MIN_SNIPPET_DURATION}s)"}

        if source_audio:
            audio_hash = generate_segment_hash(source_audio)
            time_part = format_time_short(start_sec) if start_sec is not None else "00-00"
            dur_part = f"{duration:02.0f}"
            filename = f"{audio_hash}_{time_part}_{dur_part}.flac"
        else:
            audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()[:6].upper()
            filename = f"{audio_hash}_manual.flac"
        file_path = speaker_dir / filename

        sf.write(str(file_path), audio_data.astype(np.float32), sample_rate, format="FLAC")
        try:
            fst = file_path.stat()
            file_mtime, file_size = fst.st_mtime, fst.st_size
        except OSError:
            file_mtime, file_size = None, None

        snippet_id = self._snippets.add(
            speaker_name=speaker_name,
            file_path=str(file_path),
            duration_sec=duration,
            user_id=user_id,
            source_audio=source_audio,
            start_sec=start_sec,
            end_sec=end_sec,
            file_mtime=file_mtime,
            file_size=file_size,
        )

        return {
            "id": snippet_id,
            "speaker_name": speaker_name,
            "file_path": str(file_path),
            "duration_sec": round(duration, 2),
        }

    def add_snippet_from_segment(
        self,
        speaker_name: str,
        wav_path: str,
        start_sec: float,
        end_sec: float,
        user_id: str = DEFAULT_USER,
        source_audio: str = None,
        audio_source: AudioSegmentSource = None,
    ) -> dict:
        # ``audio_source`` is the request-wide cached loader (one decode for the
        # whole request).  Without it this falls back to the historical
        # behaviour of decoding the ENTIRE file for this one span.
        if audio_source is not None:
            waveform, sr = audio_source.segment(start_sec, end_sec)
        else:
            waveform, sr = load_audio_segment(wav_path, start_sec, end_sec)
        audio_data = waveform.numpy().squeeze()
        return self.add_snippet(
            speaker_name=speaker_name,
            audio_data=audio_data,
            user_id=user_id,
            sample_rate=sr,
            source_audio=source_audio or wav_path,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    def list_speakers(self, user_id: str = DEFAULT_USER) -> list[dict]:
        snippet_info = self._snippets.all_speakers(user_id=user_id)
        voiceprint_info = self._db.list_all(user_id=user_id)

        all_names = set(list(snippet_info.keys()) + list(voiceprint_info.keys()))
        speakers = []
        for name in sorted(all_names):
            sn = snippet_info.get(name, {"count": 0, "total_duration": 0.0})
            vp = voiceprint_info.get(name, {})
            speakers.append({
                "name": name,
                "snippet_count": sn["count"],
                "total_duration_sec": round(sn["total_duration"], 2),
                "has_voiceprint": bool(vp),
                "pending": is_pending_profile(name, vp),
                "pitch_hz": round(vp.get("pitch_hz", 0), 1),
                "energy_rms": round(vp.get("energy_rms", 0), 4),
                # Only set on auto-learned profiles (see _learn_unknown_speaker):
                # how far the profile's own snippets disagree with each other,
                # and with the other groups it was separated from.
                "purity": vp.get("purity") or None,
            })
        return speakers

    def list_snippets(self, speaker_name: str, user_id: str = DEFAULT_USER) -> list[dict]:
        return self._snippets.list_by_speaker(speaker_name, user_id=user_id)

    def refine_speaker(self, speaker_name: str, user_id: str = DEFAULT_USER) -> dict:
        snippets = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        if not snippets:
            return {"error": f"No snippets found for '{speaker_name}'"}
        self._auto_refine(speaker_name, user_id=user_id)
        vp = self._db.get(speaker_name, user_id=user_id)
        if not vp:
            return {"error": f"Voiceprint could not be built for '{speaker_name}' (embedding session unavailable?)"}
        return {
            "status": "refined",
            "speaker_name": speaker_name,
            "snippet_count": len(snippets),
            "total_duration_sec": round(sum(s.get("duration_sec", 0.0) for s in snippets), 2),
            "pitch_hz": round(vp.get("pitch_hz", 0.0), 1),
            "energy_rms": round(vp.get("energy_rms", 0.0), 4),
        }

    def delete_snippet(self, snippet_id: int, user_id: str = DEFAULT_USER) -> dict:
        sn = self._snippets.get(snippet_id, user_id=user_id)
        if not sn:
            return {"error": "Snippet not found"}

        speaker_name = sn["speaker_name"]
        file_path = Path(sn["file_path"])
        if file_path.exists():
            file_path.unlink()

        self._snippets.delete(snippet_id, user_id=user_id)

        remaining = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        if not remaining:
            self._db.delete(speaker_name, user_id=user_id)
            self._cleanup_speaker_dir(speaker_name, user_id)
            return {"status": "deleted", "speaker_removed": True}

        self._auto_refine(speaker_name, user_id=user_id)
        return {"status": "deleted", "speaker_removed": False}

    def delete_speaker_bulk(self, speaker_name: str, user_id: str = DEFAULT_USER) -> dict:
        """Delete a speaker, all snippets, and files without intermediate refine steps."""
        snippets = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        for sn in snippets:
            file_path = Path(sn["file_path"])
            if file_path.exists():
                file_path.unlink()
            self._snippets.delete(sn["id"], user_id=user_id)

        self._db.delete(speaker_name, user_id=user_id)
        self._cleanup_speaker_dir(speaker_name, user_id)

        return {"status": "deleted", "name": speaker_name, "snippets_removed": len(snippets)}

    def rename_speaker(self, old_name: str, new_name: str, user_id: str = DEFAULT_USER) -> dict:
        if new_name == old_name:
            return {"error": "New name is the same as the current name"}
        existing_vp = self._db.get(new_name, user_id=user_id)
        snippet_count = self._snippets.count(new_name, user_id=user_id)
        if existing_vp or snippet_count > 0:
            return {"error": f"Speaker '{new_name}' already exists"}

        count = self._snippets.rename_speaker(old_name, new_name, user_id=user_id)

        vp = self._db.get(old_name, user_id=user_id)
        if vp:
            self._db.save(
                name=new_name, user_id=user_id,
                embedding=vp["embedding"],
                pitch_hz=vp.get("pitch_hz", 0),
                pitch_std=vp.get("pitch_std", 0),
                energy_rms=vp.get("energy_rms", 0),
                spectral_centroid=vp.get("spectral_centroid", 0),
                spectral_rolloff=vp.get("spectral_rolloff", 0),
                total_speech_sec=vp.get("total_speech_sec", 0),
                sample_count=vp.get("sample_count", 0),
            )
            self._db.delete(old_name, user_id=user_id)

        old_dir = self._voices_dir / user_id / old_name
        new_dir = self._voices_dir / user_id / new_name
        if old_dir.exists():
            if new_dir.exists():
                for f in old_dir.iterdir():
                    shutil.move(str(f), str(new_dir / f.name))
                old_dir.rmdir()
            else:
                old_dir.rename(new_dir)

        old_prefix = str(old_dir)
        new_prefix = str(new_dir)
        paths_updated = 0
        for sn in self._snippets.list_by_speaker(new_name, user_id=user_id):
            fp = sn["file_path"]
            try:
                rel = Path(fp).relative_to(old_prefix)
            except ValueError:
                continue
            if self._snippets.update_file_path(sn["id"], str(new_prefix / rel), user_id=user_id):
                paths_updated += 1

        transcripts_updated = 0
        try:
            from asr_mcp.db.manager import TranscriptDB
            transcripts_updated = TranscriptDB(self._db._db).rename_speaker(
                old_name, new_name, user_id=user_id
            )
        except Exception as e:
            logger.warning("Failed to update transcripts for rename %r -> %r: %s",
                           old_name, new_name, e)

        return {
            "status": "renamed",
            "from": old_name,
            "to": new_name,
            "snippets_moved": count,
            "paths_updated": paths_updated,
            "transcripts_updated": transcripts_updated,
        }

    def merge_speakers(self, primary_name: str, secondary_name: str, user_id: str = DEFAULT_USER) -> dict:
        secondary_snippets = self._snippets.list_by_speaker(secondary_name, user_id=user_id)
        if not secondary_snippets:
            return {"error": f"No snippets found for '{secondary_name}'"}

        primary_dir = self._voices_dir / user_id / primary_name
        primary_dir.mkdir(parents=True, exist_ok=True)

        for sn in secondary_snippets:
            old_path = Path(sn["file_path"])
            if old_path.exists():
                new_path = primary_dir / old_path.name
                shutil.move(str(old_path), str(new_path))
                self._snippets.delete(sn["id"], user_id=user_id)
                self._snippets.add(
                    speaker_name=primary_name,
                    file_path=str(new_path),
                    duration_sec=sn["duration_sec"],
                    user_id=user_id,
                    source_audio=sn.get("source_audio"),
                    start_sec=sn.get("start_sec"),
                    end_sec=sn.get("end_sec"),
                )

        self._db.delete(secondary_name, user_id=user_id)
        self._cleanup_speaker_dir(secondary_name, user_id)

        self._auto_refine(primary_name, user_id=user_id)

        return {
            "status": "merged",
            "primary": primary_name,
            "secondary": secondary_name,
            "snippets_moved": len(secondary_snippets),
        }

    def rescan_voices_dir(
        self, user_id: str = DEFAULT_USER, progress_callback=None, force: bool = False,
    ) -> dict:
        """Scan the voices directory; add new snippets and rebuild stale voiceprints.

        Two phases:
          1. Scan: register audio files that are new (or new directories) and
             fingerprint-check every already-registered file (mtime+size) to
             detect CHANGED snippets.
          2. Rebuild: invalidate the voiceprint of any speaker whose snippet
             changed, then recompute the voiceprint for every speaker that is
             new, changed, or missing (has snippets but no voiceprint row).

        ``progress_callback(evt: dict)`` is invoked (optionally) after each
        file, so an SSE stream can reflect live scan/add/refine state.
        """
        user_dir = self._voices_dir / user_id
        if not user_dir.exists():
            return {"scanned": 0, "added": 0, "invalidated": 0, "rebuilt": 0, "speakers": []}

        def _progress(evt: dict):
            if progress_callback is None:
                return
            try:
                progress_callback(evt)
            except Exception as e:
                logger.warning("Progress callback error: %s", e)

        existing = {}
        for speaker_name in self._snippets.all_speakers(user_id=user_id):
            for sn in self._snippets.list_by_speaker(speaker_name, user_id=user_id):
                existing[sn["file_path"]] = sn

        audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".mkv", ".mp4", ".webm", ".opus"}
        added = 0
        scanned = 0
        new_speakers = set()
        changed_speakers = set()
        speakers_found = []

        total_files = 0
        for speaker_dir in sorted(user_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for audio_file in speaker_dir.iterdir():
                if audio_file.is_file() and audio_file.suffix.lower() in audio_exts:
                    total_files += 1

        for speaker_dir in sorted(user_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_name = speaker_dir.name
            speakers_found.append(speaker_name)

            for audio_file in sorted(speaker_dir.iterdir()):
                if not audio_file.is_file() or audio_file.suffix.lower() not in audio_exts:
                    continue
                scanned += 1
                _progress({
                    "stage": "Scanning voices directory",
                    "speaker": speaker_name,
                    "scanned": scanned,
                    "added": added,
                    "progress": (scanned / total_files) if total_files else 0.0,
                })

                fp = str(audio_file)
                sn = existing.get(fp)
                try:
                    fst = audio_file.stat()
                    mtime, size = fst.st_mtime, fst.st_size
                except OSError:
                    mtime, size = None, None

                if sn is None:
                    try:
                        waveform, sr = load_audio(fp)
                        duration = waveform.shape[-1] / SAMPLE_RATE
                        if duration < MIN_SNIPPET_DURATION:
                            continue

                        audio_data = waveform.numpy().squeeze()
                        sf.write(fp, audio_data.astype(np.float32), sr)
                        try:
                            fst = audio_file.stat()
                            mtime, size = fst.st_mtime, fst.st_size
                        except OSError:
                            mtime, size = None, None

                        self._snippets.add(
                            speaker_name=speaker_name,
                            file_path=fp,
                            duration_sec=duration,
                            user_id=user_id,
                            file_mtime=mtime,
                            file_size=size,
                        )
                        existing[fp] = {
                            "id": None,
                            "file_mtime": mtime,
                            "file_size": size,
                        }
                        added += 1
                        new_speakers.add(speaker_name)
                        _progress({
                            "stage": f"Added snippet for {speaker_name}",
                            "speaker": speaker_name,
                            "scanned": scanned,
                            "added": added,
                            "progress": (scanned / total_files) if total_files else 0.0,
                        })
                    except Exception as e:
                        logger.warning("Failed to scan %s: %s", audio_file, e)
                else:
                    stored_mtime = sn.get("file_mtime")
                    stored_size = sn.get("file_size")
                    if stored_mtime is None or stored_size is None:
                        if sn.get("id") is not None and (mtime is not None or size is not None):
                            self._snippets.update_fingerprint(
                                sn["id"], mtime, size, user_id=user_id
                            )
                    elif (stored_mtime != mtime or stored_size != size):
                        changed_speakers.add(speaker_name)

        return self._rebuild_after_rescan(
            user_id=user_id,
            speakers_found=speakers_found,
            new_speakers=new_speakers,
            changed_speakers=changed_speakers,
            scanned=scanned,
            added=added,
            progress_callback=_progress,
            force_rebuild=force,
        )

    def _rebuild_after_rescan(
        self,
        user_id: str,
        speakers_found: list,
        new_speakers: set,
        changed_speakers: set,
        scanned: int,
        added: int,
        progress_callback=None,
        force_rebuild: bool = False,
    ) -> dict:
        """Invalidate changed voiceprints and rebuild all missing ones.

        Speakers rebuilt = new dirs with snippets ∪ speakers whose snippet
        changed (voiceprint invalidated first) ∪ speakers that have snippets
        but no voiceprint row (missing), or all speakers when force_rebuild=True.
        """
        invalidated = 0
        rebuilt = 0

        if force_rebuild:
            targets = set(speakers_found)
        else:
            targets = set(new_speakers) | set(changed_speakers)
            for speaker_name in speakers_found:
                if speaker_name in targets:
                    continue
                if self._snippets.count(speaker_name, user_id=user_id) > 0 \
                        and self._db.get(speaker_name, user_id=user_id) is None:
                    targets.add(speaker_name)

        for speaker_name in sorted(targets):
            if speaker_name in changed_speakers:
                if self._db.delete(speaker_name, user_id=user_id):
                    invalidated += 1
                progress_callback and progress_callback({
                    "stage": f"Invalidated voiceprint for {speaker_name}",
                    "speaker": speaker_name,
                    "scanned": scanned,
                    "added": added,
                })
            if progress_callback:
                progress_callback({
                    "stage": f"Rebuilding voiceprint for {speaker_name}",
                    "speaker": speaker_name,
                    "scanned": scanned,
                    "added": added,
                })
            self._auto_refine(speaker_name, user_id=user_id, progress_callback=progress_callback)
            rebuilt += 1

        return {
            "scanned": scanned,
            "added": added,
            "invalidated": invalidated,
            "rebuilt": rebuilt,
            "speakers": speakers_found,
        }

    async def auto_collect_from_diarization_async(self, *args, **kwargs) -> dict:
        """Run :meth:`auto_collect_from_diarization` off the event loop.

        Lesson 19: an SSE producer must not do heavy work inline.  Auto-collect
        decodes audio, embeds every snippet and clusters vectors; all of it is
        blocking, and every caller lives in an ``async def`` request body.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.auto_collect_from_diarization,
                                    *args, **kwargs),
        )

    def auto_collect_from_diarization(
        self,
        audio_path: str,
        segments: list[dict],
        user_id: str = DEFAULT_USER,
        source_id: str | None = None,
        learn_new: bool = False,
        vad_sections: list | None = None,
        source: AudioSegmentSource | None = None,
    ) -> dict:
        """Grow voiceprints from a diarization result.

        Two distinct behaviours, deliberately separated:

        * **collect** -- add snippets to an ALREADY-NAMED voiceprint. Only ever
          touches profiles that are registered and not pending.
        * **learn** (``learn_new=True``) -- create a *pending* profile for a
          generic ``Speaker N`` cluster that spoke long enough to be a person
          rather than a cough. The profile collects snippets immediately but
          is excluded from matching (see ``_load_known_speakers``) until the
          user names it, so this can never invent an identity.

        ``segments`` should be the boundary set the user actually SEES — for
        ``/transcribe/upload`` and ``/attribution/upload`` that is the **turns**
        from ``asr_router._prepare_turns``, not the raw diarization segments
        (phase P2 of docs/plans/boundary-and-voice-quality-plan.md, finding F1).
        A snippet cut on the raw segments is cut at a boundary the transcript
        never shows, and since a learned embedding is later used as a reference
        for cutting boundaries, that compounds run over run.

        ``vad_sections`` (the diarizer's uncollapsed Silero pass) trims every
        snippet edge onto a speech-section edge; ``source`` is a request-wide
        cached loader so the file is decoded once instead of once per segment.

        Returns ``{"collected": [...], "pending_created": [...],
        "pending_extended": [...], "learn_skipped": [...]}``. (Earlier versions
        returned a bare list; the call sites only used ``len()``, and returning
        the breakdown is what lets the API tell the user a new profile is
        waiting for a name.)
        """
        if source_id is None:
            source_id = audio_path
        if source is None:
            source = AudioSegmentSource(audio_path)
        collected = []
        pending_created = []
        pending_extended = []
        learn_skipped: list[dict] = []
        speaker_totals = {}
        for name, info in self._snippets.all_speakers(user_id=user_id).items():
            speaker_totals[name] = info["total_duration"]

        by_speaker: dict[str, list[dict]] = {}
        to_learn: dict[str, list[dict]] = {}
        skipped_uncertain = 0
        for seg in segments:
            sp = seg.get("speaker", "")
            if not sp:
                continue
            dur = seg.get("end", 0) - seg.get("start", 0)
            # Uncertainty policy: never feed UNKNOWN / OVERLAP / generic
            # "Speaker N" / suppressed or low-confidence segments into the DB.
            if not eligible_for_auto_collect(seg):
                skipped_uncertain += 1
                # ...but a generic "Speaker N" that survived every other check
                # is a candidate to LEARN, not to collect.
                if learn_new and is_generic_speaker(sp) and eligible_for_learning(seg):
                    if dur >= AUTO_COLLECT_MIN_DURATION:
                        to_learn.setdefault(sp, []).append(seg)
                continue
            if dur < AUTO_COLLECT_MIN_DURATION:
                continue
            by_speaker.setdefault(sp, []).append(seg)
        if skipped_uncertain:
            logger.info(
                "Auto-collect: skipped %d segment(s) with uncertain speaker identity",
                skipped_uncertain,
            )

        for speaker_name, segs in by_speaker.items():
            # ONLY auto-collect for genuinely registered known speakers (e.g. "Gergely Papp", "Ismael Capel")
            # NEVER create auto-collected voiceprints for generic/unregistered "Speaker N" or timestamped clusters.
            if is_spurious_speaker_name(speaker_name):
                continue
            if not self._db.get(speaker_name, user_id=user_id):
                continue

            if len(segs) < AUTO_COLLECT_MIN_SPEAKER_SEGMENTS:
                continue
            current_total = speaker_totals.get(speaker_name, 0.0)
            if current_total >= AUTO_COLLECT_MAX_TOTAL_SEC:
                logger.info(
                    "Auto-collect: %s at snippet cap (%.0fs >= %.0fs), skipping",
                    speaker_name, current_total, AUTO_COLLECT_MAX_TOTAL_SEC,
                )
                continue

            for seg in segs:
                seg_start, seg_end = self._snippet_span(
                    seg, vad_sections,
                )

                # Per-chunk length is capped by chunk_end below (MAX_SEGMENT_SEC);
                # current_total is the speaker's CUMULATIVE total, so it must only
                # be gated by MAX_TOTAL_SEC — gating it on MAX_SEGMENT_SEC made
                # every speaker with >=300s of snippets silently un-collectable.
                while seg_start < seg_end and current_total < AUTO_COLLECT_MAX_TOTAL_SEC:
                    chunk_end = min(seg_start + AUTO_COLLECT_MAX_SEGMENT_SEC, seg_end)
                    dur = chunk_end - seg_start

                    existing = self._snippets.find_duplicate(source_id, seg_start, user_id=user_id)
                    if existing:
                        if existing["duration_sec"] >= dur:
                            seg_start = chunk_end
                            continue
                        old_path = Path(existing["file_path"])
                        if old_path.exists():
                            old_path.unlink()
                        self._snippets.delete(existing["id"], user_id=user_id)
                        current_total -= existing["duration_sec"]

                    result = self.add_snippet_from_segment(
                        speaker_name=speaker_name,
                        wav_path=audio_path,
                        start_sec=seg_start,
                        end_sec=chunk_end,
                        user_id=user_id,
                        source_audio=source_id,
                        audio_source=source,
                    )
                    if "error" not in result:
                        speaker_totals[speaker_name] = current_total + dur
                        current_total += dur
                        collected.append(result)

                    seg_start = chunk_end

            self._auto_refine(speaker_name, user_id=user_id)

        for cluster, segs in to_learn.items():
            parts, status = self._split_cluster_by_cohesion(
                cluster, segs, source,
            )
            if status.startswith("unverified"):
                if _learning_cfg().get("fail_closed", True):
                    # FAIL CLOSED.  A blend we could not verify must not be
                    # learned as a profile: the profile is a *reference for
                    # cutting future boundaries*, so learning an unverified
                    # blend does not just mislabel one run, it corrupts the
                    # next one (plan F1/F4).  ``learning.fail_closed: false``
                    # is the rollback to the previous fail-open behaviour.
                    logger.info(
                        "Not learning %s: cohesion check could not be verified "
                        "(%s). Nothing is learned rather than learning a blend.",
                        cluster, status,
                    )
                    learn_skipped.append({"cluster": cluster, "reason": status})
                    continue
                # Rollback arm: the pre-P2 behaviour, learn the whole cluster.
                parts = [{"label": cluster, "segs": segs, "vectors": [],
                          "intra_max": None, "inter_min": None}]
            for part in parts:
                outcome = self._learn_unknown_speaker(
                    cluster=part["label"], segs=part["segs"],
                    audio_path=audio_path, user_id=user_id,
                    source_id=source_id, source=source,
                    vad_sections=vad_sections,
                    vectors=part.get("vectors") or None,
                    intra_max=part.get("intra_max"),
                    inter_min=part.get("inter_min"),
                    groups=len(parts),
                )
                if outcome.get("skipped"):
                    learn_skipped.append({"cluster": cluster,
                                          "reason": outcome["skipped"]})
                    continue
                if outcome.get("name"):
                    (pending_extended if outcome.get("existing")
                     else pending_created).append(outcome)

        return {
            "collected": collected,
            "pending_created": pending_created,
            "pending_extended": pending_extended,
            "learn_skipped": learn_skipped,
            "audio_loads": source.loads,
        }

    # -- learning internals ---------------------------------------------------

    def _snippet_span(self, seg: dict, vad_sections) -> tuple[float, float]:
        """The span a snippet is actually cut from, trimmed onto VAD edges."""
        start, end = float(seg["start"]), float(seg["end"])
        if not vad_sections:
            return start, end
        return trim_span_to_vad(start, end, vad_sections)

    def _embed_span(self, source: AudioSegmentSource, seg: dict):
        """One segment embedding, or ``None`` when it is unusable."""
        try:
            # The cached source already resamples to SAMPLE_RATE and returns a
            # (1, n) shaped waveform, so the sample count is shape[-1], never
            # len().  One decode for the whole request, not one per segment.
            audio, sr = source.segment(seg["start"], seg["end"])
            if audio is None or audio.shape[-1] < int(0.5 * SAMPLE_RATE):
                return None
            vec = np.asarray(
                extract_embedding(audio, sr, self._emb_session()),
                dtype=np.float32,
            ).reshape(-1)
            if not np.all(np.isfinite(vec)):
                return None
            return vec
        except Exception as e:
            logger.debug("Cohesion embed failed for %s: %s", seg, e)
            return None

    def _split_cluster_by_cohesion(
        self, cluster: str, segs: list[dict], source: AudioSegmentSource,
    ) -> tuple[list[dict], str]:
        """Split one diarization cluster into cohesive groups before learning it.

        Returns ``(parts, status)``.  ``status`` is ``"ok"``, ``"opted_out"``
        (``learning.split_clusters`` is false — an explicit operator decision,
        so the whole cluster is one part) or ``"unverified:<why>"``.

        Note the granularity the caller now passes: the TURNS, not the raw
        diarization segments (phase P2).  Turns are coarser — ``_merge_into_turns``
        folds same-speaker segments across gaps of up to
        ``uncertainty.turn_merge_gap_sec`` — so a cluster whose segments all sit
        within that gap of each other arrives as ONE span, and one span cannot
        be checked for being a blend.  That case is reported as
        ``unverified:too_few_segments`` and nothing is learned.  It is the
        correct outcome rather than a gap: a single whole-file span has exactly
        the property step 14 exists to prevent (one long snippet dominating a
        profile), and a recording whose every second belongs to one unknown
        speaker is the case where the user should register a voiceprint
        deliberately.

        A cluster is NOT a person.  On a real 2-host podcast with inserted clips
        from several other speakers (ZO249, 3439s) the cluster the pipeline
        called ``Speaker 6`` was a blend of two voices: three of its spans
        ranked one registered colleague nearest (cosine 0.533-0.589) and three
        others ranked the host nearest (0.386-0.531), with the two groups
        0.368-0.448 apart while each group sat at 0.175-0.291 internally.
        Learning that as ONE profile would have asserted a single name for two
        people — and worse, the blended profile is then used by
        ``_refine_boundaries_with_vad`` and the second pass, so it corrupts the
        turn boundaries of every later run, not just the printed label.

        So each contributing segment is embedded on its own and the cluster is
        re-clustered at ``learning.max_intra_cluster_dist``.  Every cohesive
        group becomes its own pending profile, and the user is told how many
        voices the cluster contained.

        **Fail closed (phase P2).**  Every path that previously returned the
        whole cluster — no embedding session, no sklearn, fewer than two usable
        vectors, a clustering exception, or fewer than two segments — now
        returns an empty part list plus an ``unverified:`` status, and the
        caller learns nothing.  A blend within the threshold is undetectable by
        construction, so "cannot tell" must never become "learn it anyway".

        Threshold provenance: 0.32 sits between the measured within-group
        maximum (0.291) and the between-group minimum (0.368) on that file.  It
        is one file with six sampled spans, so it is a judgement call, not a
        calibrated value — hence ``learning.split_clusters`` exists to turn the
        whole thing off, and ``learning.fail_closed`` to restore the old
        fail-open behaviour without turning the split off.
        """
        cfg = _learning_cfg()
        if not cfg.get("split_clusters", True):
            return ([{"label": cluster, "segs": segs, "vectors": [],
                      "intra_max": None, "inter_min": None}], "opted_out")
        if self._emb_session() is None:
            return [], "unverified:no_embedding_session"
        if len(segs) < 2:
            return [], "unverified:too_few_segments"
        try:
            from sklearn.cluster import AgglomerativeClustering
        except Exception:
            return [], "unverified:no_sklearn"

        vectors, kept = [], []
        for seg in sorted(segs, key=lambda s: s["start"]):
            vec = self._embed_span(source, seg)
            if vec is None:
                continue
            vectors.append(vec)
            kept.append(seg)

        if len(vectors) < 2:
            return [], "unverified:too_few_embeddings"

        try:
            threshold = float(cfg.get("max_intra_cluster_dist", 0.32))
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold,
                metric="cosine", linkage="average",
            ).fit_predict(np.stack(vectors))
        except Exception as e:
            logger.warning("Cohesion split failed for %s: %s", cluster, e)
            return [], "unverified:clustering_failed"

        groups: dict[int, list[dict]] = {}
        vecs: dict[int, list[np.ndarray]] = {}
        for label, seg, vec in zip(labels, kept, vectors):
            groups.setdefault(int(label), []).append(seg)
            vecs.setdefault(int(label), []).append(vec)

        ordered = sorted(
            groups.items(),
            key=lambda kv: -sum(s["end"] - s["start"] for s in kv[1]),
        )
        parts = []
        for i, (key, group) in enumerate(ordered):
            parts.append({
                "label": cluster if i == 0 else "%s#%d" % (cluster, i + 1),
                "segs": group,
                "vectors": vecs.get(key, []),
                "intra_max": _max_pairwise_distance(vecs.get(key)),
                "inter_min": None,
            })
        centroids = [_centroid(p["vectors"]) for p in parts]
        if len(parts) > 1:
            min_inter = float(cfg.get("min_inter_dist", 0.32))
            for i, p in enumerate(parts):
                others = [centroids[j] for j in range(len(parts)) if j != i]
                mine = centroids[i]
                if mine is None or any(o is None for o in others):
                    p["inter_min"] = None
                    continue
                p["inter_min"] = min(_cosine_distance(mine, o) for o in others)
            worst = max(
                (p["inter_min"] for p in parts if p["inter_min"] is not None),
                default=None,
            )
            if worst is not None and worst < min_inter:
                # The groups we just made sit CLOSER to each other than the bar
                # that produced them: the split is an artefact, not evidence of
                # two people, and learning it would invent a division.
                logger.info(
                    "Cohesion split of %s not trusted: groups only %.3f apart "
                    "(min_inter_dist=%.2f) -- not learning this cluster",
                    cluster, worst, min_inter,
                )
                return [], "unverified:groups_too_close"
        if len(parts) > 1:
            logger.info(
                "Cluster %s contains more than one voice: split into %d cohesive "
                "group(s) at max_intra_cluster_dist=%.2f (sizes %s, intra_max %s, "
                "inter_min %s) -- each is learned as its own pending profile",
                cluster, len(parts), threshold,
                [len(p["segs"]) for p in parts],
                [None if p["intra_max"] is None else round(p["intra_max"], 3)
                 for p in parts],
                [None if p["inter_min"] is None else round(p["inter_min"], 3)
                 for p in parts],
            )
        return parts, "ok"

    def _purity(self, intra_max, inter_min, groups: int) -> dict:
        ratio = None
        if intra_max is not None and inter_min:
            ratio = round(float(intra_max) / float(inter_min), 4)
        return {
            "intra_max": None if intra_max is None else round(float(intra_max), 4),
            "inter_min": None if inter_min is None else round(float(inter_min), 4),
            "purity_ratio": ratio,
            "groups": int(groups or 1),
        }

    def _find_pending_to_extend(
        self, centroid, user_id: str, source_id: str, cluster: str,
    ) -> tuple[str | None, float | None]:
        """Which pending profile, if any, is this cluster the SAME person as?

        The primary key is the **voice**, not the source filename (phase P2,
        finding F4).  Keying on the filename stem meant the same colleague in a
        second recording minted a SECOND pending profile, and the second
        ``confirm`` was then refused because the name was taken -- leaving the
        user with no way forward except the merge dialog.

        One deliberate exception: a pending profile learned from **this same
        recording** under a *different* cluster label is not extended by voice
        alone.  The clustering threshold already decided those are two speakers
        in this file, and overriding that decision on embedding similarity is
        how two people end up merged.  (It also keeps the "two clusters, one
        recording, two profiles" behaviour that
        ``test_the_same_recording_extends_one_pending_profile`` pins.)
        """
        if centroid is None:
            return None, None
        rows = self._db.list_all(user_id=user_id, include_pending=True)
        prefix = _origin_prefix(source_id)
        label_tokens = _name_tokens(cluster)
        thr = float(_learning_cfg().get("extend_max_dist", 0.32))
        ranked = []
        for name, vp in sorted(rows.items()):
            if not is_pending_profile(name, vp):
                continue
            emb = vp.get("embedding")
            if emb is None or len(emb) == 0:
                continue
            ranked.append((_cosine_distance(centroid, emb), name))
        ranked.sort()
        for dist, name in ranked:
            if dist > thr:
                break
            same_source = prefix and prefix in name
            if same_source and label_tokens and not _tokens_contain(
                    _name_tokens(name), label_tokens):
                continue
            logger.info(
                "Pending profile %r is %.3f from cluster %r (extend_max_dist=%.2f)"
                " -- extending it rather than minting a new one",
                name, dist, cluster, thr,
            )
            return name, dist
        return None, (ranked[0][0] if ranked else None)

    def _learn_unknown_speaker(
        self,
        cluster: str,
        segs: list[dict],
        audio_path: str,
        user_id: str = DEFAULT_USER,
        source_id: str | None = None,
        source: AudioSegmentSource | None = None,
        vad_sections: list | None = None,
        vectors: list | None = None,
        intra_max: float | None = None,
        inter_min: float | None = None,
        groups: int = 1,
    ) -> dict:
        """Create or extend a PENDING profile for an unidentified cluster.

        Extends an existing pending profile when one already covers this person:
        by embedding distance first (so a colleague in a *second recording*
        accumulates into the first profile), and by the source-filename/label
        name match as the fallback.
        """
        if source_id is None:
            source_id = audio_path
        speech = sum(max(0.0, seg["end"] - seg["start"]) for seg in segs)
        if len(segs) < LEARN_MIN_SEGMENTS or speech < LEARN_MIN_SPEECH_SEC:
            return {}
        cfg = _learning_cfg()
        # FAIL CLOSED, per group: a group whose own segments disagree more than
        # ``max_profile_intra_dist`` is not one person, so nothing is learned
        # from it even though its siblings passed.
        max_intra = float(cfg.get("max_profile_intra_dist", 0.32))
        if intra_max is not None and intra_max > max_intra:
            logger.info(
                "Not learning %r: its own %d span(s) are up to %.3f apart "
                "(max_profile_intra_dist=%.2f) -- that is not one person",
                cluster, len(segs), intra_max, max_intra,
            )
            return {"skipped": "impure_group"}

        purity = self._purity(intra_max, inter_min, groups)
        # Exact TOKEN match on the sanitised cluster label: "Speaker 1" must not
        # match "Speaker 10" (a plain substring test did exactly that).
        label_tokens = _name_tokens(re.sub(r"[^A-Za-z0-9._-]+", "_", cluster or "").strip("_"))

        rows = self._db.list_all(user_id=user_id, include_pending=True)
        known = set(rows) | set(self._snippets.all_speakers(user_id=user_id))
        prefix = _origin_prefix(source_id)

        target, dist = (None, None)
        if vectors:
            target, dist = self._find_pending_to_extend(
                _centroid(vectors), user_id, source_id, cluster,
            )
        if not target:
            for cand in sorted(known):
                if not is_pending_profile(cand, rows.get(cand)):
                    continue
                if prefix in cand and (not label_tokens or _tokens_contain(
                        _name_tokens(cand), label_tokens)):
                    target = cand
                    break
        existing = bool(target)
        if not target:
            target = pending_profile_name(source_id, cluster)
            n = 2
            # Snippet DIRECTORIES count as taken too, not just the DB row: a
            # profile learned while the embedding session was unavailable has
            # snippets and no row, and the old check minted "... #2" over it.
            while target in known or (self._voices_dir / user_id / target).exists():
                target = f"{pending_profile_name(source_id, cluster)} #{n}"
                n += 1
        added = []
        # Extending must be IDEMPOTENT: the same recording re-processed must not
        # add the same snippet twice, or the profile fills with duplicates and
        # the embedding is computed over doubly-counted audio.
        have = {os.path.basename(x["file_path"])
                for x in self._snippets.list_by_speaker(target, user_id=user_id)}
        src_hash = generate_segment_hash(audio_path)
        # Per-snippet and per-profile caps on the LEARN path.  The collect-only
        # AUTO_COLLECT_MAX_* pair did not apply here, so one 200s turn could
        # dominate a profile of 5s snippets.
        max_snippet = float(cfg.get("max_snippet_sec") or 0.0)
        max_total = float(cfg.get("max_profile_total_sec") or 0.0)
        current_total = sum(
            float(x.get("duration_sec") or 0.0)
            for x in self._snippets.list_by_speaker(target, user_id=user_id)
        )
        if max_total and current_total >= max_total:
            logger.info(
                "Pending profile %r is at its learn cap (%.0fs >= %.0fs)",
                target, current_total, max_total,
            )
        for seg in sorted(segs, key=lambda x: x["start"]):
            seg_start, seg_end = self._snippet_span(seg, vad_sections)
            # Mirror add_snippet()'s filename exactly (minus the duration part):
            # the hash is of the source audio and the time part of the start.
            key = f"{src_hash}_{format_time_short(seg['start'])}"
            if any(n.startswith(key + "_") for n in have):
                continue
            while seg_start < seg_end and not (max_total and current_total >= max_total):
                chunk_end = seg_end if not max_snippet else min(
                    seg_start + max_snippet, seg_end,
                )
                if max_total:
                    chunk_end = min(chunk_end, seg_start + (max_total - current_total))
                have.add(key)
                r = self.add_snippet_from_segment(
                    speaker_name=target, wav_path=audio_path,
                    start_sec=seg_start, end_sec=chunk_end, user_id=user_id,
                    source_audio=source_id, audio_source=source,
                )
                if "error" not in r:
                    added.append(r)
                    current_total += (chunk_end - seg_start)
                seg_start = chunk_end
        if not added:
            # Nothing new (a repeat of the same recording). Still report the
            # profile: the user needs to know this speaker is STILL unnamed,
            # and silently returning nothing hides a pending profile that is
            # accumulating nothing. `snippets: 0` keeps it honest.
            if not existing:
                return {}
            return {"name": target, "cluster": cluster, "snippets": 0,
                    "speech_sec": round(speech, 1), "existing": True,
                    "purity": purity}
        # Build the profile so it is ready the moment the user names it. This
        # writes a voiceprint row flagged pending, which matching ignores.
        try:
            self._auto_refine(target, user_id=user_id, purity=purity)
        except Exception as e:
            logger.warning("Pending profile refine failed for %s: %s", target, e)
        # MUST be set AFTER the refine: _auto_refine CREATES the row, and a new
        # row defaults to pending=False. Without this the profile would be
        # matchable from its very first moment -- the exact false attribution
        # this whole feature exists to avoid.
        self._db.set_pending(target, True, user_id=user_id)
        logger.info(
            "Learned new speaker: %s (%s, %d snippet(s), %.1fs, purity %s) -- "
            "PENDING, excluded from matching until it is named",
            target, cluster, len(added), speech, purity,
        )
        return {"name": target, "cluster": cluster, "snippets": len(added),
                "speech_sec": round(speech, 1), "existing": existing,
                "purity": purity, "extend_distance": dist}

    def _rename_snippet_dir(self, old_name: str, new_name: str,
                            user_id: str = DEFAULT_USER) -> bool:
        """Move a speaker's snippets + DB rows to a new name."""
        old_dir = self._voices_dir / user_id / old_name
        new_dir = self._voices_dir / user_id / new_name
        if old_dir.exists() and not new_dir.exists():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            old_dir.rename(new_dir)
        moved = 0
        for sn in self._snippets.list_by_speaker(old_name, user_id=user_id):
            try:
                # Re-point at the NEW directory. Copying file_path verbatim is
                # the obvious thing and it is wrong: the directory just moved,
                # so every later load of that snippet 404s on disk.
                old_path = sn["file_path"]
                new_path = str(new_dir / os.path.basename(old_path)) \
                    if str(old_dir) in old_path else old_path
                self._snippets.add(
                    speaker_name=new_name, file_path=new_path,
                    duration_sec=sn.get("duration_sec", 0.0), user_id=user_id,
                    source_audio=sn.get("source_audio"),
                    start_sec=sn.get("start_sec"), end_sec=sn.get("end_sec"),
                    file_mtime=sn.get("file_mtime"), file_size=sn.get("file_size"))
                self._snippets.delete(sn["id"], user_id=user_id)
                moved += 1
            except Exception as e:
                logger.warning("Could not move snippet %s: %s", sn.get("id"), e)
        return bool(moved or new_dir.exists())

    def pending_profiles(self, user_id: str = DEFAULT_USER) -> list[dict]:
        """Unnamed profiles awaiting a name, newest first."""
        out = []
        vps = self._db.list_all(user_id=user_id, include_pending=True)
        totals = self._snippets.all_speakers(user_id=user_id)
        # Union of rows and snippet dirs: a profile learned while embedding was
        # unavailable has snippets but no row.
        for name in sorted(set(vps) | set(totals)):
            vp = vps.get(name, {})
            if not is_pending_profile(name, vp):
                continue
            sn = totals.get(name, {})
            out.append({
                "name": name,
                "snippet_count": sn.get("count", 0),
                "total_duration_sec": round(sn.get("total_duration", 0.0), 2),
                "created_at": vp.get("created_at"),
                "purity": vp.get("purity") or None,
            })
        return out

    def pending_candidates(self, pending_name: str, user_id: str = DEFAULT_USER,
                           limit: int = 5) -> list[dict]:
        """Which registered speakers does this pending profile resemble?

        A pending profile is very often a person who is ALREADY registered --
        the user themselves picked up by the loopback, or a colleague they
        registered months ago. ``confirm_pending`` rightly refuses a name that
        is already taken, but the correct action there is a merge, not a
        rename, and the user has no way to reach one: the merge dialog only
        lists named speakers.

        Returns every registered profile ranked by combined distance, with the
        live-turn gates applied as advisory flags rather than a filter -- the
        caller decides what to show.  ``likely`` marks the candidates that
        clear ``live_attribution.min_match_confidence`` /
        ``min_match_margin``; a pending profile scored against a *profile* is
        a cleaner comparison than a 3s live turn is, so those bars are
        conservative.

        Returns ``[]`` when the profile has no embedding yet (learned while the
        embedding session was unavailable); there is nothing to compare.
        """
        from asr_mcp.streaming.attribution import config as live_cfg
        from asr_mcp.speaker.matcher import find_best_match

        vp = self._db.get(pending_name, user_id=user_id)
        emb = (vp or {}).get("embedding")
        if emb is None or len(emb) == 0:
            return []
        # Registered profiles only: a pending profile has no identity to lend.
        registered = self._db.list_all(user_id=user_id, include_pending=False)
        if not registered:
            return []
        _, _, _, distances = find_best_match(
            emb, (vp or {}).get("pitch_hz", 0.0), (vp or {}).get("energy_rms", 0.0),
            registered,
        )
        ranked = sorted(distances.items(), key=lambda kv: kv[1]["combined"])
        cfg = live_cfg()
        min_conf = float(cfg["min_match_confidence"])
        min_margin = float(cfg["min_match_margin"])
        out = []
        for i, (name, d) in enumerate(ranked[:max(1, limit)]):
            runner_up = ranked[i + 1][1]["combined"] if i + 1 < len(ranked) else 1.0
            margin = runner_up - d["combined"]
            out.append({
                "name": name,
                "distance": round(d["combined"], 4),
                "confidence": round(d.get("confidence", 0.0), 3),
                "margin": round(margin, 4),
                "snippet_count": self._snippets.all_speakers(
                    user_id=user_id).get(name, {}).get("count", 0),
                "likely": bool(d.get("confidence", 0.0) >= min_conf and margin >= min_margin),
            })
        return out

    def merge_pending_into(self, pending_name: str, target_name: str,
                           user_id: str = DEFAULT_USER) -> dict:
        """Fold a pending profile into an EXISTING speaker.

        The pending profile's snippets become the target's, the pending profile
        ceases to exist, and the target is re-refined over the larger corpus.
        This is the correct resolution when the learner turns out to be someone
        already registered: naming them would create a second profile for one
        person, and the uncertainty policy would then split their speech
        between two names forever.
        """
        pending_name = (pending_name or "").strip()
        target_name = (target_name or "").strip()
        if not pending_name or not target_name:
            return {"error": "Both a pending profile and a target speaker are required"}
        if pending_name == target_name:
            return {"error": "A pending profile cannot be merged into itself"}
        vp = self._db.get(pending_name, user_id=user_id)
        if not vp and pending_name not in self._snippets.all_speakers(user_id=user_id):
            return {"error": f"No pending profile named {pending_name!r}"}
        if not is_pending_profile(pending_name, vp):
            return {"error": f"{pending_name!r} is not a pending profile"}
        if not self._db.get(target_name, user_id=user_id) and \
                target_name not in self._snippets.all_speakers(user_id=user_id):
            return {"error": f"No speaker named {target_name!r} to merge into"}
        result = self.merge_speakers(target_name, pending_name, user_id=user_id)
        if result.get("error"):
            return result
        logger.info("Pending profile %r merged into existing speaker %r (%d snippets)",
                    pending_name, target_name, result.get("snippets_moved", 0))
        return {**result, "status": "merged_into_existing",
                "from": pending_name, "into": target_name}

    def confirm_pending(self, pending_name: str, new_name: str,
                        user_id: str = DEFAULT_USER) -> dict:
        """Give a pending profile a real name, making it matchable.

        Refuses a name that is already taken (unless it is the same pending
        profile being renamed) so confirming can never silently overwrite a
        colleague's existing profile.
        """
        new_name = (new_name or "").strip()
        if not new_name:
            return {"error": "A name is required"}
        vp = self._db.get(pending_name, user_id=user_id)
        sn = self._snippets.all_speakers(user_id=user_id).get(pending_name)
        if not vp and not sn:
            return {"error": f"No pending profile named {pending_name!r}"}
        if not is_pending_profile(pending_name, vp):
            return {"error": f"{pending_name!r} is not a pending profile"}
        if new_name != pending_name and (
                self._db.get(new_name, user_id=user_id)
                or new_name in self._snippets.all_speakers(user_id=user_id)):
            return {"error": f"A voiceprint named {new_name!r} already exists"}
        # The snippets are what make the profile usable, so they move whenever
        # the name changes -- whether or not a voiceprint row existed to rename.
        self._rename_snippet_dir(pending_name, new_name, user_id=user_id)
        if not self._db.rename(pending_name, new_name, user_id=user_id):
            if not (self._db.get(new_name, user_id=user_id)
                    or new_name in self._snippets.all_speakers(user_id=user_id)):
                return {"error": f"Could not rename {pending_name!r}"}
        self._db.set_pending(new_name, False, user_id=user_id)
        self._auto_refine(new_name, user_id=user_id)
        logger.info("Pending profile %r confirmed as %r (now matchable)",
                    pending_name, new_name)
        return {"ok": True, "name": new_name, "renamed_from": pending_name}

    def _auto_refine(self, speaker_name: str, user_id: str = DEFAULT_USER,
                     progress_callback=None, purity: Optional[dict] = None):
        if self._emb_session() is None:
            logger.warning("No embedding session, skipping auto-refine for %s", speaker_name)
            return

        snippet_files = self._snippets.list_by_speaker(speaker_name, user_id=user_id)
        if not snippet_files:
            return

        waveforms = []
        durations = []
        for sn in snippet_files:
            try:
                waveform, sr = load_audio(sn["file_path"])
                dur = waveform.shape[-1] / SAMPLE_RATE
                if dur >= MIN_SNIPPET_DURATION:
                    waveforms.append(waveform)
                    durations.append(dur)
            except Exception as e:
                logger.warning("Failed to load snippet %s: %s", sn["file_path"], e)

        if not waveforms:
            return

        embeddings = []
        for i, waveform in enumerate(waveforms):
            try:
                emb = extract_embedding(waveform, SAMPLE_RATE, self._emb_session())
                embeddings.append(emb)
            except Exception as e:
                logger.warning("Failed to embed snippet: %s", e)
                embeddings.append(None)
            if progress_callback:
                progress_callback({
                    "stage": f"Embedding snippet {i + 1}/{len(waveforms)} for {speaker_name}",
                    "speaker": speaker_name,
                })

        valid = [(e, d) for e, d in zip(embeddings, durations) if e is not None]
        if not valid:
            return

        embeddings_arr = [e for e, _ in valid]
        durations_arr = [d for _, d in valid]
        total_dur = sum(durations_arr)

        # Duration weighting, with the length of any ONE snippet capped.
        # Uncapped, one 200s snippet in a profile of 5s snippets carried 95% of
        # the weight and the "profile" became that one recording.  The cap is
        # relative to the profile's own MEDIAN snippet, so it is a no-op for a
        # profile whose snippets are of comparable length: with a ratio of R a
        # single outlier can never weigh more than R/(R + n - 1) <= R/R = 1.0,
        # and for R = 4 with two or more snippets, never more than 0.5.
        weights = np.array(durations_arr, dtype=np.float64)
        ratio = float(_learning_cfg().get("max_refine_weight_ratio") or 0.0)
        if ratio > 0 and weights.size > 1:
            median = float(np.median(weights))
            if median > 0:
                weights = np.minimum(weights, ratio * median)
        weights = weights / weights.sum() if weights.sum() > 0 else \
            np.full(weights.shape, 1.0 / max(1, weights.size))
        new_embedding = np.zeros_like(np.asarray(embeddings_arr[0]))
        for emb, w in zip(embeddings_arr, weights):
            new_embedding += emb * w

        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            new_embedding = new_embedding / norm

        combined = torch.cat(waveforms, dim=-1)
        pitch_hz, pitch_std = compute_pitch(combined, SAMPLE_RATE)
        energy_rms = compute_energy(combined)

        # pending is left as None here on purpose: VoiceprintDB.save preserves
        # the stored flag, so an auto-refine cannot promote a pending profile.
        # purity is None for every caller but the learner, and save() preserves
        # it the same way -- a plain refine must not erase the measurement taken
        # when the profile was learned.
        self._db.save(
            name=speaker_name, user_id=user_id,
            embedding=new_embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            spectral_centroid=0.0,
            spectral_rolloff=0.0,
            total_speech_sec=total_dur, sample_count=len(valid),
            mfcc=None,
            purity=purity,
        )
        logger.info("Auto-refined voiceprint for %s (user=%s): %.1fs, %d snippets",
                     speaker_name, user_id, total_dur, len(valid))

    def _cleanup_speaker_dir(self, speaker_name: str, user_id: str = DEFAULT_USER):
        speaker_dir = self._voices_dir / user_id / speaker_name
        if speaker_dir.exists():
            remaining = list(speaker_dir.iterdir())
            if not remaining:
                speaker_dir.rmdir()

    # ── Legacy API (backward compat) ─────────────────────────────────

    def register_from_segments(self, name, wav_path, segments, user_id=DEFAULT_USER, min_duration=1.5):
        all_audio = []
        total_duration = 0.0
        for seg in segments:
            dur = seg.get("end", 0) - seg.get("start", 0)
            if dur < min_duration:
                continue
            chunk, _ = load_audio_segment(wav_path, seg["start"], seg["end"])
            all_audio.append(chunk)
            total_duration += dur

        if not all_audio:
            return {"error": "No valid segments"}

        combined = torch.cat(all_audio, dim=-1)
        embedding = extract_embedding(combined, SAMPLE_RATE, self._emb_session())
        pitch_hz, pitch_std = compute_pitch(combined, SAMPLE_RATE)
        energy_rms = compute_energy(combined)

        self._db.save(
            name=name, user_id=user_id, embedding=embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            total_speech_sec=total_duration, sample_count=len(all_audio),
        )
        return {"name": name, "total_speech_sec": round(total_duration, 2), "sample_count": len(all_audio)}

    def register_from_audio(self, name, wav_path, start_sec, end_sec, user_id=DEFAULT_USER):
        waveform, sr = load_audio_segment(wav_path, start_sec, end_sec)
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._emb_session())
        pitch_hz, pitch_std = compute_pitch(waveform, SAMPLE_RATE)
        energy_rms = compute_energy(waveform)
        total_duration = end_sec - start_sec

        self._db.save(
            name=name, user_id=user_id, embedding=embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            total_speech_sec=total_duration, sample_count=1,
        )
        return {"name": name, "total_speech_sec": round(total_duration, 2), "sample_count": 1}

    def identify_in_audio(self, wav_path, start_sec=0.0, end_sec=None, top_k=5, user_id=DEFAULT_USER):
        waveform, sr = load_audio_segment(wav_path, start_sec, end_sec or 99999)
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._emb_session())
        return self._db.search(embedding, user_id=user_id, top_k=top_k)

    def get_voiceprint(self, name, user_id=DEFAULT_USER):
        return self._db.get(name, user_id=user_id)

    def delete_voiceprint(self, name, user_id=DEFAULT_USER):
        return self._db.delete(name, user_id=user_id)
