import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from asr_mcp.db.manager import DatabaseManager, SnippetDB, VoiceprintDB, DEFAULT_USER
from asr_mcp.speaker.embedding import (
    extract_embedding, batch_embed_files, compute_pitch, compute_energy,
)
from asr_mcp.voiceprint.utils import load_audio, load_audio_segment

logger = logging.getLogger("asr_mcp.voiceprint.service")

SAMPLE_RATE = 16000
AUTO_COLLECT_MIN_DURATION = 1.5
AUTO_COLLECT_MAX_TOTAL_SEC = 600.0
MIN_SNIPPET_DURATION = 1.5


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

        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()[:12]
        filename = f"{int(time.time())}_{audio_hash}.flac"
        file_path = speaker_dir / filename

        sf.write(str(file_path), audio_data.astype(np.float32), sample_rate, format="FLAC")

        snippet_id = self._snippets.add(
            speaker_name=speaker_name,
            file_path=str(file_path),
            duration_sec=duration,
            user_id=user_id,
            source_audio=source_audio,
            start_sec=start_sec,
            end_sec=end_sec,
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
    ) -> dict:
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
                "pitch_hz": round(vp.get("pitch_hz", 0), 1),
                "energy_rms": round(vp.get("energy_rms", 0), 4),
            })
        return speakers

    def list_snippets(self, speaker_name: str, user_id: str = DEFAULT_USER) -> list[dict]:
        return self._snippets.list_by_speaker(speaker_name, user_id=user_id)

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

    def rename_speaker(self, old_name: str, new_name: str, user_id: str = DEFAULT_USER) -> dict:
        existing_vp = self._db.get(new_name, user_id=user_id)
        if existing_vp:
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

        return {
            "status": "renamed",
            "from": old_name,
            "to": new_name,
            "snippets_moved": count,
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

    def rescan_voices_dir(self, user_id: str = DEFAULT_USER) -> dict:
        user_dir = self._voices_dir / user_id
        if not user_dir.exists():
            return {"scanned": 0, "added": 0, "speakers": []}

        existing_paths = set()
        for sn in self._snippets.all_speakers(user_id=user_id).values():
            pass
        all_snippets = []
        for speaker_name in self._snippets.all_speakers(user_id=user_id):
            for sn in self._snippets.list_by_speaker(speaker_name, user_id=user_id):
                existing_paths.add(sn["file_path"])

            audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        added = 0
        scanned = 0
        speakers_found = []

        for speaker_dir in sorted(user_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_name = speaker_dir.name
            speakers_found.append(speaker_name)

            for audio_file in sorted(speaker_dir.iterdir()):
                if not audio_file.is_file() or audio_file.suffix.lower() not in audio_exts:
                    continue
                scanned += 1
                if str(audio_file) in existing_paths:
                    continue

                try:
                    waveform, sr = load_audio(str(audio_file))
                    duration = waveform.shape[-1] / SAMPLE_RATE
                    if duration < MIN_SNIPPET_DURATION:
                        continue

                    audio_data = waveform.numpy().squeeze()
                    sf.write(str(audio_file), audio_data.astype(np.float32), sr)

                    self._snippets.add(
                        speaker_name=speaker_name,
                        file_path=str(audio_file),
                        duration_sec=duration,
                        user_id=user_id,
                    )
                    existing_paths.add(str(audio_file))
                    added += 1
                except Exception as e:
                    logger.warning("Failed to scan %s: %s", audio_file, e)

        for speaker_name in speakers_found:
            sn_count = self._snippets.count(speaker_name, user_id=user_id)
            if sn_count > 0:
                self._auto_refine(speaker_name, user_id=user_id)

        return {
            "scanned": scanned,
            "added": added,
            "speakers": speakers_found,
        }

    def auto_collect_from_diarization(
        self,
        audio_path: str,
        segments: list[dict],
        user_id: str = DEFAULT_USER,
    ) -> list[dict]:
        collected = []
        speaker_totals = {}
        for name, info in self._snippets.all_speakers(user_id=user_id).items():
            speaker_totals[name] = info["total_duration"]

        by_speaker: dict[str, list[dict]] = {}
        for seg in segments:
            sp = seg.get("speaker", "")
            if not sp:
                continue
            dur = seg.get("end", 0) - seg.get("start", 0)
            if dur < AUTO_COLLECT_MIN_DURATION:
                continue
            by_speaker.setdefault(sp, []).append(seg)

        for speaker_name, segs in by_speaker.items():
            current_total = speaker_totals.get(speaker_name, 0.0)
            if current_total >= AUTO_COLLECT_MAX_TOTAL_SEC:
                continue

            for seg in segs:
                current_total = speaker_totals.get(speaker_name, 0.0)
                if current_total >= AUTO_COLLECT_MAX_TOTAL_SEC:
                    break

                dur = seg["end"] - seg["start"]
                result = self.add_snippet_from_segment(
                    speaker_name=speaker_name,
                    wav_path=audio_path,
                    start_sec=seg["start"],
                    end_sec=seg["end"],
                    user_id=user_id,
                    source_audio=audio_path,
                )
                if "error" not in result:
                    speaker_totals[speaker_name] = current_total + dur
                    collected.append(result)

        for speaker_name in by_speaker:
            self._auto_refine(speaker_name, user_id=user_id)

        return collected

    def _auto_refine(self, speaker_name: str, user_id: str = DEFAULT_USER):
        if self._embedding_session is None:
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
        for waveform in waveforms:
            try:
                emb = extract_embedding(waveform, SAMPLE_RATE, self._embedding_session)
                embeddings.append(emb)
            except Exception as e:
                logger.warning("Failed to embed snippet: %s", e)
                embeddings.append(None)

        valid = [(e, d) for e, d in zip(embeddings, durations) if e is not None]
        if not valid:
            return

        embeddings_arr = [e for e, _ in valid]
        durations_arr = [d for _, d in valid]
        total_dur = sum(durations_arr)

        weights = np.array(durations_arr) / total_dur
        new_embedding = np.zeros_like(embeddings_arr[0])
        for emb, w in zip(embeddings_arr, weights):
            new_embedding += emb * w

        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            new_embedding = new_embedding / norm

        existing = self._db.get(speaker_name, user_id=user_id)
        if existing and existing.get("embedding") is not None:
            old_emb = existing["embedding"]
            old_dur = existing.get("total_speech_sec", 0)
            if old_dur + total_dur > 0:
                blend_weight = old_dur / (old_dur + total_dur)
                new_embedding = blend_weight * old_emb + (1 - blend_weight) * new_embedding
                norm = np.linalg.norm(new_embedding)
                if norm > 0:
                    new_embedding = new_embedding / norm
            total_dur += old_dur

        combined = torch.cat(waveforms, dim=-1)
        pitch_hz, pitch_std = compute_pitch(combined, SAMPLE_RATE)
        energy_rms = compute_energy(combined)

        self._db.save(
            name=speaker_name, user_id=user_id,
            embedding=new_embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std, energy_rms=energy_rms,
            total_speech_sec=total_dur, sample_count=len(valid),
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
        embedding = extract_embedding(combined, SAMPLE_RATE, self._embedding_session)
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
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._embedding_session)
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
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._embedding_session)
        return self._db.search(embedding, user_id=user_id, top_k=top_k)

    def get_voiceprint(self, name, user_id=DEFAULT_USER):
        return self._db.get(name, user_id=user_id)

    def delete_voiceprint(self, name, user_id=DEFAULT_USER):
        return self._db.delete(name, user_id=user_id)
