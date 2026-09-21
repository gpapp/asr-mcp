import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from asr_mcp.db.manager import DatabaseManager, VoiceprintDB
from asr_mcp.speaker.embedding import extract_embedding, batch_embed_files, compute_pitch, compute_energy
from asr_mcp.voiceprint.utils import load_audio_segment

logger = logging.getLogger("asr_mcp.voiceprint.service")

SAMPLE_RATE = 16000


class VoiceprintService:
    def __init__(self, data_dir: Path, db_manager: DatabaseManager, embedding_session=None):
        self._data_dir = data_dir
        self._db = VoiceprintDB(db_manager)
        self._embedding_session = embedding_session

    def set_embedding_session(self, session):
        self._embedding_session = session

    async def initialize(self):
        count = self._db.count()
        logger.info("VoiceprintService initialized — %d voiceprints in database", count)

    def register_from_segments(
        self,
        name: str,
        wav_path: str,
        segments: list[dict],
        min_duration: float = 1.5,
    ) -> dict:
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
            name=name,
            embedding=embedding,
            pitch_hz=pitch_hz,
            pitch_std=pitch_std,
            energy_rms=energy_rms,
            total_speech_sec=total_duration,
            sample_count=len(all_audio),
        )

        return {
            "name": name,
            "total_speech_sec": round(total_duration, 2),
            "sample_count": len(all_audio),
        }

    def register_from_audio(
        self,
        name: str,
        wav_path: str,
        start_sec: float,
        end_sec: float,
    ) -> dict:
        waveform, sr = load_audio_segment(wav_path, start_sec, end_sec)
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._embedding_session)
        pitch_hz, pitch_std = compute_pitch(waveform, SAMPLE_RATE)
        energy_rms = compute_energy(waveform)
        total_duration = (end_sec - start_sec)

        self._db.save(
            name=name,
            embedding=embedding,
            pitch_hz=pitch_hz,
            pitch_std=pitch_std,
            energy_rms=energy_rms,
            total_speech_sec=total_duration,
            sample_count=1,
        )

        return {
            "name": name,
            "total_speech_sec": round(total_duration, 2),
            "sample_count": 1,
        }

    def refine_voiceprint(
        self,
        name: str,
        segments_dir: Path,
        min_duration: float = 1.5,
        block_sec: float = 600.0,
    ) -> dict:
        existing = self._db.get(name)
        segment_files = []
        for ext in ["*.wav", "*.mp3", "*.flac"]:
            segment_files.extend(segments_dir.glob(ext))

        if not segment_files:
            return {"error": "No segment files found"}

        waveforms = []
        durations = []
        for sf_path in segment_files:
            try:
                from asr_mcp.voiceprint.utils import load_audio
                waveform, sr = load_audio(str(sf_path))
                dur = waveform.shape[-1] / SAMPLE_RATE
                if dur >= min_duration:
                    waveforms.append(waveform)
                    durations.append(dur)
            except Exception as e:
                logger.warning("Failed to load %s: %s", sf, e)

        if not waveforms:
            return {"error": "No valid audio segments"}

        embeddings = batch_embed_files(
            waveforms, [SAMPLE_RATE] * len(waveforms), durations,
            self._embedding_session, block_sec=block_sec,
        )

        valid = [(e, d) for e, d in zip(embeddings, durations) if e is not None]
        if not valid:
            return {"error": "No embeddings computed"}

        embeddings_arr = [e for e, _ in valid]
        durations_arr = [d for _, d in valid]
        total_dur = sum(durations_arr)

        # Duration-weighted average
        weights = np.array(durations_arr) / total_dur
        new_embedding = np.zeros_like(embeddings_arr[0])
        for emb, w in zip(embeddings_arr, weights):
            new_embedding += emb * w

        # L2 normalize
        norm = np.linalg.norm(new_embedding)
        if norm > 0:
            new_embedding = new_embedding / norm

        # Blend with existing if present
        if existing and existing.get("embedding") is not None:
            old_emb = existing["embedding"]
            old_dur = existing.get("total_speech_sec", 0)
            blend_weight = old_dur / (old_dur + total_dur)
            new_embedding = blend_weight * old_emb + (1 - blend_weight) * new_embedding
            norm = np.linalg.norm(new_embedding)
            if norm > 0:
                new_embedding = new_embedding / norm
            total_dur += old_dur

        # Compute pitch/energy from combined audio
        combined = torch.cat(waveforms, dim=-1)
        pitch_hz, pitch_std = compute_pitch(combined, SAMPLE_RATE)
        energy_rms = compute_energy(combined)

        self._db.save(
            name=name,
            embedding=new_embedding,
            pitch_hz=pitch_hz,
            pitch_std=pitch_std,
            energy_rms=energy_rms,
            total_speech_sec=total_dur,
            sample_count=len(valid),
        )

        return {
            "name": name,
            "total_speech_sec": round(total_dur, 2),
            "sample_count": len(valid),
            "status": "refined" if existing else "created",
        }

    def identify_in_audio(
        self,
        wav_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        top_k: int = 5,
    ) -> list[dict]:
        waveform, sr = load_audio_segment(wav_path, start_sec, end_sec or 99999)
        embedding = extract_embedding(waveform, SAMPLE_RATE, self._embedding_session)
        return self._db.search(embedding, top_k=top_k)

    def list_voiceprints(self) -> dict[str, dict]:
        return self._db.list_all()

    def get_voiceprint(self, name: str) -> Optional[dict]:
        return self._db.get(name)

    def delete_voiceprint(self, name: str) -> bool:
        return self._db.delete(name)
