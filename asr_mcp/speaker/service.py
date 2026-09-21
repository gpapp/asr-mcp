import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from asr_mcp.db.manager import DatabaseManager, VoiceprintDB

logger = logging.getLogger("asr_mcp.speaker.service")


class SpeakerService:
    def __init__(self, data_dir: Path, db_manager: DatabaseManager):
        self._data_dir = data_dir
        self._db = VoiceprintDB(db_manager)

    async def initialize(self):
        count = self._db.count()
        logger.info("SpeakerService initialized — %d voiceprints in database", count)

    def register_speaker(
        self, name: str, embedding: np.ndarray,
        pitch_hz: float = 0.0, pitch_std: float = 0.0,
        energy_rms: float = 0.0, spectral_centroid: float = 0.0,
        spectral_rolloff: float = 0.0, total_speech_sec: float = 0.0,
        sample_count: int = 0, mfcc: Optional[dict] = None,
    ) -> dict:
        self._db.save(
            name=name, embedding=embedding,
            pitch_hz=pitch_hz, pitch_std=pitch_std,
            energy_rms=energy_rms,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            total_speech_sec=total_speech_sec,
            sample_count=sample_count, mfcc=mfcc,
        )
        logger.info("Registered speaker: %s (speech=%.1fs)", name, total_speech_sec)
        return {"name": name, "status": "registered"}

    def identify_speaker(
        self, embedding: np.ndarray, top_k: int = 5,
    ) -> list[dict]:
        results = self._db.search(embedding, top_k=top_k)
        return [
            {"name": name, "distance": dist, "pitch_hz": info.get("pitch_hz", 0)}
            for name, dist, info in results
        ]

    def get_speaker_info(self, name: str) -> Optional[dict]:
        return self._db.get(name)

    def list_speakers(self) -> dict[str, dict]:
        return self._db.list_all()

    def remove_speaker(self, name: str) -> bool:
        deleted = self._db.delete(name)
        if deleted:
            logger.info("Removed speaker: %s", name)
        return deleted

    def to_dict(self) -> dict[str, dict]:
        return self._db.to_dict()
