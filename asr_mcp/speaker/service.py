import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from asr_mcp.db.manager import DEFAULT_USER, DatabaseManager, VoiceprintDB

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
        user_id: str = DEFAULT_USER,
    ) -> dict:
        self._db.save(
            name=name, embedding=embedding, user_id=user_id,
            pitch_hz=pitch_hz, pitch_std=pitch_std,
            energy_rms=energy_rms,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            total_speech_sec=total_speech_sec,
            sample_count=sample_count, mfcc=mfcc,
        )
        logger.info("Registered speaker: %s (user=%s, speech=%.1fs)",
                    name, user_id, total_speech_sec)
        return {"name": name, "status": "registered"}

    def identify_speaker(
        self, embedding: np.ndarray, top_k: int = 5,
        user_id: str = DEFAULT_USER,
    ) -> list[dict]:
        results = self._db.search(embedding, user_id=user_id, top_k=top_k)
        return [
            {"name": name, "distance": dist, "pitch_hz": info.get("pitch_hz", 0)}
            for name, dist, info in results
        ]

    def get_speaker_info(self, name: str,
                         user_id: str = DEFAULT_USER) -> Optional[dict]:
        return self._db.get(name, user_id=user_id)

    def list_speakers(self, user_id: str = DEFAULT_USER) -> dict[str, dict]:
        return self._db.list_all(user_id=user_id)

    def remove_speaker(self, name: str,
                       user_id: str = DEFAULT_USER) -> bool:
        deleted = self._db.delete(name, user_id=user_id)
        if deleted:
            logger.info("Removed speaker: %s (user=%s)", name, user_id)
        return deleted

    def to_dict(self, user_id: str = DEFAULT_USER) -> dict[str, dict]:
        return self._db.to_dict(user_id=user_id)
