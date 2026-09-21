import datetime
import json
import struct
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session as SASession

from asr_mcp.db.models import (
    Base,
    SessionModel,
    SnippetModel,
    TranscriptModel,
    VoiceprintModel,
    init_db,
)

DEFAULT_USER = "default"


class DatabaseManager:
    def __init__(self, db_path: str):
        self._engine = None
        self._SessionLocal = None
        self._db_path = db_path
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._SessionLocal = init_db(self._db_path)

    def get_session(self) -> SASession:
        return self._SessionLocal()

    def close(self):
        if self._engine:
            self._engine.dispose()


class VoiceprintDB:
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding.tolist())

    @staticmethod
    def _deserialize_embedding(data: bytes) -> np.ndarray:
        count = len(data) // 4
        return np.array(struct.unpack(f"{count}f", data), dtype=np.float32)

    @staticmethod
    def _serialize_mfcc(mfcc: dict) -> bytes:
        return json.dumps(mfcc).encode("utf-8")

    @staticmethod
    def _deserialize_mfcc(data: bytes) -> dict:
        return json.loads(data.decode("utf-8")) if data else {}

    def save(self, name: str, embedding: np.ndarray, user_id: str = DEFAULT_USER,
             pitch_hz: float = 0.0, pitch_std: float = 0.0, energy_rms: float = 0.0,
             spectral_centroid: float = 0.0, spectral_rolloff: float = 0.0,
             total_speech_sec: float = 0.0, sample_count: int = 0,
             mfcc: Optional[dict] = None):
        with self._db.get_session() as session:
            existing = session.query(VoiceprintModel).filter_by(
                user_id=user_id, name=name
            ).first()
            emb_bytes = self._serialize_embedding(embedding)
            mfcc_bytes = self._serialize_mfcc(mfcc) if mfcc else None
            now = datetime.datetime.utcnow()
            if existing:
                existing.embedding = emb_bytes
                existing.mfcc = mfcc_bytes
                existing.pitch_hz = pitch_hz
                existing.pitch_std = pitch_std
                existing.energy_rms = energy_rms
                existing.spectral_centroid = spectral_centroid
                existing.spectral_rolloff = spectral_rolloff
                existing.total_speech_sec = total_speech_sec
                existing.sample_count = sample_count
                existing.updated_at = now
            else:
                vp = VoiceprintModel(
                    user_id=user_id,
                    name=name,
                    embedding=emb_bytes,
                    mfcc=mfcc_bytes,
                    pitch_hz=pitch_hz,
                    pitch_std=pitch_std,
                    energy_rms=energy_rms,
                    spectral_centroid=spectral_centroid,
                    spectral_rolloff=spectral_rolloff,
                    total_speech_sec=total_speech_sec,
                    sample_count=sample_count,
                    created_at=now,
                    updated_at=now,
                )
                session.add(vp)
            session.commit()

    def get(self, name: str, user_id: str = DEFAULT_USER) -> Optional[dict]:
        with self._db.get_session() as session:
            vp = session.query(VoiceprintModel).filter_by(
                user_id=user_id, name=name
            ).first()
            if not vp:
                return None
            return self._row_to_dict(vp)

    def list_all(self, user_id: str = DEFAULT_USER) -> dict[str, dict]:
        with self._db.get_session() as session:
            rows = session.query(VoiceprintModel).filter_by(user_id=user_id).all()
            return {row.name: self._row_to_dict(row) for row in rows}

    def delete(self, name: str, user_id: str = DEFAULT_USER) -> bool:
        with self._db.get_session() as session:
            vp = session.query(VoiceprintModel).filter_by(
                user_id=user_id, name=name
            ).first()
            if not vp:
                return False
            session.delete(vp)
            session.commit()
            return True

    def rename(self, old_name: str, new_name: str, user_id: str = DEFAULT_USER) -> bool:
        with self._db.get_session() as session:
            vp = session.query(VoiceprintModel).filter_by(
                user_id=user_id, name=old_name
            ).first()
            if not vp:
                return False
            existing = session.query(VoiceprintModel).filter_by(
                user_id=user_id, name=new_name
            ).first()
            if existing:
                return False
            vp.name = new_name
            session.commit()
            return True

    def count(self, user_id: str = DEFAULT_USER) -> int:
        with self._db.get_session() as session:
            return session.query(VoiceprintModel).filter_by(user_id=user_id).count()

    def search(self, embedding: np.ndarray, user_id: str = DEFAULT_USER, top_k: int = 5) -> list[tuple[str, float, dict]]:
        all_vps = self.list_all(user_id=user_id)
        results = []
        for name, vp_dict in all_vps.items():
            dist = 1.0 - float(np.dot(embedding, vp_dict["embedding"]) /
                               (np.linalg.norm(embedding) * np.linalg.norm(vp_dict["embedding"]) + 1e-8))
            results.append((name, dist, vp_dict))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def _row_to_dict(self, vp: VoiceprintModel) -> dict:
        return {
            "user_id": vp.user_id,
            "pitch_hz": vp.pitch_hz or 0.0,
            "pitch_std": vp.pitch_std or 0.0,
            "energy_rms": vp.energy_rms or 0.0,
            "spectral_centroid": vp.spectral_centroid or 0.0,
            "spectral_rolloff": vp.spectral_rolloff or 0.0,
            "total_speech_sec": vp.total_speech_sec or 0.0,
            "sample_count": vp.sample_count or 0,
            "embedding": self._deserialize_embedding(vp.embedding),
            "mfcc": self._deserialize_mfcc(vp.mfcc) if vp.mfcc else {},
        }

    def to_dict(self, user_id: str = DEFAULT_USER) -> dict[str, dict]:
        return self.list_all(user_id=user_id)


class SnippetDB:
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def add(self, speaker_name: str, file_path: str, duration_sec: float,
            user_id: str = DEFAULT_USER, source_audio: str = None,
            start_sec: float = None, end_sec: float = None) -> int:
        with self._db.get_session() as session:
            sn = SnippetModel(
                user_id=user_id,
                speaker_name=speaker_name,
                file_path=file_path,
                duration_sec=duration_sec,
                source_audio=source_audio,
                start_sec=start_sec,
                end_sec=end_sec,
            )
            session.add(sn)
            session.commit()
            return sn.id

    def list_by_speaker(self, speaker_name: str, user_id: str = DEFAULT_USER) -> list[dict]:
        with self._db.get_session() as session:
            rows = (
                session.query(SnippetModel)
                .filter_by(user_id=user_id, speaker_name=speaker_name)
                .order_by(SnippetModel.created_at)
                .all()
            )
            return [self._row_to_dict(r) for r in rows]

    def get(self, snippet_id: int, user_id: str = DEFAULT_USER) -> Optional[dict]:
        with self._db.get_session() as session:
            sn = session.query(SnippetModel).filter_by(
                id=snippet_id, user_id=user_id
            ).first()
            if not sn:
                return None
            return self._row_to_dict(sn)

    def delete(self, snippet_id: int, user_id: str = DEFAULT_USER) -> bool:
        with self._db.get_session() as session:
            sn = session.query(SnippetModel).filter_by(
                id=snippet_id, user_id=user_id
            ).first()
            if not sn:
                return False
            session.delete(sn)
            session.commit()
            return True

    def rename_speaker(self, old_name: str, new_name: str, user_id: str = DEFAULT_USER) -> int:
        with self._db.get_session() as session:
            count = (
                session.query(SnippetModel)
                .filter_by(user_id=user_id, speaker_name=old_name)
                .update({"speaker_name": new_name})
            )
            session.commit()
            return count

    def total_duration(self, speaker_name: str, user_id: str = DEFAULT_USER) -> float:
        with self._db.get_session() as session:
            from sqlalchemy import func
            result = (
                session.query(func.sum(SnippetModel.duration_sec))
                .filter_by(user_id=user_id, speaker_name=speaker_name)
                .scalar()
            )
            return result or 0.0

    def find_duplicate(self, source_audio: str, start_sec: float, user_id: str = DEFAULT_USER) -> Optional[dict]:
        with self._db.get_session() as session:
            sn = session.query(SnippetModel).filter_by(
                user_id=user_id, source_audio=source_audio, start_sec=start_sec
            ).first()
            if not sn:
                return None
            return self._row_to_dict(sn)

    def count(self, speaker_name: str, user_id: str = DEFAULT_USER) -> int:
        with self._db.get_session() as session:
            return (
                session.query(SnippetModel)
                .filter_by(user_id=user_id, speaker_name=speaker_name)
                .count()
            )

    def all_speakers(self, user_id: str = DEFAULT_USER) -> dict[str, dict]:
        with self._db.get_session() as session:
            from sqlalchemy import func
            rows = (
                session.query(
                    SnippetModel.speaker_name,
                    func.count(SnippetModel.id).label("count"),
                    func.sum(SnippetModel.duration_sec).label("total_duration"),
                )
                .filter_by(user_id=user_id)
                .group_by(SnippetModel.speaker_name)
                .all()
            )
            return {
                r.speaker_name: {"count": r.count, "total_duration": r.total_duration or 0.0}
                for r in rows
            }

    def _row_to_dict(self, sn: SnippetModel) -> dict:
        return {
            "id": sn.id,
            "user_id": sn.user_id,
            "speaker_name": sn.speaker_name,
            "file_path": sn.file_path,
            "duration_sec": sn.duration_sec,
            "source_audio": sn.source_audio,
            "start_sec": sn.start_sec,
            "end_sec": sn.end_sec,
            "created_at": sn.created_at.isoformat() if sn.created_at else None,
        }


class SessionDB:
    def __init__(self, db_manager: DatabaseManager, ttl_seconds: int = 3600):
        self._db = db_manager
        self._ttl_seconds = ttl_seconds

    def create(self, session_id: str, data: Optional[dict] = None) -> str:
        now = datetime.datetime.utcnow()
        with self._db.get_session() as session:
            sm = SessionModel(
                id=session_id,
                data=json.dumps(data or {}),
                created_at=now,
                expires_at=now + datetime.timedelta(seconds=self._ttl_seconds),
                last_accessed=now,
            )
            session.add(sm)
            session.commit()
        return session_id

    def get(self, session_id: str) -> Optional[dict]:
        with self._db.get_session() as session:
            sm = session.get(SessionModel, session_id)
            if not sm:
                return None
            if sm.expires_at < datetime.datetime.utcnow():
                session.delete(sm)
                session.commit()
                return None
            sm.last_accessed = datetime.datetime.utcnow()
            sm.expires_at = datetime.datetime.utcnow() + datetime.timedelta(seconds=self._ttl_seconds)
            session.commit()
            return json.loads(sm.data)

    def set_data(self, session_id: str, data: dict):
        with self._db.get_session() as session:
            sm = session.get(SessionModel, session_id)
            if sm:
                sm.data = json.dumps(data)
                sm.last_accessed = datetime.datetime.utcnow()
                session.commit()

    def delete(self, session_id: str) -> bool:
        with self._db.get_session() as session:
            sm = session.get(SessionModel, session_id)
            if not sm:
                return False
            session.delete(sm)
            session.commit()
            return True

    def cleanup_expired(self) -> int:
        with self._db.get_session() as session:
            now = datetime.datetime.utcnow()
            expired = session.query(SessionModel).filter(SessionModel.expires_at < now).all()
            count = len(expired)
            for sm in expired:
                session.delete(sm)
            session.commit()
            return count

    def list_all(self) -> list[dict]:
        with self._db.get_session() as session:
            rows = session.query(SessionModel).all()
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(),
                    "expires_at": r.expires_at.isoformat(),
                    "last_accessed": r.last_accessed.isoformat(),
                }
                for r in rows
            ]


class TranscriptDB:
    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    def save(self, audio_filename: str, result: dict, session_id: Optional[str] = None) -> int:
        with self._db.get_session() as session:
            tm = TranscriptModel(
                session_id=session_id,
                audio_filename=audio_filename,
                result=json.dumps(result),
            )
            session.add(tm)
            session.commit()
            return tm.id

    def get(self, transcript_id: int) -> Optional[dict]:
        with self._db.get_session() as session:
            tm = session.get(TranscriptModel, transcript_id)
            if not tm:
                return None
            return {
                "id": tm.id,
                "session_id": tm.session_id,
                "audio_filename": tm.audio_filename,
                "result": json.loads(tm.result),
                "created_at": tm.created_at.isoformat(),
            }

    def list_by_session(self, session_id: str) -> list[dict]:
        with self._db.get_session() as session:
            rows = (
                session.query(TranscriptModel)
                .filter(TranscriptModel.session_id == session_id)
                .order_by(TranscriptModel.created_at.desc())
                .all()
            )
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "audio_filename": r.audio_filename,
                    "result": json.loads(r.result),
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    def delete(self, transcript_id: int) -> bool:
        with self._db.get_session() as session:
            tm = session.get(TranscriptModel, transcript_id)
            if not tm:
                return False
            session.delete(tm)
            session.commit()
            return True
