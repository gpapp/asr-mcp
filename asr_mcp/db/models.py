import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class VoiceprintModel(Base):
    __tablename__ = "voiceprints"

    user_id = Column(String(255), primary_key=True, default="default")
    name = Column(String(255), primary_key=True)
    embedding = Column(LargeBinary, nullable=False)
    mfcc = Column(LargeBinary, nullable=True)
    pitch_hz = Column(Float, nullable=True)
    pitch_std = Column(Float, nullable=True)
    energy_rms = Column(Float, nullable=True)
    spectral_centroid = Column(Float, nullable=True)
    spectral_rolloff = Column(Float, nullable=True)
    total_speech_sec = Column(Float, default=0.0)
    sample_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(255), primary_key=True)
    data = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_accessed = Column(DateTime, default=datetime.datetime.utcnow)


class TranscriptModel(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, default="default", index=True)
    audio_filename = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    total_speakers = Column(Integer, default=0)
    audio_duration_sec = Column(Float, default=0.0)
    processing_time_sec = Column(Float, default=0.0)
    result = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ApiTokenModel(Base):
    __tablename__ = "api_tokens"

    user_id = Column(String(255), primary_key=True)
    token_hash = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SnippetModel(Base):
    __tablename__ = "snippets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False, default="default", index=True)
    speaker_name = Column(String(255), nullable=False, index=True)
    file_path = Column(String(1024), nullable=False)
    duration_sec = Column(Float, default=0.0)
    source_audio = Column(String(1024), nullable=True)
    start_sec = Column(Float, nullable=True)
    end_sec = Column(Float, nullable=True)
    file_mtime = Column(Float, nullable=True)
    file_size = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db(db_path: str) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(snippets)"))}
        for name, decl in (("file_mtime", "REAL"), ("file_size", "INTEGER")):
            if name not in cols:
                conn.execute(text(f"ALTER TABLE snippets ADD COLUMN {name} {decl}"))
    return sessionmaker(bind=engine)
