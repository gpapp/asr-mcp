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
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class VoiceprintModel(Base):
    __tablename__ = "voiceprints"

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
    session_id = Column(String(255), nullable=True)
    audio_filename = Column(String(512), nullable=False)
    result = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db(db_path: str) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
