import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
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
    # A pending profile was created by auto-learning from an unidentified
    # cluster. It collects snippets but is EXCLUDED from voiceprint matching
    # until the user gives it a name, so learning someone can never produce a
    # confident misattribution (uncertainty policy, lesson 30).
    pending = Column(Boolean, default=False, nullable=False, index=True)
    # A JSON blob describing how tightly the profile's own snippets agree with
    # each other and with the profiles they were separated from
    # ({"intra_max": .., "inter_min": .., "purity_ratio": .., "groups": ..}).
    # Retained so a learned profile can be SHOWN to be a blend candidate rather
    # than silently asserted as one person.
    purity = Column(Text, nullable=True)
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


class CredentialModel(Base):
    """A username/password account created through self-service registration.

    The second credential store, next to the htpasswd file. htpasswd is the
    operator's and stays read-only; this table is what this app owns and writes
    (see ``asr_mcp.auth.store.CredentialStore``).

    ``user_id`` is the vault key -- the namespace every voiceprint, snippet,
    transcript row and every ``voices/<user_id>/`` directory is scoped by -- so
    a username is an identity decision, not just a login string. That is why
    ``user_id`` is validated for length and character set rather than accepted
    as typed, and why a *disabled* row keeps it occupied.

    ``email`` is NOT the key. An address can be reassigned by its provider, so
    keying a vault on one would mean a reassigned address reaching the previous
    occupant's voiceprints. The address exists to prove the account and to
    enforce one-per-account, which is what the partial unique index is for.
    """

    __tablename__ = "credentials"

    user_id = Column(String(64), primary_key=True)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(254), nullable=False, default="")
    # Both verification timestamps are separate on purpose: ``_sent_at`` records
    # that a message went out (auditable, and what a resend button needs) while
    # ``_verified_at`` is the only thing that says the account may sign in.
    email_verified_at = Column(Float, nullable=True)
    verification_token = Column(String(64), nullable=True)
    verification_sent_at = Column(Float, nullable=True)
    created_at = Column(Float, nullable=False)
    updated_at = Column(Float, nullable=False)
    # Set, then either cleared or kept: disabling keeps the username occupied so
    # it cannot be registered again over a vault that still exists.
    disabled_at = Column(Float, nullable=True)

    __table_args__ = (
        Index(
            "credentials_email_unique", "email", unique=True,
            sqlite_where=text("email <> ''"),
        ),
    )


class GoogleIdentityModel(Base):
    """A verified Google account, linked to one vault key.

    The primary key is ``(provider, subject)`` and deliberately NOT
    ``(provider, subject, user_id)``: ``subject`` is Google's stable identifier
    for one person, so it must resolve to exactly one vault forever. A second
    link for the same subject is a contradiction rather than a second grant --
    silently reassigning would let whoever holds a working Google login for one
    address take over a different vault.

    The vault key is the normalised email address, so ``user_id`` here is also
    the ``voices/<user_id>/`` directory name and the ``user_id`` column of every
    voiceprint and snippet row. That is the accepted cost of a human-readable
    identity (see AGENTS.md): the address is PII in this database.
    """

    __tablename__ = "google_identities"

    provider = Column(String(32), primary_key=True, default="google")
    subject = Column(String(255), primary_key=True)
    user_id = Column(String(254), nullable=False, index=True)
    email = Column(String(254), nullable=False, default="")
    display_name = Column(String(120), nullable=False, default="")
    created_at = Column(Float, nullable=False)
    # Bumped only on a successful sign-in. A subject that is not linked must not
    # be able to keep a row alive by being replayed.
    last_used_at = Column(Float, nullable=True)


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
    # Outlier-review state: set by VoiceprintService.consolidate_speaker and
    # cleared by the user keeping the snippet or removing it. review_score and
    # review_reason are the EVIDENCE for the accusation, not just a boolean.
    review_flag = Column(Integer, nullable=False, default=0)
    review_score = Column(Float, nullable=True)
    review_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def init_db(db_path: str) -> sessionmaker:
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        # Additive migrations for databases created before a column existed.
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(snippets)"))}
        for name, decl in (("file_mtime", "REAL"), ("file_size", "INTEGER"),
                           ("review_flag", "INTEGER NOT NULL DEFAULT 0"),
                           ("review_score", "REAL"), ("review_reason", "TEXT")):
            if name not in cols:
                conn.execute(text(f"ALTER TABLE snippets ADD COLUMN {name} {decl}"))
        vp_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(voiceprints)"))}
        if "pending" not in vp_cols:
            conn.execute(text(
                "ALTER TABLE voiceprints ADD COLUMN pending BOOLEAN NOT NULL DEFAULT 0"))
        if "purity" not in vp_cols:
            conn.execute(text("ALTER TABLE voiceprints ADD COLUMN purity TEXT"))
        # credentials/google_identities are created by create_all above, so these
        # ALTERs only fire for a column added in a later version of this file.
        cred_cols = {row[1] for row in conn.execute(text("PRAGMA table_info(credentials)"))}
        for name, decl in (("email", "TEXT NOT NULL DEFAULT ''"),
                           ("email_verified_at", "REAL"),
                           ("verification_token", "TEXT"),
                           ("verification_sent_at", "REAL"),
                           ("disabled_at", "REAL")):
            if name not in cred_cols:
                conn.execute(text(f"ALTER TABLE credentials ADD COLUMN {name} {decl}"))
        conn.commit()
    return sessionmaker(bind=engine)
