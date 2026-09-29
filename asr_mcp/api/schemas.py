from typing import Optional

from pydantic import BaseModel, field_validator


class DiarizeRequest(BaseModel):
    wav_path: str
    num_speakers: Optional[int] = None
    diarization_threshold: Optional[float] = None
    vad_threshold: Optional[float] = None
    vad_min_speech_duration_ms: Optional[int] = None
    known_speakers: Optional[dict[str, dict]] = None
    # ISO 639-1 code (e.g. "hu") or "auto" to let the backend detect it.
    # Only used by the transcribe endpoint; diarize ignores it.
    language: str = "auto"


class DiarizeResult(BaseModel):
    start: float
    end: float
    # None when the speaker identity is uncertain (see speaker/uncertainty.py)
    speaker: Optional[str] = None
    uncertain: bool = False
    attribution_reason: Optional[str] = None


class DiarizeResponse(BaseModel):
    segments: list[DiarizeResult]
    total_time_sec: float
    total_speakers: int = 0
    audio_duration_sec: float = 0.0
    uncertain_segments: int = 0
    error: Optional[str] = None


class TranscribePathsRequest(BaseModel):
    wav_paths: list[str]
    language: str = "en"

    @field_validator("wav_paths")
    @classmethod
    def validate_paths(cls, v):
        if not v:
            raise ValueError("At least one path is required")
        return v


class TimedSegment(BaseModel):
    start: float
    end: float
    text: str
    # 0..1 decode confidence (whisper: exp(avg_logprob)); None when the
    # backend does not report one (cohere/qwen3)
    confidence: Optional[float] = None


class TranscribeResult(BaseModel):
    text: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    # None means the speaker could NOT be established — the text is still
    # valid (uncertainty policy: suppress identity, not content).
    speaker: Optional[str] = None
    audio_duration_sec: float = 0.0
    inference_time_sec: float = 0.0
    tokens_generated: int = 0
    segments: Optional[list[TimedSegment]] = None
    error: Optional[str] = None
    # Attribution metadata (see asr_mcp/speaker/uncertainty.py)
    speaker_confidence: Optional[float] = None
    speaker_source: Optional[str] = None      # known_voiceprint | diarization_cluster | unknown
    uncertain: bool = False
    attribution_reason: Optional[str] = None


class TranscribeResponse(BaseModel):
    results: list[TranscribeResult]
    total_time_sec: float


class AttributionItem(BaseModel):
    """One already-transcribed span on the file timeline.

    Supplied by the live client so the server can re-attribute text it has
    already decoded (see ``POST /api/asr/attribution``). Only the timings and
    the text matter — the speaker is always recomputed.
    """
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


class AttributionRequest(BaseModel):
    wav_path: str
    items: list[AttributionItem]
    num_speakers: Optional[int] = None
    diarization_threshold: Optional[float] = None
    vad_threshold: Optional[float] = None
    known_speakers: Optional[dict[str, dict]] = None
    #: Free-form provenance echoed back to the client (asr source, gaps, ...).
    metadata: Optional[dict] = None


class AttributionResponse(BaseModel):
    results: list[TranscribeResult]
    total_time_sec: float
    audio_duration_sec: float = 0.0
    total_speakers: int = 0
    uncertain_segments: int = 0
    #: Seconds spent in diarization. Transcription is not run, so this is the
    #: whole cost of the request.
    processing_time_sec: float = 0.0
    metadata: Optional[dict] = None
    error: Optional[str] = None


class SpeakerRegisterRequest(BaseModel):
    name: str
    wav_path: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    segments: Optional[list[dict]] = None


class SpeakerIdentifyRequest(BaseModel):
    wav_path: str
    start_sec: float = 0.0
    end_sec: Optional[float] = None
    top_k: int = 5


class SpeakerIdentifyResponse(BaseModel):
    name: str
    distance: float
    pitch_hz: float = 0.0


class SpeakerInfoResponse(BaseModel):
    name: str
    pitch_hz: float = 0.0
    pitch_std: float = 0.0
    energy_rms: float = 0.0
    total_speech_sec: float = 0.0
    sample_count: int = 0


class SpeakerListResponse(BaseModel):
    speakers: dict[str, SpeakerInfoResponse]
    count: int


class SpeakerRenameRequest(BaseModel):
    new_name: str


class SpeakerMergeRequest(BaseModel):
    primary: str
    secondary: str


class SpeakerSnippetInfo(BaseModel):
    id: int
    speaker_name: str
    file_path: str
    duration_sec: float
    source_audio: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    created_at: Optional[str] = None


class VoiceprintSpeakerInfo(BaseModel):
    name: str
    snippet_count: int = 0
    total_duration_sec: float = 0.0
    has_voiceprint: bool = False
    pitch_hz: float = 0.0
    energy_rms: float = 0.0


class VoiceprintSpeakerListResponse(BaseModel):
    speakers: list[VoiceprintSpeakerInfo]
    count: int


class VoiceprintSnippetListResponse(BaseModel):
    speaker_name: str
    snippets: list[SpeakerSnippetInfo]
    count: int


class RescanResponse(BaseModel):
    scanned: int = 0
    added: int = 0
    invalidated: int = 0
    rebuilt: int = 0
    speakers: list[str] = []


class HealthResponse(BaseModel):
    status: str
    model_status: str
    voiceprint_count: int = 0
    version: str = "2.0.0"
