from typing import Optional

from pydantic import BaseModel, field_validator


class DiarizeRequest(BaseModel):
    wav_path: str
    num_speakers: Optional[int] = None
    diarization_threshold: Optional[float] = None
    vad_threshold: Optional[float] = None
    vad_min_speech_duration_ms: Optional[int] = None
    known_speakers: Optional[dict[str, dict]] = None


class DiarizeResult(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizeResponse(BaseModel):
    segments: list[DiarizeResult]
    total_time_sec: float
    total_speakers: int = 0
    audio_duration_sec: float = 0.0
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


class TranscribeResult(BaseModel):
    text: str = ""
    start: Optional[float] = None
    end: Optional[float] = None
    speaker: Optional[str] = None
    audio_duration_sec: float = 0.0
    inference_time_sec: float = 0.0
    tokens_generated: int = 0
    segments: Optional[list[TimedSegment]] = None
    error: Optional[str] = None


class TranscribeResponse(BaseModel):
    results: list[TranscribeResult]
    total_time_sec: float


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
    speakers: list[str] = []


class HealthResponse(BaseModel):
    status: str
    model_status: str
    voiceprint_count: int = 0
    version: str = "2.0.0"
