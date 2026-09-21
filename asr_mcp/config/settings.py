import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # GPU / Execution
    cuda_device: str = Field(default="cuda:0", description="CUDA device ordinal")
    torch_dtype: str = Field(default="float16", description="Torch dtype for inference")
    gpu_memory_limit_gb: float = Field(default=4.0, description="Max GPU memory limit")
    cpu_threads: int = Field(default=max(1, os.cpu_count() - 1))

    # ASR Model (Cohere Transcribe ONNX)
    model_repo: str = Field(default="onnx-community/cohere-transcribe-03-2026-ONNX")
    model_dir: Path = Path("./models/cohere-transcribe")
    encoder_model_type: str = "_q4"
    decoder_model_type: str = "_q4"

    # Embedding Model (ECAPA-TDNN)
    embedding_model_repo: str = Field(default="Wespeaker/wespeaker-ecapa-tdnn512-LM")
    embedding_model_filename: str = "voxceleb_ECAPA512_LM.onnx"
    embedding_model_dir: Path = Path("./models/ecapa-tdnn")

    # VAD Model (Silero)
    vad_model_repo: str = Field(default="onnx-community/silero-vad")
    vad_model_dir: Path = Path("./models/silero-vad")
    vad_model_type: str = ""

    # ASR Architecture
    n_layers: int = 8
    heads: int = 8
    head_dim: int = 128
    max_ctx: int = 1024
    max_new_tokens: int = 448

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    request_timeout: int = 120
    max_request_size_mb: int = 200
    max_batch_size: int = 10
    max_audio_duration_sec: int = 600

    # Security
    htpasswd_path: str = "/app/htpasswd"
    session_secret: str = Field(..., description="Secret key for session management")
    api_keys: list[str] = Field(default_factory=list)
    enable_cors: bool = True
    cors_origins: list[str] = Field(default=["*"])

    # Rate limiting
    enable_rate_limit: bool = True
    rate_limit: str = "60/minute"

    # Paths
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./logs")
    model_cache_dir: Path = Path("./models")
    db_path: str = ""

    # Diarization defaults
    diarization_threshold: float = 0.35
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250

    # HuggingFace
    hf_token: Optional[str] = None

    @field_validator(
        "data_dir", "log_dir", "model_dir", "embedding_model_dir",
        "vad_model_dir", "model_cache_dir", mode="before"
    )
    @classmethod
    def ensure_paths(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v

    @property
    def api_key_set(self) -> set[str]:
        return set(self.api_keys)

    def model_post_init(self, __context):
        if not self.db_path:
            self.db_path = str(self.data_dir / "asr_mcp.db")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
