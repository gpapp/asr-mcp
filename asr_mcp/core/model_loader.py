import logging
import sys
from pathlib import Path
from typing import Optional

import onnxruntime as ort
from huggingface_hub import hf_hub_download

from asr_mcp.core.model_state import ModelState, state
from asr_mcp.config.settings import Settings

logger = logging.getLogger("asr_mcp.core.model_loader")


def get_session_options(settings: Settings = None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = settings.cpu_threads if settings else 1
    so.inter_op_num_threads = 1
    return so


def _get_providers(settings: Settings = None) -> list[str]:
    available = ort.get_available_providers()
    if settings and "CUDAExecutionProvider" in available:
        logger.info("Using CUDAExecutionProvider")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info("Falling back to CPUExecutionProvider")
    return ["CPUExecutionProvider"]


def ensure_model(settings: Settings) -> Path:
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    encoder_file = f"encoder{settings.encoder_model_type}.onnx"
    decoder_file = f"decoder{settings.decoder_model_type}.onnx"

    encoder_path = model_dir / encoder_file
    decoder_path = model_dir / decoder_file

    if not encoder_path.exists():
        logger.info("Downloading encoder model from %s", settings.model_repo)
        encoder_path = Path(hf_hub_download(
            repo_id=settings.model_repo,
            filename=encoder_file,
            local_dir=str(model_dir),
            token=settings.hf_token,
        ))

    if not decoder_path.exists():
        logger.info("Downloading decoder model from %s", settings.model_repo)
        decoder_path = Path(hf_hub_download(
            repo_id=settings.model_repo,
            filename=decoder_file,
            local_dir=str(model_dir),
            token=settings.hf_token,
        ))

    return model_dir


def ensure_vad_model(settings: Settings) -> Path:
    model_dir = Path(settings.vad_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = "silero_vad.onnx"
    model_path = model_dir / filename

    if not model_path.exists():
        logger.info("Downloading VAD model from %s", settings.vad_model_repo)
        model_path = Path(hf_hub_download(
            repo_id=settings.vad_model_repo,
            filename=filename,
            local_dir=str(model_dir),
            token=settings.hf_token,
        ))

    return model_path


def ensure_embedding_model(settings: Settings) -> Path:
    model_dir = Path(settings.embedding_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / settings.embedding_model_filename

    if not model_path.exists():
        logger.info("Downloading embedding model from %s", settings.embedding_model_repo)
        model_path = Path(hf_hub_download(
            repo_id=settings.embedding_model_repo,
            filename=settings.embedding_model_filename,
            local_dir=str(model_dir),
            token=settings.hf_token,
        ))

    return model_path


def load_models(settings: Settings) -> None:
    providers = _get_providers(settings)
    so = get_session_options(settings)

    model_dir = ensure_model(settings)
    encoder_file = f"encoder{settings.encoder_model_type}.onnx"
    decoder_file = f"decoder{settings.decoder_model_type}.onnx"

    logger.info("Loading encoder session (providers=%s)", providers)
    state.encoder_session = ort.InferenceSession(
        str(model_dir / encoder_file), sess_options=so, providers=providers
    )

    logger.info("Loading decoder session (CPU only)")
    cpu_so = get_session_options(settings)
    state.decoder_session = ort.InferenceSession(
        str(model_dir / decoder_file), sess_options=cpu_so, providers=["CPUExecutionProvider"]
    )

    vad_path = ensure_vad_model(settings)
    logger.info("Loading VAD session (CPU)")
    state.vad_session = ort.InferenceSession(
        str(vad_path), sess_options=so, providers=["CPUExecutionProvider"]
    )

    emb_path = ensure_embedding_model(settings)
    logger.info("Loading embedding session (providers=%s)", providers)
    state.embedding_session = ort.InferenceSession(
        str(emb_path), sess_options=so, providers=providers
    )

    state.settings = settings

    logger.info("All models loaded successfully")


def reload_encoder_session(settings: Settings, force_cpu: bool = False) -> None:
    so = get_session_options(settings)
    model_dir = Path(settings.model_dir)
    encoder_file = f"encoder{settings.encoder_model_type}.onnx"
    providers = ["CPUExecutionProvider"] if force_cpu else _get_providers(settings)
    state.encoder_session = ort.InferenceSession(
        str(model_dir / encoder_file), sess_options=so, providers=providers
    )


def reload_embedding_session(settings: Settings, force_cpu: bool = False) -> None:
    so = get_session_options(settings)
    emb_path = ensure_embedding_model(settings)
    providers = ["CPUExecutionProvider"] if force_cpu else _get_providers(settings)
    state.embedding_session = ort.InferenceSession(
        str(emb_path), sess_options=so, providers=providers
    )
