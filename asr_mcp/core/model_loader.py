import logging
import sys
from pathlib import Path
from typing import Optional

import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download

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


def _download_with_external_data(
    repo_id: str, filename: str, model_dir: Path, hf_token: Optional[str]
) -> Path:
    """Download an ONNX file plus its external .onnx_data files."""
    model_path = model_dir / filename

    if not model_path.exists():
        logger.info("Downloading %s from %s", filename, repo_id)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
            token=hf_token,
        )

    data_file = filename + "_data"
    data_path = model_dir / data_file
    if not data_path.exists():
        logger.info("Downloading external data %s from %s", data_file, repo_id)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=data_file,
                local_dir=str(model_dir),
                token=hf_token,
            )
        except Exception:
            pass

    for i in range(1, 10):
        shard = f"{data_file}_{i}"
        shard_path = model_dir / shard
        if shard_path.exists():
            continue
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=shard,
                local_dir=str(model_dir),
                token=hf_token,
            )
        except Exception:
            break

    return model_path


def ensure_model(settings: Settings) -> Path:
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    encoder_file = f"onnx/encoder_model{settings.encoder_model_type}.onnx"
    decoder_file = f"onnx/decoder_model_merged{settings.decoder_model_type}.onnx"

    _download_with_external_data(settings.model_repo, encoder_file, model_dir, settings.hf_token)
    _download_with_external_data(settings.model_repo, decoder_file, model_dir, settings.hf_token)

    tokenizer_file = "tokenizer.json"
    tokenizer_path = model_dir / tokenizer_file
    if not tokenizer_path.exists():
        logger.info("Downloading %s from %s", tokenizer_file, settings.model_repo)
        hf_hub_download(
            repo_id=settings.model_repo,
            filename=tokenizer_file,
            local_dir=str(model_dir),
            token=settings.hf_token,
        )

    return model_dir


def ensure_vad_model(settings: Settings) -> Path:
    model_dir = Path(settings.vad_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = "onnx/model.onnx"
    return _download_with_external_data(
        settings.vad_model_repo, filename, model_dir, settings.hf_token
    )


def ensure_embedding_model(settings: Settings) -> Path:
    model_dir = Path(settings.embedding_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    return _download_with_external_data(
        settings.embedding_model_repo,
        settings.embedding_model_filename,
        model_dir,
        settings.hf_token,
    )


def _load_tokenizer(model_dir: Path):
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        logger.warning("tokenizer.json not found at %s", tokenizer_path)
        return None
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        logger.info("Loaded tokenizer from %s (vocab_size=%d)", tokenizer_path, tokenizer.get_vocab_size())
        return tokenizer
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        return None


def _build_prompt_ids(tokenizer, language: str = "en") -> list[int]:
    token_to_id = tokenizer.get_vocab()
    lang_token = f"<|{language}|>"
    prompt_tokens = [
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "<|emo:undefined|>",
        lang_token,
        lang_token,
        "<|pnc|>",
        "<|noitn|>",
        "<|timestamp|>",
        "<|nodiarize|>",
    ]
    return [token_to_id[t] for t in prompt_tokens if t in token_to_id]


def load_models(settings: Settings) -> None:
    providers = _get_providers(settings)
    so = get_session_options(settings)

    model_dir = ensure_model(settings)
    encoder_file = f"onnx/encoder_model{settings.encoder_model_type}.onnx"
    decoder_file = f"onnx/decoder_model_merged{settings.decoder_model_type}.onnx"

    logger.info("Loading encoder session (providers=%s)", providers)
    state.encoder_session = ort.InferenceSession(
        str(model_dir / encoder_file), sess_options=so, providers=providers
    )

    enc_inputs = [inp.name for inp in state.encoder_session.get_inputs()]
    enc_outputs = [out.name for out in state.encoder_session.get_outputs()]
    logger.info("Encoder inputs: %s", enc_inputs)
    logger.info("Encoder outputs: %s", enc_outputs)

    logger.info("Loading decoder session (CPU only)")
    cpu_so = get_session_options(settings)
    state.decoder_session = ort.InferenceSession(
        str(model_dir / decoder_file), sess_options=cpu_so, providers=["CPUExecutionProvider"]
    )

    dec_inputs = [inp.name for inp in state.decoder_session.get_inputs()]
    dec_outputs = [out.name for out in state.decoder_session.get_outputs()]
    logger.info("Decoder inputs: %s", dec_inputs)
    logger.info("Decoder outputs: %s", dec_outputs)

    state.tokenizer = _load_tokenizer(model_dir)
    if state.tokenizer:
        state.prompt_ids = _build_prompt_ids(state.tokenizer)
        token_to_id = state.tokenizer.get_vocab()
        state.tokens = {v: k for k, v in token_to_id.items()}
        state.eos_token_id = token_to_id.get("endoftext", 3)
        state.decoder_start_token_id = 13764
        logger.info("Decoder prompt token IDs: %s", state.prompt_ids)
        logger.info("EOS token ID: %d", state.eos_token_id)
    else:
        state.prompt_ids = [13764, 13902, 14190, 14021, 14074, 14254, 13912]
        state.eos_token_id = 3
        state.decoder_start_token_id = 13764
        logger.warning("Tokenizer not loaded, using hardcoded prompt IDs: %s", state.prompt_ids)

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
    encoder_file = f"onnx/encoder_model{settings.encoder_model_type}.onnx"
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
