import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("asr_mcp.core.model_loader")

# Preload CUDA libraries BEFORE any ORT CUDA session is created: importing torch
# dlopens libcudnn/libcublas into the global namespace (RTLD_GLOBAL), which ORT's
# CUDAExecutionProvider needs at runtime. On the nvidia/cuda:12.2.0 runtime base
# image cuDNN exists only inside the pip nvidia-cudnn-cu12 wheel, and ORT cannot
# locate it on its own, so without this preload CUDA sessions silently fall back
# to CPU and the per-run arena-shrink options then crash with "no arena allocator
# for gpu:0".
def _preload_cuda_libs() -> bool:
    try:
        import torch  # noqa: F401
        # Ensure cuDNN/cuBLAS are in the global namespace exactly when torch
        # bundles them (pip nvidia-cudnn-cu12 / nvidia-cublas-cu12 wheels).
        for lib in ("libcudnn.so.9", "libcublas.so.12", "libcublasLt.so.12"):
            try:
                import ctypes
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        return True
    except Exception:
        logger.warning("torch/cuDNN preload failed: CUDAExecutionProvider may fail", exc_info=True)
        return False


_PRELOADED = _preload_cuda_libs()

import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download

from asr_mcp.core.model_state import ModelState, state
from asr_mcp.config.settings import Settings


def get_session_options(settings: Settings = None) -> ort.SessionOptions:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = settings.cpu_threads if settings else 1
    so.inter_op_num_threads = 1
    return so


def _cuda_provider_options(settings: Settings = None, kind: str = "encoder") -> dict:
    total_gb = float(settings.gpu_memory_limit_gb) if settings else 4.0
    total_bytes = int(total_gb * (1024 ** 3))
    if kind == "encoder":
        limit = int(total_bytes * 0.625)  # ~2.5GB of default 4GB budget
    else:
        limit = min(total_bytes // 4, 768 * 1024 * 1024)  # embedding ≤768MB
    return {
        "gpu_mem_limit": limit,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
    }


def _get_providers(settings: Settings = None, kind: str = "encoder") -> list:
    available = ort.get_available_providers()
    if settings and "CUDAExecutionProvider" in available:
        opts = _cuda_provider_options(settings, kind)
        logger.info("Using CUDAExecutionProvider (%s, gpu_mem_limit=%d MiB)",
                    kind, opts["gpu_mem_limit"] // (1024 * 1024))
        return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
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


def load_vad_session(settings: Settings = None) -> ort.InferenceSession:
    if not settings:
        from asr_mcp.config.settings import get_settings
        settings = get_settings()
    vad_path = ensure_vad_model(settings)
    so = get_session_options(settings)
    logger.info("Loading VAD session (CPU)")
    return ort.InferenceSession(
        str(vad_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def load_embedding_session(settings: Settings = None):
    if not settings:
        from asr_mcp.config.settings import get_settings
        settings = get_settings()
    emb_path = ensure_embedding_model(settings)
    emb_providers = _get_providers(settings, "embedding")
    so = get_session_options(settings)
    logger.info("Loading embedding session (providers=%s)", emb_providers)
    return ort.InferenceSession(
        str(emb_path), sess_options=so, providers=emb_providers
    )


def reload_embedding_session(settings: Settings, force_cpu: bool = False) -> None:
    import gc
    so = get_session_options(settings)
    emb_path = ensure_embedding_model(settings)
    providers = ["CPUExecutionProvider"] if force_cpu else _get_providers(settings, "embedding")
    state.embedding_session = None
    gc.collect()
    state.embedding_session = ort.InferenceSession(
        str(emb_path), sess_options=so, providers=providers
    )
