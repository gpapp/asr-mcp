import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger("asr_mcp.core.model_state")

EMBEDDING_CACHE_MAX_SIZE = 5000

# Shrink the GPU arena after every run — without this the arena only
# releases on full session reload (reload_encoder_session / reload_embedding_session).
GPU_SHRINK_RUN_OPTIONS = ort.RunOptions()
GPU_SHRINK_RUN_OPTIONS.add_run_config_entry(
    "memory.enable_memory_arena_shrinkage", "gpu:0"
)


class KVCachePool:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int,
                 max_ctx: int, pool_size: int = 4, device: str = "cpu"):
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._max_ctx = max_ctx
        self._device = device
        self._pool: list[dict] = []
        self._lock = threading.Lock()
        self._pool_size = pool_size

    def _create_cache(self) -> dict:
        cache = {}
        for i in range(self._num_layers):
            k = np.zeros((1, self._num_heads, 0, self._head_dim), dtype=np.float32)
            v = np.zeros((1, self._num_heads, 0, self._head_dim), dtype=np.float32)
            cache[f"past_key_values.{i}.key"] = k
            cache[f"past_key_values.{i}.value"] = v
        return cache

    @contextmanager
    def acquire(self):
        cache = None
        with self._lock:
            if self._pool:
                cache = self._pool.pop()
        if cache is None:
            cache = self._create_cache()
        try:
            yield cache
        finally:
            with self._lock:
                if len(self._pool) < self._pool_size:
                    self._pool.append(cache)


class LRUCache:
    def __init__(self, max_size: int = EMBEDDING_CACHE_MAX_SIZE):
        self._max_size = max_size
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._order: list[str] = []

    @staticmethod
    def _make_key(data: np.ndarray) -> str:
        return hashlib.md5(data.tobytes()).hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key].copy()
            return None

    def get_by_data(self, data: np.ndarray) -> Optional[np.ndarray]:
        key = self._make_key(data)
        return self.get(key)

    def put(self, key: str, value: np.ndarray):
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            elif len(self._cache) >= self._max_size:
                oldest = self._order.pop(0)
                del self._cache[oldest]
            self._cache[key] = value.copy()
            self._order.append(key)

    def put_data(self, data: np.ndarray, value: np.ndarray):
        key = self._make_key(data)
        self.put(key, value)

    def __len__(self):
        with self._lock:
            return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()


class ModelState:
    def __init__(self):
        self.encoder_session = None
        self.decoder_session = None
        self.embedding_session = None
        self.vad_session = None
        self.tokens = None
        self.tokenizer = None
        self.prompt_ids = None
        self.eos_token_id = 3
        self.decoder_start_token_id = 13764
        self.device = "cpu"
        self.settings = None
        self._last_used: float = 0.0
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return all([
            self.encoder_session is not None,
            self.decoder_session is not None,
            self.embedding_session is not None,
            self.vad_session is not None,
        ])

    @property
    def any_loaded(self) -> bool:
        return any([
            self.encoder_session is not None,
            self.decoder_session is not None,
            self.embedding_session is not None,
            self.vad_session is not None,
        ])

    def touch(self):
        """Record last usage time."""
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since last usage. Returns 0 if never used."""
        if self._last_used == 0:
            return 0
        return time.monotonic() - self._last_used

    def unload_models(self):
        """Unload all loaded models to free VRAM."""
        import gc
        with self._lock:
            loaded = [n for n in ("encoder_session", "decoder_session", "embedding_session", "vad_session") if getattr(self, n) is not None]
            if not loaded:
                return
            logger.info("Unloading models (idle %.0fs): %s", self.idle_seconds(), ", ".join(loaded))
            self.encoder_session = None
            self.decoder_session = None
            self.embedding_session = None
            self.vad_session = None
            self._last_used = 0.0
        gc.collect()
        log_gpu_memory("models unloaded")
        logger.info("Models unloaded")

    def unload_encoder(self):
        import gc
        with self._lock:
            if self.encoder_session is None:
                return
            logger.info("Unloading encoder session (VRAM)")
            self.encoder_session = None
        gc.collect()
        log_gpu_memory("encoder unloaded")

    def unload_embedding(self):
        import gc
        with self._lock:
            if self.embedding_session is None:
                return
            logger.info("Unloading embedding session (VRAM)")
            self.embedding_session = None
        gc.collect()
        log_gpu_memory("embedding unloaded")


    def reload_models(self):
        """Load any missing models (full cold start or granular fill)."""
        if self.is_ready:
            self.touch()
            return
        if not self.settings:
            from asr_mcp.config.settings import get_settings
            self.settings = get_settings()
        from asr_mcp.core.model_loader import load_models, reload_encoder_session, reload_embedding_session
        try:
            cold = (
                self.tokens is None
                and self.encoder_session is None
                and self.decoder_session is None
                and self.embedding_session is None
                and self.vad_session is None
            )
            if cold:
                logger.info("Cold-loading all models...")
                load_models(self.settings)
            else:
                if self.encoder_session is None:
                    logger.info("Loading encoder session (partial)...")
                    reload_encoder_session(self.settings)
                if self.embedding_session is None:
                    logger.info("Loading embedding session (partial)...")
                    reload_embedding_session(self.settings)
                if self.decoder_session is None or self.vad_session is None or self.tokens is None:
                    logger.info("Loading remaining models (full reload)...")
                    load_models(self.settings)
            self.touch()
            if self.is_ready:
                logger.info("Models ready")
            else:
                logger.warning("Model load incomplete")
        except Exception as e:
            logger.error("Failed to reload models: %s", e)

    def ensure_ready(self):
        """Ensure models are loaded, reload if needed, and record usage."""
        if not self.is_ready:
            self.reload_models()
        if self.is_ready:
            self.touch()

    def clear_gpu_memory(self):
        import gc
        self.encoder_session = None
        self.decoder_session = None
        self.embedding_session = None
        self.vad_session = None
        gc.collect()


state = ModelState()
executor: Optional[ThreadPoolExecutor] = None


def run_embedding(input_feed: dict) -> list[np.ndarray]:
    import onnxruntime as ort
    output_names = [o.name for o in state.embedding_session.get_outputs()]
    return state.embedding_session.run(output_names, input_feed,
                                       run_options=GPU_SHRINK_RUN_OPTIONS)


def is_gpu_oom(err: BaseException) -> bool:
    s = str(err).lower()
    return (
        "failed to allocate memory" in s
        or "out of memory" in s
        or "out_of_memory" in s
        or "available memory of" in s
        or "smaller than requested bytes" in s
    )


def log_gpu_memory(tag: str) -> None:
    """Best-effort GPU memory usage log (for OOM diagnosis)."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
            logger.info("GPU memory [%s]: %s MiB used / %s MiB total",
                        tag, " | ".join(l.split(",")[0] for l in lines),
                        " | ".join(l.split(",")[1] for l in lines))
    except Exception:
        logger.debug("GPU memory query unavailable [%s]", tag, exc_info=True)
