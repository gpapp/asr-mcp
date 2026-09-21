import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("asr_mcp.core.model_state")

EMBEDDING_CACHE_MAX_SIZE = 5000


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
        self.prompt_ids = None
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

    def touch(self):
        """Record last usage time."""
        self._last_used = time.monotonic()

    def idle_seconds(self) -> float:
        """Seconds since last usage. Returns 0 if never used."""
        if self._last_used == 0:
            return 0
        return time.monotonic() - self._last_used

    def unload_models(self):
        """Unload all GPU models to free VRAM."""
        import gc
        with self._lock:
            if not self.is_ready:
                return
            logger.info("Unloading GPU models (idle %.0fs)", self.idle_seconds())
            self.encoder_session = None
            self.decoder_session = None
            self.embedding_session = None
            self.vad_session = None
            self._last_used = 0.0
        gc.collect()
        logger.info("GPU models unloaded")

    def reload_models(self):
        """Reload models if they were unloaded."""
        if self.is_ready:
            self.touch()
            return
        if not self.settings:
            logger.warning("Cannot reload models: no settings stored")
            return
        logger.info("Auto-reloading models after idle period...")
        from asr_mcp.core.model_loader import load_models
        try:
            load_models(self.settings)
            self.touch()
            logger.info("Models reloaded successfully")
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
    return state.embedding_session.run(output_names, input_feed)
