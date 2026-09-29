# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
# faster-whisper (requirements) depends on the CPU onnxruntime wheel whose files
# collide with onnxruntime-gpu (same package dir); the RUN below re-installs the
# GPU wheel last so the CUDA execution provider stays intact (a silent CPU
# fallback would then crash on GPU_SHRINK_RUN_OPTIONS).
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install \
    -r requirements.txt && \
    python3 -m pip install --force-reinstall --no-deps \
    "onnxruntime-gpu>=1.22.0"

COPY asr_mcp ./asr_mcp
COPY tests ./tests

# Test tooling is installed in its own layer (and kept out of requirements.txt)
# so adding/removing it never invalidates the cached heavy dependency layer
# above. Run with: docker compose exec asr-mcp python3 -m pytest tests/ -q
RUN python3 -m pip install --no-cache-dir pytest

RUN mkdir -p /app/data /app/logs /app/models /app/voices

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    MODEL_CACHE_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "asr_mcp.server:app", "--host", "0.0.0.0", "--port", "8080"]
