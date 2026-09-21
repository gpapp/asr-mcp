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
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install \
    -r requirements.txt

COPY asr_mcp ./asr_mcp

RUN mkdir -p /app/data /app/logs /app/models /app/voices

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    MODEL_CACHE_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "asr_mcp.server:app", "--host", "0.0.0.0", "--port", "8080"]
