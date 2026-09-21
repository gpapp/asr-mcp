FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY asr_mcp ./asr_mcp

RUN mkdir -p /app/data /app/logs /app/models

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    MODEL_CACHE_DIR=/app/models \
    PYTHONPATH=/app

EXPOSE 8080

CMD ["python3", "-m", "uvicorn", "asr_mcp.server:app", "--host", "0.0.0.0", "--port", "8080"]
