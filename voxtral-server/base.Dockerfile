# base.Dockerfile — Built once during AMI prep (~23GB).
# Contains vLLM runtime, system deps, model weights, and heavy Python packages.
# Cached on the AMI so runtime Docker builds take seconds (only thin app layer on top).
FROM vllm/vllm-openai:v0.18.0

# System deps: FFmpeg (SRT ingest + audio decode), aiortc build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsrtp2-dev \
        libopus-dev \
        libvpx-dev \
        libffi-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pre-install heavy Python deps that rarely change
RUN pip install --no-cache-dir "mistral-common[soundfile]" aiortc aioice

# Pre-download model (~8GB) — cached in image layers
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Voxtral-Mini-4B-Realtime-2602')"
