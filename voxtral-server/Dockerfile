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

WORKDIR /app

# Install voxtral-server Python package
COPY pyproject.toml .
COPY voxtral_server/ voxtral_server/
COPY tests/ tests/
RUN pip install --no-cache-dir . "mistral-common[soundfile]"

# Entrypoint: starts vLLM then voxtral-server
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8091

ENTRYPOINT ["/app/entrypoint.sh"]
