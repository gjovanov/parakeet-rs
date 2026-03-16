#!/bin/bash
#
# Parakeet-rs Server Launcher
#
# Reads configuration from .env (GPU/CPU mode, model paths, etc.)
#
# Usage: ./start-server.sh [extra args...]
#   or:  sudo ./start-server.sh
#
# Override .env values via environment:
#   USE_GPU=false ./start-server.sh   # force CPU
#   PORT=3000 ./start-server.sh       # custom port
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Save caller's overrides before sourcing .env
_PORT="${PORT:-}"
_MAX="${MAX_CONCURRENT_SESSIONS:-}"
_USE_GPU="${USE_GPU:-}"
_ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-}"

# Load .env if present (caller env vars take precedence below)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Restore caller overrides (.env must not clobber explicit env vars)
[ -n "$_PORT" ] && PORT="$_PORT"
[ -n "$_MAX" ] && MAX_CONCURRENT_SESSIONS="$_MAX"
[ -n "$_USE_GPU" ] && USE_GPU="$_USE_GPU"
[ -n "$_ORT_DYLIB_PATH" ] && ORT_DYLIB_PATH="$_ORT_DYLIB_PATH"

PORT="${PORT:-8080}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-10}"
USE_GPU="${USE_GPU:-false}"

# Set ORT library path based on GPU mode
if [ "$USE_GPU" = "false" ] || [ -z "$USE_GPU" ]; then
    # CPU mode
    ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-./ort-cpu/libonnxruntime.so}"
    export LD_LIBRARY_PATH="$(cd "$(dirname "$ORT_DYLIB_PATH")" && pwd):${LD_LIBRARY_PATH:-}"
    MODE_LABEL="CPU"
else
    # GPU mode (cuda, tensorrt, true)
    ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-./ort-gpu/libonnxruntime.so}"
    export LD_LIBRARY_PATH="$(cd "$(dirname "$ORT_DYLIB_PATH")" && pwd):/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    MODE_LABEL="GPU ($USE_GPU)"
fi

export ORT_DYLIB_PATH
export USE_GPU

# Build command args
ARGS=(
    --port "$PORT"
    --tdt-model "${TDT_MODEL_PATH:-./tdt}"
    --canary-model "${CANARY_MODEL_PATH:-./canary}"
    --diar-model "${DIAR_MODEL_PATH:-./diar_streaming_sortformer_4spk-v2.onnx}"
    --vad-model "${VAD_MODEL_PATH:-./silero_vad.onnx}"
    --frontend "${FRONTEND_PATH:-./frontend}"
    --media-dir "${MEDIA_DIR:-./media}"
    --max-sessions "$MAX_CONCURRENT_SESSIONS"
)

if [ -n "$PUBLIC_IP" ]; then
    ARGS+=(--public-ip "$PUBLIC_IP")
fi

if [ -n "$FAB_URL" ]; then
    ARGS+=(--fab-url "$FAB_URL")
fi

if [ "${SPEEDY_MODE:-true}" = "true" ]; then
    ARGS+=(--speedy)
fi

echo "Starting parakeet-rs ($MODE_LABEL) on port $PORT..."
echo "  ORT_DYLIB_PATH=$ORT_DYLIB_PATH"

# Port 80 requires root — re-exec with sudo -E if needed
if [ "$PORT" -le 1024 ] && [ "$(id -u)" -ne 0 ]; then
    exec sudo -E "$0" "$@"
fi

LOG_FILE="${LOG_FILE:-/tmp/parakeet-server-${PORT}.log}"
echo "Logging to: $LOG_FILE"
exec ./target/release/parakeet-server "${ARGS[@]}" "$@" >> "$LOG_FILE" 2>&1
