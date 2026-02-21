#!/bin/bash
#
# Parakeet-rs CPU Server Launcher
#
# Usage: ./start-server.sh [extra args...]
#   or:  sudo ./start-server.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Save caller's overrides before sourcing .env
_PORT="${PORT:-}"
_MAX="${MAX_CONCURRENT_SESSIONS:-}"

# Load .env if present (caller env vars take precedence below)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Restore caller overrides (.env must not clobber explicit env vars)
[ -n "$_PORT" ] && PORT="$_PORT"
[ -n "$_MAX" ] && MAX_CONCURRENT_SESSIONS="$_MAX"

# CPU mode: force correct values (override stale .env entries)
export ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so"
export USE_GPU="false"
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

PORT="${PORT:-8080}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-10}"

# Build command args
ARGS=(
    --port "$PORT"
    --tdt-model ./tdt
    --canary-model ./canary
    --diar-model ./diar_streaming_sortformer_4spk-v2.onnx
    --vad-model ./silero_vad.onnx
    --frontend ./frontend
    --media-dir ./media
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

echo "Starting parakeet-rs (CPU mode) on port $PORT..."

# Port 80 requires root — re-exec with sudo -E if needed
if [ "$PORT" -le 1024 ] && [ "$(id -u)" -ne 0 ]; then
    exec sudo -E "$0" "$@"
fi

LOG_FILE="${LOG_FILE:-/tmp/parakeet-server-${PORT}.log}"
echo "Logging to: $LOG_FILE"
exec ./target/release/parakeet-server "${ARGS[@]}" "$@" >> "$LOG_FILE" 2>&1
