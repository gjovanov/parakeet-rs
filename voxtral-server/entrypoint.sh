#!/bin/bash
set -e

MODEL_ID="${VLLM_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
VLLM_PORT="${VOXTRAL_VLLM_PORT:-8002}"
VLLM_GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
APP_PORT="${VOXTRAL_PORT:-8091}"

echo "=== Voxtral Container ==="
echo "  Model:    $MODEL_ID"
echo "  vLLM:     port $VLLM_PORT (gpu_util=$VLLM_GPU_UTIL, max_len=$VLLM_MAX_LEN)"
echo "  App:      port $APP_PORT"
echo ""

# --- Start vLLM in background ---
echo "[entrypoint] Starting vLLM..."
vllm serve "$MODEL_ID" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len "$VLLM_MAX_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_UTIL" \
    &
VLLM_PID=$!

# --- Wait for vLLM health ---
echo "[entrypoint] Waiting for vLLM to become ready..."
TIMEOUT=180
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[entrypoint] vLLM ready after ${ELAPSED}s"
        break
    fi
    # Check if vLLM crashed
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM process exited"
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "[entrypoint] ERROR: vLLM did not become ready within ${TIMEOUT}s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# --- Start voxtral-server ---
echo "[entrypoint] Starting voxtral-server on port $APP_PORT..."
export VOXTRAL_VLLM_URL="ws://localhost:${VLLM_PORT}/v1/realtime"
export VOXTRAL_PORT="$APP_PORT"

python3 -m uvicorn voxtral_server.main:app \
    --host 0.0.0.0 \
    --port "$APP_PORT" \
    --log-level warning \
    &
APP_PID=$!

echo "[entrypoint] Both services running (vLLM=$VLLM_PID, app=$APP_PID)"

# --- Graceful shutdown ---
shutdown() {
    echo "[entrypoint] Shutting down..."
    kill $APP_PID 2>/dev/null
    kill $VLLM_PID 2>/dev/null
    wait $APP_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    echo "[entrypoint] Stopped"
    exit 0
}
trap shutdown SIGTERM SIGINT

# --- Monitor both processes ---
while true; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[entrypoint] vLLM exited unexpectedly"
        kill $APP_PID 2>/dev/null
        exit 1
    fi
    if ! kill -0 $APP_PID 2>/dev/null; then
        echo "[entrypoint] voxtral-server exited unexpectedly"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done
