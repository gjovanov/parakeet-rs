#!/bin/bash
#
# Parakeet-rs Run Script
# Starts the WebRTC transcription server in CPU or GPU mode
#
# Usage: ./run.sh [cpu|gpu]
#        Default: gpu
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default mode
MODE="${1:-gpu}"

# Validate mode
if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: ./run.sh [cpu|gpu]"
    exit 1
fi

echo "========================================"
echo "  Parakeet-rs WebRTC Transcriber"
echo "  Mode: $MODE"
echo "========================================"
echo ""

# Check prerequisites
check_prerequisites() {
    # Check .env
    if [ ! -f ".env" ]; then
        echo "Error: .env file not found"
        echo "Run ./init.sh first to download models and create configuration"
        exit 1
    fi

    # Check models
    if [ ! -d "tdt" ] || [ ! -f "tdt/encoder-model.onnx" ]; then
        echo "Error: TDT model not found"
        echo "Run ./init.sh first to download models"
        exit 1
    fi

    if [ ! -f "diar_streaming_sortformer_4spk-v2.onnx" ]; then
        echo "Error: Diarization model not found"
        echo "Run ./init.sh first to download models"
        exit 1
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker not found"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "Error: Docker daemon not running"
        exit 1
    fi

    # Check GPU for GPU mode
    if [ "$MODE" = "gpu" ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            echo "Warning: nvidia-smi not found, GPU may not be available"
        elif ! nvidia-smi &> /dev/null; then
            echo "Warning: NVIDIA GPU not detected, falling back to CPU inside container"
        fi
    fi
}

# Select compose file based on mode
get_compose_file() {
    if [ "$MODE" = "gpu" ]; then
        echo "docker-compose.gpu.yml"
    else
        echo "docker-compose.yml"
    fi
}

# Get container name based on mode
get_container_name() {
    if [ "$MODE" = "gpu" ]; then
        echo "parakeet-transcriber-gpu"
    else
        echo "parakeet-transcriber"
    fi
}

# Get port based on mode
get_port() {
    if [ "$MODE" = "gpu" ]; then
        # Read from .env or default
        grep -E "^GPU_PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "8090"
    else
        grep -E "^PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "8080"
    fi
}

# Start the container
start_container() {
    local compose_file=$(get_compose_file)
    local container_name=$(get_container_name)

    echo "[1/3] Starting container with $compose_file..."

    # Stop existing container if running
    docker compose -f "$compose_file" down 2>/dev/null || true

    # Start container
    docker compose -f "$compose_file" up -d

    echo "  [DONE] Container started"
}

# Wait for container to be ready
wait_for_container() {
    local container_name=$(get_container_name)

    echo "[2/3] Waiting for container to be ready..."

    local retries=30
    while [ $retries -gt 0 ]; do
        if docker exec "$container_name" echo "ready" &> /dev/null; then
            echo "  [DONE] Container ready"
            return 0
        fi
        sleep 1
        retries=$((retries - 1))
    done

    echo "Error: Container failed to start"
    docker compose -f "$(get_compose_file)" logs
    exit 1
}

# Start the transcriber
start_transcriber() {
    local container_name=$(get_container_name)
    local port=$(get_port)

    echo "[3/3] Starting transcriber..."

    # Check if audio file exists for demo
    local audio_cmd=""
    if [ -f "broadcast_2.wav" ]; then
        audio_cmd="ffmpeg -re -stream_loop -1 -i /tmp/broadcast_2.wav -f s16le -ar 16000 -ac 1 - 2>/dev/null |"
        echo "  [INFO] Using broadcast_2.wav for demo audio"
    else
        echo "  [INFO] No demo audio file found"
        echo "  [INFO] Pipe audio to container: docker exec -i $container_name sh -c 'cat | /app/webrtc_transcriber --speedy'"
    fi

    # Start transcriber in background
    if [ -n "$audio_cmd" ]; then
        docker exec -d "$container_name" sh -c "$audio_cmd /app/webrtc_transcriber --speedy 2>&1 | tee /tmp/transcriber.log"
    else
        docker exec -d "$container_name" sh -c "/app/webrtc_transcriber --speedy 2>&1 | tee /tmp/transcriber.log"
    fi

    # Wait for server to start
    sleep 3

    # Check if transcriber is running
    if docker exec "$container_name" pgrep -f webrtc_transcriber &> /dev/null; then
        echo "  [DONE] Transcriber started"
    else
        echo "  [WARN] Transcriber may have failed to start"
        echo "  Check logs: docker exec $container_name cat /tmp/transcriber.log"
    fi
}

# Print status
print_status() {
    local container_name=$(get_container_name)
    local port=$(get_port)

    echo ""
    echo "========================================"
    echo "  Server Running!"
    echo "========================================"
    echo ""
    echo "Access:"
    echo "  WebSocket streaming: http://localhost:$port/"
    echo "  WebRTC streaming:    http://localhost:$port/index-webrtc.html"
    echo ""
    echo "Commands:"
    echo "  View logs:    docker exec $container_name cat /tmp/transcriber.log"
    echo "  Follow logs:  docker exec $container_name tail -f /tmp/transcriber.log"
    echo "  Stop:         docker compose -f $(get_compose_file) down"
    echo "  Restart:      ./run.sh $MODE"
    echo ""

    # Show initial logs
    echo "Initial logs:"
    echo "----------------------------------------"
    sleep 1
    docker exec "$container_name" head -20 /tmp/transcriber.log 2>/dev/null || echo "(waiting for logs...)"
    echo "----------------------------------------"
}

# Main
main() {
    check_prerequisites
    start_container
    wait_for_container
    start_transcriber
    print_status
}

main "$@"
