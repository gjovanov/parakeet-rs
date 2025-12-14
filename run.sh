#!/bin/bash
#
# Parakeet-rs Run Script
# Starts the WebRTC transcription server
#
# Usage: ./run.sh [runtime] [mode] [file]
#        runtime: cpu|gpu (default: gpu)
#        mode: speedy|low-latency|ultra-low-latency|extreme-low-latency|pause-based|lookahead|vad-speedy|vad-pause-based|vad-sliding-window|asr|parallel|pause-parallel (default: speedy)
#        file: audio file path (default: broadcast_2.wav)
#
# Examples:
#        ./run.sh                              # GPU + speedy + broadcast_2.wav
#        ./run.sh gpu speedy                   # GPU + speedy + broadcast_2.wav
#        ./run.sh gpu speedy myaudio.wav       # GPU + speedy + custom audio
#        ./run.sh cpu low-latency podcast.wav  # CPU + low-latency + custom audio
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
RUNTIME="${1:-gpu}"
MODE="${2:-speedy}"
AUDIO_FILE="${3:-broadcast_2.wav}"

# Valid options
VALID_RUNTIMES="cpu gpu"
VALID_MODES="speedy low-latency ultra-low-latency extreme-low-latency pause-based lookahead vad-speedy vad-pause-based vad-sliding-window asr parallel pause-parallel"

# Validate runtime
if [[ ! " $VALID_RUNTIMES " =~ " $RUNTIME " ]]; then
    echo "Error: Invalid runtime '$RUNTIME'"
    echo "Valid runtimes: $VALID_RUNTIMES"
    echo ""
    echo "Usage: ./run.sh [runtime] [mode]"
    exit 1
fi

# Validate mode
if [[ ! " $VALID_MODES " =~ " $MODE " ]]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Valid modes: $VALID_MODES"
    echo ""
    echo "Usage: ./run.sh [runtime] [mode]"
    exit 1
fi

# Convert mode to CLI flag
get_mode_flag() {
    case "$MODE" in
        speedy)              echo "--speedy" ;;
        low-latency)         echo "--low-latency" ;;
        ultra-low-latency)   echo "--ultra-low-latency" ;;
        extreme-low-latency) echo "--extreme-low-latency" ;;
        pause-based)         echo "--pause-based" ;;
        lookahead)           echo "--lookahead" ;;
        vad-speedy)          echo "--vad-speedy" ;;
        vad-pause-based)     echo "--vad-pause-based" ;;
        vad-sliding-window)  echo "--vad-sliding-window" ;;
        asr)                 echo "--asr" ;;
        parallel)            echo "--parallel" ;;
        pause-parallel)      echo "--pause-parallel" ;;
        *)                   echo "--speedy" ;;
    esac
}

echo "========================================"
echo "  Parakeet-rs WebRTC Transcriber"
echo "  Runtime: $RUNTIME"
echo "  Mode:    $MODE"
echo "  Audio:   $AUDIO_FILE"
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

    if [ ! -f "silero_vad.onnx" ]; then
        echo "Error: VAD model not found"
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

    # Check GPU for GPU runtime
    if [ "$RUNTIME" = "gpu" ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            echo "Warning: nvidia-smi not found, GPU may not be available"
        elif ! nvidia-smi &> /dev/null; then
            echo "Warning: NVIDIA GPU not detected, falling back to CPU inside container"
        fi
    fi
}

# Select compose file based on runtime
get_compose_file() {
    if [ "$RUNTIME" = "gpu" ]; then
        echo "docker-compose.gpu.yml"
    else
        echo "docker-compose.yml"
    fi
}

# Get container name based on runtime
get_container_name() {
    if [ "$RUNTIME" = "gpu" ]; then
        echo "parakeet-transcriber-gpu"
    else
        echo "parakeet-transcriber"
    fi
}

# Get port based on runtime
get_port() {
    if [ "$RUNTIME" = "gpu" ]; then
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
    local mode_flag=$(get_mode_flag)

    echo "[3/3] Starting transcriber with $mode_flag..."

    # Check if audio file exists
    local audio_cmd=""
    if [ -f "$AUDIO_FILE" ]; then
        # Get the filename for container path
        local audio_basename=$(basename "$AUDIO_FILE")
        local container_audio_path="/tmp/$audio_basename"

        # Copy audio file to container if not already mounted
        if ! docker exec "$container_name" test -f "$container_audio_path" 2>/dev/null; then
            echo "  [INFO] Copying $AUDIO_FILE to container..."
            docker cp "$AUDIO_FILE" "$container_name:$container_audio_path"
        fi

        audio_cmd="ffmpeg -re -stream_loop -1 -i $container_audio_path -f s16le -ar 16000 -ac 1 - 2>/dev/null |"
        echo "  [INFO] Using $AUDIO_FILE for audio input"
    else
        echo "  [WARN] Audio file not found: $AUDIO_FILE"
        echo "  [INFO] Starting without audio input"
        echo "  [INFO] Pipe audio to container: docker exec -i $container_name sh -c 'cat | /app/webrtc_transcriber $mode_flag'"
    fi

    # Start transcriber in background
    if [ -n "$audio_cmd" ]; then
        docker exec -d "$container_name" sh -c "$audio_cmd /app/webrtc_transcriber $mode_flag 2>&1 | tee /tmp/transcriber.log"
    else
        docker exec -d "$container_name" sh -c "/app/webrtc_transcriber $mode_flag 2>&1 | tee /tmp/transcriber.log"
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
    local mode_flag=$(get_mode_flag)

    echo ""
    echo "========================================"
    echo "  Server Running!"
    echo "========================================"
    echo ""
    echo "Configuration:"
    echo "  Runtime: $RUNTIME"
    echo "  Mode:    $MODE ($mode_flag)"
    echo "  Audio:   $AUDIO_FILE"
    echo "  Port:    $port"
    echo ""
    echo "Access:"
    echo "  Web UI: http://localhost:$port/"
    echo ""
    echo "Commands:"
    echo "  View logs:    docker exec $container_name cat /tmp/transcriber.log"
    echo "  Follow logs:  docker exec $container_name tail -f /tmp/transcriber.log"
    echo "  Stop:         docker compose -f $(get_compose_file) down"
    echo "  Restart:      ./run.sh $RUNTIME $MODE"
    echo ""

    # Show initial logs
    echo "Initial logs:"
    echo "----------------------------------------"
    sleep 1
    docker exec "$container_name" head -20 /tmp/transcriber.log 2>/dev/null || echo "(waiting for logs...)"
    echo "----------------------------------------"
}

# Show help
show_help() {
    echo "Parakeet-rs Run Script"
    echo ""
    echo "Usage: ./run.sh [runtime] [mode] [file]"
    echo ""
    echo "Arguments:"
    echo "  runtime    Execution runtime (default: gpu)"
    echo "             cpu  - CPU-only execution"
    echo "             gpu  - GPU-accelerated execution (requires NVIDIA GPU)"
    echo ""
    echo "  mode       Transcription mode (default: speedy)"
    echo ""
    echo "             Standard modes (continuous processing):"
    echo "             speedy              - Best balance of latency and quality (~0.3-1.5s)"
    echo "             pause-based         - Conservative pause-based (~0.5-2.0s)"
    echo "             low-latency         - Time-based, fixed latency (~3.5s)"
    echo "             ultra-low-latency   - Faster time-based (~2.5s)"
    echo "             extreme-low-latency - Fastest response (~1.3s)"
    echo "             lookahead           - Best quality with future context (~1.0-3.0s)"
    echo ""
    echo "             VAD-triggered modes (utterance-based):"
    echo "             vad-speedy          - Short pause detection (~0.3s pause)"
    echo "             vad-pause-based     - Longer pause detection (~0.7s pause)"
    echo "             vad-sliding-window  - Multi-segment buffered transcription"
    echo ""
    echo "             Other modes:"
    echo "             asr                 - Pure streaming without VAD"
    echo "             parallel            - Multi-threaded sliding window inference"
    echo "             pause-parallel      - Pause-triggered parallel with ordered output"
    echo ""
    echo "  file       Audio file path (default: broadcast_2.wav)"
    echo "             Path to a WAV file for audio input"
    echo "             The file will be copied to the container and looped"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                              # GPU + speedy + broadcast_2.wav"
    echo "  ./run.sh gpu                          # GPU + speedy + broadcast_2.wav"
    echo "  ./run.sh cpu                          # CPU + speedy + broadcast_2.wav"
    echo "  ./run.sh gpu speedy                   # GPU + speedy + broadcast_2.wav"
    echo "  ./run.sh gpu speedy myaudio.wav       # GPU + speedy + custom audio"
    echo "  ./run.sh cpu low-latency podcast.wav  # CPU + low-latency + custom audio"
    echo "  ./run.sh gpu lookahead interview.wav  # GPU + lookahead + custom audio"
    echo ""
}

# Main
main() {
    # Show help if requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_help
        exit 0
    fi

    check_prerequisites
    start_container
    wait_for_container
    start_transcriber
    print_status
}

main "$@"
