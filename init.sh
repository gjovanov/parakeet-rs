#!/bin/bash
#
# Parakeet-rs Initialization Script
# Sets up system deps, ONNX Runtime (CPU or GPU), models, and builds the binary.
# Idempotent — safe to re-run; skips steps that are already done.
#
# Usage:
#   ./init.sh                  # CPU mode (default)
#   ./init.sh --gpu            # GPU mode (CUDA, downloads ORT GPU + CUDA 13 build)
#   ./init.sh --whisper        # CPU + Whisper GGML models
#   ./init.sh --gpu --whisper  # GPU + Whisper GGML models
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ORT_VERSION="1.24.3"
ORT_CPU_DIR="./ort-cpu"
ORT_GPU_DIR="./ort-gpu"

# Parse args
GPU_MODE=true
WHISPER_MODE=true
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU_MODE=true ;;
        --whisper) WHISPER_MODE=true ;;
    esac
done

if [ "$GPU_MODE" = true ]; then
    echo "========================================"
    echo "  Parakeet-rs GPU Initialization"
    echo "========================================"
else
    echo "========================================"
    echo "  Parakeet-rs CPU Initialization"
    echo "========================================"
fi
echo ""

# ─── 1. System dependencies ──────────────────────────────────────────

install_system_deps() {
    local pkgs=(cmake clang pkg-config libssl-dev libopus-dev libasound2-dev ffmpeg curl)
    local missing=()

    for pkg in "${pkgs[@]}"; do
        if ! dpkg -s "$pkg" &>/dev/null; then
            missing+=("$pkg")
        fi
    done

    if [ ${#missing[@]} -eq 0 ]; then
        echo "[1/6] System dependencies already installed"
    else
        echo "[1/6] Installing system dependencies: ${missing[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${missing[@]}"
    fi
}

# ─── 2. ONNX Runtime ─────────────────────────────────────────────────

install_onnxruntime_cpu() {
    if [ -f "$ORT_CPU_DIR/libonnxruntime.so" ]; then
        echo "[2/6] ONNX Runtime CPU $ORT_VERSION already installed in $ORT_CPU_DIR/"
        return 0
    fi

    echo "[2/6] Installing ONNX Runtime CPU $ORT_VERSION..."
    local arch
    arch=$(uname -m)
    [ "$arch" != "aarch64" ] && arch="x64"

    local tarball="onnxruntime-linux-${arch}-${ORT_VERSION}.tgz"
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${tarball}"
    local tmpdir
    tmpdir=$(mktemp -d)

    curl -L --progress-bar "$url" -o "$tmpdir/$tarball"
    tar -xzf "$tmpdir/$tarball" -C "$tmpdir"

    local extract_dir
    extract_dir=$(find "$tmpdir" -maxdepth 1 -type d -name "onnxruntime-*" | head -1)

    mkdir -p "$ORT_CPU_DIR"
    cp "$extract_dir"/lib/libonnxruntime.so* "$ORT_CPU_DIR/"

    # Create symlink for load-dynamic
    local so_file
    so_file=$(ls "$ORT_CPU_DIR"/libonnxruntime.so.*.*.* 2>/dev/null | head -1)
    if [ -n "$so_file" ]; then
        ln -sf "$(basename "$so_file")" "$ORT_CPU_DIR/libonnxruntime.so"
    fi

    rm -rf "$tmpdir"
    echo "  [DONE] ONNX Runtime CPU installed to $ORT_CPU_DIR/"
    ls -lh "$ORT_CPU_DIR/"
}

install_onnxruntime_gpu() {
    if [ -f "$ORT_GPU_DIR/libonnxruntime_providers_cuda.so" ]; then
        echo "[2/6] ONNX Runtime GPU $ORT_VERSION already installed in $ORT_GPU_DIR/"
        return 0
    fi

    echo "[2/6] Installing ONNX Runtime GPU $ORT_VERSION (CUDA 13)..."

    # Check for CUDA toolkit
    if ! command -v nvidia-smi &>/dev/null; then
        echo "  [WARN] nvidia-smi not found. GPU acceleration requires NVIDIA drivers."
        echo "  [WARN] Falling back to CPU-only. Install NVIDIA drivers and re-run with --gpu."
        install_onnxruntime_cpu
        return 0
    fi

    local arch
    arch=$(uname -m)
    [ "$arch" != "aarch64" ] && arch="x64"

    # Try CUDA 13 build first (for Blackwell/sm_120), fall back to regular GPU build
    local tarball="onnxruntime-linux-${arch}-gpu_cuda13-${ORT_VERSION}.tgz"
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${tarball}"
    local tmpdir
    tmpdir=$(mktemp -d)

    echo "  Downloading ORT GPU (CUDA 13)..."
    if ! curl -L --fail --progress-bar "$url" -o "$tmpdir/$tarball" 2>/dev/null; then
        echo "  [INFO] CUDA 13 build not available, trying standard GPU build..."
        tarball="onnxruntime-linux-${arch}-gpu-${ORT_VERSION}.tgz"
        url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${tarball}"
        curl -L --progress-bar "$url" -o "$tmpdir/$tarball"
    fi

    tar -xzf "$tmpdir/$tarball" -C "$tmpdir"

    local extract_dir
    extract_dir=$(find "$tmpdir" -maxdepth 1 -type d -name "onnxruntime-*" | head -1)

    mkdir -p "$ORT_GPU_DIR"
    cp "$extract_dir"/lib/libonnxruntime.so* "$ORT_GPU_DIR/"
    cp "$extract_dir"/lib/libonnxruntime_providers_*.so "$ORT_GPU_DIR/"

    # Create symlink for load-dynamic
    local so_file
    so_file=$(ls "$ORT_GPU_DIR"/libonnxruntime.so.*.*.* 2>/dev/null | head -1)
    if [ -n "$so_file" ]; then
        ln -sf "$(basename "$so_file")" "$ORT_GPU_DIR/libonnxruntime.so"
    fi

    rm -rf "$tmpdir"
    echo "  [DONE] ONNX Runtime GPU installed to $ORT_GPU_DIR/"
    ls -lh "$ORT_GPU_DIR/"
}

# ─── 3. Download models ──────────────────────────────────────────────

load_hf_token() {
    if [ -f ".env" ]; then
        HF_TOKEN=$(grep -E "^HF_TOKEN=" .env 2>/dev/null | cut -d'=' -f2 || echo "")
    fi
    HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
}

download_file() {
    local url="$1"
    local output="$2"

    if [ -f "$output" ]; then
        echo "  [SKIP] $output already exists"
        return 0
    fi

    echo "  [DOWN] $output"
    local auth_args=()
    if [ -n "$HF_TOKEN" ]; then
        auth_args=(-H "Authorization: Bearer $HF_TOKEN")
    fi
    curl -L --progress-bar "${auth_args[@]}" "$url" -o "$output"
    echo "  [DONE] $output"
}

download_models() {
    load_hf_token

    if [ -n "$HF_TOKEN" ]; then
        echo "[3/6] Downloading models (with HF auth)..."
    else
        echo "[3/6] Downloading models..."
    fi

    # TDT model (English, fast)
    echo "  --- TDT 0.6B (English ASR) ---"
    mkdir -p tdt
    local tdt_url="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"
    download_file "$tdt_url/encoder-model.onnx" "tdt/encoder-model.onnx"
    download_file "$tdt_url/encoder-model.onnx.data" "tdt/encoder-model.onnx.data"
    download_file "$tdt_url/decoder_joint-model.onnx" "tdt/decoder_joint-model.onnx"
    download_file "$tdt_url/vocab.txt" "tdt/vocab.txt"

    # Sortformer diarization
    echo "  --- Sortformer (Speaker Diarization, up to 4 speakers) ---"
    download_file \
        "https://huggingface.co/altunenes/parakeet-rs/resolve/main/diar_streaming_sortformer_4spk-v2.onnx" \
        "diar_streaming_sortformer_4spk-v2.onnx"

    # Canary 1B INT8 (Multilingual)
    echo "  --- Canary 1B INT8 (Multilingual ASR: en/de/fr/es) ---"
    mkdir -p canary
    local canary_url="https://huggingface.co/istupakov/canary-1b-v2-onnx/resolve/main"
    download_file "$canary_url/encoder-model.int8.onnx" "canary/encoder-model.int8.onnx"
    download_file "$canary_url/decoder-model.int8.onnx" "canary/decoder-model.int8.onnx"
    download_file "$canary_url/vocab.txt" "canary/vocab.txt"
    download_file "$canary_url/config.json" "canary/config.json"

    # Whisper models (optional, only if --whisper flag)
    if [ "$WHISPER_MODE" = true ]; then
        echo "  --- Whisper GGML Models ---"
        mkdir -p whisper
        local whisper_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

        # large-v3-turbo (809M params, best speed/quality for German)
        download_file "$whisper_url/ggml-large-v3-turbo.bin" "whisper/ggml-large-v3-turbo.bin"

        # large-v3-turbo Q5_0 quantized (smaller, slightly lower quality)
        download_file "$whisper_url/ggml-large-v3-turbo-q5_0.bin" "whisper/ggml-large-v3-turbo-q5_0.bin"

        # large-v3 (1.55B params, reference accuracy)
        download_file "$whisper_url/ggml-large-v3.bin" "whisper/ggml-large-v3.bin"

        # medium (769M params, lower-bound reference)
        download_file "$whisper_url/ggml-medium.bin" "whisper/ggml-medium.bin"
    fi
}

# ─── 4. Build binary ─────────────────────────────────────────────────

build_binary() {
    local bin="target/release/parakeet-server"
    local features="server,sortformer"

    if [ "$GPU_MODE" = true ]; then
        features="server,sortformer,cuda"
    fi

    if [ "$WHISPER_MODE" = true ]; then
        if [ "$GPU_MODE" = true ]; then
            features="${features},whisper-cuda"
        else
            features="${features},whisper"
        fi
    fi

    # Always rebuild if source changed (let cargo decide)
    echo "[4/6] Building parakeet-server (features: $features)..."
    if [ "$GPU_MODE" = true ]; then
        export ORT_DYLIB_PATH="$ORT_GPU_DIR/libonnxruntime.so"

        # Auto-detect CUDA toolkit for whisper-cuda (needs nvcc matching GPU arch)
        # Prefer highest version available (e.g. CUDA 13 for Blackwell/sm_120)
        if [[ "$features" == *"whisper-cuda"* ]]; then
            local cuda_dir=""
            for d in /usr/local/cuda-13.* /usr/local/cuda-13 /usr/local/cuda-12.* /usr/local/cuda-12 /usr/local/cuda; do
                if [ -f "$d/bin/nvcc" ]; then
                    cuda_dir="$d"
                    break
                fi
            done
            if [ -n "$cuda_dir" ]; then
                echo "  [INFO] Using CUDA toolkit: $cuda_dir (for whisper-cuda)"
                export CUDACXX="$cuda_dir/bin/nvcc"
                export CUDA_PATH="$cuda_dir"
                export CMAKE_CUDA_COMPILER="$cuda_dir/bin/nvcc"
            fi
        fi
    else
        export ORT_DYLIB_PATH="$ORT_CPU_DIR/libonnxruntime.so"
    fi
    cargo build --release --bin parakeet-server --features "$features"
    echo "  [DONE] Built $bin"
}

# ─── 5. Create .env ──────────────────────────────────────────────────

create_env() {
    if [ -f ".env" ]; then
        echo "[5/6] .env already exists (skipping)"
        return 0
    fi

    echo "[5/6] Creating .env..."

    local public_ip=""
    public_ip=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "")

    if [ "$GPU_MODE" = true ]; then
        cat > .env << 'EOF'
# Parakeet-rs Configuration (GPU mode)
USE_GPU=cuda
ORT_DYLIB_PATH=./ort-gpu/libonnxruntime.so
INTRA_THREADS=2
INTER_THREADS=1
TDT_MODEL_PATH=./tdt
CANARY_MODEL_PATH=./canary
DIAR_MODEL_PATH=./diar_streaming_sortformer_4spk-v2.onnx
PORT=8080
PUBLIC_IP=
MAX_CONCURRENT_SESSIONS=10
HF_TOKEN=
EOF
    else
        cat > .env << 'EOF'
# Parakeet-rs Configuration (CPU mode)
USE_GPU=false
ORT_DYLIB_PATH=./ort-cpu/libonnxruntime.so
INTRA_THREADS=4
INTER_THREADS=1
TDT_MODEL_PATH=./tdt
CANARY_MODEL_PATH=./canary
DIAR_MODEL_PATH=./diar_streaming_sortformer_4spk-v2.onnx
PORT=8080
PUBLIC_IP=
MAX_CONCURRENT_SESSIONS=10
HF_TOKEN=
EOF
    fi

    # Append Whisper model path if whisper mode enabled
    if [ "$WHISPER_MODE" = true ]; then
        echo "WHISPER_MODEL_PATH=./whisper/ggml-large-v3-turbo.bin" >> .env
    fi

    if [ -n "$public_ip" ]; then
        sed -i "s/^PUBLIC_IP=$/PUBLIC_IP=$public_ip/" .env
        echo "  [INFO] Detected public IP: $public_ip"
    fi

    echo "  [DONE] Created .env"
}

# ─── 6. Verify ────────────────────────────────────────────────────────

verify_setup() {
    echo "[6/6] Verifying setup..."

    local ok=true

    # Check binary
    if [ -f "target/release/parakeet-server" ]; then
        echo "  [OK] parakeet-server binary"
    else
        echo "  [FAIL] parakeet-server binary not found"
        ok=false
    fi

    # Check ORT
    if [ "$GPU_MODE" = true ]; then
        if [ -f "$ORT_GPU_DIR/libonnxruntime_providers_cuda.so" ]; then
            echo "  [OK] ORT GPU (CUDA) library"
        else
            echo "  [FAIL] ORT GPU library not found"
            ok=false
        fi
    else
        if [ -f "$ORT_CPU_DIR/libonnxruntime.so" ]; then
            echo "  [OK] ORT CPU library"
        else
            echo "  [FAIL] ORT CPU library not found"
            ok=false
        fi
    fi

    # Check models
    for model_dir in tdt canary; do
        if [ -d "$model_dir" ] && [ "$(ls -A "$model_dir")" ]; then
            echo "  [OK] $model_dir model"
        else
            echo "  [FAIL] $model_dir model missing"
            ok=false
        fi
    done

    if [ -f "diar_streaming_sortformer_4spk-v2.onnx" ]; then
        echo "  [OK] diar_streaming_sortformer_4spk-v2.onnx"
    else
        echo "  [FAIL] diar_streaming_sortformer_4spk-v2.onnx missing"
        ok=false
    fi

    if [ "$WHISPER_MODE" = true ]; then
        if [ -d "whisper" ] && ls whisper/*.bin &>/dev/null; then
            echo "  [OK] whisper models"
            for f in whisper/*.bin; do
                echo "       $(ls -lh "$f")"
            done
        else
            echo "  [FAIL] whisper models missing"
            ok=false
        fi
    fi

    if [ "$GPU_MODE" = true ] && command -v nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')
        echo "  [INFO] GPU: $gpu_name"
    fi

    if [ "$ok" = false ]; then
        echo ""
        echo "  [WARN] Some checks failed — review above."
    fi
}

# ─── Main ─────────────────────────────────────────────────────────────

install_system_deps

if [ "$GPU_MODE" = true ]; then
    install_onnxruntime_gpu
else
    install_onnxruntime_cpu
fi

download_models
build_binary
create_env
verify_setup

echo ""
echo "========================================"
echo "  Initialization Complete!"
echo "========================================"
echo ""
if [ "$GPU_MODE" = true ]; then
    echo "Mode: GPU (CUDA)"
else
    echo "Mode: CPU"
fi
echo ""
echo "Next steps:"
echo "  1. Review .env (edit PUBLIC_IP, HF_TOKEN, TURN settings, etc.)"
echo "  2. Run: ./start-server.sh"
echo ""
