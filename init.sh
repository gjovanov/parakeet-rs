#!/bin/bash
#
# Parakeet-rs CPU Initialization Script
# Sets up system deps, ONNX Runtime, models, and builds the binary.
# Idempotent — safe to re-run; skips steps that are already done.
#
# Usage: ./init.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ORT_VERSION="1.22.0"
ORT_LIB="/usr/local/lib/libonnxruntime.so"

echo "========================================"
echo "  Parakeet-rs CPU Initialization"
echo "========================================"
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
        echo "[1/5] System dependencies already installed"
    else
        echo "[1/5] Installing system dependencies: ${missing[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${missing[@]}"
    fi
}

# ─── 2. ONNX Runtime CPU ─────────────────────────────────────────────

install_onnxruntime() {
    if [ -f "$ORT_LIB" ]; then
        echo "[2/5] ONNX Runtime CPU $ORT_VERSION already installed"
        return 0
    fi

    echo "[2/5] Installing ONNX Runtime CPU $ORT_VERSION..."
    local arch
    arch=$(uname -m)
    if [ "$arch" = "aarch64" ]; then
        arch="aarch64"
    else
        arch="x64"
    fi

    local tarball="onnxruntime-linux-${arch}-${ORT_VERSION}.tgz"
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${tarball}"
    local tmpdir
    tmpdir=$(mktemp -d)

    curl -L --progress-bar "$url" -o "$tmpdir/$tarball"
    tar -xzf "$tmpdir/$tarball" -C "$tmpdir"

    local extract_dir="$tmpdir/onnxruntime-linux-${arch}-${ORT_VERSION}"
    sudo cp "$extract_dir"/lib/libonnxruntime.so* /usr/local/lib/
    sudo cp -r "$extract_dir"/include/* /usr/local/include/ 2>/dev/null || true
    sudo ldconfig

    rm -rf "$tmpdir"
    echo "  [DONE] ONNX Runtime installed to /usr/local/lib/"
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
        echo "[3/5] Downloading models (with HF auth)..."
    else
        echo "[3/5] Downloading models..."
    fi

    # TDT model
    echo "  --- TDT (Multilingual Transcription) ---"
    mkdir -p tdt
    local tdt_url="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"
    download_file "$tdt_url/encoder-model.onnx" "tdt/encoder-model.onnx"
    download_file "$tdt_url/encoder-model.onnx.data" "tdt/encoder-model.onnx.data"
    download_file "$tdt_url/decoder_joint-model.onnx" "tdt/decoder_joint-model.onnx"
    download_file "$tdt_url/vocab.txt" "tdt/vocab.txt"

    # Sortformer diarization
    echo "  --- Sortformer (Speaker Diarization) ---"
    download_file \
        "https://huggingface.co/altunenes/parakeet-rs/resolve/main/diar_streaming_sortformer_4spk-v2.onnx" \
        "diar_streaming_sortformer_4spk-v2.onnx"

    # Canary INT8
    echo "  --- Canary INT8 (Multilingual ASR) ---"
    mkdir -p canary
    local canary_url="https://huggingface.co/istupakov/canary-1b-v2-onnx/resolve/main"
    download_file "$canary_url/encoder-model.int8.onnx" "canary/encoder-model.int8.onnx"
    download_file "$canary_url/decoder-model.int8.onnx" "canary/decoder-model.int8.onnx"
    download_file "$canary_url/vocab.txt" "canary/vocab.txt"
    download_file "$canary_url/config.json" "canary/config.json"

    # Silero VAD
    echo "  --- Silero VAD (Voice Activity Detection) ---"
    download_file \
        "https://huggingface.co/snakers4/silero-vad/resolve/master/files/silero_vad.onnx" \
        "silero_vad.onnx"
}

# ─── 4. Build binary ─────────────────────────────────────────────────

build_binary() {
    local bin="target/release/examples/webrtc_transcriber"

    if [ -f "$bin" ]; then
        echo "[4/5] Binary already built at $bin"
        return 0
    fi

    echo "[4/5] Building webrtc_transcriber (release, sortformer)..."
    export ORT_DYLIB_PATH="$ORT_LIB"
    cargo build --release --example webrtc_transcriber --features sortformer
    echo "  [DONE] Built $bin"
}

# ─── 5. Create .env ──────────────────────────────────────────────────

create_env() {
    if [ -f ".env" ]; then
        echo "[5/5] .env already exists (skipping)"
        return 0
    fi

    echo "[5/5] Creating .env with CPU defaults..."

    local public_ip=""
    public_ip=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "")

    cat > .env << 'EOF'
# Parakeet-rs Configuration (CPU mode)
USE_GPU=false
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
PORT=80
PUBLIC_IP=
MAX_CONCURRENT_SESSIONS=10
SPEEDY_MODE=true
RUST_LOG=info
HF_TOKEN=
EOF

    if [ -n "$public_ip" ]; then
        sed -i "s/^PUBLIC_IP=$/PUBLIC_IP=$public_ip/" .env
        echo "  [INFO] Detected public IP: $public_ip"
    fi

    echo "  [DONE] Created .env"
}

# ─── Main ─────────────────────────────────────────────────────────────

install_system_deps
install_onnxruntime
download_models
build_binary
create_env

echo ""
echo "========================================"
echo "  Initialization Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Review .env (edit PUBLIC_IP, HF_TOKEN, etc.)"
echo "  2. Run: ./start-server.sh"
echo ""
