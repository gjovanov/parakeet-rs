# Deployment

[Architecture](architecture.md) | [API Reference](api.md) | [Transcription Modes](transcription-modes.md) | [Frontend](frontend.md) | [FAB Teletext](fab-teletext.md) | [Testing](testing.md) | [Deployment](deployment.md)

---

## Quick Start

```bash
# 1. Build the server binary
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  cargo build --release --features "server,sortformer"

# 2. Configure environment
cp .env .env.backup   # backup existing
# Edit .env with your model paths, network settings, etc.

# 3. Start the server
./start-server.sh
# Or run directly:
sudo -E ./target/release/parakeet-server \
  --port 80 \
  --tdt-model ./tdt \
  --canary-model ./canary \
  --diar-model ./diar_streaming_sortformer_4spk-v2.onnx \
  --vad-model ./silero_vad.onnx \
  --frontend ./frontend \
  --media-dir ./media
```

The server will be available at `http://localhost:80`.

## CLI Arguments

| Argument | Env Variable | Default | Description |
|----------|-------------|---------|-------------|
| `--port` | `PORT` | `8080` | HTTP/WebSocket server port |
| `--tdt-model` | `TDT_MODEL_PATH` | `./tdt` | Path to TDT model directory |
| `--canary-model` | `CANARY_MODEL_PATH` | `./canary` | Path to Canary 1B model directory |
| `--canary-flash-model` | `CANARY_FLASH_MODEL_PATH` | — | Path to Canary 180M Flash model (optional) |
| `--diar-model` | `DIAR_MODEL_PATH` | `./diar_streaming_sortformer_4spk-v2.onnx` | Diarization model path |
| `--vad-model` | `VAD_MODEL_PATH` | `./silero_vad.onnx` | Silero VAD model path |
| `--frontend` | `FRONTEND_PATH` | `./frontend` | Frontend static files directory |
| `--media-dir` | `MEDIA_DIR` | `./media` | Media files directory |
| `--public-ip` | `PUBLIC_IP` | auto-detected | Public IP for WebRTC ICE candidates |
| `--max-sessions` | `MAX_CONCURRENT_SESSIONS` | `10` | Maximum concurrent sessions |
| `--fab-url` | `FAB_URL` | — | FAB endpoint URL |
| `--fab-enabled` | `FAB_ENABLED` | `true` when URL set | Enable FAB forwarding |
| `--fab-send-type` | `FAB_SEND_TYPE` | `growing` | `"growing"` or `"confirmed"` |
| `--speedy` | `SPEEDY_MODE` | default | Speedy latency mode |
| `--pause-based` | — | — | Pause-based latency mode |
| `--low-latency` | — | — | Low latency mode |
| `--ultra-low-latency` | — | — | Ultra-low latency mode |
| `--extreme-low-latency` | — | — | Extreme-low latency mode |
| `--lookahead` | — | — | Lookahead mode |

## Environment Variables

### GPU Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | — | GPU acceleration: `true`, `cuda`, `tensorrt`, or empty for CPU |
| `INTRA_THREADS` | `4` | ONNX Runtime intra-op threads (CPU: 4, GPU: 2) |
| `INTER_THREADS` | `1` | ONNX Runtime inter-op threads |
| `ORT_DYLIB_PATH` | — | Path to `libonnxruntime.so` (required for `load-dynamic`) |
| `LD_LIBRARY_PATH` | — | Must include CUDA/cuDNN libs for GPU mode |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `80` | Server port (ports ≤1024 require root) |
| `MAX_CONCURRENT_SESSIONS` | `10` | Maximum simultaneous sessions |
| `MAX_PARALLEL_THREADS` | auto | Max threads for parallel modes (auto-detected from RAM/CPU) |
| `MEDIA_DIR` | `./media` | Audio file storage directory |
| `FRONTEND_PATH` | `./frontend` | Static frontend files |
| `RUST_LOG` | `info` | Log level: `error`, `warn`, `info`, `debug`, `trace` |

### WebRTC / ICE Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PUBLIC_IP` | auto-detected | Public IP for NAT traversal |
| `WS_HOST` | `PUBLIC_IP` | WebSocket host override |
| `TURN_SERVER` | — | TURN server URL (e.g. `turn:host:3478`) |
| `TURN_USERNAME` | — | TURN credentials |
| `TURN_PASSWORD` | — | TURN credentials |
| `FORCE_RELAY` | `false` | Force TURN relay mode (skip STUN, skip NAT 1:1 mapping) |

### FAB Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `FAB_URL` | — | FAB endpoint URL |
| `FAB_ENABLED` | `true` when URL set | Global FAB enable/disable |
| `FAB_SEND_TYPE` | `growing` | `"growing"` (cumulative) or `"confirmed"` (finalized only) |

### SRT Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SRT_ENCODER_IP` | — | IP address of the SRT encoder/source |
| `SRT_CHANNELS` | — | JSON array of channel configs (see below) |
| `SRT_LATENCY` | `200000` | SRT latency in microseconds (200ms default) |
| `SRT_RCVBUF` | `2097152` | SRT receive buffer size in bytes (2MB default) |

#### SRT Channel Format

```json
[
  {"name": "ORF1", "port": "24001"},
  {"name": "ORF2", "port": "24002"},
  {"name": "KIDS", "port": "24011"}
]
```

Current ORF channel configuration:

| Channel | Port |
|---------|------|
| ORF1 | 24001 |
| ORF2 | 24002 |
| KIDS | 24011 |
| ORFS | 24004 |
| ORF-B | 24013 |
| ORF-K | 24019 |
| ORF-NOE | 24012 |
| ORF-OOE | 24014 |
| ORF-S | 24015 |
| ORF-ST | 24018 |
| ORF-T | 24016 |
| ORF-V | 24017 |
| ORF-W | 24011 |
| ORF-SI | 24016 |

## Model Setup

Download models from HuggingFace and place them in the configured directories.

### Parakeet TDT 0.6B (English)

Download from [HuggingFace](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx):

```
tdt/
  encoder-model.onnx
  encoder-model.onnx.data
  decoder_joint-model.onnx
  vocab.txt
```

### Canary 1B (Multilingual: en, de, fr, es)

```
canary/
  encoder.onnx
  decoder.onnx
  tokenizer.json
  (additional model files)
```

### Sortformer v2 (Speaker Diarization)

Download from [HuggingFace](https://huggingface.co/altunenes/parakeet-rs/blob/main/diar_streaming_sortformer_4spk-v2.onnx):

```
diar_streaming_sortformer_4spk-v2.onnx
```

### Silero VAD

```
silero_vad.onnx
```

## GPU Support

Build with GPU feature flags:

```bash
# CUDA
cargo build --release --features "server,sortformer,cuda"

# TensorRT
cargo build --release --features "server,sortformer,tensorrt"
```

Set environment for GPU:

```bash
USE_GPU=cuda
INTRA_THREADS=2
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/path/to/onnxruntime/lib
```

Available feature flags: `cuda`, `tensorrt`, `coreml`, `directml`, `rocm`, `openvino`, `webgpu`

## start-server.sh

The `start-server.sh` script is a convenience launcher for CPU mode:

1. Loads `.env` if present
2. Forces CPU mode (`USE_GPU=false`)
3. Sets `ORT_DYLIB_PATH` to `/usr/local/lib/libonnxruntime.so`
4. Builds CLI arguments from environment variables
5. Re-executes with `sudo -E` if port ≤1024 and not already root
6. Runs `./target/release/parakeet-server` with all arguments

```bash
# Basic usage
./start-server.sh

# With extra arguments
./start-server.sh --lookahead

# Run in background with logging
sudo bash -c './start-server.sh >> /tmp/parakeet-server.log 2>&1 &'
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| `Failed to load library libonnxruntime_providers_shared.so` | Set `LD_LIBRARY_PATH` to include `target/release` directory |
| `libcudnn.so.9: cannot open` | Add CUDA lib paths to `LD_LIBRARY_PATH` |
| `USE_GPU=true but no GPU features compiled` | Rebuild with `--features cuda` or `--features tensorrt` |
| `Permission denied` on port 80 | Run with `sudo` or use a port > 1024 |
| Inference is slow (>1 second) | Verify `USE_GPU=cuda` is set; check logs for GPU provider messages |
| Server respawns after kill | Use `sudo killall -9 parakeet-server` and verify with `pgrep` |
