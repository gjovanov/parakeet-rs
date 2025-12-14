# Deployment Guide

> **Navigation**: [Index](./README.md) | [Architecture](./architecture.md) | [API Reference](./api-reference.md) | [Latency Modes](./latency-modes.md) | [Frontend](./frontend.md) | Deployment

## Configuration

### Environment Variables

#### Server Configuration

| Variable | CLI Arg | Default | Description |
|----------|---------|---------|-------------|
| `PORT` | `--port` | `8080` | HTTP/WebSocket server port |
| `PUBLIC_IP` | `--public-ip` | auto-detect | Public IP for WebRTC ICE candidates |
| `FRONTEND_PATH` | `--frontend` | `./frontend` | Path to frontend static files |
| `MEDIA_DIR` | `--media-dir` | `./media` | Media directory for audio files |
| `MAX_CONCURRENT_SESSIONS` | `--max-sessions` | `10` | Maximum concurrent sessions |
| `WS_HOST` | - | auto-detect | WebSocket host for frontend config |
| `RUST_LOG` | - | `info` | Log level (error, warn, info, debug, trace) |

#### Model Configuration

| Variable | CLI Arg | Default | Description |
|----------|---------|---------|-------------|
| `TDT_MODEL_PATH` | `--tdt-model` | `./tdt` | Path to TDT model directory |
| `DIAR_MODEL_PATH` | `--diar-model` | `*.onnx` | Path to Sortformer diarization model |
| `VAD_MODEL_PATH` | `--vad-model` | `silero_vad.onnx` | Path to Silero VAD model |

#### TURN Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TURN_SERVER` | - | TURN server URL (e.g., `turns:server.com:443`) |
| `TURN_USERNAME` | - | TURN authentication username |
| `TURN_PASSWORD` | - | TURN authentication password |

#### GPU Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Enable GPU acceleration (`true`, `cuda`, `tensorrt`, `rocm`) |
| `INTRA_THREADS` | `4` | ONNX Runtime intra-op parallelism |
| `INTER_THREADS` | `1` | ONNX Runtime inter-op parallelism |

**Supported GPU Providers:**
- `cuda` - NVIDIA CUDA (requires `--features cuda`)
- `tensorrt` - NVIDIA TensorRT (requires `--features tensorrt`)
- `rocm` - AMD ROCm (requires `--features rocm`)
- `coreml` - Apple CoreML (requires `--features coreml`)
- `directml` - Windows DirectML (requires `--features directml`)
- `openvino` - Intel OpenVINO (requires `--features openvino`)

### Example .env File

```bash
# Server
PORT=8080
PUBLIC_IP=203.0.113.50
WS_HOST=transcribe.example.com
MEDIA_DIR=./media
MAX_CONCURRENT_SESSIONS=10

# Models
TDT_MODEL_PATH=/app/models/tdt
CANARY_MODEL_PATH=/app/models/canary
DIAR_MODEL_PATH=/app/models/diar_streaming_sortformer_4spk-v2.onnx
VAD_MODEL_PATH=/app/models/silero_vad.onnx

# GPU (optional - requires compilation with --features cuda)
USE_GPU=false
INTRA_THREADS=4
INTER_THREADS=1

# TURN (for NAT traversal)
TURN_SERVER=turns:coturn.example.com:443?transport=udp
TURN_USERNAME=myuser
TURN_PASSWORD=mysecret

# Logging
RUST_LOG=info
```

### CLI Arguments

```bash
cargo run --release --example webrtc_transcriber --features sortformer -- \
  --port 8080 \
  --tdt-model ./tdt \
  --diar-model ./diar_streaming_sortformer_4spk-v2.onnx \
  --vad-model ./silero_vad.onnx \
  --frontend ./frontend \
  --media-dir ./media \
  --public-ip 203.0.113.50 \
  --max-sessions 10
```

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone and enter directory
git clone https://github.com/your-org/parakeet-rs.git
cd parakeet-rs

# 2. Download models
# - TDT model → ./tdt/
# - Diarization model → ./diar_streaming_sortformer_4spk-v2.onnx
# - VAD model → ./silero_vad.onnx

# 3. Create media directory
mkdir -p media

# 4. Create .env file
cat > .env << EOF
PUBLIC_IP=$(curl -s ifconfig.me)
TDT_MODEL_PATH=/app/models/tdt
DIAR_MODEL_PATH=/app/models/diar_streaming_sortformer_4spk-v2.onnx
VAD_MODEL_PATH=/app/models/silero_vad.onnx
MEDIA_DIR=/app/media
EOF

# 5. Build and start
docker-compose build
docker-compose up -d

# 6. Open browser
echo "Open http://$(curl -s ifconfig.me):8080"
```

### Option 2: Local Development

```bash
# 1. Build
cargo build --release --example webrtc_transcriber --features sortformer

# 2. Create media directory
mkdir -p media

# 3. Run the server
./target/release/examples/webrtc_transcriber \
  --tdt-model ./tdt \
  --diar-model ./diar_streaming_sortformer_4spk-v2.onnx \
  --vad-model ./silero_vad.onnx \
  --frontend ./frontend \
  --media-dir ./media \
  --port 8080

# 4. Open http://localhost:8080 in browser
```

---

## Docker Deployment

### Dockerfile

The multi-stage Dockerfile builds a minimal runtime image:

```dockerfile
# Build stage
FROM rust:latest AS builder
WORKDIR /app
COPY . .
RUN cargo build --release --example webrtc_transcriber --features sortformer

# Runtime stage
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y libssl3 libopus0 ffmpeg curl
COPY --from=builder /app/target/release/examples/webrtc_transcriber /app/
COPY frontend/ /app/frontend/

# Environment variables
ENV PORT=8080
ENV PUBLIC_IP=""
ENV TDT_MODEL_PATH=/app/models/tdt
ENV DIAR_MODEL_PATH=/app/models/diarization/model.onnx
ENV VAD_MODEL_PATH=/app/models/silero_vad.onnx
ENV MEDIA_DIR=/app/media
ENV TURN_SERVER=""
ENV TURN_USERNAME=""
ENV TURN_PASSWORD=""

ENTRYPOINT ["/app/webrtc_transcriber"]
CMD ["--frontend", "/app/frontend"]
```

### Docker Compose

```yaml
services:
  transcriber:
    build:
      context: .
      dockerfile: docker/Dockerfile.transcriber
    container_name: parakeet-transcriber
    network_mode: host  # Required for WebRTC UDP

    volumes:
      - ./tdt:/app/models/tdt:ro
      - ./diar_streaming_sortformer_4spk-v2.onnx:/app/models/diarization/model.onnx:ro
      - ./silero_vad.onnx:/app/models/silero_vad.onnx:ro
      - ./media:/app/media

    environment:
      - PORT=8080
      - PUBLIC_IP=${PUBLIC_IP:-}
      - MEDIA_DIR=/app/media
      - TURN_SERVER=${TURN_SERVER:-}
      - TURN_USERNAME=${TURN_USERNAME:-}
      - TURN_PASSWORD=${TURN_PASSWORD:-}
```

**Note:** `network_mode: host` is required for WebRTC UDP traffic. For Kubernetes, use `hostNetwork: true`.

---

## GPU Deployment

### Building with GPU Support

```bash
# NVIDIA CUDA
cargo build --release --example webrtc_transcriber --features "cuda,sortformer"

# NVIDIA TensorRT (faster inference)
cargo build --release --example webrtc_transcriber --features "tensorrt,sortformer"

# AMD ROCm
cargo build --release --example webrtc_transcriber --features "rocm,sortformer"
```

### Docker GPU Deployment

Use `docker-compose.gpu.yml` for GPU acceleration:

```bash
# Build GPU image
docker-compose -f docker-compose.gpu.yml build

# Run with GPU
docker-compose -f docker-compose.gpu.yml up -d
```

**Requirements:**
- NVIDIA Container Toolkit installed
- NVIDIA drivers installed on host
- CUDA 12.2+ compatible GPU

### GPU Environment Variables

```bash
# Enable GPU
USE_GPU=true

# Or specify provider explicitly
USE_GPU=cuda
USE_GPU=tensorrt

# GPU thread tuning (lower values recommended)
INTRA_THREADS=2
INTER_THREADS=1
```

### GPU Performance Notes

- GPU inference is typically 5-10x faster than CPU
- For parallel modes, GPU workers share the same GPU device
- Reduce `INTRA_THREADS` to avoid GPU contention
- TensorRT provides fastest inference (requires TensorRT installation)

---

## Production Considerations

### TURN Server Setup

For NAT traversal in production, configure a TURN server:

```bash
# .env
TURN_SERVER=turns:coturn.example.com:443?transport=udp
TURN_USERNAME=myuser
TURN_PASSWORD=mysecret
```

Popular TURN servers:
- [coturn](https://github.com/coturn/coturn) - Open source
- [Twilio TURN](https://www.twilio.com/stun-turn) - Managed service
- [Cloudflare TURN](https://developers.cloudflare.com/calls/) - Managed service

### PUBLIC_IP Configuration

Critical for WebRTC to work in containerized environments:

```bash
# Get public IP
export PUBLIC_IP=$(curl -s ifconfig.me)

# Or set explicitly
export PUBLIC_IP=203.0.113.50
```

The server uses this IP for ICE candidate generation.

### Resource Limits

Each session consumes:
- 1 FFmpeg process
- 1 transcription thread
- ~100-500MB RAM per session (depending on model)
- CPU scales with transcription complexity

Set `MAX_CONCURRENT_SESSIONS` to limit resource usage:

```bash
export MAX_CONCURRENT_SESSIONS=5
```

### Health Monitoring

The `/health` endpoint returns "OK" when the server is running:

```bash
curl http://localhost:8080/health
```

For monitoring, check:
- Health endpoint responsiveness
- Active session count via `/api/sessions`
- WebSocket connection state
- FFmpeg process health

---

## Performance Tuning

### Latency Breakdown

| Component | Typical Latency |
|-----------|-----------------|
| FFmpeg audio processing | 20-50ms |
| Opus encoding | 1-2ms |
| WebRTC jitter buffer | 50-150ms |
| Network RTT | 10-100ms |
| ASR processing | 30-100ms |
| **Total audio latency** | **100-400ms** |
| **Transcription latency** | **300ms-3s** (depends on mode) |

### Optimization Tips

1. **Choose the right mode** for your use case:
   - `speedy` for lowest latency with good quality
   - `vad_speedy` for utterance-based transcription
   - `lookahead` for highest accuracy

2. **Use TURN over UDP** (port 443) for best NAT traversal

3. **Set `PUBLIC_IP` explicitly** in containerized environments

4. **Limit concurrent sessions** with `MAX_CONCURRENT_SESSIONS`

5. **Monitor with `/health` endpoint** for production deployments

6. **Use `RUST_LOG=debug`** for troubleshooting

### Multi-Session Scaling

- Each session spawns:
  - 1 FFmpeg process (audio decoding)
  - 1 transcription thread
  - 1 WebRTC audio track
- Memory usage scales with buffer size and number of sessions
- CPU usage scales with transcription complexity and session count
- Consider GPU acceleration for high session counts

---

## Troubleshooting

### No Audio Playback

1. Check WebRTC connection in browser console
2. Verify ICE candidates are being exchanged
3. Ensure TURN server is configured for NAT traversal
4. Click "Play" button (autoplay may be blocked)
5. Check session state is "running" in Sessions tab

### High Latency

1. Use `speedy` or `vad_speedy` mode for lowest latency
2. Check network conditions (RTT shown in UI)
3. Ensure server has adequate CPU for real-time processing
4. Reduce concurrent sessions if CPU is overloaded

### Connection Failures

1. Verify `PUBLIC_IP` is set correctly
2. Check firewall allows UDP traffic (especially for WebRTC)
3. Configure TURN server for restrictive networks
4. Check browser console for WebRTC errors
5. Verify session exists and is in "running" state

### Session Issues

1. **Session not starting**: Check model and media paths are valid
2. **Transcription stuck**: Check server logs for ONNX errors
3. **No subtitles**: Verify WebSocket connection in browser console
4. **FFmpeg errors**: Ensure media file format is supported (WAV, MP3)

### Debug Logging

Enable debug logging:

```bash
RUST_LOG=debug ./webrtc_transcriber ...
```

Or for specific modules:

```bash
RUST_LOG=webrtc_transcriber=debug,parakeet_rs=debug ./webrtc_transcriber ...
```

---

## Related Documentation

- [Architecture](./architecture.md) - System design overview
- [API Reference](./api-reference.md) - Complete API documentation
- [Latency Modes](./latency-modes.md) - Mode selection guide
