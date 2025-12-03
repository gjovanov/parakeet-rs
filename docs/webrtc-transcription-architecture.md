# Building Real-Time Speech Transcription with WebRTC: A Deep Dive into parakeet-rs

Real-time speech transcription has become increasingly important for accessibility, live captioning, meeting transcription, and broadcast subtitling. In this post, we'll explore how `parakeet-rs` implements a production-ready WebRTC-based transcription system that delivers sub-second latency while maintaining high-quality output with speaker diarization.

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [The Audio Pipeline](#the-audio-pipeline)
3. [WebRTC Signaling and Media Flow](#webrtc-signaling-and-media-flow)
4. [Real-Time Transcription Engine](#real-time-transcription-engine)
5. [Pause-Based vs Time-Based Confirmation](#pause-based-vs-time-based-confirmation)
6. [Speaker Diarization Integration](#speaker-diarization-integration)
7. [Configuration via Environment Variables](#configuration-via-environment-variables)
8. [Deployment with Docker](#deployment-with-docker)
9. [Quick Start Guide](#quick-start-guide)
10. [Performance Tuning](#performance-tuning)

---

## System Architecture Overview

The parakeet-rs WebRTC transcriber consists of several interconnected components:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WebRTC Transcription Server                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Audio Source                     Server                      Browser       │
│  ───────────                     ────────                    ─────────      │
│                                                                             │
│  ┌─────────┐    stdin     ┌──────────────────────────┐                     │
│  │  ffmpeg │ ──────────►  │    Audio Thread          │                     │
│  │ (PCM)   │   s16le      │    ┌────────────────┐    │    WebRTC           │
│  └─────────┘   16kHz      │    │ Opus Encoder   │────┼──────────►  🔊      │
│                           │    └────────────────┘    │    Audio            │
│                           │           │              │                     │
│                           │           ▼              │                     │
│                           │    ┌────────────────┐    │                     │
│                           │    │ Transcriber    │    │                     │
│                           │    │ (TDT + Diar)   │    │                     │
│                           │    └────────────────┘    │                     │
│                           │           │              │                     │
│                           └───────────┼──────────────┘                     │
│                                       │                                     │
│                                       ▼                                     │
│                           ┌──────────────────────────┐                     │
│                           │    HTTP/WS Server        │                     │
│                           │    (Axum)                │    WebSocket        │
│                           │    ┌────────────────┐    │   ──────────►  📝   │
│                           │    │ /ws            │────┼───  Subtitles       │
│                           │    │ /api/config    │    │                     │
│                           │    │ /health        │    │                     │
│                           │    │ /* (frontend)  │    │                     │
│                           │    └────────────────┘    │                     │
│                           └──────────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Audio Ingestion**: Raw PCM audio (16kHz, mono, 16-bit signed little-endian) from stdin
2. **WebRTC Server**: Axum-based HTTP/WebSocket server for signaling and audio streaming
3. **Transcription Engine**: NVIDIA's TDT (Token-and-Duration Transducer) model via ONNX runtime
4. **Diarization**: Sortformer-based speaker identification
5. **Frontend**: Browser-based client with WebRTC audio playback and subtitle display

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend (serves static files) |
| `/ws` | WebSocket | Signaling and subtitles |
| `/api/config` | GET | Runtime configuration (JSON) |
| `/health` | GET | Health check ("OK") |

---

## The Audio Pipeline

### Input Processing

Audio enters the system through stdin, typically piped from FFmpeg:

```bash
ffmpeg -re -i input.wav -f s16le -ar 16000 -ac 1 - | \
  /app/webrtc_transcriber --tdt-model /app/models/tdt \
    --diar-model /app/models/diarization/model.onnx --speedy
```

The `-re` flag ensures real-time playback speed, essential for live streaming scenarios.

### Ring Buffer Architecture

The transcription engine uses a ring buffer to maintain a sliding window of audio:

```rust
pub struct StreamingConfig {
    /// How much audio to buffer before processing (default: 10s)
    pub buffer_size_secs: f32,

    /// How often to process the buffer (default: 0.3s)
    pub process_interval_secs: f32,

    /// How far from buffer end to consider "confirmed" (default: 0.5s)
    pub confirm_threshold_secs: f32,

    /// Enable pause-based confirmation (default: true)
    pub pause_based_confirm: bool,

    /// Pause duration to trigger confirmation (default: 0.3s)
    pub pause_threshold_secs: f32,
}
```

The ring buffer serves multiple purposes:
- **Context for ASR**: Neural models perform better with surrounding context
- **Re-processing tolerance**: Allows correction of previous hypotheses
- **Smooth output**: Prevents jarring text changes

### Opus Encoding for WebRTC

Audio is encoded to Opus format for WebRTC transmission:

```rust
let encoder = opus::Encoder::new(
    SAMPLE_RATE as u32,  // 16000
    opus::Channels::Mono,
    opus::Application::Voip, // Optimized for speech
)?;

// Encode 20ms frames (320 samples at 16kHz)
let frame_samples = (SAMPLE_RATE * 20) / 1000; // 320
let encoded = encoder.encode(&samples, &mut opus_buffer)?;
```

Opus provides excellent compression for speech (typically 12-20 kbps) while maintaining quality.

---

## WebRTC Signaling and Media Flow

### Signaling Protocol

The server uses WebSocket for SDP and ICE candidate exchange:

```
Browser                    Server
   │                          │
   │──── { type: "ready" }───▶│
   │                          │
   │◀── { type: "welcome" }───│
   │                          │
   │◀─── { type: "offer",  ───│
   │       sdp: "..." }       │
   │                          │
   │──── { type: "answer", ──▶│
   │       sdp: "..." }       │
   │                          │
   │◀── { type: "ice-candidate" }
   │                          │
   │── { type: "ice-candidate" } ──▶
   │                          │
   │◀═══ RTP Audio Stream ════│
   │                          │
   │◀── { type: "subtitle" }──│
```

### ICE and NAT Traversal

For NAT traversal, the server supports TURN/STUN:

```rust
// Server-side ICE configuration
let mut media_engine = MediaEngine::default();
media_engine.register_default_codecs()?;

let mut setting_engine = SettingEngine::default();

// Set public IP for host candidates (crucial for Docker)
if let Some(public_ip) = &args.public_ip {
    setting_engine.set_nat_1to1_ips(
        vec![public_ip.clone()],
        RTCIceCandidateType::Host
    );
}
```

---

## Real-Time Transcription Engine

### The TDT Model

Token-and-Duration Transducer (TDT) from NVIDIA NeMo provides:
- Word-level timestamps with high accuracy
- Fast inference (~50ms for 10s of audio on CPU)
- Multi-language support

### Token Confirmation Strategy

The key challenge in streaming ASR is deciding when to "confirm" (finalize) tokens:

```
Time:    0s        2s        4s        6s        8s       10s
Audio:   [=========================================]
         ↑                                        ↑
    Buffer Start                            Buffer End

         [CONFIRMED ZONE]  [PENDING ZONE]
         ◀───────────────▶ ◀────────────▶
              Emit these    May change
```

Tokens in the **confirmed zone** are emitted immediately and won't change. Tokens in the **pending zone** may be revised as more audio arrives.

---

## Pause-Based vs Time-Based Confirmation

### The Problem with Pure Time-Based Confirmation

Time-based confirmation (e.g., "confirm everything 2 seconds behind") works but has issues:
- Words may be cut mid-sentence
- Doesn't respect natural speech boundaries
- Fixed latency regardless of content

### Pause Detection Algorithm

Pause-based confirmation detects natural speech pauses using RMS energy:

```rust
fn detect_silence(&mut self, samples: &[f32]) -> bool {
    // Calculate RMS energy
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_squares / samples.len() as f32).sqrt();

    // Compare against threshold (configurable)
    rms < self.config.silence_energy_threshold // e.g., 0.008
}
```

### Available Latency Modes

The transcriber supports multiple latency modes, each optimized for different use cases:

| Mode | Buffer | Interval | Confirm | Expected Latency | Pause Detection |
|------|--------|----------|---------|------------------|-----------------|
| `--speedy` | 8.0s | 0.2s | 0.4s | ~0.3-1.5s | ✅ Yes |
| `--pause-based` | 10.0s | 0.3s | 0.5s | ~0.5-2.0s | ✅ Yes |
| `--low-latency` | 10.0s | 1.5s | 2.0s | ~3.5s | ❌ No |
| `--ultra-low-latency` | 8.0s | 1.0s | 1.5s | ~2.5s | ❌ No |
| `--extreme-low-latency` | 5.0s | 0.5s | 0.8s | ~1.3s | ❌ No |
| `--lookahead` | 10.0s | 0.3s | 0.5s | ~1.0-3.0s | ✅ Yes |

#### Mode Details

**`--speedy`** (Recommended)
- Best balance of latency and quality
- Uses pause detection to confirm at natural speech boundaries
- Aggressive timings (0.2s interval, 0.4s confirm threshold)
- Ideal for: Live captioning, real-time subtitles, interactive applications

**`--pause-based`**
- Similar to speedy but with more conservative timings
- Better accuracy at the cost of slightly higher latency
- Ideal for: High-quality transcription where accuracy is more important than speed

**`--low-latency`**
- Time-based confirmation without pause detection
- Predictable, fixed latency
- Ideal for: Broadcast scenarios with consistent delay requirements

**`--ultra-low-latency`**
- Faster than low-latency with smaller buffer
- Good for applications needing faster response
- Ideal for: Live interviews, Q&A sessions

**`--extreme-low-latency`**
- Fastest possible response time
- May sacrifice some accuracy for speed
- Ideal for: Real-time voice assistants, gaming, interactive voice applications

**`--lookahead`**
- Best transcription quality
- Uses future context for better accuracy
- Processes segments with knowledge of subsequent audio
- Ideal for: Post-processing, archival transcription, highest quality requirements

#### Choosing the Right Mode

```
Quality vs Latency Spectrum:

  Lower Latency                                    Higher Quality
  ◄─────────────────────────────────────────────────────────────►

  extreme    ultra      speedy     pause-     low-       lookahead
  -low-      -low-                 based      latency
  latency    latency

  ~1.3s      ~2.5s      ~0.3-1.5s  ~0.5-2.0s  ~3.5s      ~1.0-3.0s
```

#### Usage Examples

```bash
# Via CLI flags
./webrtc_transcriber --speedy
./webrtc_transcriber --low-latency
./webrtc_transcriber --lookahead

# Via run.sh script
./run.sh gpu speedy
./run.sh cpu low-latency
./run.sh gpu lookahead

# Via environment variable (speedy mode only)
SPEEDY_MODE=1 ./webrtc_transcriber
```

---

## Speaker Diarization Integration

### Sortformer Model

Speaker diarization identifies "who spoke when" using NVIDIA's Sortformer:

```rust
let diarizer = StreamingDiarizer::new(
    SortformerOnnxModel::from_path(&args.diar_model)?,
    DiarizationConfig {
        max_speakers: 4,
        chunk_duration_secs: 5.0,
        ..Default::default()
    },
)?;
```

Output format includes speaker ID:

```
[Speaker 0] Hello, how are you today? [0.50s-2.30s]
[Speaker 1] I'm doing great, thanks! [2.45s-4.10s]
```

---

## Configuration via Environment Variables

All configuration is done via environment variables, making deployment flexible and containerization simple.

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | HTTP/WebSocket server port |
| `PUBLIC_IP` | auto-detect | Public IP for WebRTC ICE candidates |
| `FRONTEND_PATH` | `./frontend` | Path to frontend static files |
| `SPEEDY_MODE` | - | Set to `1` for low-latency mode |
| `RUST_LOG` | `info` | Log level (error, warn, info, debug, trace) |

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TDT_MODEL_PATH` | `.` | Path to TDT model directory |
| `DIAR_MODEL_PATH` | `*.onnx` | Path to diarization ONNX model |

### TURN Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TURN_SERVER` | - | TURN server URL (e.g., `turns:server.com:443`) |
| `TURN_USERNAME` | - | TURN authentication username |
| `TURN_PASSWORD` | - | TURN authentication password |

### Example .env File

```bash
# Server
PORT=8080
PUBLIC_IP=203.0.113.50
SPEEDY_MODE=1

# Models
TDT_MODEL_PATH=/app/models/tdt
DIAR_MODEL_PATH=/app/models/diarization/model.onnx

# TURN (for NAT traversal)
TURN_SERVER=turns:coturn.example.com:443?transport=udp
TURN_USERNAME=myuser
TURN_PASSWORD=mysecret

# Logging
RUST_LOG=info
```

### Frontend Auto-Configuration

The frontend automatically fetches configuration from the server:

```javascript
// frontend/js/config.js
export async function loadConfig() {
  const response = await fetch('/api/config');
  const serverConfig = await response.json();
  // Returns: { wsUrl, iceServers, audio, subtitles, ... }
}
```

The `/api/config` endpoint returns:

```json
{
  "wsUrl": "ws://203.0.113.50:8080/ws",
  "iceServers": [
    { "urls": "stun:stun.l.google.com:19302" },
    { "urls": "turns:coturn.example.com:443", "username": "myuser", "credential": "mysecret" }
  ],
  "audio": { "sampleRate": 16000, "channels": 1 },
  "subtitles": { "maxSegments": 1000, "autoScroll": true },
  "speakerColors": ["#4A90D9", "#50C878", "#E9967A", "#DDA0DD", ...]
}
```

This eliminates hardcoding server URLs or TURN credentials in the frontend.

---

## Deployment with Docker

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
ENV SPEEDY_MODE=""
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

    environment:
      - PORT=8080
      - PUBLIC_IP=${PUBLIC_IP:-}
      - SPEEDY_MODE=${SPEEDY_MODE:-1}
      - TURN_SERVER=${TURN_SERVER:-}
      - TURN_USERNAME=${TURN_USERNAME:-}
      - TURN_PASSWORD=${TURN_PASSWORD:-}

    stdin_open: true
    tty: true
```

---

## Quick Start Guide

### Option 1: Docker (Recommended)

```bash
# 1. Clone and enter directory
git clone https://github.com/your-org/parakeet-rs.git
cd parakeet-rs

# 2. Download models
# - TDT model → ./tdt/
# - Diarization model → ./diar_streaming_sortformer_4spk-v2.onnx

# 3. Create .env file
cat > .env << EOF
PUBLIC_IP=$(curl -s ifconfig.me)
SPEEDY_MODE=1
EOF

# 4. Build and start
docker-compose build
docker-compose up -d

# 5. Stream audio (in a separate terminal)
ffmpeg -re -i your-audio.wav -f s16le -ar 16000 -ac 1 - | \
  docker exec -i parakeet-transcriber /app/webrtc_transcriber

# 6. Open browser
echo "Open http://$(curl -s ifconfig.me):8080"
```

### Option 2: Local Development

```bash
# 1. Build
cargo build --release --example webrtc_transcriber --features sortformer

# 2. Set environment and run (terminal 1)
export PUBLIC_IP=localhost
export SPEEDY_MODE=1
export TDT_MODEL_PATH=./tdt
export DIAR_MODEL_PATH=./diar_streaming_sortformer_4spk-v2.onnx

./target/release/examples/webrtc_transcriber --frontend ./frontend

# 3. Stream audio (terminal 2)
ffmpeg -re -i test.wav -f s16le -ar 16000 -ac 1 - | \
  ./target/release/examples/webrtc_transcriber

# 4. Open http://localhost:8080 in browser
```

### Streaming Audio Sources

```bash
# From a file (real-time)
ffmpeg -re -i audio.wav -f s16le -ar 16000 -ac 1 -

# From microphone (Linux)
ffmpeg -f alsa -i default -f s16le -ar 16000 -ac 1 -

# From microphone (macOS)
ffmpeg -f avfoundation -i ":0" -f s16le -ar 16000 -ac 1 -

# From RTMP stream
ffmpeg -i rtmp://server/live/stream -f s16le -ar 16000 -ac 1 -

# From YouTube live
ffmpeg -i "$(yt-dlp -g https://youtube.com/watch?v=...)" -f s16le -ar 16000 -ac 1 -
```

---

## Performance Tuning

### Latency Breakdown

| Component | Typical Latency |
|-----------|-----------------|
| Audio capture | 20-50ms |
| Opus encoding | 1-2ms |
| WebRTC jitter buffer | 50-150ms |
| Network RTT | 10-100ms |
| ASR processing | 30-100ms |
| **Total audio latency** | **100-400ms** |
| **Transcription latency** | **300ms-2s** (depends on mode) |

### Optimization Tips

1. **Use `SPEEDY_MODE=1`** for lowest transcription latency
2. **Use TURN over UDP** (port 443) for best NAT traversal
3. **Set `PUBLIC_IP` explicitly** in containerized environments
4. **Monitor with `/health` endpoint** for production deployments
5. **Use `RUST_LOG=debug`** for troubleshooting

---

## Troubleshooting

### No Audio Playback

1. Check WebRTC connection in browser console
2. Verify ICE candidates are being exchanged
3. Ensure TURN server is configured for NAT traversal
4. Click "Play" button (autoplay may be blocked)

### High Latency

1. Use `SPEEDY_MODE=1` for lowest latency
2. Check network conditions (RTT shown in UI)
3. Ensure server has adequate CPU for real-time processing

### Connection Failures

1. Verify `PUBLIC_IP` is set correctly
2. Check firewall allows UDP traffic (especially for WebRTC)
3. Configure TURN server for restrictive networks
4. Check browser console for WebRTC errors

---

## Conclusion

Building a real-time transcription system requires careful orchestration of:

1. **Efficient audio pipelines** with ring buffers and proper pacing
2. **WebRTC for low-latency delivery** with proper NAT traversal
3. **Smart confirmation strategies** that balance latency vs quality
4. **Environment-based configuration** for flexible deployment
5. **Robust edge case handling** for production reliability

The parakeet-rs implementation demonstrates that sub-second transcription latency is achievable while maintaining high accuracy and speaker identification.

### Further Reading

- [WebRTC.rs Documentation](https://webrtc.rs/)
- [NVIDIA NeMo TDT](https://docs.nvidia.com/nemo-framework/)
- [Opus Codec Specification](https://opus-codec.org/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

*This document is part of the parakeet-rs project. For the latest code and updates, visit the repository.*
