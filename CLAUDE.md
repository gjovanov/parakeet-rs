# Parakeet-RS

Real-time ASR (Automatic Speech Recognition) with two server backends sharing one frontend.

## Architecture: Two Servers, One Frontend

| Server | Stack | Models | Port |
|--------|-------|--------|------|
| **parakeet-server** (Rust) | ONNX Runtime / whisper.cpp | parakeet-tdt, canary-1b, whisper | 8080 |
| **voxtral-server** (Python) | vLLM (GPU) | voxtral-mini-4b | 8090 |

Both serve the same `frontend/` directory with identical REST/WebSocket/WebRTC API contracts.

## Models
- **parakeet-tdt** — English only, fast CTC/TDT model (NVIDIA), CPU/GPU
- **canary-1b** — Multilingual (en, de, fr, es), encoder-decoder (NVIDIA), CPU/GPU
- **whisper** — 99+ languages via whisper.cpp (GGML), CPU/GPU
- **voxtral-mini-4b** — 13 languages, natively streaming via vLLM, GPU only

## Transcription Modes
- **speedy** — Low-latency streaming with pause-based word confirmation
- **growing_segments** — Word-by-word PARTIAL updates building toward FINAL sentences
- **pause_segmented** — Segment audio by acoustic pauses, transcribe each chunk once

## Parakeet Server (Rust)

### Build
```bash
cargo build --release --bin parakeet-server --features "server,sortformer"
# Add ,whisper for Whisper support; ,whisper-cuda for GPU Whisper
```

### Run
```bash
export ORT_DYLIB_PATH=./ort-cpu/libonnxruntime.so
export LD_LIBRARY_PATH=$(dirname $ORT_DYLIB_PATH)
./start-server.sh
```

### Test
```bash
cargo test                        # all tests (173)
cargo test -- --test-threads=1    # reliable (env var tests are flaky multi-threaded)
```

## Voxtral Server (Python)

Standalone Python backend for Mistral's Voxtral-Mini-4B-Realtime via vLLM sidecar.

### Setup
```bash
cd voxtral-server
./init.sh           # Creates venv, installs vLLM + deps, downloads model
```

### Run (two terminals)
```bash
# Terminal 1: vLLM GPU server
./start-vllm.sh     # Port 8001

# Terminal 2: voxtral-server
./start.sh           # Port 8090
```

### Benchmark
```bash
python3 scripts/benchmark_voxtral.py --duration 300    # 5-min benchmark
```

### Config (env vars, prefixed VOXTRAL_)
- `VOXTRAL_VLLM_URL` — vLLM WebSocket URL (default: ws://localhost:8001/v1/realtime)
- `VOXTRAL_PORT` — Server port (default: 8090)
- `VOXTRAL_MEDIA_DIR` — Media directory (default: ../media, shared with parakeet-rs)
- `VOXTRAL_FRONTEND_PATH` — Frontend directory (default: ../frontend, shared)

## Project Structure
```
src/                              # Rust parakeet-server
  lib.rs                          # Library exports
  canary.rs                       # Canary-1B ONNX model
  realtime_canary.rs              # Canary streaming transcriber
  realtime_tdt.rs                 # TDT streaming transcriber
  pause_segmented.rs              # Canary pause-segmented mode
  pause_segmented_tdt.rs          # TDT pause-segmented mode
  whisper.rs                      # Whisper GGML model (whisper-rs)
  realtime_whisper.rs             # Whisper streaming transcriber
  pause_segmented_whisper.rs      # Whisper pause-segmented mode
  growing_text.rs                 # GrowingTextMerger (tail-overwrite + dedup)
  streaming_transcriber.rs        # Core StreamingTranscriber trait
  model_registry.rs               # Model discovery from env vars
  session.rs                      # Session lifecycle management
  vad.rs                          # Silero VAD for pause detection
  bin/server/
    main.rs                       # Server entry point (axum + WebRTC)
    config.rs                     # LatencyMode enum (3 modes)
    state.rs                      # AppState
    transcription/
      mod.rs                      # Main transcription loop
      factory.rs                  # Transcriber creation dispatch
      configs.rs                  # Mode-specific config tuning
      emitters.rs                 # Subtitle emission + normalize_text()
      audio_pipeline.rs           # FFmpeg → PCM → noise cancel → Opus → RTP
    api/
      sessions.rs                 # Session CRUD + start
      models.rs                   # Model/mode listing
voxtral-server/                   # Python voxtral-server
  voxtral_server/
    main.py                       # FastAPI app entry point
    config.py                     # Settings (VOXTRAL_ prefixed env vars)
    models.py                     # Pydantic models (API contract)
    state.py                      # AppState (sessions, broadcast)
    api/                          # REST endpoints (sessions, media, models)
    ws/handler.py                 # WebSocket + WebRTC signaling
    transcription/
      vllm_client.py              # vLLM /v1/realtime WebSocket client
      session_runner.py           # FFmpeg + vLLM + subtitle orchestrator
    audio/
      ffmpeg_source.py            # FFmpeg → PCM 16kHz
      webrtc_track.py             # aiortc AudioStreamTrack
    media/manager.py              # Media file listing/upload
  init.sh                         # Full setup script
  start-vllm.sh                   # Start vLLM GPU server
  start.sh                        # Start voxtral-server
frontend/                         # Web UI (shared by both servers)
scripts/
  benchmark_broadcast1.py         # Benchmark for parakeet-server
  benchmark_voxtral.py            # Benchmark for voxtral-server
```

## Key Architecture
- **Parakeet server**: ONNX Runtime with `load-dynamic` feature, WebRTC audio streaming, in-process inference
- **Voxtral server**: vLLM sidecar (GPU), FastAPI + aiortc, background WebSocket reader for streaming deltas
- Both use: FFmpeg → PCM → WebRTC (audio) + WebSocket (subtitles)
- Frontend is a dumb display — no dedup, no filtering, pass-through rendering

## Environment Variables (parakeet-server)
See `.env.example` for all options. Key ones:
- `ORT_DYLIB_PATH` — Path to libonnxruntime.so
- `TDT_MODEL_PATH`, `CANARY_MODEL_PATH` — Model directories
- `WHISPER_MODEL_PATH` — Whisper GGML model directory
- `DIAR_MODEL_PATH` — Diarization model (sortformer)
- `USE_GPU` — `cuda` for GPU, empty/`false` for CPU
- `PORT` — Server port (default: 8080)
