# Parakeet-RS

Real-time ASR (Automatic Speech Recognition) server in Rust using ONNX Runtime.

## Models
- **parakeet-tdt** — English only, fast CTC/TDT model (NVIDIA)
- **canary-1b** — Multilingual (en, de, fr, es), encoder-decoder model (NVIDIA)

## Transcription Modes
- **speedy** — Low-latency streaming with pause-based word confirmation
- **growing_segments** — Word-by-word PARTIAL updates building toward FINAL sentences (best for live subtitles)
- **pause_segmented** — Segment audio by acoustic pauses, transcribe each chunk once (precise timestamps)

## Build
```bash
cargo build --release --bin parakeet-server --features "server,sortformer"
```

## Run
```bash
# Set env vars (or use .env)
export ORT_DYLIB_PATH=./ort-cpu/libonnxruntime.so
export LD_LIBRARY_PATH=$(dirname $ORT_DYLIB_PATH)
./start-server.sh
```

## Test
```bash
cargo test                        # all tests (173)
cargo test -- --test-threads=1    # reliable (env var tests are flaky multi-threaded)
```

## Project Structure
```
src/
  lib.rs                          # Library exports
  canary.rs                       # Canary-1B ONNX model
  realtime_canary.rs              # Canary streaming transcriber
  realtime_tdt.rs                 # TDT streaming transcriber
  pause_segmented.rs              # Canary pause-segmented mode
  pause_segmented_tdt.rs          # TDT pause-segmented mode
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
frontend/                         # Web UI served by the server
```

## Key Architecture
- ONNX Runtime with `load-dynamic` feature — requires `ORT_DYLIB_PATH` at runtime
- WebRTC for real-time audio streaming (server → browser)
- Growing segments pipeline: push_audio → GrowingTextMerger → echo dedup → cross-cycle buffering → FINAL/PARTIAL emission
- Canary uses sliding buffer re-transcription; TDT uses incremental CTC decoding

## Environment Variables
See `.env.example` for all options. Key ones:
- `ORT_DYLIB_PATH` — Path to libonnxruntime.so
- `TDT_MODEL_PATH`, `CANARY_MODEL_PATH` — Model directories
- `DIAR_MODEL_PATH` — Diarization model (sortformer)
- `USE_GPU` — `cuda` for GPU, empty/`false` for CPU
- `PORT` — Server port (default: 8080)
