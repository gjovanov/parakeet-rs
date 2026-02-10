# Architecture

[Architecture](architecture.md) | [API Reference](api.md) | [Transcription Modes](transcription-modes.md) | [Frontend](frontend.md) | [FAB Teletext](fab-teletext.md) | [Testing](testing.md) | [Deployment](deployment.md)

---

## System Overview

```mermaid
graph TD
    subgraph Inputs["Audio Inputs"]
        MF["Media Files<br/>(WAV, MP3, etc.)"]
        SRT["SRT Live Streams<br/>(14 ORF channels)"]
        UL["File Upload<br/>(up to 2GB)"]
    end

    subgraph Server["Axum HTTP/WS Server"]
        API["REST API<br/>/api/*"]
        WSH["WebSocket Handler<br/>/ws/:session_id"]
        SM["Session Manager<br/>(max 10 concurrent)"]
        MR["Model Registry<br/>(TDT, Canary, Flash)"]
        MM["Media Manager<br/>(scan + upload)"]
        SC["SRT Config<br/>(channel mapping)"]
    end

    subgraph Pipeline["Audio Processing Pipeline"]
        FF["FFmpeg<br/>decode to PCM 16kHz"]
        NC["RNNoise<br/>noise cancellation"]
        OP["Opus Encoder<br/>WebRTC audio"]
        TF["Transcriber Factory<br/>13 modes"]
    end

    subgraph Models["ONNX Models"]
        TDT["Parakeet TDT 0.6B<br/>English, token timestamps"]
        CAN["Canary 1B<br/>KV-cached decoder"]
        CF["Canary 180M Flash<br/>fast multilingual"]
        VAD["Silero VAD<br/>speech boundaries"]
        SF["Sortformer v2<br/>4-speaker diarization"]
    end

    subgraph TextProc["Text Processing"]
        GT["GrowingTextMerger<br/>anchor-based tail-overwrite"]
        SB["Sentence Buffer<br/>completion + boundaries"]
        DD["Deduplication<br/>containment coefficient"]
    end

    subgraph Output["Output"]
        SUB["WebSocket Subtitles<br/>(partial + final)"]
        RTC["WebRTC Audio<br/>(Opus/RTP)"]
        FAB["FAB Teletext<br/>(HTTP GET, 42x2 chars)"]
        VOD["VoD Transcript<br/>(JSON download)"]
    end

    MF --> FF
    SRT --> FF
    UL --> MM --> FF
    FF --> NC --> TF
    FF --> OP --> RTC
    TF --> TDT & CAN & CF
    VAD -.-> TF
    SF -.-> TF
    TF --> GT --> SB --> DD
    DD --> SUB
    DD --> FAB
    GT --> VOD

    API --> SM
    WSH --> SM
    SM --> MR
    SC --> SM
```

## Audio Pipeline

The audio pipeline transforms input sources into PCM samples for transcription and Opus packets for WebRTC playback.

```mermaid
flowchart LR
    SRC["Audio Source<br/>(File or SRT)"] --> FFM["FFmpeg<br/>-f s16le -ar 16000 -ac 1"]
    FFM --> PCM["PCM i16 → f32<br/>(normalize to -1.0..1.0)"]
    PCM --> NC{"RNNoise<br/>enabled?"}
    NC -- yes --> RNN["RNNoiseProcessor<br/>(480-sample frames)"]
    NC -- no --> SPLIT
    RNN --> SPLIT["Split"]
    SPLIT --> TX["Transcriber<br/>(via mpsc channel)"]
    SPLIT --> ENC["Opus Encoder<br/>(960 samples/frame)"]
    ENC --> RTP["RTP Packets<br/>(WebRTC track)"]
```

Key details:
- **FFmpeg** decodes any audio format to raw PCM: 16kHz, mono, signed 16-bit little-endian
- For **SRT streams**, FFmpeg uses `-fflags +nobuffer+genpts` and `-flags low_delay` for minimal latency
- **RNNoise** processes 480-sample frames (30ms at 16kHz) for real-time noise suppression
- **Opus encoder** produces 20ms frames (960 samples at 48kHz after resampling) for WebRTC delivery
- Audio samples are sent to the transcriber via `std::sync::mpsc` channel in 2560-sample chunks (160ms)

## Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: POST /api/sessions
    Created --> Starting: POST /api/sessions/:id/start
    Starting --> Running: transcription thread spawned
    Running --> Completed: media file ended
    Running --> Stopped: DELETE /api/sessions/:id
    Running --> Running: SRT stream (continuous)
    Completed --> [*]
    Stopped --> [*]

    note right of Created
        Session created with model, media,
        mode, language, and config
    end note

    note right of Running
        Audio pipeline active,
        WebSocket broadcasting,
        FAB forwarding (if enabled)
    end note
```

Sessions are explicitly started after creation. This two-step process allows:
1. WebSocket clients to connect before transcription begins (avoiding race conditions)
2. Configuration to be finalized before resource allocation

## Request Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant A as Axum Server
    participant SM as Session Manager
    participant F as Factory
    participant T as Transcriber
    participant WS as WebSocket

    B->>A: POST /api/sessions<br/>{model_id, media_id, mode}
    A->>SM: create_session()
    SM-->>A: session_id
    A-->>B: {success: true, data: {id, state: "created"}}

    B->>A: WS /ws/:session_id
    A->>WS: upgrade connection
    WS-->>B: connected (subtitle + WebRTC signaling)

    B->>A: POST /api/sessions/:id/start
    A->>SM: start_session()
    SM->>F: create_transcriber(mode, model)
    F-->>SM: Box<dyn StreamingTranscriber>
    SM->>T: spawn transcription thread

    loop Audio Processing
        T->>T: FFmpeg → PCM → transcribe
        T->>WS: broadcast subtitle (partial)
        T->>WS: broadcast subtitle (final)
    end

    T-->>SM: transcription complete
    SM-->>B: {type: "end"}
```

## Server Binary Structure

The server binary lives in `src/bin/server/` with the following module layout:

| Module | File | Description |
|--------|------|-------------|
| **main** | `main.rs` | Entry point, CLI args (clap), router setup, server startup |
| **api** | `api/mod.rs` | API handler re-exports |
| **api::sessions** | `api/sessions.rs` | Session CRUD, start/stop, transcript download |
| **api::models** | `api/models.rs` | Model listing, mode listing, `ApiResponse<T>` envelope |
| **api::config** | `api/config.rs` | Frontend config (WS URL, ICE servers, FAB settings) |
| **api::media** | `api/media.rs` | Media file listing, upload, delete |
| **api::srt** | `api/srt.rs` | SRT stream listing |
| **api::noise** | `api/noise.rs` | Noise cancellation options |
| **api::diarization** | `api/diarization.rs` | Diarization options |
| **config** | `config.rs` | `LatencyMode` enum, `RuntimeConfig` |
| **state** | `state.rs` | `AppState` (shared server state) |
| **srt_config** | `srt_config.rs` | SRT channel configuration from env |
| **fab_forwarder** | `fab_forwarder.rs` | FAB teletext forwarding, dedup, splitting |
| **transcription** | `transcription/mod.rs` | Session orchestration, `AudioSource` enum |
| **transcription::factory** | `transcription/factory.rs` | Mode-to-transcriber mapping |
| **transcription::configs** | `transcription/configs.rs` | Per-mode configuration factories |
| **transcription::emitters** | `transcription/emitters.rs` | Partial/final/streaming subtitle emission |
| **transcription::audio_pipeline** | `transcription/audio_pipeline.rs` | FFmpeg spawn, PCM read, Opus encode |
| **transcription::vod** | `transcription/vod.rs` | VoD batch transcription |
| **webrtc_handlers** | `webrtc_handlers/mod.rs` | WebSocket upgrade, WebRTC signaling |
| **webrtc_handlers::audio** | `webrtc_handlers/audio.rs` | Opus encoder wrapper |

## Library Modules

The core library (`src/lib.rs`) exposes these public modules:

| Module | Description |
|--------|-------------|
| `canary` | Canary 1B model: encoder, decoder with KV cache, tokenizer |
| `canary_flash` | Canary 180M Flash model with `DecoderKVCache` |
| `realtime_canary` | Streaming Canary transcriber (sliding window) |
| `realtime_canary_flash` | Streaming Canary Flash transcriber |
| `realtime_canary_vad` | VAD-triggered Canary transcriber |
| `realtime_tdt` | Streaming TDT transcriber with diarization |
| `realtime_tdt_vad` | VAD-triggered TDT transcriber (requires sortformer) |
| `parallel_canary` | Multi-threaded parallel Canary inference |
| `parallel_tdt` | Multi-threaded parallel TDT inference |
| `pause_parallel_canary` | Pause-triggered parallel Canary inference |
| `pause_parallel_tdt` | Pause-triggered parallel TDT inference |
| `streaming_transcriber` | `StreamingTranscriber` trait and `TranscriberFactory` |
| `growing_text` | `GrowingTextMerger` — anchor-based tail-overwrite text merging |
| `sentence_buffer` | Sentence boundary detection and completion |
| `vod_transcriber` | VoD batch transcription (10-min chunks) |
| `model_registry` | `ModelRegistry` — model discovery and type mapping |
| `media_manager` | `MediaManager` — file scanning and upload handling |
| `session` | `SessionManager`, `TranscriptionSession`, `SessionState` |
| `vad` | Silero VAD wrapper, `VadSegmenter` |
| `noise_cancellation` | RNNoise wrapper |
| `sortformer` / `sortformer_stream` | Sortformer v2 diarization (feature-gated) |

## Canary 1B KV Cache

The Canary 1B decoder uses a KV (key-value) cache for O(n) incremental decoding instead of O(n^2) full re-computation:

```mermaid
flowchart LR
    subgraph First["First Token"]
        E1["Encoder Output<br/>[B, T, 1024]"] --> D1["Decoder<br/>(no cache)"]
        D1 --> H1["decoder_hidden_states<br/>[10, B, 1, 1024]"]
        D1 --> T1["token_0"]
    end

    subgraph Cached["Subsequent Tokens"]
        H1 --> D2["Decoder<br/>(with cache)"]
        TN["last token only"] --> D2
        D2 --> H2["decoder_hidden_states<br/>[10, B, n+1, 1024]"]
        D2 --> TN1["token_n"]
        H2 --> D2
    end
```

- **Decoder ONNX I/O**: `decoder_mems` [10, B, mems_len, 1024] input, `decoder_hidden_states` [10, B, seq_len, 1024] output
- **Cached path**: Feed `decoder_hidden_states` back as `decoder_mems`, pass only last token as `input_ids`
- **Performance**: Cached decoding achieves ~10% WER vs ~60% WER with full re-computation on German test audio
- **Fallback**: `greedy_decode()` tries cached path first, falls back to full decode if cache shape mismatches
