# Architecture Overview

> **Navigation**: [Index](./README.md) | Architecture | [API Reference](./api-reference.md) | [Latency Modes](./latency-modes.md) | [Frontend](./frontend.md) | [Deployment](./deployment.md)

## System Architecture

The parakeet-rs WebRTC transcriber is a **multi-session transcription server** that supports multiple concurrent transcription sessions with different models and media files. Each session streams audio via WebRTC and subtitles via WebSocket.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Multi-Session WebRTC Transcription Server                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  Media Files                        Server                        Browser         │
│  ───────────                       ────────                      ─────────        │
│                                                                                   │
│  ┌─────────┐                 ┌──────────────────────────────┐                    │
│  │ Media   │                 │    Session Manager           │                    │
│  │ Manager │◄───────────────►│    ┌────────────────────┐   │                    │
│  │(./media)│                 │    │ Session 1          │   │    WebRTC          │
│  └─────────┘                 │    │  ├─ FFmpeg Process │───┼──────────►  🔊     │
│                              │    │  ├─ Opus Encoder   │   │    Audio           │
│  ┌─────────┐                 │    │  └─ Transcriber    │   │                    │
│  │ Model   │                 │    └────────────────────┘   │                    │
│  │Registry │◄───────────────►│    ┌────────────────────┐   │    WebSocket       │
│  │(TDT/    │                 │    │ Session 2 ...      │───┼──────────►  📝     │
│  │ Canary) │                 │    └────────────────────┘   │    Subtitles       │
│  └─────────┘                 │              │              │                    │
│                              └──────────────┼──────────────┘                    │
│                                             │                                    │
│                                             ▼                                    │
│                              ┌──────────────────────────────┐                    │
│                              │    HTTP/WS Server (Axum)     │                    │
│                              │    ┌────────────────────┐    │                    │
│                              │    │ /api/sessions      │    │                    │
│                              │    │ /api/models        │    │                    │
│                              │    │ /api/media         │    │                    │
│                              │    │ /api/modes         │    │                    │
│                              │    │ /ws/:session_id    │    │                    │
│                              │    └────────────────────┘    │                    │
│                              └──────────────────────────────┘                    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Session Manager

Manages multiple concurrent transcription sessions with independent models and configurations.

Each session maintains:
- `id`: Unique session identifier
- `model_id`: Selected transcription model
- `media_id`: Reference to media file
- `mode`: Latency mode (speedy, vad_speedy, etc.)
- `language`: Target language for transcription
- `state`: Current state (starting, running, completed, stopped)
- `progress_secs`: Current playback position
- `duration_secs`: Total media duration
- `client_count`: Number of connected WebRTC clients

### 2. Model Registry

Discovers and loads transcription models (TDT, Canary) and diarization models from environment variables.

**Supported Models:**

| Model | Description | Languages |
|-------|-------------|-----------|
| **TDT** | Token-and-Duration Transducer from NVIDIA NeMo | 25 languages |
| **Canary** | Multilingual transcription model | Configurable |
| **VAD+TDT** | Voice Activity Detection + TDT | 25 languages |
| **VAD+Canary** | Voice Activity Detection + Canary | Configurable |

### 3. Media Manager

Handles audio file uploads, storage, and lifecycle in the `./media` directory.

- Supports WAV and MP3 formats
- Maximum upload size: 1GB
- Automatic duration detection via ffprobe

### 4. WebRTC Server

Axum-based HTTP/WebSocket server for signaling and audio streaming.

- HTTP REST API for session/media management
- WebSocket for signaling and subtitle delivery
- WebRTC for ultra-low-latency audio (~100-400ms)

### 5. Transcription Engine

Supports multiple model types with different processing strategies:

- **TDT (Token-and-Duration Transducer)**: Word-level timestamps, fast inference
- **Canary**: Multilingual support with language-aware processing
- **VAD+TDT/Canary**: Voice Activity Detection triggered transcription

### 6. Speaker Diarization

Sortformer-based speaker identification (up to 4 speakers).

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

---

## Multi-Session Architecture

The server supports multiple concurrent transcription sessions, each with:
- Independent model selection (TDT or Canary)
- Independent latency mode
- Independent media file
- Independent language setting
- Per-session WebRTC audio track
- Per-session subtitle broadcast channel

### Session Lifecycle

```
┌─────────┐     POST /api/sessions      ┌──────────┐
│ Created │◄────────────────────────────│  Client  │
└────┬────┘                             └──────────┘
     │
     │ POST /api/sessions/:id/start
     ▼
┌─────────┐
│ Running │◄─── Audio streaming + Transcription
└────┬────┘
     │
     │ DELETE /api/sessions/:id
     │ or audio completes
     ▼
┌───────────┐
│ Completed │
└───────────┘
```

### Session States

| State | Description |
|-------|-------------|
| `starting` | Session created, waiting for start |
| `running` | Transcription in progress |
| `completed` | Audio finished, transcription complete |
| `stopped` | Manually stopped by user |

---

## Audio Pipeline

### Input Processing

Audio is processed from media files stored in the media directory. The server uses FFmpeg with real-time pacing (`-re` flag) to simulate live streaming:

```bash
ffmpeg -re -i /media/input.wav -f s16le -ar 16000 -ac 1 -loglevel error -
```

The `-re` flag ensures real-time playback speed, essential for synchronized audio and subtitle delivery.

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

The server uses WebSocket (`/ws/:session_id`) for SDP and ICE candidate exchange:

```
Browser                    Server
   │                          │
   │──── WS Connect ─────────▶│  /ws/{session_id}
   │                          │
   │◀── { type: "welcome",    │  Session info + client ID
   │      session: {...} }────│
   │                          │
   │──── { type: "ready" }───▶│  Request offer
   │                          │
   │◀─── { type: "offer",  ───│  SDP offer
   │       sdp: "..." }       │
   │                          │
   │──── { type: "answer", ──▶│  SDP answer
   │       sdp: "..." }       │
   │                          │
   │◀── { type: "ice-candidate" }  ICE candidates
   │                          │
   │── { type: "ice-candidate" } ──▶
   │                          │
   │◀═══ RTP Audio Stream ════│  Opus encoded audio
   │                          │
   │◀── { type: "subtitle" }──│  Transcription segments
   │                          │
   │◀── { type: "status" }────│  Progress updates
   │                          │
   │◀── { type: "end" }───────│  Stream complete
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

## Data Flow Summary

1. **Client uploads media** → Media Manager stores in `./media`
2. **Client creates session** → Session Manager allocates resources
3. **Client starts session** → FFmpeg process spawns, transcription begins
4. **Client joins via WebSocket** → WebRTC peer connection established
5. **Audio flows** → FFmpeg → Opus Encoder → WebRTC → Browser
6. **Transcription flows** → Transcriber → Subtitle broadcast → WebSocket → Browser
7. **Session completes** → Resources released, state updated

---

## Related Documentation

- [API Reference](./api-reference.md) - Complete REST and WebSocket API
- [Latency Modes](./latency-modes.md) - 10 transcription modes explained
- [Deployment](./deployment.md) - Configuration and Docker setup
