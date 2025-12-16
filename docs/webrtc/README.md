# WebRTC Transcription Server Documentation

Real-time speech transcription with WebRTC audio delivery and live subtitles.

## Overview

The parakeet-rs WebRTC transcriber is a **multi-session transcription server** that supports multiple concurrent transcription sessions with different models and media files. Each session streams audio via WebRTC (~100-400ms latency) and subtitles via WebSocket.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](./architecture.md) | System architecture, components, and data flow |
| [API Reference](./api-reference.md) | REST API and WebSocket endpoints |
| [Latency Modes](./latency-modes.md) | 12 transcription modes with trade-offs |
| [Frontend Guide](./frontend.md) | Web UI components and JavaScript API |
| [Deployment](./deployment.md) | Configuration, Docker, and production setup |

## Quick Start

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

## Features

- **Multi-session support** - Run multiple concurrent transcription sessions
- **Multiple models** - TDT (25 languages), Canary (multilingual), VAD variants
- **12 latency modes** - From ~1.3s (extreme low latency) to highest quality (lookahead), plus parallel modes
- **Noise cancellation** - RNNoise (lightweight) or DeepFilterNet3 (high-quality) for cleaner audio
- **Speaker diarization** - Sortformer identifies up to 4 speakers (toggleable per session)
- **WebRTC audio** - Ultra-low latency audio delivery (~100-400ms)
- **RESTful API** - Programmatic session and media management
- **Modern web UI** - Multi-session interface with real-time subtitles

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                Multi-Session WebRTC Transcription Server          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────┐     ┌──────────────────────┐     ┌─────────┐       │
│  │ Media   │────►│   Session Manager    │────►│ Browser │       │
│  │ Manager │     │  ┌────────────────┐  │     │         │       │
│  └─────────┘     │  │ Session 1      │  │     │ WebRTC  │       │
│                  │  │  └─Transcriber │──┼────►│  Audio  │       │
│  ┌─────────┐     │  ├────────────────┤  │     │         │       │
│  │ Model   │────►│  │ Session 2 ...  │──┼────►│WebSocket│       │
│  │Registry │     │  └────────────────┘  │     │Subtitles│       │
│  └─────────┘     └──────────────────────┘     └─────────┘       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | POST | Create new transcription session |
| `/api/sessions/:id/start` | POST | Start transcription |
| `/api/media/upload` | POST | Upload media file |
| `/ws/:session_id` | WS | Join session for audio + subtitles |

See [API Reference](./api-reference.md) for complete documentation.

## Supported Languages

25 European languages via Canary model:

German, English, French, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, Czech, Slovak, Hungarian, Romanian, Bulgarian, Croatian, Slovenian, Estonian, Latvian, Lithuanian, Finnish, Swedish, Danish, Greek, Maltese

## Further Reading

- [WebRTC.rs Documentation](https://webrtc.rs/)
- [NVIDIA NeMo TDT](https://docs.nvidia.com/nemo-framework/)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Opus Codec](https://opus-codec.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
