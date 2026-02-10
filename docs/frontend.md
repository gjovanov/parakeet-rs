# Frontend

[Architecture](architecture.md) | [API Reference](api.md) | [Transcription Modes](transcription-modes.md) | [Frontend](frontend.md) | [FAB Teletext](fab-teletext.md) | [Testing](testing.md) | [Deployment](deployment.md)

---

## Overview

The frontend is a vanilla JavaScript web application served as static files from the `frontend/` directory. It provides a UI for managing transcription sessions, viewing live subtitles, and controlling WebRTC audio playback.

```mermaid
graph TD
    subgraph UI["Web UI (index.html)"]
        CTRL["Session Controls<br/>Create / Start / Stop"]
        LIVE["Live Subtitle Display<br/>Partial + Final"]
        TRANS["Transcript Panel<br/>Scrolling history"]
        AUDIO["Audio Controls<br/>Play / Mute / Volume"]
        EXPORT["Export<br/>Download transcript"]
    end

    subgraph Modules["JavaScript Modules"]
        SM["session-manager.js<br/>Session CRUD + lifecycle"]
        WSM["websocket.js<br/>WS connection + reconnect"]
        WRTC["webrtc.js<br/>Peer connection + ICE"]
        SUB["subtitles.js<br/>Text processing + display"]
        FU["file-upload.js<br/>Media upload handling"]
        AUD["audio.js<br/>Audio playback controls"]
        UTL["utils.js<br/>Shared utilities"]
    end

    CTRL --> SM
    SM --> WSM
    WSM --> SUB
    SM --> WRTC
    WRTC --> AUD
    CTRL --> FU
    SUB --> LIVE & TRANS
    TRANS --> EXPORT
```

## JavaScript Modules

| Module | File | Description |
|--------|------|-------------|
| **Session Manager** | `session-manager.js` | Session CRUD via REST API, lifecycle coordination, UI state management |
| **WebSocket** | `websocket.js` | WebSocket connection, automatic reconnection, message routing |
| **WebRTC** | `webrtc.js` | RTCPeerConnection setup, SDP offer/answer, ICE candidate exchange |
| **Subtitles** | `subtitles.js` | `_processSegment()` pipeline: growing text merging, teletext splitting, deduplication, display |
| **File Upload** | `file-upload.js` | Drag-and-drop and button-based media upload via multipart POST |
| **Audio** | `audio.js` | WebRTC audio element management, play/mute/volume controls |
| **Utils** | `utils.js` | Shared utility functions |

## Session Flow

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Web UI
    participant API as REST API
    participant WS as WebSocket
    participant RTC as WebRTC

    U->>UI: Select model + media + mode
    UI->>API: POST /api/sessions
    API-->>UI: {id: "abc123", state: "created"}

    UI->>WS: Connect /ws/abc123
    WS-->>UI: Connected

    UI->>RTC: Create RTCPeerConnection
    RTC->>WS: {type: "offer", sdp: "..."}
    WS-->>RTC: {type: "answer", sdp: "..."}
    RTC-->>RTC: ICE candidate exchange

    U->>UI: Click Start
    UI->>API: POST /api/sessions/abc123/start
    API-->>UI: {state: "running"}

    loop Transcription
        WS-->>UI: {type: "subtitle", is_final: false}
        UI->>UI: _processSegment() → live display
        WS-->>UI: {type: "subtitle", is_final: true}
        UI->>UI: append to transcript
    end

    WS-->>UI: {type: "end"}
    UI->>UI: Show completion
```

## Subtitle Processing Pipeline

The `_processSegment()` function in `subtitles.js` is the core text processing pipeline:

```mermaid
flowchart TD
    MSG["WebSocket Message<br/>{type: subtitle}"] --> ROUTE{"is_final?"}

    ROUTE -- "no (partial)" --> GT["Use growing_text<br/>(cumulative sentence)"]
    GT --> SPLIT["splitForTeletext()<br/>(42x2 = 84 chars max)"]
    SPLIT --> DISPLAY["Update live<br/>subtitle area"]

    ROUTE -- "yes (final)" --> DEDUP["isDuplicateOfHistory()<br/>(15-entry history)"]
    DEDUP -- "duplicate" --> SKIP["Skip"]
    DEDUP -- "new" --> APPEND["appendToTranscript()"]
    APPEND --> SCROLL["Auto-scroll<br/>transcript panel"]
```

### `splitForTeletext(text, maxChars)`

Splits text into lines of at most 84 characters (teletext constraint: 42 chars/line x 2 lines), preferring breaks at:
1. Sentence boundaries (`.` `!` `?`)
2. Clause separators (`,` `;` `:` `–` `—`)
3. Word boundaries (spaces)

### `isDuplicateOfHistory(text, history)`

Checks if text is a duplicate of any of the last 15 entries using:
- Exact match
- Substring containment
- Containment coefficient: `|A∩B| / min(|A|, |B|)` with threshold 0.75 and minimum 3 shared words

## WebRTC Audio Playback

The browser receives audio via WebRTC for synchronized playback alongside subtitles.

```mermaid
flowchart LR
    SRV["Server<br/>Opus/RTP Track"] --> PC["RTCPeerConnection"]
    PC --> RE["Remote Stream"]
    RE --> AE["Audio Element<br/>(autoplay)"]
```

### Reconnection Logic

When the WebRTC connection fails:

1. ICE connection state changes to `"failed"` or `"disconnected"`
2. 5-second timeout before closing
3. Peer connection is closed
4. Automatic reconnect with exponential backoff
5. New SDP offer/answer exchange

### ICE Configuration

The frontend fetches ICE server configuration from `GET /api/config`:

- **STUN**: `stun:stun.l.google.com:19302` (unless relay-only mode)
- **TURN**: Configured server with both UDP and TCP transport
- **Transport policy**: `"relay"` when `FORCE_RELAY=true`, `"all"` otherwise
