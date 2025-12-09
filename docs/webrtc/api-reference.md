# API Reference

> **Navigation**: [Index](./README.md) | [Architecture](./architecture.md) | API Reference | [Latency Modes](./latency-modes.md) | [Frontend](./frontend.md) | [Deployment](./deployment.md)

## REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend (serves static files) |
| `/health` | GET | Health check ("OK") |
| `/api/config` | GET | Runtime configuration (JSON) |
| `/api/models` | GET | List available transcription models |
| `/api/modes` | GET | List available latency modes |
| `/api/media` | GET | List uploaded media files |
| `/api/media/upload` | POST | Upload media file (multipart/form-data) |
| `/api/media/:id` | DELETE | Delete a media file |
| `/api/sessions` | GET | List all sessions |
| `/api/sessions` | POST | Create new session |
| `/api/sessions/:id` | GET | Get session info |
| `/api/sessions/:id/start` | POST | Start transcription |
| `/api/sessions/:id` | DELETE | Stop session |
| `/ws/:session_id` | WebSocket | Join session for subtitles and audio |

---

## Response Format

All API responses follow a consistent format:

### Success Response

```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

### Error Response

```json
{
  "success": false,
  "data": null,
  "error": "Error message"
}
```

---

## Session Endpoints

### Create Session

```http
POST /api/sessions
Content-Type: application/json
```

**Request Body:**

```json
{
  "model_id": "tdt-en",
  "media_id": "abc123-uuid",
  "mode": "speedy",
  "language": "en"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Transcription model ID |
| `media_id` | string | Yes | Uploaded media file ID |
| `mode` | string | No | Latency mode (default: `speedy`) |
| `language` | string | No | Language code (default: `de`) |

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "session-uuid",
    "model_id": "tdt-en",
    "model_name": "TDT English",
    "media_id": "media-uuid",
    "media_filename": "interview.wav",
    "mode": "speedy",
    "language": "en",
    "state": "starting",
    "progress_secs": 0,
    "duration_secs": 120.0,
    "client_count": 0
  }
}
```

### List Sessions

```http
GET /api/sessions
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "session-uuid",
      "model_id": "tdt-en",
      "model_name": "TDT English",
      "media_id": "media-uuid",
      "media_filename": "interview.wav",
      "mode": "speedy",
      "language": "en",
      "state": "running",
      "progress_secs": 45.2,
      "duration_secs": 120.0,
      "client_count": 2
    }
  ]
}
```

### Get Session

```http
GET /api/sessions/:id
```

**Response:** Same as session object above.

### Start Session

```http
POST /api/sessions/:id/start
```

Starts transcription for the session. The session must be in `starting` state.

**Response:** Updated session object with `state: "running"`.

### Stop Session

```http
DELETE /api/sessions/:id
```

Stops the session and releases all resources.

**Response:**

```json
{
  "success": true,
  "data": null
}
```

---

## Media Endpoints

### List Media Files

```http
GET /api/media
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "media-uuid",
      "filename": "interview.wav",
      "size_bytes": 15360000,
      "duration_secs": 120.0
    }
  ]
}
```

### Upload Media File

```http
POST /api/media/upload
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: Audio file (WAV or MP3, max 1GB)

**Response:**

```json
{
  "success": true,
  "data": {
    "id": "media-uuid",
    "filename": "interview.wav",
    "size_bytes": 15360000,
    "duration_secs": 120.0
  }
}
```

### Delete Media File

```http
DELETE /api/media/:id
```

**Response:**

```json
{
  "success": true,
  "data": null
}
```

---

## Model and Mode Endpoints

### List Models

```http
GET /api/models
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "tdt-en",
      "display_name": "TDT English",
      "model_path": "/app/models/tdt",
      "diarization_path": "/app/models/diar.onnx"
    },
    {
      "id": "canary-1b",
      "display_name": "Canary Multilingual",
      "model_path": "/app/models/canary"
    }
  ]
}
```

### List Modes

```http
GET /api/modes
```

**Response:**

```json
{
  "success": true,
  "data": [
    {
      "id": "speedy",
      "name": "Speedy (~0.3-1.5s)",
      "description": "Best balance of latency and quality. Uses pause detection."
    },
    {
      "id": "vad_speedy",
      "name": "VAD Speedy (~0.3s pause)",
      "description": "Silero VAD triggered. Transcribes complete utterances after short pauses."
    }
  ]
}
```

See [Latency Modes](./latency-modes.md) for complete mode documentation.

---

## Configuration Endpoint

### Get Config

```http
GET /api/config
```

Returns runtime configuration for the frontend.

**Response:**

```json
{
  "wsUrl": "ws://203.0.113.50:8080/ws",
  "iceServers": [
    { "urls": "stun:stun.l.google.com:19302" },
    {
      "urls": "turns:coturn.example.com:443",
      "username": "myuser",
      "credential": "mysecret"
    }
  ],
  "speakerColors": [
    "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
    "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98"
  ]
}
```

---

## WebSocket API

### Connection

```
ws://host:port/ws/:session_id
```

Connect to a specific session to receive audio via WebRTC and subtitles via WebSocket.

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `welcome` | Server → Client | Session info and client ID |
| `ready` | Client → Server | Request WebRTC offer |
| `offer` | Server → Client | SDP offer |
| `answer` | Client → Server | SDP answer |
| `ice-candidate` | Bidirectional | ICE candidate exchange |
| `subtitle` | Server → Client | Transcription segment |
| `status` | Server → Client | Progress update |
| `end` | Server → Client | Stream completed |
| `error` | Server → Client | Error message |

### Welcome Message

Sent immediately upon connection:

```json
{
  "type": "welcome",
  "client_id": "abc123",
  "session": {
    "id": "session-uuid",
    "model_id": "tdt-en",
    "state": "running",
    "progress_secs": 45.2,
    "duration_secs": 120.0
  }
}
```

### Ready Message

Client sends to request WebRTC offer:

```json
{
  "type": "ready"
}
```

### Offer/Answer Messages

SDP exchange for WebRTC:

```json
{
  "type": "offer",
  "sdp": "v=0\r\no=- ..."
}
```

```json
{
  "type": "answer",
  "sdp": "v=0\r\no=- ..."
}
```

### ICE Candidate Message

```json
{
  "type": "ice-candidate",
  "candidate": {
    "candidate": "candidate:...",
    "sdpMLineIndex": 0,
    "sdpMid": "audio"
  }
}
```

### Subtitle Message

Transcription segment with speaker info:

```json
{
  "type": "subtitle",
  "text": "Hello, how are you today?",
  "speaker": 0,
  "start": 0.50,
  "end": 2.30,
  "is_final": true,
  "inference_time_ms": 45
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Transcribed text |
| `speaker` | number | Speaker ID (0-3) |
| `start` | number | Start time in seconds |
| `end` | number | End time in seconds |
| `is_final` | boolean | Whether segment is finalized |
| `inference_time_ms` | number | Inference time in milliseconds |

### Status Message

Progress update:

```json
{
  "type": "status",
  "progress_secs": 45.2,
  "total_duration": 120.0
}
```

### End Message

Stream completed:

```json
{
  "type": "end",
  "total_duration": 120.0
}
```

### Error Message

```json
{
  "type": "error",
  "message": "Error description"
}
```

---

## Example: Complete Session Flow

```bash
# 1. Upload a media file
curl -X POST http://localhost:8080/api/media/upload \
  -F "file=@interview.wav"
# Response: {"success":true,"data":{"id":"media-123",...}}

# 2. Create a session
curl -X POST http://localhost:8080/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"model_id":"tdt-en","media_id":"media-123","mode":"speedy","language":"en"}'
# Response: {"success":true,"data":{"id":"session-456",...}}

# 3. Start the session
curl -X POST http://localhost:8080/api/sessions/session-456/start
# Response: {"success":true,"data":{"state":"running",...}}

# 4. Connect via WebSocket
# ws://localhost:8080/ws/session-456

# 5. Stop the session when done
curl -X DELETE http://localhost:8080/api/sessions/session-456
# Response: {"success":true,"data":null}
```

---

## Related Documentation

- [Architecture](./architecture.md) - System design and data flow
- [Frontend](./frontend.md) - JavaScript client implementation
- [Deployment](./deployment.md) - Server configuration
