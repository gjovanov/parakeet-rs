# Multi-Session, Multi-Model Transcription Architecture

## Overview

This document outlines the architecture for extending parakeet-rs to support:
- Multiple transcription models (Parakeet TDT, Canary 1B, future Whisper v3)
- Dynamic media file selection and upload
- Concurrent transcription sessions that users can join
- Clean, extensible architecture following best practices

---

## Current Architecture Analysis

### Backend (webrtc_transcriber.rs)
```
┌─────────────────────────────────────────────────────────────┐
│                   webrtc_transcriber                         │
├─────────────────────────────────────────────────────────────┤
│  stdin (PCM) ──► Single Transcriber ──► Subtitles ──► WS   │
│       │                                                      │
│       └──► Opus Encoder ──► Single Track ──► All Clients    │
└─────────────────────────────────────────────────────────────┘
```

**Limitations:**
- Single hardcoded TDT model
- Single audio source (stdin)
- All clients share the same session
- No session management
- No media file management

### Frontend (index-webrtc.html)
- Simple connect/disconnect
- No model selection
- No file selection
- No session browser

---

## Proposed Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Session Transcription Server                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Media     │    │              Session Manager                        │ │
│  │   Manager   │    │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │             │    │  │Session 1 │  │Session 2 │  │Session N │          │ │
│  │ ./media/    │───►│  │TDT Model │  │Canary 1B │  │TDT Model │          │ │
│  │ *.wav       │    │  │file_a.wav│  │file_b.wav│  │file_c.wav│          │ │
│  │ *.mp3→wav   │    │  │3 clients │  │1 client  │  │5 clients │          │ │
│  └─────────────┘    │  └──────────┘  └──────────┘  └──────────┘          │ │
│                     └─────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────────┐ │
│  │   Model     │    │                    API Layer                        │ │
│  │   Registry  │    │  POST /api/sessions      - Create session           │ │
│  │             │    │  GET  /api/sessions      - List sessions            │ │
│  │ - TDT 0.6B  │    │  GET  /api/sessions/:id  - Get session details      │ │
│  │ - Canary 1B │    │  DELETE /api/sessions/:id - Stop session            │ │
│  │ - (Whisper) │    │  POST /api/media/upload  - Upload media file        │ │
│  └─────────────┘    │  GET  /api/media         - List media files         │ │
│                     │  GET  /api/models        - List available models    │ │
│                     │  WS   /ws/:session_id    - Join session             │ │
│                     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Backend Implementation Plan

### Phase 1: Core Abstractions

#### 1.1 Model Registry Trait (`src/model_registry.rs`)

```rust
/// Trait for transcription models that can be used in streaming sessions
pub trait StreamingTranscriber: Send + Sync {
    /// Model identifier
    fn model_id(&self) -> &str;

    /// Human-readable name
    fn display_name(&self) -> &str;

    /// Push audio samples and get transcription results
    fn push_audio(&mut self, samples: &[f32]) -> Result<TranscriptionChunk>;

    /// Finalize and get remaining transcription
    fn finalize(&mut self) -> Result<TranscriptionChunk>;

    /// Reset state for new stream
    fn reset(&mut self);
}

/// Transcription chunk result (model-agnostic)
pub struct TranscriptionChunk {
    pub segments: Vec<TranscriptionSegment>,
    pub is_final: bool,
}

pub struct TranscriptionSegment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub speaker: Option<usize>,
    pub confidence: Option<f32>,
}
```

#### 1.2 Model Registry (`src/model_registry.rs`)

```rust
/// Available transcription models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    ParakeetTDT,
    Canary1B,
    // Future: WhisperV3,
}

/// Model configuration
pub struct ModelConfig {
    pub model_type: ModelType,
    pub model_path: PathBuf,
    pub diarization_path: Option<PathBuf>,
    pub execution_config: ExecutionConfig,
}

/// Registry for available models
pub struct ModelRegistry {
    models: HashMap<String, ModelConfig>,
}

impl ModelRegistry {
    /// Load models from configuration
    pub fn from_config(config: &AppConfig) -> Result<Self>;

    /// List available models
    pub fn list_models(&self) -> Vec<ModelInfo>;

    /// Create transcriber instance for a model
    pub fn create_transcriber(&self, model_id: &str) -> Result<Box<dyn StreamingTranscriber>>;
}
```

#### 1.3 Canary 1B Integration (`src/canary.rs`)

**Note:** Canary 1B uses encoder-decoder architecture (different from TDT transducer).
Need to check for ONNX exports on HuggingFace.

```rust
/// Canary 1B streaming transcriber wrapper
pub struct CanaryStreaming {
    model: CanaryModel,  // New model implementation
    config: CanaryConfig,
    audio_buffer: VecDeque<f32>,
    // ... state management
}

impl StreamingTranscriber for CanaryStreaming {
    // Implementation
}
```

### Phase 2: Session Management

#### 2.1 Session Types (`src/session.rs`)

```rust
/// Unique session identifier
pub type SessionId = String;

/// Session state
#[derive(Debug, Clone, Serialize)]
pub enum SessionState {
    Starting,
    Running,
    Paused,
    Stopped,
    Error(String),
}

/// Transcription session
pub struct TranscriptionSession {
    pub id: SessionId,
    pub model_id: String,
    pub media_file: PathBuf,
    pub state: SessionState,
    pub created_at: DateTime<Utc>,
    pub client_count: AtomicU64,

    // Broadcast channels
    subtitle_tx: broadcast::Sender<String>,
    audio_track: Arc<TrackLocalStaticRTP>,

    // Control
    running: Arc<AtomicBool>,
}

impl TranscriptionSession {
    /// Create new session
    pub fn new(
        model_id: String,
        media_file: PathBuf,
        model_registry: &ModelRegistry,
    ) -> Result<Self>;

    /// Start transcription in background thread
    pub fn start(&self) -> Result<()>;

    /// Stop transcription
    pub fn stop(&self);

    /// Subscribe to subtitles
    pub fn subscribe_subtitles(&self) -> broadcast::Receiver<String>;

    /// Get audio track for WebRTC
    pub fn audio_track(&self) -> Arc<TrackLocalStaticRTP>;
}
```

#### 2.2 Session Manager (`src/session_manager.rs`)

```rust
/// Manages all active transcription sessions
pub struct SessionManager {
    sessions: RwLock<HashMap<SessionId, Arc<TranscriptionSession>>>,
    model_registry: Arc<ModelRegistry>,
    media_manager: Arc<MediaManager>,
    max_sessions: usize,
}

impl SessionManager {
    /// Create new session
    pub async fn create_session(
        &self,
        model_id: &str,
        media_file: &str,
    ) -> Result<SessionId>;

    /// Get session by ID
    pub async fn get_session(&self, id: &SessionId) -> Option<Arc<TranscriptionSession>>;

    /// List all sessions
    pub async fn list_sessions(&self) -> Vec<SessionInfo>;

    /// Stop and remove session
    pub async fn stop_session(&self, id: &SessionId) -> Result<()>;

    /// Cleanup stopped sessions
    pub async fn cleanup(&self);
}
```

### Phase 3: Media Management

#### 3.1 Media Manager (`src/media_manager.rs`)

```rust
/// Supported media formats
pub enum MediaFormat {
    Wav,
    Mp3,
}

/// Media file info
#[derive(Debug, Clone, Serialize)]
pub struct MediaFile {
    pub id: String,
    pub filename: String,
    pub path: PathBuf,
    pub format: MediaFormat,
    pub duration_secs: Option<f32>,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
}

/// Manages media files in ./media directory
pub struct MediaManager {
    media_dir: PathBuf,
    files: RwLock<HashMap<String, MediaFile>>,
}

impl MediaManager {
    /// Scan media directory for files
    pub async fn scan(&self) -> Result<()>;

    /// List all media files
    pub async fn list_files(&self) -> Vec<MediaFile>;

    /// Upload new file (handles mp3→wav conversion)
    pub async fn upload(&self, filename: &str, data: Bytes) -> Result<MediaFile>;

    /// Get file path for transcription
    pub async fn get_wav_path(&self, file_id: &str) -> Result<PathBuf>;

    /// Delete media file
    pub async fn delete(&self, file_id: &str) -> Result<()>;
}
```

#### 3.2 Audio Conversion (`src/audio_convert.rs`)

```rust
/// Convert MP3 to WAV using ffmpeg
pub async fn convert_mp3_to_wav(input: &Path, output: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args([
            "-i", input.to_str().unwrap(),
            "-ar", "16000",      // 16kHz sample rate
            "-ac", "1",          // Mono
            "-f", "wav",
            "-y",                // Overwrite
            output.to_str().unwrap(),
        ])
        .status()
        .await?;

    if !status.success() {
        return Err(Error::Conversion("ffmpeg failed".into()));
    }
    Ok(())
}

/// Get audio duration using ffprobe
pub async fn get_duration(path: &Path) -> Result<f32>;
```

### Phase 4: API Layer

#### 4.1 REST Endpoints (`src/api.rs`)

```rust
// Session endpoints
POST   /api/sessions              // Create new session
GET    /api/sessions              // List all sessions
GET    /api/sessions/:id          // Get session details
DELETE /api/sessions/:id          // Stop session

// Media endpoints
GET    /api/media                 // List media files
POST   /api/media/upload          // Upload new file
DELETE /api/media/:id             // Delete media file

// Model endpoints
GET    /api/models                // List available models

// Config endpoint (existing)
GET    /api/config                // Get frontend config
```

#### 4.2 WebSocket Protocol Update

```javascript
// Client -> Server
{ "type": "join", "session_id": "abc123" }         // Join existing session
{ "type": "ready" }                                 // Ready for WebRTC
{ "type": "answer", "sdp": "..." }                 // SDP answer
{ "type": "ice-candidate", "candidate": {...} }   // ICE candidate

// Server -> Client
{ "type": "session_info", "session": {...} }      // Session details
{ "type": "offer", "sdp": "..." }                 // SDP offer
{ "type": "subtitle", "text": "...", ... }        // Transcription
{ "type": "status", "state": "running", ... }     // Session status
{ "type": "error", "message": "..." }             // Error
```

### Phase 5: Application State

#### 5.1 Updated AppState

```rust
/// Shared application state
pub struct AppState {
    // Session management
    pub session_manager: Arc<SessionManager>,

    // Model registry
    pub model_registry: Arc<ModelRegistry>,

    // Media management
    pub media_manager: Arc<MediaManager>,

    // WebRTC API (shared)
    pub webrtc_api: webrtc::api::API,

    // Runtime config
    pub config: RuntimeConfig,
}
```

---

## Frontend Implementation Plan

### Phase 1: UI Components

#### 1.1 Session Browser Component

```javascript
// frontend/js/modules/session-browser.js

export class SessionBrowser extends EventEmitter {
    constructor(container) {
        this.container = container;
        this.sessions = [];
    }

    async refresh() {
        const response = await fetch('/api/sessions');
        this.sessions = await response.json();
        this.render();
    }

    render() {
        // Render session cards with:
        // - Model name
        // - Media file name
        // - Client count
        // - Duration/progress
        // - Join button
    }

    async joinSession(sessionId) {
        this.emit('join', sessionId);
    }
}
```

#### 1.2 Session Creator Component

```javascript
// frontend/js/modules/session-creator.js

export class SessionCreator extends EventEmitter {
    constructor(container) {
        this.container = container;
        this.models = [];
        this.mediaFiles = [];
    }

    async init() {
        await Promise.all([
            this.loadModels(),
            this.loadMediaFiles()
        ]);
        this.render();
    }

    async loadModels() {
        const response = await fetch('/api/models');
        this.models = await response.json();
    }

    async loadMediaFiles() {
        const response = await fetch('/api/media');
        this.mediaFiles = await response.json();
    }

    async createSession() {
        const modelId = this.modelSelect.value;
        const mediaId = this.mediaSelect.value;

        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, media_id: mediaId })
        });

        const session = await response.json();
        this.emit('created', session);
    }
}
```

#### 1.3 Media Upload Component

```javascript
// frontend/js/modules/media-upload.js

export class MediaUpload extends EventEmitter {
    constructor(container) {
        this.container = container;
    }

    render() {
        // File input for .wav and .mp3
        // Progress bar
        // Upload button
    }

    async upload(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/media/upload', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const mediaFile = await response.json();
            this.emit('uploaded', mediaFile);
        }
    }
}
```

### Phase 2: Updated Main Application

#### 2.1 New HTML Structure (`frontend/index-webrtc.html`)

```html
<!-- Session Control Panel -->
<section id="session-panel">
    <!-- Tab navigation -->
    <div class="tabs">
        <button class="tab active" data-tab="browse">Browse Sessions</button>
        <button class="tab" data-tab="create">Create Session</button>
        <button class="tab" data-tab="upload">Upload Media</button>
    </div>

    <!-- Browse existing sessions -->
    <div id="tab-browse" class="tab-content active">
        <div id="session-list"></div>
        <button id="refresh-sessions">Refresh</button>
    </div>

    <!-- Create new session -->
    <div id="tab-create" class="tab-content">
        <div class="form-group">
            <label for="model-select">Model:</label>
            <select id="model-select">
                <option value="parakeet-tdt">Parakeet TDT 0.6B</option>
                <option value="canary-1b">Canary 1B</option>
            </select>
        </div>
        <div class="form-group">
            <label for="media-select">Media File:</label>
            <select id="media-select"></select>
        </div>
        <button id="create-session">Start Transcription</button>
    </div>

    <!-- Upload media -->
    <div id="tab-upload" class="tab-content">
        <input type="file" id="media-file" accept=".wav,.mp3">
        <div id="upload-progress"></div>
        <button id="upload-btn">Upload</button>
    </div>
</section>
```

### Phase 3: Application Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Flow                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User opens page                                         │
│     └─► Load config from /api/config                        │
│     └─► Load models from /api/models                        │
│     └─► Load media from /api/media                          │
│     └─► Load sessions from /api/sessions                    │
│                                                              │
│  2. User browses sessions                                   │
│     └─► Click "Join" on existing session                    │
│     └─► WebSocket connects to /ws?session=<id>              │
│     └─► WebRTC established with session's audio track       │
│                                                              │
│  3. User creates new session                                │
│     └─► Select model from dropdown                          │
│     └─► Select media file from dropdown                     │
│     └─► POST /api/sessions                                  │
│     └─► Session created, auto-join                          │
│                                                              │
│  4. User uploads media                                      │
│     └─► Select .wav or .mp3 file                            │
│     └─► POST /api/media/upload (multipart)                  │
│     └─► Server converts mp3→wav if needed                   │
│     └─► File appears in media dropdown                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
parakeet-rs/
├── src/
│   ├── lib.rs                    # Library exports
│   ├── model_registry.rs         # NEW: Model abstraction & registry
│   ├── canary.rs                 # NEW: Canary 1B implementation
│   ├── streaming_transcriber.rs  # NEW: Common streaming trait
│   ├── session.rs                # NEW: Session management
│   ├── session_manager.rs        # NEW: Session lifecycle
│   ├── media_manager.rs          # NEW: Media file management
│   ├── audio_convert.rs          # NEW: ffmpeg conversion
│   └── ... (existing files)
│
├── examples/
│   └── webrtc_transcriber.rs     # UPDATED: Multi-session support
│
├── frontend/
│   ├── index-webrtc.html         # UPDATED: New UI
│   ├── css/
│   │   └── styles.css            # UPDATED: New components
│   └── js/
│       ├── main-webrtc.js        # UPDATED: Session handling
│       ├── config.js             # UPDATED: New config fields
│       └── modules/
│           ├── session-browser.js  # NEW
│           ├── session-creator.js  # NEW
│           ├── media-upload.js     # NEW
│           ├── webrtc.js           # UPDATED: Session param
│           └── ... (existing)
│
├── media/                         # NEW: Media storage directory
│   └── .gitkeep
│
└── models/                        # Model storage
    ├── tdt/                       # Existing TDT model
    └── canary/                    # NEW: Canary 1B model
```

---

## Configuration

### Environment Variables

```bash
# Server
PORT=8080
PUBLIC_IP=
WS_HOST=

# GPU
USE_GPU=true
INTRA_THREADS=2
INTER_THREADS=1

# Models
TDT_MODEL_PATH=/app/models/tdt
CANARY_MODEL_PATH=/app/models/canary
DIAR_MODEL_PATH=/app/models/diarization/model.onnx

# Media
MEDIA_DIR=/app/media
MAX_UPLOAD_SIZE_MB=500

# Sessions
MAX_CONCURRENT_SESSIONS=10
SESSION_TIMEOUT_SECS=3600
```

### Docker Volume Updates

```yaml
# docker-compose.yml
volumes:
  - ./tdt:/app/models/tdt:ro
  - ./canary:/app/models/canary:ro
  - ./diar_streaming_sortformer_4spk-v2.onnx:/app/models/diarization/model.onnx:ro
  - ./media:/app/media                    # NEW: Read-write for uploads
```

---

## Implementation Order

### Phase 1: Foundation (Backend)
1. Create `StreamingTranscriber` trait
2. Implement trait for existing `RealtimeTDTDiarized`
3. Create `ModelRegistry` with TDT only
4. Create basic `MediaManager` (list files)
5. Create basic `SessionManager` (single session)

### Phase 2: Multi-Session (Backend)
1. Implement full session lifecycle
2. Add session creation/destruction
3. Update WebSocket to route by session
4. Test multiple concurrent sessions

### Phase 3: Media Management (Backend)
1. Implement file upload endpoint
2. Add mp3→wav conversion
3. Add duration detection
4. Add file deletion

### Phase 4: Frontend - Session UI
1. Create session browser component
2. Create session creator component
3. Update WebRTC client for session routing
4. Test session joining

### Phase 5: Frontend - Media UI
1. Create media upload component
2. Add progress indicator
3. Refresh media list after upload
4. Test full flow

### Phase 6: Canary 1B Integration
1. Research ONNX export availability
2. Implement `CanaryModel`
3. Implement `CanaryStreaming`
4. Add to model registry
5. Test with sessions

### Phase 7: Polish & Documentation
1. Error handling improvements
2. Loading states
3. Documentation
4. Docker updates

---

## Extensibility for Future Models

To add a new model (e.g., Whisper v3):

1. **Create model implementation:**
   ```rust
   // src/whisper.rs
   pub struct WhisperStreaming { ... }
   impl StreamingTranscriber for WhisperStreaming { ... }
   ```

2. **Add to ModelType enum:**
   ```rust
   pub enum ModelType {
       ParakeetTDT,
       Canary1B,
       WhisperV3,  // NEW
   }
   ```

3. **Register in ModelRegistry:**
   ```rust
   impl ModelRegistry {
       pub fn create_transcriber(&self, model_id: &str) -> Result<Box<dyn StreamingTranscriber>> {
           match model_type {
               ModelType::ParakeetTDT => ...,
               ModelType::Canary1B => ...,
               ModelType::WhisperV3 => Box::new(WhisperStreaming::new(...)?),
           }
       }
   }
   ```

4. **Add environment variable:**
   ```bash
   WHISPER_MODEL_PATH=/app/models/whisper
   ```

5. **Update Docker volumes:**
   ```yaml
   - ./whisper:/app/models/whisper:ro
   ```

---

## Design Decisions

1. **Session persistence**: **Ephemeral** - Sessions are stored in-memory and lost on restart. Simpler implementation, suitable for demo/internal use.

2. **Authentication**: **No auth** - Anyone can create/join sessions. For internal/demo use cases.

3. **Max upload size**: **1 GB** - Approximately 2 hours of audio.

4. **Max concurrent sessions**: 10 (configurable via `MAX_CONCURRENT_SESSIONS`)

## Open Questions

1. **Canary 1B ONNX availability**: Need to verify if ONNX exports exist on HuggingFace or if we need to export from NeMo.

2. **Session audio isolation**: Each session should have its own audio track but can share the WebRTC API instance for resource efficiency.
