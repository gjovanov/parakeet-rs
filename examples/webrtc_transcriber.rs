//! WebRTC Real-Time Transcription Server with Multi-Session Support
//!
//! Supports multiple concurrent transcription sessions with different models and media files.
//! Each session streams audio via WebRTC and subtitles via WebSocket.
//!
//! API Endpoints:
//!   GET    /api/models                - List available models
//!   GET    /api/media                 - List media files
//!   POST   /api/media/upload          - Upload media file (multipart)
//!   DELETE /api/media/:id             - Delete media file
//!   GET    /api/sessions              - List all sessions
//!   POST   /api/sessions              - Create new session {model_id, media_id}
//!   GET    /api/sessions/:id          - Get session info
//!   POST   /api/sessions/:id/start    - Start transcription
//!   DELETE /api/sessions/:id          - Stop session
//!   WS     /ws/:session_id            - Join session for subtitles and audio
//!
//! Usage:
//!   cargo run --release --example webrtc_transcriber --features sortformer
//!
//! Architecture:
//!   SessionManager -> Session 1 -> Subtitles -> WebSocket
//!        |                |
//!        |                +-> Opus -> WebRTC -> Browser
//!        +-> Session N -> (same per session)
//!
//!   MediaManager -> ./media/ (wav, mp3)
//!   ModelRegistry -> Parakeet TDT, Canary 1B

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, Path, State,
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use bytes::Bytes;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use parakeet_rs::{
    MediaManager, MediaManagerConfig, ModelRegistry, SessionManager, SessionState,
    SharedMediaManager, SharedModelRegistry, SharedSessionManager, TranscriptionSession,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;
use webrtc::{
    api::{
        interceptor_registry::register_default_interceptors,
        media_engine::{MediaEngine, MIME_TYPE_OPUS},
        APIBuilder,
    },
    ice::network_type::NetworkType,
    ice_transport::{
        ice_candidate::RTCIceCandidateInit,
        ice_credential_type::RTCIceCredentialType,
        ice_server::RTCIceServer,
    },
    interceptor::registry::Registry,
    peer_connection::{
        configuration::RTCConfiguration,
        policy::ice_transport_policy::RTCIceTransportPolicy,
        sdp::session_description::RTCSessionDescription,
        RTCPeerConnection,
    },
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::{
        track_local_static_rtp::TrackLocalStaticRTP, TrackLocal, TrackLocalWriter,
    },
};

/// Transcription latency mode
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LatencyMode {
    #[default]
    Speedy,
    PauseBased,
    LowLatency,
    UltraLowLatency,
    ExtremeLowLatency,
    Lookahead,
    /// VAD-triggered mode: transcribe only after speech pauses (Silero VAD)
    VadSpeedy,
    VadPauseBased,
    /// Pure streaming ASR mode: continuous transcription without VAD
    Asr,
}

impl LatencyMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            LatencyMode::Speedy => "speedy",
            LatencyMode::PauseBased => "pause_based",
            LatencyMode::LowLatency => "low_latency",
            LatencyMode::UltraLowLatency => "ultra_low_latency",
            LatencyMode::ExtremeLowLatency => "extreme_low_latency",
            LatencyMode::Lookahead => "lookahead",
            LatencyMode::VadSpeedy => "vad_speedy",
            LatencyMode::VadPauseBased => "vad_pause_based",
            LatencyMode::Asr => "asr",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            LatencyMode::Speedy => "Speedy (~0.3-1.5s)",
            LatencyMode::PauseBased => "Pause-Based (~0.5-2.0s)",
            LatencyMode::LowLatency => "Low Latency (~3.5s)",
            LatencyMode::UltraLowLatency => "Ultra-Low Latency (~2.5s)",
            LatencyMode::ExtremeLowLatency => "Extreme-Low Latency (~1.3s)",
            LatencyMode::Lookahead => "Lookahead (~1.0-3.0s)",
            LatencyMode::VadSpeedy => "VAD Speedy (~0.3s pause)",
            LatencyMode::VadPauseBased => "VAD Pause-Based (~0.7s pause)",
            LatencyMode::Asr => "ASR (Pure streaming)",
        }
    }

    /// Check if this mode uses VAD-triggered transcription
    pub fn is_vad_mode(&self) -> bool {
        matches!(self, LatencyMode::VadSpeedy | LatencyMode::VadPauseBased)
    }

    /// Get the underlying VAD mode string for VadConfig::from_mode()
    pub fn vad_mode_str(&self) -> &'static str {
        match self {
            LatencyMode::VadSpeedy => "speedy",
            LatencyMode::VadPauseBased => "pause_based",
            _ => "speedy", // Default for non-VAD modes
        }
    }

    pub fn all() -> &'static [LatencyMode] {
        &[
            LatencyMode::Speedy,
            LatencyMode::PauseBased,
            LatencyMode::LowLatency,
            LatencyMode::UltraLowLatency,
            LatencyMode::ExtremeLowLatency,
            LatencyMode::Lookahead,
            LatencyMode::VadSpeedy,
            LatencyMode::VadPauseBased,
            LatencyMode::Asr,
        ]
    }
}

#[derive(Parser)]
#[command(name = "webrtc_transcriber")]
#[command(about = "Multi-session WebRTC transcription server")]
struct Args {
    /// HTTP/WebSocket server port
    #[arg(long, env = "PORT", default_value = "8080")]
    port: u16,

    /// Path to TDT model directory
    #[arg(long, env = "TDT_MODEL_PATH", default_value = "./tdt")]
    tdt_model: String,

    /// Path to diarization model (ONNX)
    #[arg(long, env = "DIAR_MODEL_PATH", default_value = "diar_streaming_sortformer_4spk-v2.onnx")]
    diar_model: String,

    /// Path to Silero VAD model (ONNX)
    #[arg(long, env = "VAD_MODEL_PATH", default_value = "silero_vad.onnx")]
    vad_model: String,

    /// Path to Whisper Large v2 model directory (optional)
    #[arg(long, env = "WHISPER_V2_MODEL_PATH")]
    whisper_v2_model: Option<String>,

    /// Path to Whisper Large v3 model directory (optional)
    #[arg(long, env = "WHISPER_V3_MODEL_PATH")]
    whisper_v3_model: Option<String>,

    /// Path to frontend directory
    #[arg(long, env = "FRONTEND_PATH", default_value = "./frontend")]
    frontend: PathBuf,

    /// Media directory for audio files
    #[arg(long, env = "MEDIA_DIR", default_value = "./media")]
    media_dir: PathBuf,

    /// Public IP address for WebRTC ICE candidates
    #[arg(long, env = "PUBLIC_IP")]
    public_ip: Option<String>,

    /// Maximum concurrent sessions
    #[arg(long, env = "MAX_CONCURRENT_SESSIONS", default_value = "10")]
    max_sessions: usize,

    // Latency mode flags (mutually exclusive)
    /// Speedy mode: Best balance of latency and quality (~0.3-1.5s latency)
    #[arg(long, env = "SPEEDY_MODE")]
    speedy: bool,

    /// Pause-based mode: Better accuracy, slightly higher latency (~0.5-2.0s)
    #[arg(long)]
    pause_based: bool,

    /// Low-latency mode: Fixed latency without pause detection (~3.5s)
    #[arg(long)]
    low_latency: bool,

    /// Ultra-low-latency mode: Faster response (~2.5s)
    #[arg(long)]
    ultra_low_latency: bool,

    /// Extreme-low-latency mode: Fastest possible (~1.3s)
    #[arg(long)]
    extreme_low_latency: bool,

    /// Lookahead mode: Best quality with future context (~1.0-3.0s)
    #[arg(long)]
    lookahead: bool,
}

impl Args {
    fn latency_mode(&self) -> LatencyMode {
        if self.lookahead {
            LatencyMode::Lookahead
        } else if self.extreme_low_latency {
            LatencyMode::ExtremeLowLatency
        } else if self.ultra_low_latency {
            LatencyMode::UltraLowLatency
        } else if self.low_latency {
            LatencyMode::LowLatency
        } else if self.pause_based {
            LatencyMode::PauseBased
        } else {
            // Default to speedy (also when SPEEDY_MODE=1)
            LatencyMode::Speedy
        }
    }
}

/// Client connection with WebRTC peer connection
struct ClientConnection {
    id: String,
    session_id: String,
    peer_connection: Arc<RTCPeerConnection>,
    ice_tx: mpsc::Sender<String>,
}

/// Runtime configuration for frontend
#[derive(Clone)]
struct RuntimeConfig {
    ws_url: String,
    turn_server: String,
    turn_username: String,
    turn_password: String,
}

/// Per-session audio track state
struct SessionAudioState {
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    /// FFmpeg process ID for killing on session stop
    ffmpeg_pid: Arc<std::sync::atomic::AtomicU32>,
}

/// Shared application state
struct AppState {
    /// Session manager
    session_manager: SharedSessionManager,
    /// Model registry
    model_registry: SharedModelRegistry,
    /// Media manager
    media_manager: SharedMediaManager,
    /// WebRTC API
    api: webrtc::api::API,
    /// Connected clients by ID
    clients: Mutex<HashMap<String, ClientConnection>>,
    /// Total client count
    client_count: AtomicU64,
    /// Runtime configuration for frontend
    config: RuntimeConfig,
    /// Per-session audio tracks
    session_audio: RwLock<HashMap<String, SessionAudioState>>,
}

/// Request to create a new session
#[derive(Debug, Deserialize)]
struct CreateSessionRequest {
    model_id: String,
    media_id: String,
    #[serde(default)]
    mode: LatencyMode,
    /// Language code for transcription (default: "de" for German)
    #[serde(default = "default_language")]
    language: String,
}

fn default_language() -> String {
    "de".to_string()
}

/// Generic API response
#[derive(Debug, Serialize)]
struct ApiResponse<T> {
    success: bool,
    data: Option<T>,
    error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn error(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
        }
    }
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("Error: This example requires the 'sortformer' feature.");
    eprintln!("Run with: cargo run --release --example webrtc_transcriber --features sortformer");
    std::process::exit(1);
}

#[cfg(feature = "sortformer")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("===========================================");
    eprintln!("  Multi-Session Transcription Server");
    eprintln!("===========================================");
    eprintln!("Port: {}", args.port);
    eprintln!("TDT Model: {}", args.tdt_model);
    eprintln!("Diarization Model: {}", args.diar_model);
    eprintln!("VAD Model: {}", args.vad_model);
    eprintln!("Media Directory: {}", args.media_dir.display());
    eprintln!("Max Sessions: {}", args.max_sessions);
    eprintln!("Frontend: {}", args.frontend.display());
    eprintln!("===========================================");
    eprintln!();

    // Set environment variables for model registry
    std::env::set_var("TDT_MODEL_PATH", &args.tdt_model);
    std::env::set_var("DIAR_MODEL_PATH", &args.diar_model);
    std::env::set_var("VAD_MODEL_PATH", &args.vad_model);
    if let Some(ref path) = args.whisper_v2_model {
        eprintln!("Whisper v2 Model: {}", path);
        std::env::set_var("WHISPER_V2_MODEL_PATH", path);
    }
    if let Some(ref path) = args.whisper_v3_model {
        eprintln!("Whisper v3 Model: {}", path);
        std::env::set_var("WHISPER_V3_MODEL_PATH", path);
    }
    std::env::set_var("MEDIA_DIR", &args.media_dir);
    std::env::set_var("MAX_CONCURRENT_SESSIONS", args.max_sessions.to_string());

    // Initialize model registry
    let model_registry = Arc::new(ModelRegistry::from_env());

    // Initialize media manager
    let media_manager = Arc::new(MediaManager::new(MediaManagerConfig {
        media_dir: args.media_dir.clone(),
        max_upload_size: 1024 * 1024 * 1024, // 1GB
    }));
    media_manager.init().await?;

    // Initialize session manager
    let session_manager = Arc::new(SessionManager::new(
        model_registry.clone(),
        media_manager.clone(),
        args.max_sessions,
    ));

    // Create WebRTC API
    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;

    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;

    // Detect host IP
    let detected_ip = std::process::Command::new("hostname")
        .arg("-I")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .and_then(|s| s.split_whitespace().next().map(String::from))
        });

    let nat_ip = args
        .public_ip
        .clone()
        .or(detected_ip.clone())
        .unwrap_or_else(|| "127.0.0.1".to_owned());

    eprintln!("WebRTC NAT 1:1 IP: {}", nat_ip);

    let mut setting_engine = webrtc::api::setting_engine::SettingEngine::default();
    setting_engine.set_nat_1to1_ips(
        vec![nat_ip],
        webrtc::ice_transport::ice_candidate_type::RTCIceCandidateType::Host,
    );
    setting_engine.set_network_types(vec![
        NetworkType::Udp4,
        NetworkType::Udp6,
        NetworkType::Tcp4,
        NetworkType::Tcp6,
    ]);

    let api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .with_setting_engine(setting_engine)
        .build();

    // Build runtime config
    let turn_server = std::env::var("TURN_SERVER").unwrap_or_default();
    let turn_username = std::env::var("TURN_USERNAME").unwrap_or_default();
    let turn_password = std::env::var("TURN_PASSWORD").unwrap_or_default();

    let ws_host = std::env::var("WS_HOST")
        .ok()
        .or_else(|| args.public_ip.clone())
        .or_else(|| detected_ip.clone())
        .unwrap_or_else(|| "localhost".to_owned());

    let ws_url = format!("ws://{}:{}/ws", ws_host, args.port);

    let runtime_config = RuntimeConfig {
        ws_url: ws_url.clone(),
        turn_server: turn_server.clone(),
        turn_username: turn_username.clone(),
        turn_password: turn_password.clone(),
    };

    eprintln!("Frontend config: ws_url={}", ws_url);

    let state = Arc::new(AppState {
        session_manager,
        model_registry,
        media_manager,
        api,
        clients: Mutex::new(HashMap::new()),
        client_count: AtomicU64::new(0),
        config: runtime_config,
        session_audio: RwLock::new(HashMap::new()),
    });

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(|| async { "OK" }))
        // Config
        .route("/api/config", get(config_handler))
        // Models
        .route("/api/models", get(list_models))
        // Media
        .route("/api/media", get(list_media))
        .route("/api/media/upload", post(upload_media))
        .route("/api/media/:id", delete(delete_media))
        // Modes
        .route("/api/modes", get(list_modes))
        // Sessions
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions", post(create_session))
        .route("/api/sessions/:id", get(get_session))
        .route("/api/sessions/:id", delete(stop_session))
        .route("/api/sessions/:id/start", post(start_session))
        // WebSocket
        .route("/ws/:session_id", get(ws_handler))
        // Static frontend
        .fallback_service(ServeDir::new(&args.frontend))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    eprintln!("Server listening on http://{}", addr);
    eprintln!("API: http://{}/api/*", addr);
    eprintln!("Frontend: http://{}", addr);
    eprintln!();

    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// API Handlers: Models
// ============================================================================

#[cfg(feature = "sortformer")]
async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<parakeet_rs::ModelInfo>>> {
    let models = state.model_registry.list_models();
    Json(ApiResponse::success(models))
}

// ============================================================================
// API Handlers: Modes
// ============================================================================

/// Mode info for API responses
#[derive(Debug, Serialize)]
struct ModeInfo {
    id: &'static str,
    name: &'static str,
    description: &'static str,
}

#[cfg(feature = "sortformer")]
async fn list_modes() -> Json<ApiResponse<Vec<ModeInfo>>> {
    let modes = vec![
        ModeInfo {
            id: "speedy",
            name: "Speedy (~0.3-1.5s)",
            description: "Best balance of latency and quality. Uses pause detection.",
        },
        ModeInfo {
            id: "pause_based",
            name: "Pause-Based (~0.5-2.0s)",
            description: "Better accuracy with pause detection, slightly higher latency.",
        },
        ModeInfo {
            id: "low_latency",
            name: "Low Latency (~3.5s)",
            description: "Fixed latency without pause detection. Predictable timing.",
        },
        ModeInfo {
            id: "ultra_low_latency",
            name: "Ultra-Low Latency (~2.5s)",
            description: "Faster response time for interactive applications.",
        },
        ModeInfo {
            id: "extreme_low_latency",
            name: "Extreme-Low Latency (~1.3s)",
            description: "Fastest possible response. May sacrifice accuracy.",
        },
        ModeInfo {
            id: "lookahead",
            name: "Lookahead (~1.0-3.0s)",
            description: "Best quality with future context. Ideal for high accuracy.",
        },
        ModeInfo {
            id: "vad_speedy",
            name: "VAD Speedy (~0.3s pause)",
            description: "Silero VAD triggered. Transcribes complete utterances after short pauses.",
        },
        ModeInfo {
            id: "vad_pause_based",
            name: "VAD Pause-Based (~0.7s pause)",
            description: "Silero VAD triggered. More accurate with longer pause detection.",
        },
        ModeInfo {
            id: "asr",
            name: "ASR (Pure streaming)",
            description: "Pure streaming ASR without VAD. Processes audio continuously with sliding window.",
        },
    ];
    Json(ApiResponse::success(modes))
}

// ============================================================================
// API Handlers: Media
// ============================================================================

#[cfg(feature = "sortformer")]
async fn list_media(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<parakeet_rs::MediaFile>>> {
    let files = state.media_manager.list_files().await;
    Json(ApiResponse::success(files))
}

#[cfg(feature = "sortformer")]
async fn upload_media(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<ApiResponse<parakeet_rs::MediaFile>>, StatusCode> {
    while let Some(field) = multipart.next_field().await.map_err(|_| StatusCode::BAD_REQUEST)? {
        let filename = field.file_name().map(|s| s.to_string());
        let data = field.bytes().await.map_err(|_| StatusCode::BAD_REQUEST)?;

        if let Some(filename) = filename {
            match state.media_manager.upload(&filename, data).await {
                Ok(file) => return Ok(Json(ApiResponse::success(file))),
                Err(e) => return Ok(Json(ApiResponse::error(e.to_string()))),
            }
        }
    }

    Ok(Json(ApiResponse::error("No file provided")))
}

#[cfg(feature = "sortformer")]
async fn delete_media(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<ApiResponse<()>> {
    match state.media_manager.delete(&id).await {
        Ok(_) => Json(ApiResponse::success(())),
        Err(e) => Json(ApiResponse::error(e.to_string())),
    }
}

// ============================================================================
// API Handlers: Sessions
// ============================================================================

#[cfg(feature = "sortformer")]
async fn list_sessions(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<parakeet_rs::SessionInfo>>> {
    let sessions = state.session_manager.list_sessions().await;
    Json(ApiResponse::success(sessions))
}

#[cfg(feature = "sortformer")]
async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Json<ApiResponse<parakeet_rs::SessionInfo>> {
    let mode_str = req.mode.as_str();
    let language = if req.language.is_empty() { "de".to_string() } else { req.language };
    match state.session_manager.create_session(&req.model_id, &req.media_id, mode_str, &language).await {
        Ok(session) => {
            let info = session.info().await;
            Json(ApiResponse::success(info))
        }
        Err(e) => Json(ApiResponse::error(e.to_string())),
    }
}

#[cfg(feature = "sortformer")]
async fn get_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<ApiResponse<parakeet_rs::SessionInfo>> {
    match state.session_manager.get_session(&id).await {
        Some(session) => {
            let info = session.info().await;
            Json(ApiResponse::success(info))
        }
        None => Json(ApiResponse::error("Session not found")),
    }
}

#[cfg(feature = "sortformer")]
async fn stop_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<ApiResponse<()>> {
    // Stop the session
    if let Err(e) = state.session_manager.stop_session(&id).await {
        return Json(ApiResponse::error(e.to_string()));
    }

    // Clean up audio state and kill ffmpeg process
    {
        let mut audio_states = state.session_audio.write().await;
        if let Some(audio_state) = audio_states.remove(&id) {
            audio_state.running.store(false, Ordering::SeqCst);

            // Kill ffmpeg process to unblock the audio streaming thread
            let pid = audio_state.ffmpeg_pid.load(Ordering::SeqCst);
            if pid != 0 {
                eprintln!("[Session {}] Killing ffmpeg process (pid: {})", id, pid);
                #[cfg(unix)]
                unsafe {
                    libc::kill(pid as i32, libc::SIGKILL);
                }
            }
        }
    }

    Json(ApiResponse::success(()))
}

#[cfg(feature = "sortformer")]
async fn start_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<ApiResponse<parakeet_rs::SessionInfo>> {
    // Get the session
    let session = match state.session_manager.get_session(&id).await {
        Some(s) => s,
        None => return Json(ApiResponse::error("Session not found")),
    };

    // Check if already running
    if session.is_running() {
        let info = session.info().await;
        return Json(ApiResponse::success(info));
    }

    // Create audio track for this session
    let audio_track = Arc::new(TrackLocalStaticRTP::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_owned(),
            clock_rate: 48000,
            channels: 1,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".to_owned(),
            rtcp_feedback: vec![],
        },
        format!("audio-{}", id),
        format!("session-{}", id),
    ));

    let running = Arc::new(AtomicBool::new(true));
    let ffmpeg_pid = Arc::new(std::sync::atomic::AtomicU32::new(0));

    // Store audio state
    {
        let mut audio_states = state.session_audio.write().await;
        audio_states.insert(
            id.clone(),
            SessionAudioState {
                audio_track: audio_track.clone(),
                running: running.clone(),
                ffmpeg_pid: ffmpeg_pid.clone(),
            },
        );
    }

    // Get model config for this session
    let model = match state.model_registry.get_model(&session.model_id) {
        Some(m) => m.clone(),
        None => return Json(ApiResponse::error("Model not found")),
    };

    // Start transcription thread
    session.start();
    session.set_state(SessionState::Running).await;

    let session_clone = session.clone();
    let wav_path = session.wav_path.clone();
    let model_path = model.model_path.clone();
    let diar_path = model.diarization_path.clone();
    let exec_config = model.exec_config.clone();
    let model_id = session.model_id.clone();
    let language = session.language.clone();

    std::thread::spawn(move || {
        run_session_transcription(
            session_clone,
            wav_path,
            model_path,
            diar_path,
            exec_config,
            audio_track,
            running,
            model_id,
            language,
            ffmpeg_pid,
        );
    });

    let info = session.info().await;
    Json(ApiResponse::success(info))
}

// ============================================================================
// WebSocket Handler
// ============================================================================

#[cfg(feature = "sortformer")]
async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(session_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, session_id, state))
}

#[cfg(feature = "sortformer")]
async fn handle_socket(socket: WebSocket, session_id: String, state: Arc<AppState>) {
    let client_id = Uuid::new_v4().to_string();
    let count = state.client_count.fetch_add(1, Ordering::SeqCst) + 1;
    eprintln!("[WebRTC] Client {} connecting to session {} (total: {})",
             &client_id[..8], session_id, count);

    // Get the session
    let session = match state.session_manager.get_session(&session_id).await {
        Some(s) => s,
        None => {
            eprintln!("[WebRTC] Session {} not found", session_id);
            state.client_count.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    // Add client to session
    session.add_client();

    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Channel for ICE candidates
    let (ice_tx, mut ice_rx) = mpsc::channel::<String>(100);

    // Subscribe to session broadcasts
    let mut subtitle_rx = session.subscribe_subtitles();
    let mut status_rx = session.subscribe_status();

    // Create peer connection
    let turn_server = std::env::var("TURN_SERVER").unwrap_or_default();
    let turn_username = std::env::var("TURN_USERNAME").unwrap_or_default();
    let turn_password = std::env::var("TURN_PASSWORD").unwrap_or_default();

    let mut ice_servers = vec![RTCIceServer {
        urls: vec!["stun:stun.l.google.com:19302".to_owned()],
        ..Default::default()
    }];

    if !turn_server.is_empty() {
        ice_servers.push(RTCIceServer {
            urls: vec![turn_server],
            username: turn_username,
            credential: turn_password,
            credential_type: RTCIceCredentialType::Password,
        });
    }

    let config = RTCConfiguration {
        ice_servers,
        ice_transport_policy: RTCIceTransportPolicy::All,
        ..Default::default()
    };

    let peer_connection = match state.api.new_peer_connection(config).await {
        Ok(pc) => Arc::new(pc),
        Err(e) => {
            eprintln!("[WebRTC] Failed to create peer connection: {}", e);
            session.remove_client();
            state.client_count.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    // Get audio track for this session
    let audio_track = {
        let audio_states = state.session_audio.read().await;
        audio_states.get(&session_id).map(|s| s.audio_track.clone())
    };

    // Add audio track if available
    if let Some(track) = audio_track {
        let rtp_sender = match peer_connection
            .add_track(track as Arc<dyn TrackLocal + Send + Sync>)
            .await
        {
            Ok(sender) => sender,
            Err(e) => {
                eprintln!("[WebRTC] Failed to add audio track: {}", e);
                session.remove_client();
                state.client_count.fetch_sub(1, Ordering::SeqCst);
                return;
            }
        };

        // Read RTCP packets
        tokio::spawn(async move {
            let mut rtcp_buf = vec![0u8; 1500];
            while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
        });
    }

    // Handle ICE candidates
    let ice_tx_clone = ice_tx.clone();
    let client_id_ice = client_id.clone();
    peer_connection.on_ice_candidate(Box::new(move |candidate| {
        let ice_tx = ice_tx_clone.clone();
        let cid = client_id_ice.clone();
        Box::pin(async move {
            if let Some(candidate) = candidate {
                eprintln!("[WebRTC] {} ICE candidate: {:?}", &cid[..8], candidate.to_json());
                if let Ok(json) = candidate.to_json() {
                    let msg = serde_json::json!({
                        "type": "ice-candidate",
                        "candidate": json
                    });
                    ice_tx.send(msg.to_string()).await.ok();
                }
            }
        })
    }));

    // Handle connection state changes
    let client_id_clone = client_id.clone();
    peer_connection.on_peer_connection_state_change(Box::new(move |state| {
        eprintln!("[WebRTC] Client {} state: {:?}", &client_id_clone[..8], state);
        Box::pin(async {})
    }));

    // Store client
    {
        let mut clients = state.clients.lock().await;
        clients.insert(
            client_id.clone(),
            ClientConnection {
                id: client_id.clone(),
                session_id: session_id.clone(),
                peer_connection: peer_connection.clone(),
                ice_tx,
            },
        );
    }

    // Send welcome message with session info
    let session_info = session.info().await;
    let welcome = serde_json::json!({
        "type": "welcome",
        "client_id": &client_id[..8],
        "session": session_info
    });
    if ws_sender.send(Message::Text(welcome.to_string())).await.is_err() {
        cleanup_client(&state, &client_id, &session).await;
        return;
    }

    // Send cached last subtitle to late-joining client
    if let Some(last_subtitle) = session.get_last_subtitle() {
        let preview: String = last_subtitle.chars().take(80).collect();
        eprintln!("[WebRTC] Sending cached subtitle to late-joining client: {}...", preview);
        if ws_sender.send(Message::Text(last_subtitle)).await.is_err() {
            cleanup_client(&state, &client_id, &session).await;
            return;
        }
    }

    // Main message loop
    loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = handle_client_message(
                            &text,
                            &peer_connection,
                            &mut ws_sender,
                        ).await {
                            eprintln!("[WebRTC] Error handling message: {}", e);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }

            Some(ice_msg) = ice_rx.recv() => {
                if ws_sender.send(Message::Text(ice_msg)).await.is_err() {
                    break;
                }
            }

            msg = subtitle_rx.recv() => {
                match msg {
                    Ok(json) => {
                        // Truncate safely for UTF-8 strings
                        let preview: String = json.chars().take(80).collect();
                        eprintln!("[WebRTC] Sending subtitle to client: {}...", preview);
                        if ws_sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("[WebRTC] Lagged {} subtitle messages", n);
                    }
                    Err(_) => break,
                }
            }

            msg = status_rx.recv() => {
                match msg {
                    Ok(json) => {
                        if ws_sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }
        }
    }

    cleanup_client(&state, &client_id, &session).await;
}

#[cfg(feature = "sortformer")]
async fn handle_client_message(
    text: &str,
    peer_connection: &Arc<RTCPeerConnection>,
    ws_sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: serde_json::Value = serde_json::from_str(text)?;

    match msg["type"].as_str() {
        Some("ready") => {
            let offer = peer_connection.create_offer(None).await?;
            peer_connection.set_local_description(offer.clone()).await?;

            let offer_msg = serde_json::json!({
                "type": "offer",
                "sdp": offer.sdp
            });
            ws_sender.send(Message::Text(offer_msg.to_string())).await?;
            eprintln!("[WebRTC] Sent offer to client");
        }
        Some("answer") => {
            if let Some(sdp) = msg["sdp"].as_str() {
                let answer = RTCSessionDescription::answer(sdp.to_owned())?;
                peer_connection.set_remote_description(answer).await?;
                eprintln!("[WebRTC] Set remote description (answer)");
            }
        }
        Some("ice-candidate") => {
            if let Some(candidate) = msg.get("candidate") {
                let ice_candidate: RTCIceCandidateInit = serde_json::from_value(candidate.clone())?;
                peer_connection.add_ice_candidate(ice_candidate).await?;
            }
        }
        _ => {}
    }

    Ok(())
}

#[cfg(feature = "sortformer")]
async fn cleanup_client(state: &Arc<AppState>, client_id: &str, session: &Arc<TranscriptionSession>) {
    let mut clients = state.clients.lock().await;
    if let Some(client) = clients.remove(client_id) {
        client.peer_connection.close().await.ok();
    }
    session.remove_client();
    let count = state.client_count.fetch_sub(1, Ordering::SeqCst) - 1;
    eprintln!("[WebRTC] Client {} disconnected (total: {})", &client_id[..8], count);
}

// ============================================================================
// Config Handler
// ============================================================================

#[cfg(feature = "sortformer")]
async fn config_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let config = &state.config;

    let mut ice_servers = vec![
        serde_json::json!({ "urls": "stun:stun.l.google.com:19302" })
    ];

    if !config.turn_server.is_empty() {
        ice_servers.push(serde_json::json!({
            "urls": config.turn_server,
            "username": config.turn_username,
            "credential": config.turn_password
        }));
    }

    let response = serde_json::json!({
        "wsUrl": config.ws_url,
        "iceServers": ice_servers,
        "speakerColors": [
            "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
            "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98"
        ]
    });

    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        response.to_string()
    )
}

// ============================================================================
// Session Transcription
// ============================================================================

/// Opus encoder wrapper
#[cfg(feature = "sortformer")]
struct OpusEncoder {
    encoder: audiopus::coder::Encoder,
    resample_buffer: Vec<i16>,
    frame_size: usize,
    output_buffer: Vec<u8>,
    sequence: u16,
    timestamp: u32,
    ssrc: u32,
}

#[cfg(feature = "sortformer")]
impl OpusEncoder {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = audiopus::coder::Encoder::new(
            audiopus::SampleRate::Hz48000,
            audiopus::Channels::Mono,
            audiopus::Application::Voip,
        )?;

        Ok(Self {
            encoder,
            resample_buffer: Vec::with_capacity(960 * 2),
            frame_size: 960,
            output_buffer: vec![0u8; 1500],
            sequence: 0,
            timestamp: 0,
            ssrc: rand::random(),
        })
    }

    fn encode(&mut self, samples_16k: &[f32]) -> Vec<Bytes> {
        let mut packets = Vec::new();

        for sample in samples_16k {
            let s16 = (*sample * 32767.0) as i16;
            self.resample_buffer.push(s16);
            self.resample_buffer.push(s16);
            self.resample_buffer.push(s16);
        }

        while self.resample_buffer.len() >= self.frame_size {
            let frame: Vec<i16> = self.resample_buffer.drain(..self.frame_size).collect();

            match self.encoder.encode(&frame, &mut self.output_buffer[12..]) {
                Ok(len) => {
                    if len > 0 {
                        let packet = self.build_rtp_packet(len);
                        packets.push(packet);
                    }
                }
                Err(e) => {
                    eprintln!("[Opus] Encode error: {}", e);
                }
            }
        }

        packets
    }

    fn build_rtp_packet(&mut self, payload_len: usize) -> Bytes {
        let mut packet = vec![0u8; 12 + payload_len];

        packet[0] = 0x80;
        packet[1] = 111;

        packet[2] = (self.sequence >> 8) as u8;
        packet[3] = self.sequence as u8;
        self.sequence = self.sequence.wrapping_add(1);

        packet[4] = (self.timestamp >> 24) as u8;
        packet[5] = (self.timestamp >> 16) as u8;
        packet[6] = (self.timestamp >> 8) as u8;
        packet[7] = self.timestamp as u8;
        self.timestamp = self.timestamp.wrapping_add(self.frame_size as u32);

        packet[8] = (self.ssrc >> 24) as u8;
        packet[9] = (self.ssrc >> 16) as u8;
        packet[10] = (self.ssrc >> 8) as u8;
        packet[11] = self.ssrc as u8;

        packet[12..12 + payload_len].copy_from_slice(&self.output_buffer[12..12 + payload_len]);

        Bytes::from(packet)
    }
}

/// Create RealtimeCanaryConfig based on latency mode
/// Canary uses different parameters since it's an encoder-decoder model without pause detection
fn create_canary_config(mode: &str, language: String) -> parakeet_rs::realtime_canary::RealtimeCanaryConfig {
    use parakeet_rs::realtime_canary::RealtimeCanaryConfig;

    match mode {
        // Speedy: Process frequently for lower latency
        "speedy" => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
        },
        // Pause-based: More buffering for better accuracy
        "pause_based" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
        },
        // Low latency: Balanced approach
        "low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 1.5,
            process_interval_secs: 1.5,
            language,
        },
        // Ultra-low latency: More frequent processing
        "ultra_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 6.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
        },
        // Extreme low latency: Most frequent processing (may reduce quality)
        "extreme_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 4.0,
            min_audio_secs: 0.5,
            process_interval_secs: 0.5,
            language,
        },
        // Lookahead: Similar to pause-based for Canary
        "lookahead" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
        },
        // Default to speedy
        _ => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
        },
    }
}

/// Create RealtimeWhisperConfig based on latency mode
#[cfg(feature = "whisper")]
fn create_whisper_config(mode: &str, language: String) -> parakeet_rs::realtime_whisper::RealtimeWhisperConfig {
    use parakeet_rs::realtime_whisper::RealtimeWhisperConfig;

    match mode {
        // Speedy: Process frequently for lower latency
        "speedy" => RealtimeWhisperConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 1.5,
            process_interval_secs: 1.0,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 3,
        },
        // Pause-based: More buffering for better accuracy
        "pause_based" => RealtimeWhisperConfig {
            buffer_size_secs: 15.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 5,
        },
        // Low latency: Balanced approach
        "low_latency" => RealtimeWhisperConfig {
            buffer_size_secs: 12.0,
            min_audio_secs: 1.5,
            process_interval_secs: 1.5,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 4,
        },
        // Ultra-low latency: More frequent processing
        "ultra_low_latency" => RealtimeWhisperConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 3,
        },
        // Extreme low latency: Most frequent processing
        "extreme_low_latency" => RealtimeWhisperConfig {
            buffer_size_secs: 6.0,
            min_audio_secs: 0.8,
            process_interval_secs: 0.8,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 2,
        },
        // Lookahead: Similar to quality mode - more context
        "lookahead" => RealtimeWhisperConfig {
            buffer_size_secs: 25.0,
            min_audio_secs: 3.0,
            process_interval_secs: 3.0,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 8,
        },
        // ASR: Pure streaming mode - balanced quality and latency
        "asr" => RealtimeWhisperConfig {
            buffer_size_secs: 20.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 5,
        },
        // Default to balanced
        _ => RealtimeWhisperConfig {
            buffer_size_secs: 15.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            task: "transcribe".to_string(),
            confirmation_words: 5,
        },
    }
}

/// Create RealtimeTDTConfig based on latency mode
#[cfg(feature = "sortformer")]
fn create_transcription_config(mode: &str) -> parakeet_rs::RealtimeTDTConfig {
    use parakeet_rs::RealtimeTDTConfig;

    match mode {
        "speedy" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.4,
            pause_based_confirm: true,
            pause_threshold_secs: 0.35,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "pause_based" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 1.5,
            confirm_threshold_secs: 2.0,
            pause_based_confirm: false,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "ultra_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 1.0,
            confirm_threshold_secs: 1.5,
            pause_based_confirm: false,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "extreme_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 5.0,
            process_interval_secs: 0.5,
            confirm_threshold_secs: 0.8,
            pause_based_confirm: false,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "lookahead" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: true,
            lookahead_segments: 2,
        },
        // Default to speedy
        _ => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.4,
            pause_based_confirm: true,
            pause_threshold_secs: 0.35,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
    }
}

/// Helper to emit transcript segments (DiarizedSegment)
#[cfg(feature = "sortformer")]
fn emit_segments(session: &TranscriptionSession, segments: &[parakeet_rs::DiarizedSegment]) {
    for segment in segments {
        let subtitle_msg = serde_json::json!({
            "type": "subtitle",
            "text": segment.text,
            "speaker": segment.speaker,
            "start": segment.start_time,
            "end": segment.end_time,
            "is_final": segment.is_final
        });

        // Send to all connected clients (ignore if no receivers - this is expected when no clients connected)
        let receiver_count = session.subtitle_tx.receiver_count();
        let subtitle_str = subtitle_msg.to_string();

        // Cache the last subtitle for late-joining clients
        session.set_last_subtitle(subtitle_str.clone());

        let _ = session.subtitle_tx.send(subtitle_str);

        // Log final segments
        if segment.is_final && receiver_count > 0 {
            eprintln!(
                "[Session {} | Speaker {}] {} [{:.2}s-{:.2}s] (sent to {} clients)",
                session.id,
                segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
                segment.text,
                segment.start_time,
                segment.end_time,
                receiver_count
            );
        } else if segment.is_final {
            eprintln!(
                "[Session {} | Speaker {}] {} [{:.2}s-{:.2}s]",
                session.id,
                segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
                segment.text,
                segment.start_time,
                segment.end_time
            );
        }
    }
}

/// Helper to emit transcript segments from StreamingTranscriber (TranscriptionSegment)
#[cfg(feature = "sortformer")]
fn emit_streaming_segments(session: &TranscriptionSession, segments: &[parakeet_rs::streaming_transcriber::TranscriptionSegment]) {
    // Debug: log segment count
    if !segments.is_empty() {
        eprintln!("[Session {}] emit_streaming_segments: {} segment(s)", session.id, segments.len());
    }

    for segment in segments {
        let subtitle_msg = serde_json::json!({
            "type": "subtitle",
            "text": segment.text,
            "speaker": segment.speaker,
            "start": segment.start_time,
            "end": segment.end_time,
            "is_final": segment.is_final
        });

        // Send to all connected clients (ignore if no receivers - this is expected when no clients connected)
        let receiver_count = session.subtitle_tx.receiver_count();
        let subtitle_str = subtitle_msg.to_string();

        // Cache the last subtitle for late-joining clients
        session.set_last_subtitle(subtitle_str.clone());

        let _ = session.subtitle_tx.send(subtitle_str);

        // Log all segments (both partial and final)
        let partial_marker = if segment.is_final { "FINAL" } else { "partial" };
        eprintln!(
            "[Session {} | {} | Speaker {}] \"{}\" [{:.2}s-{:.2}s] (receivers: {})",
            session.id,
            partial_marker,
            segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
            segment.text,
            segment.start_time,
            segment.end_time,
            receiver_count
        );
    }
}

#[cfg(feature = "sortformer")]
fn run_session_transcription(
    session: Arc<TranscriptionSession>,
    wav_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
    diar_path: Option<std::path::PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    model_id: String,
    language: String,
    ffmpeg_pid: Arc<std::sync::atomic::AtomicU32>,
) {
    use parakeet_rs::streaming_transcriber::StreamingTranscriber;
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::sync::mpsc as std_mpsc;
    use std::time::Instant;

    eprintln!("[Session {}] Starting transcription for {}", session.id, wav_path.display());

    // Get duration using ffprobe
    let duration_secs = get_audio_duration(&wav_path).unwrap_or(0.0);
    eprintln!("[Session {}] Total duration: {:.2}s", session.id, duration_secs);

    // Check if this is Canary, Whisper, or TDT model
    let is_canary = model_id == "canary-1b";
    let is_whisper = model_id.starts_with("whisper-large-");

    // Channel to send audio samples to transcription thread
    // Use a very large buffer to handle CPU transcription being slower than real-time
    // At 16kHz with 320 samples per chunk:
    // - 5000 chunks = ~100 seconds of audio buffer (old)
    // - 50000 chunks = ~1000 seconds of audio buffer (~16 minutes)
    // This allows transcription to lag behind audio by up to 16 minutes before dropping
    // On CPU, TDT+diarization runs at ~25% of real-time, so this handles ~4 minutes of audio
    let (audio_tx, audio_rx) = std_mpsc::sync_channel::<Vec<f32>>(50000);

    // Get the mode for transcription config
    let mode = session.mode.clone();
    eprintln!("[Session {}] Using transcription mode: {}", session.id, mode);

    // Check if this is a VAD mode
    let is_vad_mode = mode.starts_with("vad_");
    let vad_base_mode = if is_vad_mode {
        mode.strip_prefix("vad_").unwrap_or("speedy").to_string()
    } else {
        mode.clone()
    };

    // Get VAD model path from env
    let vad_model_path = std::env::var("VAD_MODEL_PATH").unwrap_or_else(|_| "silero_vad.onnx".to_string());

    // Spawn transcription thread (runs independently of audio streaming)
    let transcription_session = session.clone();
    let transcription_running = running.clone();
    let transcription_thread = std::thread::spawn(move || {
        // Create the appropriate transcriber based on model type and mode
        let mut transcriber: Box<dyn StreamingTranscriber> = if is_vad_mode {
            // VAD-triggered transcription
            if is_canary {
                // VAD + Canary
                use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};

                // Use buffered mode for better transcription quality
                // This accumulates 2-3 seconds of speech before transcribing
                let config = RealtimeCanaryVadConfig::buffered(language.clone());

                eprintln!("[Session {}] Creating VAD+Canary transcriber from {:?} (language: {}, vad_mode: {})",
                    transcription_session.id, model_path, language, &vad_base_mode);

                match RealtimeCanaryVad::new(&model_path, &vad_model_path, Some(exec_config), Some(config)) {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!("[Session {}] Failed to create VAD+Canary transcriber: {}",
                            transcription_session.id, e);
                        return;
                    }
                }
            } else if is_whisper {
                // VAD + Whisper
                #[cfg(feature = "whisper")]
                {
                    use parakeet_rs::realtime_whisper_vad::{RealtimeWhisperVad, RealtimeWhisperVadConfig};
                    use parakeet_rs::vad::VadConfig;

                    let vad_config = VadConfig::from_mode(&vad_base_mode);
                    let config = RealtimeWhisperVadConfig {
                        vad: vad_config,
                        language: language.clone(),
                        task: "transcribe".to_string(),
                        min_buffer_duration: 0.0,  // Immediate mode
                        max_buffer_duration: 25.0, // Whisper max is 30s
                        long_pause_threshold: 1.5, // Default pause threshold
                    };

                    eprintln!("[Session {}] Creating VAD+Whisper transcriber from {:?} (language: {}, vad_mode: {})",
                        transcription_session.id, model_path, language, &vad_base_mode);

                    match RealtimeWhisperVad::new(&model_path, &vad_model_path, Some(exec_config), Some(config)) {
                        Ok(t) => Box::new(t),
                        Err(e) => {
                            eprintln!("[Session {}] Failed to create VAD+Whisper transcriber: {}",
                                transcription_session.id, e);
                            return;
                        }
                    }
                }
                #[cfg(not(feature = "whisper"))]
                {
                    eprintln!("[Session {}] Whisper support requires the 'whisper' feature flag",
                        transcription_session.id);
                    return;
                }
            } else {
                // VAD + TDT with diarization
                use parakeet_rs::realtime_tdt_vad::{RealtimeTdtVad, RealtimeTdtVadConfig};
                use parakeet_rs::vad::VadConfig;

                let diar_path = match diar_path {
                    Some(p) => p,
                    None => {
                        eprintln!("[Session {}] No diarization model configured for TDT",
                            transcription_session.id);
                        return;
                    }
                };

                let vad_config = VadConfig::from_mode(&vad_base_mode);
                let config = RealtimeTdtVadConfig {
                    vad: vad_config,
                    enable_diarization: true,
                };

                eprintln!("[Session {}] Creating VAD+TDT transcriber from {:?} (vad_mode: {})",
                    transcription_session.id, model_path, &vad_base_mode);

                match RealtimeTdtVad::new(&model_path, Some(&diar_path), &vad_model_path, Some(exec_config), Some(config)) {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!("[Session {}] Failed to create VAD+TDT transcriber: {}",
                            transcription_session.id, e);
                        return;
                    }
                }
            }
        } else if is_canary {
            // Use Canary for multilingual transcription (non-VAD)
            use parakeet_rs::realtime_canary::RealtimeCanary;

            // Create config based on mode (speedy, pause_based, extreme_low_latency, etc.)
            let canary_config = create_canary_config(&mode, language.clone());

            eprintln!("[Session {}] Creating Canary transcriber from {:?} (language: {}, mode: {})",
                transcription_session.id, model_path, language, mode);

            match RealtimeCanary::new(&model_path, Some(exec_config), Some(canary_config)) {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!("[Session {}] Failed to create Canary transcriber: {}",
                        transcription_session.id, e);
                    return;
                }
            }
        } else if is_whisper {
            // Use Whisper for multilingual transcription (non-VAD)
            #[cfg(feature = "whisper")]
            {
                use parakeet_rs::realtime_whisper::RealtimeWhisper;

                // Create config based on mode
                let whisper_config = create_whisper_config(&mode, language.clone());

                eprintln!("[Session {}] Creating Whisper transcriber from {:?} (language: {}, mode: {})",
                    transcription_session.id, model_path, language, mode);

                match RealtimeWhisper::new(&model_path, Some(exec_config), Some(whisper_config)) {
                    Ok(mut t) => {
                        // Set cancellation token for responsive shutdown during long inference
                        t.set_cancellation_token(transcription_running.clone());
                        Box::new(t)
                    },
                    Err(e) => {
                        eprintln!("[Session {}] Failed to create Whisper transcriber: {}",
                            transcription_session.id, e);
                        return;
                    }
                }
            }
            #[cfg(not(feature = "whisper"))]
            {
                eprintln!("[Session {}] Whisper support requires the 'whisper' feature flag",
                    transcription_session.id);
                return;
            }
        } else {
            // Use TDT with diarization (non-VAD)
            use parakeet_rs::RealtimeTDTDiarized;

            let diar_path = match diar_path {
                Some(p) => p,
                None => {
                    eprintln!("[Session {}] No diarization model configured for TDT",
                        transcription_session.id);
                    return;
                }
            };

            let config = create_transcription_config(&mode);

            eprintln!("[Session {}] Creating TDT transcriber from {:?}",
                transcription_session.id, model_path);

            match RealtimeTDTDiarized::new(&model_path, &diar_path, Some(exec_config), Some(config)) {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!("[Session {}] Failed to create TDT transcriber: {}",
                        transcription_session.id, e);
                    return;
                }
            }
        };

        let model_type = match (is_vad_mode, is_canary, is_whisper) {
            (true, true, _) => "VAD+Canary",
            (true, _, true) => "VAD+Whisper",
            (true, _, _) => "VAD+TDT",
            (false, true, _) => "Canary",
            (false, _, true) => "Whisper",
            (false, _, _) => "TDT",
        };
        eprintln!("[Session {}] Transcription thread started ({})",
            transcription_session.id, model_type);

        // Process audio samples as they arrive
        // IMPORTANT: Batch multiple chunks together before calling push_audio
        // This dramatically improves efficiency by reducing transcriber call overhead
        let mut chunks_processed = 0u64;
        while transcription_running.load(Ordering::SeqCst) {
            // First, wait for at least one batch
            let first_batch = match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(batch) => batch,
                Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
            };

            // Collect the first batch
            let mut all_samples = first_batch;

            // Drain any additional pending batches to reduce latency and improve efficiency
            while let Ok(batch) = audio_rx.try_recv() {
                all_samples.extend(batch);
            }

            chunks_processed += 1;

            // Process all collected samples through transcriber in one call
            match transcriber.push_audio(&all_samples) {
                Ok(result) => {
                    emit_streaming_segments(&transcription_session, &result.segments);
                }
                Err(e) => {
                    if chunks_processed % 100 == 0 {
                        eprintln!("[Session {}] push_audio error (batch {}): {}",
                            transcription_session.id, chunks_processed, e);
                    }
                }
            }

            // Check if we should stop after inference completes (for responsive shutdown)
            if !transcription_running.load(Ordering::SeqCst) {
                eprintln!("[Session {}] Stop requested, exiting transcription loop", transcription_session.id);
                break;
            }
        }

        // Only finalize if we weren't stopped (avoid expensive operation on shutdown)
        if transcription_running.load(Ordering::SeqCst) {
            // Drain any remaining batches before finalizing
            let mut remaining_samples = Vec::new();
            while let Ok(batch) = audio_rx.try_recv() {
                remaining_samples.extend(batch);
            }
            if !remaining_samples.is_empty() {
                transcriber.push_audio(&remaining_samples).ok();
            }

            // Finalize transcription
            eprintln!("[Session {}] Finalizing transcription...", transcription_session.id);
            match transcriber.finalize() {
                Ok(result) => {
                    emit_streaming_segments(&transcription_session, &result.segments);
                }
                Err(e) => {
                    eprintln!("[Session {}] Finalization error: {}", transcription_session.id, e);
                }
            }
        } else {
            eprintln!("[Session {}] Skipping finalization (session stopped)", transcription_session.id);
        }

        eprintln!("[Session {}] Transcription thread finished", transcription_session.id);
    });

    // Initialize Opus encoder
    let mut opus_encoder = match OpusEncoder::new() {
        Ok(enc) => enc,
        Err(e) => {
            eprintln!("[Session {}] Failed to create Opus encoder: {}", session.id, e);
            return;
        }
    };

    // Spawn ffmpeg with -re for real-time pacing
    let mut ffmpeg = match Command::new("ffmpeg")
        .args([
            "-re",                    // Real-time pacing (the magic flag!)
            "-i", wav_path.to_str().unwrap_or(""),
            "-f", "s16le",            // PCM signed 16-bit little-endian
            "-ar", "16000",           // 16kHz sample rate
            "-ac", "1",               // Mono
            "-loglevel", "error",     // Suppress ffmpeg output
            "-",                      // Output to stdout
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => {
            // Store ffmpeg PID so it can be killed on session stop
            ffmpeg_pid.store(child.id(), Ordering::SeqCst);
            eprintln!("[Session {}] ffmpeg spawned with pid {}", session.id, child.id());
            child
        },
        Err(e) => {
            eprintln!("[Session {}] Failed to spawn ffmpeg: {}", session.id, e);
            let _ = session.status_tx.send(serde_json::json!({
                "type": "error",
                "message": format!("Failed to spawn ffmpeg: {}", e)
            }).to_string());
            return;
        }
    };

    eprintln!("[Session {}] Started ffmpeg with real-time pacing", session.id);

    let mut stdout = ffmpeg.stdout.take().expect("Failed to get ffmpeg stdout");

    // Create tokio runtime for async WebRTC writes
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Read 20ms chunks (320 samples at 16kHz = 640 bytes)
    let chunk_samples = 320;
    let bytes_per_chunk = chunk_samples * 2;
    let mut byte_buffer = vec![0u8; bytes_per_chunk];
    let mut total_samples: usize = 0;
    let mut last_status_time = Instant::now();

    // Audio streaming loop - ONLY handles audio, never blocks on transcription
    while running.load(Ordering::SeqCst) {
        // Read PCM from ffmpeg stdout (blocks at real-time rate due to -re)
        match stdout.read_exact(&mut byte_buffer) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                eprintln!("[Session {}] End of audio stream", session.id);
                break;
            }
            Err(e) => {
                eprintln!("[Session {}] Error reading from ffmpeg: {}", session.id, e);
                break;
            }
        }

        // Convert bytes to f32 samples
        let samples: Vec<f32> = byte_buffer
            .chunks(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        total_samples += samples.len();
        let current_time = total_samples as f32 / 16000.0;

        // Send samples to transcription thread (non-blocking - may drop if channel full)
        if let Err(std_mpsc::TrySendError::Full(_)) = audio_tx.try_send(samples.clone()) {
            // Only log occasionally to avoid spam
            if total_samples % (16000 * 30) < 320 { // Log every ~30 seconds
                eprintln!("[Session {}] WARNING: Transcription buffer full at {:.1}s - CPU cannot keep up with real-time. \
Consider using GPU or shorter audio files.",
                    session.id, current_time);
            }
        }

        // Encode to Opus and send via WebRTC immediately
        let rtp_packets = opus_encoder.encode(&samples);
        for packet in &rtp_packets {
            let track = audio_track.clone();
            let pkt = packet.clone();
            rt.block_on(async {
                if let Err(e) = track.write(&pkt).await {
                    let err_str = e.to_string();
                    if !err_str.contains("no receiver") && !err_str.contains("ErrClosedPipe") {
                        eprintln!("[Session {}] WebRTC write error: {}", session.id, e);
                    }
                }
            });
        }

        // Update progress every second
        if last_status_time.elapsed().as_secs() >= 1 {
            rt.block_on(session.set_progress(current_time));

            let status_msg = serde_json::json!({
                "type": "status",
                "progress_secs": current_time,
                "total_duration": duration_secs
            });
            session.status_tx.send(status_msg.to_string()).ok();
            last_status_time = Instant::now();
        }
    }

    // Signal end of audio and kill ffmpeg (don't wait - it might still be streaming)
    drop(audio_tx);
    // Kill ffmpeg explicitly to prevent orphan processes
    let _ = ffmpeg.kill();
    let _ = ffmpeg.wait();

    // Wait for transcription to finish
    transcription_thread.join().ok();

    // Mark session as completed
    rt.block_on(session.set_state(SessionState::Completed));
    session.stop();

    let end_msg = serde_json::json!({
        "type": "end",
        "total_duration": duration_secs
    });
    session.status_tx.send(end_msg.to_string()).ok();

    eprintln!("[Session {}] Complete. Duration: {:.2}s", session.id, duration_secs);
}

/// Get audio duration using ffprobe
#[cfg(feature = "sortformer")]
fn get_audio_duration(path: &std::path::Path) -> Option<f32> {
    use std::process::Command;

    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path.to_str()?,
        ])
        .output()
        .ok()?;

    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()
}

// Random number generator for SSRC
mod rand {
    pub fn random<T: Default + From<u8>>() -> T
    where
        T: std::ops::BitOr<Output = T> + std::ops::Shl<usize, Output = T>,
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        T::from((nanos & 0xFF) as u8)
            | (T::from(((nanos >> 8) & 0xFF) as u8) << 8)
            | (T::from(((nanos >> 16) & 0xFF) as u8) << 16)
            | (T::from(((nanos >> 24) & 0xFF) as u8) << 24)
    }
}
