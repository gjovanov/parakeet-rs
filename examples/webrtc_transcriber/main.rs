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
//!   GET    /api/modes                 - List available transcription modes
//!   GET    /api/noise-cancellation    - List noise cancellation options
//!   GET    /api/diarization           - List diarization options
//!   GET    /api/sessions              - List all sessions
//!   POST   /api/sessions              - Create new session {model_id, media_id, mode, language, noise_cancellation, diarization}
//!   GET    /api/sessions/:id          - Get session info
//!   POST   /api/sessions/:id/start    - Start transcription
//!   DELETE /api/sessions/:id          - Stop session
//!   WS     /ws/:session_id            - Join session for subtitles and audio
//!
//! Usage:
//!   cargo run --release --example webrtc_transcriber --features sortformer

mod api;
mod config;
mod srt_config;
mod state;
mod transcription;
mod webrtc_handlers;

use axum::{
    routing::{delete, get, post},
    Router,
};
use clap::Parser;
use config::{LatencyMode, RuntimeConfig};
use srt_config::SrtConfig;
use parakeet_rs::{MediaManager, MediaManagerConfig, ModelRegistry, SessionManager};
use state::AppState;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tower_http::{cors::CorsLayer, services::ServeDir};
use webrtc::api::{
    interceptor_registry::register_default_interceptors,
    media_engine::MediaEngine,
    APIBuilder,
};
use webrtc::ice::network_type::NetworkType;

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
            // Default to speedy
            LatencyMode::Speedy
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
    // Install panic hook to log crashes
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();
        let timestamp = {
            let now = std::time::SystemTime::now();
            let duration = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            let secs = duration.as_secs();
            // Format as simple timestamp
            format!("{}", secs)
        };

        let panic_msg = if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };

        let location = panic_info.location().map(|loc| {
            format!("{}:{}:{}", loc.file(), loc.line(), loc.column())
        }).unwrap_or_else(|| "unknown location".to_string());

        let crash_msg = format!(
            "\n=== PANIC at {} ===\nLocation: {}\nMessage: {}\nBacktrace:\n{}\n==================\n",
            timestamp, location, panic_msg, backtrace
        );

        // Log to stderr
        eprintln!("{}", crash_msg);

        // Try to log to file
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("crash.log")
        {
            use std::io::Write;
            let _ = file.write_all(crash_msg.as_bytes());
        }
    }));

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

    let mut registry = webrtc::interceptor::registry::Registry::new();
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

    // Initialize SRT configuration (optional)
    let srt_config = SrtConfig::from_env();
    if srt_config.is_some() {
        eprintln!("SRT streams: enabled");
    } else {
        eprintln!("SRT streams: not configured (set SRT_ENCODER_IP and SRT_CHANNELS to enable)");
    }

    let state = Arc::new(AppState {
        session_manager,
        model_registry,
        media_manager,
        api,
        clients: Mutex::new(HashMap::new()),
        client_count: AtomicU64::new(0),
        config: runtime_config,
        session_audio: RwLock::new(HashMap::new()),
        parallel_configs: RwLock::new(HashMap::new()),
        pause_configs: RwLock::new(HashMap::new()),
        srt_config,
    });

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(|| async { "OK" }))
        // Config
        .route("/api/config", get(api::config_handler))
        // Models
        .route("/api/models", get(api::list_models))
        // Media
        .route("/api/media", get(api::list_media))
        .route("/api/media/upload", post(api::upload_media))
        .route("/api/media/:id", delete(api::delete_media))
        // Modes
        .route("/api/modes", get(api::list_modes))
        // Noise cancellation
        .route("/api/noise-cancellation", get(api::list_noise_cancellation))
        // Diarization
        .route("/api/diarization", get(api::list_diarization))
        // SRT streams
        .route("/api/srt-streams", get(api::list_srt_streams))
        // Sessions
        .route("/api/sessions", get(api::list_sessions))
        .route("/api/sessions", post(api::create_session))
        .route("/api/sessions/:id", get(api::get_session))
        .route("/api/sessions/:id", delete(api::stop_session))
        .route("/api/sessions/:id/start", post(api::start_session))
        // WebSocket
        .route("/ws/:session_id", get(webrtc_handlers::ws_handler))
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
