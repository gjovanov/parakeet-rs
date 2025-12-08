//! Application state types for the WebRTC transcription server

use crate::config::RuntimeConfig;
use parakeet_rs::{SharedMediaManager, SharedModelRegistry, SharedSessionManager};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;

/// Client connection with WebRTC peer connection
pub struct ClientConnection {
    pub id: String,
    pub session_id: String,
    pub peer_connection: Arc<RTCPeerConnection>,
    pub ice_tx: mpsc::Sender<String>,
}

/// Per-session audio track state
pub struct SessionAudioState {
    pub audio_track: Arc<TrackLocalStaticRTP>,
    pub running: Arc<AtomicBool>,
    /// FFmpeg process ID for killing on session stop
    pub ffmpeg_pid: Arc<std::sync::atomic::AtomicU32>,
}

/// Shared application state
pub struct AppState {
    /// Session manager
    pub session_manager: SharedSessionManager,
    /// Model registry
    pub model_registry: SharedModelRegistry,
    /// Media manager
    pub media_manager: SharedMediaManager,
    /// WebRTC API
    pub api: webrtc::api::API,
    /// Connected clients by ID
    pub clients: Mutex<HashMap<String, ClientConnection>>,
    /// Total client count
    pub client_count: AtomicU64,
    /// Runtime configuration for frontend
    pub config: RuntimeConfig,
    /// Per-session audio tracks
    pub session_audio: RwLock<HashMap<String, SessionAudioState>>,
}
