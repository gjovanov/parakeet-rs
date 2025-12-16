//! API handlers for session management

use super::models::ApiResponse;
use crate::config::LatencyMode;
use crate::state::{AppState, SessionAudioState};
use crate::transcription::run_session_transcription;
use axum::{extract::{Path, State}, Json};
use parakeet_rs::SessionState;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use webrtc::api::media_engine::MIME_TYPE_OPUS;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;

/// Configuration for parallel sliding window mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads (default: 8)
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
    /// Buffer size in seconds (default: 6)
    #[serde(default = "default_buffer_size")]
    pub buffer_size_secs: usize,
}

fn default_num_threads() -> usize {
    8
}

fn default_buffer_size() -> usize {
    6
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 8,
            buffer_size_secs: 6,
        }
    }
}

/// Configuration for pause detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseConfig {
    /// Silence duration (ms) to trigger segment boundary (150-600, default: 300)
    #[serde(default = "default_pause_threshold_ms")]
    pub pause_threshold_ms: u32,

    /// RMS energy threshold for silence detection (0.003-0.02, default: 0.008)
    #[serde(default = "default_silence_energy")]
    pub silence_energy_threshold: f32,

    /// Maximum segment duration in seconds (3.0-15.0, default: 5.0)
    #[serde(default = "default_max_segment_secs")]
    pub max_segment_secs: f32,

    /// Context buffer in seconds for parallel modes (0.0-3.0, default: 0.0)
    /// Higher values improve accuracy at segment boundaries but cause text overlap
    #[serde(default = "default_context_buffer_secs")]
    pub context_buffer_secs: f32,
}

fn default_pause_threshold_ms() -> u32 {
    300
}

fn default_silence_energy() -> f32 {
    0.008
}

fn default_max_segment_secs() -> f32 {
    5.0
}

fn default_context_buffer_secs() -> f32 {
    0.0
}

impl Default for PauseConfig {
    fn default() -> Self {
        Self {
            pause_threshold_ms: default_pause_threshold_ms(),
            silence_energy_threshold: default_silence_energy(),
            max_segment_secs: default_max_segment_secs(),
            context_buffer_secs: default_context_buffer_secs(),
        }
    }
}

/// Request to create a new session
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    pub model_id: String,
    pub media_id: String,
    #[serde(default)]
    pub mode: LatencyMode,
    /// Language code for transcription (default: "de" for German)
    #[serde(default = "default_language")]
    pub language: String,
    /// Configuration for parallel mode (only used when mode is "parallel")
    #[serde(default)]
    pub parallel_config: Option<ParallelConfig>,
    /// Noise cancellation type ("none", "rnnoise", "deepfilternet3")
    #[serde(default = "default_noise_cancellation")]
    pub noise_cancellation: String,
    /// Whether to enable speaker diarization
    #[serde(default)]
    pub diarization: bool,
    /// Pause detection configuration (only used for pause-related modes)
    #[serde(default)]
    pub pause_config: Option<PauseConfig>,
}

fn default_language() -> String {
    "de".to_string()
}

fn default_noise_cancellation() -> String {
    "none".to_string()
}

/// List all sessions
pub async fn list_sessions(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<parakeet_rs::SessionInfo>>> {
    let sessions = state.session_manager.list_sessions().await;
    Json(ApiResponse::success(sessions))
}

/// Create a new session
pub async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSessionRequest>,
) -> Json<ApiResponse<parakeet_rs::SessionInfo>> {
    let mode_str = req.mode.as_str();
    let language = if req.language.is_empty() { "de".to_string() } else { req.language };
    let noise_cancellation = if req.noise_cancellation.is_empty() {
        "none".to_string()
    } else {
        req.noise_cancellation
    };

    // Determine diarization model name if enabled
    let diarization_model = if req.diarization {
        Some("Sortformer v2 (4 speakers)".to_string())
    } else {
        None
    };

    match state
        .session_manager
        .create_session(
            &req.model_id,
            &req.media_id,
            mode_str,
            &language,
            &noise_cancellation,
            req.diarization,
            diarization_model,
        )
        .await
    {
        Ok(session) => {
            let info = session.info().await;

            // Store parallel config if provided
            if let Some(parallel_config) = req.parallel_config {
                let mut configs = state.parallel_configs.write().await;
                configs.insert(session.id.clone(), parallel_config);
            }

            // Store pause config if provided
            if let Some(pause_config) = req.pause_config {
                let mut configs = state.pause_configs.write().await;
                configs.insert(session.id.clone(), pause_config);
            }

            Json(ApiResponse::success(info))
        }
        Err(e) => Json(ApiResponse::error(e.to_string())),
    }
}

/// Get a specific session
pub async fn get_session(
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

/// Stop and remove a session
pub async fn stop_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Json<ApiResponse<()>> {
    // First try to stop the session (may fail if session doesn't exist)
    let _ = state.session_manager.stop_session(&id).await;

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

    // Clean up parallel config
    {
        let mut configs = state.parallel_configs.write().await;
        configs.remove(&id);
    }

    // Clean up pause config
    {
        let mut configs = state.pause_configs.write().await;
        configs.remove(&id);
    }

    // Always remove the session from the manager
    let _ = state.session_manager.remove_session(&id).await;

    Json(ApiResponse::success(()))
}

/// Start a session
pub async fn start_session(
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

    // Get parallel config if any
    let parallel_config = {
        let configs = state.parallel_configs.read().await;
        configs.get(&id).cloned()
    };

    // Get pause config if any
    let pause_config = {
        let configs = state.pause_configs.read().await;
        configs.get(&id).cloned()
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
            parallel_config,
            pause_config,
        );
    });

    let info = session.info().await;
    Json(ApiResponse::success(info))
}
