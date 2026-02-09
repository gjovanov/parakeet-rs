//! API handlers for session management

use super::models::ApiResponse;
use crate::config::LatencyMode;
use crate::state::{AppState, FabConfig, SessionAudioState};
use crate::transcription::{run_session_transcription, AudioSource};
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
    /// Media file ID (for file-based sessions) - mutually exclusive with srt_channel_id
    #[serde(default)]
    pub media_id: Option<String>,
    /// SRT channel ID (for SRT stream sessions) - mutually exclusive with media_id
    #[serde(default)]
    pub srt_channel_id: Option<usize>,
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
    /// Sentence completion mode ("off", "minimal", "balanced", "complete")
    #[serde(default = "default_sentence_completion")]
    pub sentence_completion: String,
    /// FAB forwarding override ("default", "enabled", "disabled")
    #[serde(default)]
    pub fab_enabled: Option<String>,
    /// FAB endpoint URL override (empty = use server default)
    #[serde(default)]
    pub fab_url: Option<String>,
    /// FAB send type override ("growing" or "confirmed", empty = use server default)
    #[serde(default)]
    pub fab_send_type: Option<String>,
}

fn default_sentence_completion() -> String {
    "minimal".to_string()
}

fn default_language() -> String {
    "de".to_string()
}

fn default_noise_cancellation() -> String {
    "none".to_string()
}

/// Get max parallel threads based on available memory or env override
/// Each parallel thread loads a model instance (~2GB each)
fn get_max_parallel_threads() -> usize {
    // Check for env override first
    if let Some(max) = std::env::var("MAX_PARALLEL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        return max;
    }

    // Auto-detect based on available system memory
    // Reserve ~4GB for system + base process, allocate ~2.5GB per thread
    let available_gb = get_available_memory_gb();
    let usable_gb = (available_gb - 4.0).max(0.0);
    let memory_based_threads = (usable_gb / 2.5).floor() as usize;

    // Also consider CPU cores (no point having more threads than cores)
    let cpu_cores = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    // Use the minimum of memory-based and CPU-based limits, with min 1 and max 8
    let max_threads = memory_based_threads.min(cpu_cores).max(1).min(8);

    eprintln!(
        "[Config] Auto-detected max parallel threads: {} (available RAM: {:.1}GB, CPUs: {})",
        max_threads, available_gb, cpu_cores
    );

    max_threads
}

/// Get available system memory in GB
fn get_available_memory_gb() -> f64 {
    // Try to read from /proc/meminfo
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemAvailable:") {
                if let Some(kb_str) = line.split_whitespace().nth(1) {
                    if let Ok(kb) = kb_str.parse::<u64>() {
                        return kb as f64 / 1024.0 / 1024.0;
                    }
                }
            }
        }
    }
    // Fallback: assume 16GB available
    16.0
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

    // Validate that exactly one of media_id or srt_channel_id is provided
    let session_result = match (&req.media_id, &req.srt_channel_id) {
        (Some(media_id), None) => {
            // File-based session
            state
                .session_manager
                .create_session(
                    &req.model_id,
                    media_id,
                    mode_str,
                    &language,
                    &noise_cancellation,
                    req.diarization,
                    diarization_model.clone(),
                    &req.sentence_completion,
                )
                .await
        }
        (None, Some(srt_channel_id)) => {
            // SRT stream session
            let srt_config = match &state.srt_config {
                Some(config) => config,
                None => return Json(ApiResponse::error("SRT streams not configured")),
            };

            let channel = match srt_config.get_channel(*srt_channel_id) {
                Some(ch) => ch,
                None => return Json(ApiResponse::error(format!(
                    "Invalid SRT channel ID: {}",
                    srt_channel_id
                ))),
            };

            let srt_url = srt_config.build_srt_url(channel);

            state
                .session_manager
                .create_srt_session(
                    &req.model_id,
                    *srt_channel_id,
                    &channel.name,
                    &srt_url,
                    mode_str,
                    &language,
                    &noise_cancellation,
                    req.diarization,
                    diarization_model,
                    &req.sentence_completion,
                )
                .await
        }
        (Some(_), Some(_)) => {
            return Json(ApiResponse::error(
                "Cannot specify both media_id and srt_channel_id"
            ));
        }
        (None, None) => {
            return Json(ApiResponse::error(
                "Must specify either media_id or srt_channel_id"
            ));
        }
    };

    match session_result {
        Ok(session) => {
            let info = session.info().await;

            // Store parallel config if provided, with thread cap to prevent OOM
            if let Some(mut parallel_config) = req.parallel_config {
                let max_threads = get_max_parallel_threads();
                if parallel_config.num_threads > max_threads {
                    eprintln!(
                        "[Session {}] Capping parallel threads from {} to {} (OOM protection)",
                        session.id, parallel_config.num_threads, max_threads
                    );
                    parallel_config.num_threads = max_threads;
                }
                let mut configs = state.parallel_configs.write().await;
                configs.insert(session.id.clone(), parallel_config);
            }

            // Store pause config if provided
            if let Some(pause_config) = req.pause_config {
                let mut configs = state.pause_configs.write().await;
                configs.insert(session.id.clone(), pause_config);
            }

            // Store FAB config
            {
                let fab_config = FabConfig {
                    enabled: match req.fab_enabled.as_deref() {
                        Some("enabled") => true,
                        Some("disabled") => false,
                        _ => state.fab_enabled,
                    },
                    url: req.fab_url.filter(|u| !u.is_empty()),
                    send_type: match req.fab_send_type.as_deref() {
                        Some("growing") => "growing".to_string(),
                        Some("confirmed") => "confirmed".to_string(),
                        _ => state.fab_send_type.clone(),
                    },
                };
                state.fab_configs.write().await.insert(session.id.clone(), fab_config);
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

    // Clean up FAB config
    {
        let mut configs = state.fab_configs.write().await;
        configs.remove(&id);
    }

    // Always remove the session from the manager
    let _ = state.session_manager.remove_session(&id).await;

    Json(ApiResponse::success(()))
}

/// Get transcript for a completed VoD session
pub async fn get_transcript(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    use axum::body::Body;
    use axum::http::{header, StatusCode};
    use axum::response::IntoResponse;

    // Get the session
    let session = match state.session_manager.get_session(&id).await {
        Some(s) => s,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiResponse::<()>::error("Session not found")),
            )
                .into_response();
        }
    };

    // Check if transcript is available
    let transcript_path = match session.transcript_path().await {
        Some(path) => path,
        None => {
            return (
                StatusCode::NOT_FOUND,
                Json(ApiResponse::<()>::error("Transcript not available")),
            )
                .into_response();
        }
    };

    // Read the transcript file
    match tokio::fs::read_to_string(&transcript_path).await {
        Ok(content) => {
            // Get filename for Content-Disposition
            let filename = transcript_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("transcript.json");

            (
                StatusCode::OK,
                [
                    (header::CONTENT_TYPE, "application/json"),
                    (
                        header::CONTENT_DISPOSITION,
                        &format!("attachment; filename=\"{}\"", filename),
                    ),
                ],
                content,
            )
                .into_response()
        }
        Err(e) => {
            eprintln!(
                "[Session {}] Failed to read transcript: {}",
                id, e
            );
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::<()>::error("Failed to read transcript")),
            )
                .into_response()
        }
    }
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

    // Spawn FAB forwarder if enabled (skip VoD batch mode)
    if session.mode != "vod" {
        let fab_config = state.fab_configs.read().await.get(&id).cloned();
        let should_run = fab_config.as_ref().map(|c| c.enabled).unwrap_or(state.fab_enabled);
        if should_run {
            let url = fab_config.as_ref().and_then(|c| c.url.clone())
                .or_else(|| state.fab_url.clone());
            let send_type = fab_config.as_ref()
                .map(|c| c.send_type.clone())
                .unwrap_or_else(|| state.fab_send_type.clone());
            if let (Some(url), Some(ref client)) = (url, &state.fab_client) {
                let subtitle_rx = session.subscribe_subtitles();
                crate::fab_forwarder::spawn_fab_forwarder(
                    id.clone(),
                    url,
                    send_type,
                    subtitle_rx,
                    client.clone(),
                );
            }
        }
    }

    let session_clone = session.clone();
    let model_path = model.model_path.clone();
    let diar_path = model.diarization_path.clone();
    let exec_config = model.exec_config.clone();
    let model_id = session.model_id.clone();
    let language = session.language.clone();
    let sentence_completion = session.sentence_completion.clone();

    // Determine audio source type
    let audio_source = if let Some(srt_url) = &session.srt_url {
        AudioSource::Srt(srt_url.clone())
    } else {
        AudioSource::File(session.wav_path.clone())
    };

    std::thread::spawn(move || {
        run_session_transcription(
            session_clone,
            audio_source,
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
            sentence_completion,
        );
    });

    let info = session.info().await;
    Json(ApiResponse::success(info))
}
