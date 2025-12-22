//! Session management for multi-stream transcription
//!
//! This module provides:
//! - Session types for representing transcription sessions
//! - SessionManager for creating, listing, and managing sessions
//! - Thread-safe session state management
//! - Support for both media files and SRT live streams

use crate::error::{Error, Result};
use crate::media_manager::SharedMediaManager;
use crate::model_registry::SharedModelRegistry;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Media source type for sessions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaSourceType {
    /// Media file from disk
    File,
    /// Live SRT stream
    SrtStream,
}

impl Default for MediaSourceType {
    fn default() -> Self {
        Self::File
    }
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    /// Session is initializing
    Starting,
    /// Session is actively transcribing
    Running,
    /// Session is paused
    Paused,
    /// Session has completed
    Completed,
    /// Session has stopped (manually or due to error)
    Stopped,
}

impl SessionState {
    pub fn as_str(&self) -> &'static str {
        match self {
            SessionState::Starting => "starting",
            SessionState::Running => "running",
            SessionState::Paused => "paused",
            SessionState::Completed => "completed",
            SessionState::Stopped => "stopped",
        }
    }
}

/// Session information for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub media_id: String,
    pub media_filename: String,
    pub state: SessionState,
    pub client_count: u64,
    pub duration_secs: f32,
    pub progress_secs: f32,
    pub created_at: u64,
    /// Latency mode for transcription
    #[serde(default)]
    pub mode: String,
    /// Language code for transcription (e.g., "de", "en")
    #[serde(default = "default_language")]
    pub language: String,
    /// Noise cancellation type ("none", "rnnoise", "deepfilternet3")
    #[serde(default)]
    pub noise_cancellation: String,
    /// Whether diarization is enabled
    #[serde(default)]
    pub diarization: bool,
    /// Diarization model name (if enabled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diarization_model: Option<String>,
    /// Media source type (file or srt_stream)
    #[serde(default)]
    pub source_type: MediaSourceType,
    /// SRT channel ID (for SRT streams)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub srt_channel_id: Option<usize>,
    /// SRT channel name (for SRT streams)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub srt_channel_name: Option<String>,
    /// SRT URL (for SRT streams)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub srt_url: Option<String>,
    /// Sentence completion mode ("off", "minimal", "balanced", "complete")
    #[serde(default = "default_sentence_completion")]
    pub sentence_completion: String,
    /// Whether transcript is available for download (VoD mode)
    #[serde(default)]
    pub transcript_available: bool,
    /// VoD progress information (only for VoD mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vod_progress: Option<VodProgressInfo>,
}

/// VoD progress information for API responses
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VodProgressInfo {
    pub total_chunks: usize,
    pub completed_chunks: usize,
    pub percent: f32,
}

fn default_sentence_completion() -> String {
    "minimal".to_string()
}

fn default_language() -> String {
    "de".to_string()
}

/// A transcription session
pub struct TranscriptionSession {
    /// Unique session ID
    pub id: String,
    /// Model ID being used
    pub model_id: String,
    /// Model display name
    pub model_name: String,
    /// Media file ID (for file sessions)
    pub media_id: String,
    /// Media filename (or SRT channel name for SRT sessions)
    pub media_filename: String,
    /// Path to WAV file for transcription (empty for SRT sessions)
    pub wav_path: PathBuf,
    /// Total duration in seconds (0 for live SRT streams)
    pub duration_secs: f32,
    /// Current state
    state: RwLock<SessionState>,
    /// Number of connected clients
    pub client_count: AtomicU64,
    /// Subtitle broadcast channel
    pub subtitle_tx: broadcast::Sender<String>,
    /// Status broadcast channel
    pub status_tx: broadcast::Sender<String>,
    /// Running flag
    pub running: Arc<AtomicBool>,
    /// Creation timestamp
    pub created_at: u64,
    /// Progress in seconds
    progress_secs: RwLock<f32>,
    /// Latency mode for transcription
    pub mode: String,
    /// Language code for transcription (e.g., "de", "en")
    pub language: String,
    /// Noise cancellation type ("none", "rnnoise", "deepfilternet3")
    pub noise_cancellation: String,
    /// Whether diarization is enabled
    pub diarization: bool,
    /// Diarization model name (if enabled)
    pub diarization_model: Option<String>,
    /// Cached last subtitle for late-joining clients (uses std::sync::RwLock for sync access)
    last_subtitle: StdRwLock<Option<String>>,
    /// Media source type (file or SRT stream)
    pub source_type: MediaSourceType,
    /// SRT channel ID (for SRT streams)
    pub srt_channel_id: Option<usize>,
    /// SRT channel name (for SRT streams)
    pub srt_channel_name: Option<String>,
    /// SRT URL (for SRT streams)
    pub srt_url: Option<String>,
    /// Sentence completion mode ("off", "minimal", "balanced", "complete")
    pub sentence_completion: String,
    /// Path to transcript.json (VoD mode)
    transcript_path: RwLock<Option<PathBuf>>,
    /// VoD progress (VoD mode)
    vod_progress: RwLock<Option<VodProgressInfo>>,
}

impl TranscriptionSession {
    /// Create a new file-based session
    pub fn new(
        model_id: String,
        model_name: String,
        media_id: String,
        media_filename: String,
        wav_path: PathBuf,
        duration_secs: f32,
        mode: String,
        language: String,
        noise_cancellation: String,
        diarization: bool,
        diarization_model: Option<String>,
        sentence_completion: String,
    ) -> Self {
        let (subtitle_tx, _) = broadcast::channel(1000);
        let (status_tx, _) = broadcast::channel(100);

        Self {
            id: Uuid::new_v4().to_string()[..8].to_string(),
            model_id,
            model_name,
            media_id,
            media_filename,
            wav_path,
            duration_secs,
            state: RwLock::new(SessionState::Starting),
            client_count: AtomicU64::new(0),
            subtitle_tx,
            status_tx,
            running: Arc::new(AtomicBool::new(false)),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            progress_secs: RwLock::new(0.0),
            mode,
            language,
            noise_cancellation,
            diarization,
            diarization_model,
            last_subtitle: StdRwLock::new(None),
            source_type: MediaSourceType::File,
            srt_channel_id: None,
            srt_channel_name: None,
            srt_url: None,
            sentence_completion,
            transcript_path: RwLock::new(None),
            vod_progress: RwLock::new(None),
        }
    }

    /// Create a new SRT stream session
    pub fn new_srt(
        model_id: String,
        model_name: String,
        srt_channel_id: usize,
        srt_channel_name: String,
        srt_url: String,
        mode: String,
        language: String,
        noise_cancellation: String,
        diarization: bool,
        diarization_model: Option<String>,
        sentence_completion: String,
    ) -> Self {
        let (subtitle_tx, _) = broadcast::channel(1000);
        let (status_tx, _) = broadcast::channel(100);

        Self {
            id: Uuid::new_v4().to_string()[..8].to_string(),
            model_id,
            model_name,
            media_id: format!("srt:{}", srt_channel_id),
            media_filename: srt_channel_name.clone(),
            wav_path: PathBuf::new(), // Empty for SRT streams
            duration_secs: 0.0,       // Unknown for live streams
            state: RwLock::new(SessionState::Starting),
            client_count: AtomicU64::new(0),
            subtitle_tx,
            status_tx,
            running: Arc::new(AtomicBool::new(false)),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            progress_secs: RwLock::new(0.0),
            mode,
            language,
            noise_cancellation,
            diarization,
            diarization_model,
            last_subtitle: StdRwLock::new(None),
            source_type: MediaSourceType::SrtStream,
            srt_channel_id: Some(srt_channel_id),
            srt_channel_name: Some(srt_channel_name),
            srt_url: Some(srt_url),
            sentence_completion,
            transcript_path: RwLock::new(None),
            vod_progress: RwLock::new(None),
        }
    }

    /// Check if this is an SRT stream session
    pub fn is_srt_stream(&self) -> bool {
        self.source_type == MediaSourceType::SrtStream
    }

    /// Get session info for API responses
    pub async fn info(&self) -> SessionInfo {
        SessionInfo {
            id: self.id.clone(),
            model_id: self.model_id.clone(),
            model_name: self.model_name.clone(),
            media_id: self.media_id.clone(),
            media_filename: self.media_filename.clone(),
            state: *self.state.read().await,
            client_count: self.client_count.load(Ordering::SeqCst),
            duration_secs: self.duration_secs,
            progress_secs: *self.progress_secs.read().await,
            created_at: self.created_at,
            mode: self.mode.clone(),
            language: self.language.clone(),
            noise_cancellation: self.noise_cancellation.clone(),
            diarization: self.diarization,
            diarization_model: self.diarization_model.clone(),
            source_type: self.source_type,
            srt_channel_id: self.srt_channel_id,
            srt_channel_name: self.srt_channel_name.clone(),
            srt_url: self.srt_url.clone(),
            sentence_completion: self.sentence_completion.clone(),
            transcript_available: self.transcript_path.read().await.is_some(),
            vod_progress: self.vod_progress.read().await.clone(),
        }
    }

    /// Get current state
    pub async fn state(&self) -> SessionState {
        *self.state.read().await
    }

    /// Set session state
    pub async fn set_state(&self, state: SessionState) {
        *self.state.write().await = state;
    }

    /// Update progress
    pub async fn set_progress(&self, progress: f32) {
        *self.progress_secs.write().await = progress;
    }

    /// Get progress
    pub async fn progress(&self) -> f32 {
        *self.progress_secs.read().await
    }

    /// Increment client count
    pub fn add_client(&self) -> u64 {
        self.client_count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Decrement client count
    pub fn remove_client(&self) -> u64 {
        let prev = self.client_count.fetch_sub(1, Ordering::SeqCst);
        if prev > 0 { prev - 1 } else { 0 }
    }

    /// Get client count
    pub fn get_client_count(&self) -> u64 {
        self.client_count.load(Ordering::SeqCst)
    }

    /// Subscribe to subtitle updates
    pub fn subscribe_subtitles(&self) -> broadcast::Receiver<String> {
        self.subtitle_tx.subscribe()
    }

    /// Subscribe to status updates
    pub fn subscribe_status(&self) -> broadcast::Receiver<String> {
        self.status_tx.subscribe()
    }

    /// Set the last subtitle for late-joining clients (synchronous)
    pub fn set_last_subtitle(&self, subtitle: String) {
        if let Ok(mut guard) = self.last_subtitle.write() {
            *guard = Some(subtitle);
        }
    }

    /// Get the last subtitle for late-joining clients (synchronous)
    pub fn get_last_subtitle(&self) -> Option<String> {
        self.last_subtitle.read().ok().and_then(|guard| guard.clone())
    }

    /// Check if session is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Start the session
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    /// Stop the session
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Set transcript path (VoD mode)
    pub async fn set_transcript_path(&self, path: PathBuf) {
        *self.transcript_path.write().await = Some(path);
    }

    /// Get transcript path (VoD mode)
    pub async fn transcript_path(&self) -> Option<PathBuf> {
        self.transcript_path.read().await.clone()
    }

    /// Set VoD progress
    pub async fn set_vod_progress(&self, total: usize, completed: usize) {
        let percent = if total > 0 {
            (completed as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        *self.vod_progress.write().await = Some(VodProgressInfo {
            total_chunks: total,
            completed_chunks: completed,
            percent,
        });
    }

    /// Clear VoD progress
    pub async fn clear_vod_progress(&self) {
        *self.vod_progress.write().await = None;
    }
}

/// Manager for transcription sessions
pub struct SessionManager {
    /// Active sessions
    sessions: RwLock<HashMap<String, Arc<TranscriptionSession>>>,
    /// Model registry
    model_registry: SharedModelRegistry,
    /// Media manager
    media_manager: SharedMediaManager,
    /// Maximum concurrent sessions
    max_sessions: usize,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(
        model_registry: SharedModelRegistry,
        media_manager: SharedMediaManager,
        max_sessions: usize,
    ) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            model_registry,
            media_manager,
            max_sessions,
        }
    }

    /// Create from environment variables
    pub fn from_env(
        model_registry: SharedModelRegistry,
        media_manager: SharedMediaManager,
    ) -> Self {
        let max_sessions = std::env::var("MAX_CONCURRENT_SESSIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);

        Self::new(model_registry, media_manager, max_sessions)
    }

    /// Create a new transcription session
    pub async fn create_session(
        &self,
        model_id: &str,
        media_id: &str,
        mode: &str,
        language: &str,
        noise_cancellation: &str,
        diarization: bool,
        diarization_model: Option<String>,
        sentence_completion: &str,
    ) -> Result<Arc<TranscriptionSession>> {
        // Check session limit
        let current_count = self.sessions.read().await.len();
        if current_count >= self.max_sessions {
            return Err(Error::Model(format!(
                "Maximum sessions reached ({})",
                self.max_sessions
            )));
        }

        // Validate model
        let model = self.model_registry.get_model(model_id).ok_or_else(|| {
            Error::Model(format!("Unknown model: {}", model_id))
        })?;

        if !model.is_available {
            return Err(Error::Model(format!(
                "Model {} not available",
                model_id
            )));
        }

        // Get media file
        let media = self.media_manager.get_file(media_id).await.ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Media file not found: {}", media_id),
            ))
        })?;

        // Get WAV path (may trigger conversion)
        let wav_path = self.media_manager.get_wav_path(media_id).await?;

        // Use default language if empty
        let lang = if language.is_empty() { "de" } else { language };

        // Create session
        let session = TranscriptionSession::new(
            model_id.to_string(),
            model.model_type.display_name().to_string(),
            media_id.to_string(),
            media.filename,
            wav_path,
            media.duration_secs.unwrap_or(0.0),
            mode.to_string(),
            lang.to_string(),
            noise_cancellation.to_string(),
            diarization,
            diarization_model,
            sentence_completion.to_string(),
        );

        let session = Arc::new(session);

        // Add to sessions
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session.id.clone(), session.clone());
        }

        eprintln!(
            "[SessionManager] Created session {} (model: {}, media: {}, mode: {}, language: {}, noise: {}, diarization: {})",
            session.id, model_id, media_id, mode, lang, noise_cancellation, diarization
        );

        Ok(session)
    }

    /// Create a new SRT stream session
    pub async fn create_srt_session(
        &self,
        model_id: &str,
        srt_channel_id: usize,
        srt_channel_name: &str,
        srt_url: &str,
        mode: &str,
        language: &str,
        noise_cancellation: &str,
        diarization: bool,
        diarization_model: Option<String>,
        sentence_completion: &str,
    ) -> Result<Arc<TranscriptionSession>> {
        // Check session limit
        let current_count = self.sessions.read().await.len();
        if current_count >= self.max_sessions {
            return Err(Error::Model(format!(
                "Maximum sessions reached ({})",
                self.max_sessions
            )));
        }

        // Validate model
        let model = self.model_registry.get_model(model_id).ok_or_else(|| {
            Error::Model(format!("Unknown model: {}", model_id))
        })?;

        if !model.is_available {
            return Err(Error::Model(format!(
                "Model {} not available",
                model_id
            )));
        }

        // Use default language if empty
        let lang = if language.is_empty() { "de" } else { language };

        // Create SRT session
        let session = TranscriptionSession::new_srt(
            model_id.to_string(),
            model.model_type.display_name().to_string(),
            srt_channel_id,
            srt_channel_name.to_string(),
            srt_url.to_string(),
            mode.to_string(),
            lang.to_string(),
            noise_cancellation.to_string(),
            diarization,
            diarization_model,
            sentence_completion.to_string(),
        );

        let session = Arc::new(session);

        // Add to sessions
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session.id.clone(), session.clone());
        }

        eprintln!(
            "[SessionManager] Created SRT session {} (model: {}, channel: {} [{}], mode: {}, language: {}, noise: {}, diarization: {})",
            session.id, model_id, srt_channel_name, srt_channel_id, mode, lang, noise_cancellation, diarization
        );

        Ok(session)
    }

    /// Get a session by ID
    pub async fn get_session(&self, id: &str) -> Option<Arc<TranscriptionSession>> {
        let sessions = self.sessions.read().await;
        sessions.get(id).cloned()
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> Vec<SessionInfo> {
        let sessions = self.sessions.read().await;
        let mut infos = Vec::with_capacity(sessions.len());

        for session in sessions.values() {
            infos.push(session.info().await);
        }

        // Sort by creation time (newest first)
        infos.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        infos
    }

    /// Stop a session
    pub async fn stop_session(&self, id: &str) -> Result<()> {
        let session = {
            let sessions = self.sessions.read().await;
            sessions.get(id).cloned()
        };

        if let Some(session) = session {
            session.stop();
            session.set_state(SessionState::Stopped).await;
            eprintln!("[SessionManager] Stopped session {}", id);
            Ok(())
        } else {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Session not found: {}", id),
            )))
        }
    }

    /// Remove a session
    pub async fn remove_session(&self, id: &str) -> Result<()> {
        let removed = {
            let mut sessions = self.sessions.write().await;
            sessions.remove(id)
        };

        if removed.is_some() {
            eprintln!("[SessionManager] Removed session {}", id);
            Ok(())
        } else {
            Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Session not found: {}", id),
            )))
        }
    }

    /// Cleanup completed or stopped sessions
    pub async fn cleanup(&self) {
        let mut sessions = self.sessions.write().await;
        let before = sessions.len();

        sessions.retain(|id, session| {
            let running = session.is_running();
            let clients = session.get_client_count();

            // Keep if running or has clients
            if running || clients > 0 {
                return true;
            }

            eprintln!("[SessionManager] Cleaning up session {}", id);
            false
        });

        let removed = before - sessions.len();
        if removed > 0 {
            eprintln!("[SessionManager] Cleaned up {} sessions", removed);
        }
    }

    /// Get session count
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Get model registry
    pub fn model_registry(&self) -> &SharedModelRegistry {
        &self.model_registry
    }

    /// Get media manager
    pub fn media_manager(&self) -> &SharedMediaManager {
        &self.media_manager
    }
}

/// Thread-safe reference to SessionManager
pub type SharedSessionManager = Arc<SessionManager>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state() {
        assert_eq!(SessionState::Running.as_str(), "running");
        assert_eq!(SessionState::Stopped.as_str(), "stopped");
    }
}
