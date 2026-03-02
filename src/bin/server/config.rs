//! Configuration types for the WebRTC transcription server

use base64::Engine;
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha1::Sha1;

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
    /// VAD sliding window mode: buffer multiple segments, transcribe with context
    VadSlidingWindow,
    /// Pure streaming ASR mode: continuous transcription without VAD
    Asr,
    /// Parallel sliding window: 8 threads processing overlapping windows
    Parallel,
    /// Pause-based parallel: dispatch on pause detection, ordered output
    PauseParallel,
    /// Confidence mode: AWS Transcribe-like streaming with confidence filtering
    Confidence,
    /// VoD batch mode: process entire file in 10-min chunks with deduplication
    Vod,
    /// Growing segments: word-by-word PARTIAL updates building toward FINAL sentences
    GrowingSegments,
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
            LatencyMode::VadSlidingWindow => "vad_sliding_window",
            LatencyMode::Asr => "asr",
            LatencyMode::Parallel => "parallel",
            LatencyMode::PauseParallel => "pause_parallel",
            LatencyMode::Confidence => "confidence",
            LatencyMode::Vod => "vod",
            LatencyMode::GrowingSegments => "growing_segments",
        }
    }

    #[allow(dead_code)]
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
            LatencyMode::VadSlidingWindow => "VAD Sliding Window (10 seg / 15s)",
            LatencyMode::Asr => "ASR (Pure streaming)",
            LatencyMode::Parallel => "Parallel Sliding Window (8 threads)",
            LatencyMode::PauseParallel => "Pause-Parallel (8 threads, ordered)",
            LatencyMode::Confidence => "Confidence (AWS-style streaming)",
            LatencyMode::Vod => "VoD Batch (10-min chunks)",
            LatencyMode::GrowingSegments => "Growing Segments (word-by-word)",
        }
    }

    /// Check if this mode uses VAD-triggered transcription
    #[allow(dead_code)]
    pub fn is_vad_mode(&self) -> bool {
        matches!(self, LatencyMode::VadSpeedy | LatencyMode::VadPauseBased | LatencyMode::VadSlidingWindow)
    }

    /// Get the underlying VAD mode string for VadConfig::from_mode()
    #[allow(dead_code)]
    pub fn vad_mode_str(&self) -> &'static str {
        match self {
            LatencyMode::VadSpeedy => "speedy",
            LatencyMode::VadPauseBased => "pause_based",
            LatencyMode::VadSlidingWindow => "sliding_window",
            _ => "speedy", // Default for non-VAD modes
        }
    }

    #[allow(dead_code)]
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
            LatencyMode::VadSlidingWindow,
            LatencyMode::Asr,
            LatencyMode::Parallel,
            LatencyMode::PauseParallel,
            LatencyMode::Confidence,
            LatencyMode::Vod,
            LatencyMode::GrowingSegments,
        ]
    }

    /// Check if this mode is confidence-based
    #[allow(dead_code)]
    pub fn is_confidence_mode(&self) -> bool {
        matches!(self, LatencyMode::Confidence)
    }

    /// Check if this mode is VoD batch mode
    #[allow(dead_code)]
    pub fn is_vod_mode(&self) -> bool {
        matches!(self, LatencyMode::Vod)
    }
}

/// Runtime configuration for frontend
#[derive(Clone)]
pub struct RuntimeConfig {
    pub ws_url: String,
    pub turn_server: String,
    pub turn_username: String,
    pub turn_password: String,
<<<<<<< HEAD
    pub turn_shared_secret: String,
    pub turn_credential_ttl: u64,
}

/// Generate ephemeral TURN credentials using HMAC-SHA1 (RFC 5389 shared-secret mode).
///
/// Returns `(username, credential)` where:
/// - `username` = `<unix_expiry_timestamp>:parakeet`
/// - `credential` = `base64(HMAC-SHA1(shared_secret, username))`
pub fn generate_turn_credentials(shared_secret: &str, ttl: u64) -> (String, String) {
    let expiry = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        + ttl;
    let username = format!("{}:parakeet", expiry);

    let mut mac =
        Hmac::<Sha1>::new_from_slice(shared_secret.as_bytes()).expect("HMAC accepts any key size");
    mac.update(username.as_bytes());
    let result = mac.finalize();
    let credential = base64::engine::general_purpose::STANDARD.encode(result.into_bytes());

    (username, credential)
=======
    /// COTURN shared secret for ephemeral credentials (overrides username/password when set)
    pub turn_shared_secret: String,
>>>>>>> 423e2252a776f67ae1aec078e6f034ba429f26f9
}
