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
    /// Growing segments: word-by-word PARTIAL updates building toward FINAL sentences
    GrowingSegments,
    /// Pause-segmented: segment audio by acoustic pauses, transcribe each chunk once
    PauseSegmented,
}

#[allow(dead_code)]
impl LatencyMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            LatencyMode::Speedy => "speedy",
            LatencyMode::GrowingSegments => "growing_segments",
            LatencyMode::PauseSegmented => "pause_segmented",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            LatencyMode::Speedy => "Speedy (~0.3-1.5s)",
            LatencyMode::GrowingSegments => "Growing Segments (word-by-word)",
            LatencyMode::PauseSegmented => "Pause-Segmented (1 chunk per pause)",
        }
    }

    pub fn all() -> &'static [LatencyMode] {
        &[
            LatencyMode::Speedy,
            LatencyMode::GrowingSegments,
            LatencyMode::PauseSegmented,
        ]
    }
}

/// Runtime configuration for frontend
#[derive(Clone)]
pub struct RuntimeConfig {
    pub ws_url: String,
    pub turn_server: String,
    pub turn_username: String,
    pub turn_password: String,
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
}
