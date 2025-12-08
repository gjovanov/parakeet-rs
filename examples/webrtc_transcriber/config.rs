//! Configuration types for the WebRTC transcription server

use serde::{Deserialize, Serialize};

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

/// Runtime configuration for frontend
#[derive(Clone)]
pub struct RuntimeConfig {
    pub ws_url: String,
    pub turn_server: String,
    pub turn_username: String,
    pub turn_password: String,
}
