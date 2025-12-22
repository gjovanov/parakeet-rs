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
            LatencyMode::VadSlidingWindow => "VAD Sliding Window (10 seg / 15s)",
            LatencyMode::Asr => "ASR (Pure streaming)",
            LatencyMode::Parallel => "Parallel Sliding Window (8 threads)",
            LatencyMode::PauseParallel => "Pause-Parallel (8 threads, ordered)",
            LatencyMode::Confidence => "Confidence (AWS-style streaming)",
            LatencyMode::Vod => "VoD Batch (10-min chunks)",
        }
    }

    /// Check if this mode uses VAD-triggered transcription
    pub fn is_vad_mode(&self) -> bool {
        matches!(self, LatencyMode::VadSpeedy | LatencyMode::VadPauseBased | LatencyMode::VadSlidingWindow)
    }

    /// Get the underlying VAD mode string for VadConfig::from_mode()
    pub fn vad_mode_str(&self) -> &'static str {
        match self {
            LatencyMode::VadSpeedy => "speedy",
            LatencyMode::VadPauseBased => "pause_based",
            LatencyMode::VadSlidingWindow => "sliding_window",
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
            LatencyMode::VadSlidingWindow,
            LatencyMode::Asr,
            LatencyMode::Parallel,
            LatencyMode::PauseParallel,
            LatencyMode::Confidence,
            LatencyMode::Vod,
        ]
    }

    /// Check if this mode is confidence-based
    pub fn is_confidence_mode(&self) -> bool {
        matches!(self, LatencyMode::Confidence)
    }

    /// Check if this mode is VoD batch mode
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
}
