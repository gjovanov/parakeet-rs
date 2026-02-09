//! API handlers for models and modes

use crate::state::AppState;
use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

/// Generic API response
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
        }
    }
}

/// List available transcription models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<parakeet_rs::ModelInfo>>> {
    let models = state.model_registry.list_models();
    Json(ApiResponse::success(models))
}

/// Mode info for API responses
#[derive(Debug, Serialize)]
pub struct ModeInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
}

/// List available transcription modes
pub async fn list_modes() -> Json<ApiResponse<Vec<ModeInfo>>> {
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
            id: "vad_sliding_window",
            name: "VAD Sliding Window (5 seg / 10s)",
            description: "Sliding window VAD. Buffers 5 segments or 10s, transcribes with context overlap for better accuracy.",
        },
        ModeInfo {
            id: "asr",
            name: "ASR (Pure streaming)",
            description: "Pure streaming ASR without VAD. Processes audio continuously with sliding window.",
        },
        ModeInfo {
            id: "parallel",
            name: "Parallel Sliding Window",
            description: "Multi-threaded parallel inference with sliding window. Uses 8 threads for Canary, 4 for TDT. Works with both models.",
        },
        ModeInfo {
            id: "pause_parallel",
            name: "Pause-Parallel (ordered)",
            description: "Pause-triggered parallel inference. Dispatches on speech pauses for natural boundaries. Ordered output. Works with both models.",
        },
        ModeInfo {
            id: "vod",
            name: "VoD Batch (10-min chunks)",
            description: "Batch transcription for pre-recorded files. Processes in 10-minute chunks with overlap and deduplication. Generates downloadable transcript.json.",
        },
    ];
    Json(ApiResponse::success(modes))
}
