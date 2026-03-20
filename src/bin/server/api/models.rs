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
            description: "Best balance of latency and quality. Uses pause detection for word confirmation.",
        },
        ModeInfo {
            id: "growing_segments",
            name: "Growing Segments (word-by-word)",
            description: "Real-time growing transcription. Shows words appearing one by one (PARTIAL) \
                          until sentence completion (FINAL). Best for live subtitle display.",
        },
        ModeInfo {
            id: "pause_segmented",
            name: "Pause-Segmented (1 chunk per pause)",
            description: "Segments audio by acoustic pauses, transcribes each speech chunk exactly once. \
                          Precise timestamps, no echo dedup. Each FINAL = one speech segment.",
        },
    ];
    Json(ApiResponse::success(modes))
}
