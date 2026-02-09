//! API handlers for diarization options

use super::models::ApiResponse;
use crate::state::AppState;
use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

/// Diarization option info
#[derive(Debug, Clone, Serialize)]
pub struct DiarizationOption {
    pub id: String,
    pub name: String,
    pub description: String,
    pub available: bool,
}

/// List available diarization options
pub async fn list_diarization(
    State(state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<DiarizationOption>>> {
    // Check if any model has diarization available
    let has_diarization = state
        .model_registry
        .list_models()
        .iter()
        .any(|m| m.supports_diarization);

    let mut options = vec![DiarizationOption {
        id: "none".to_string(),
        name: "None".to_string(),
        description: "No speaker diarization".to_string(),
        available: true,
    }];

    if has_diarization {
        options.push(DiarizationOption {
            id: "sortformer".to_string(),
            name: "Sortformer v2 (4 speakers)".to_string(),
            description: "NVIDIA Sortformer streaming speaker identification".to_string(),
            available: true,
        });
    }

    Json(ApiResponse::success(options))
}
