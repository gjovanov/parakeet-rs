//! API handlers for noise cancellation options

use super::models::ApiResponse;
use crate::state::AppState;
use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

/// Noise cancellation option info
#[derive(Debug, Clone, Serialize)]
pub struct NoiseCancellationOption {
    pub id: String,
    pub name: String,
    pub description: String,
    pub available: bool,
}

/// List available noise cancellation options
pub async fn list_noise_cancellation(
    State(_state): State<Arc<AppState>>,
) -> Json<ApiResponse<Vec<NoiseCancellationOption>>> {
    let options = vec![
        NoiseCancellationOption {
            id: "none".to_string(),
            name: "None".to_string(),
            description: "No noise cancellation".to_string(),
            available: true,
        },
        NoiseCancellationOption {
            id: "rnnoise".to_string(),
            name: "RNNoise".to_string(),
            description: "Lightweight neural network noise suppression (built-in)".to_string(),
            available: true, // Always available - built-in weights
        },
        NoiseCancellationOption {
            id: "deepfilternet3".to_string(),
            name: "DeepFilterNet3".to_string(),
            description: "High-quality deep filtering noise suppression".to_string(),
            // DeepFilterNet3 falls back to RNNoise when not implemented
            available: true,
        },
    ];
    Json(ApiResponse::success(options))
}
