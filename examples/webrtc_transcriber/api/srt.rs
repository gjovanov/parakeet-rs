//! SRT streams API endpoint

use crate::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
struct SrtStreamsResponse {
    success: bool,
    streams: Vec<SrtStreamInfo>,
    configured: bool,
}

#[derive(Serialize)]
struct SrtStreamInfo {
    id: usize,
    name: String,
    port: String,
    display: String,
}

/// List available SRT streams
/// GET /api/srt-streams
pub async fn list_srt_streams(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let (streams, configured) = if let Some(ref srt_config) = state.srt_config {
        let streams = srt_config
            .list_streams()
            .into_iter()
            .map(|s| SrtStreamInfo {
                id: s.id,
                name: s.name,
                port: s.port,
                display: s.display,
            })
            .collect();
        (streams, true)
    } else {
        (vec![], false)
    };

    Json(SrtStreamsResponse {
        success: true,
        streams,
        configured,
    })
}
