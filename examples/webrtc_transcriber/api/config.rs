//! API handler for frontend configuration

use crate::state::AppState;
use axum::{extract::State, response::IntoResponse};
use std::sync::Arc;

/// Return frontend configuration (WebSocket URL, ICE servers, etc.)
pub async fn config_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = &state.config;

    let mut ice_servers = vec![
        serde_json::json!({ "urls": "stun:stun.l.google.com:19302" })
    ];

    if !config.turn_server.is_empty() {
        ice_servers.push(serde_json::json!({
            "urls": config.turn_server,
            "username": config.turn_username,
            "credential": config.turn_password
        }));
    }

    let response = serde_json::json!({
        "wsUrl": config.ws_url,
        "iceServers": ice_servers,
        "speakerColors": [
            "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
            "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98"
        ]
    });

    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        response.to_string()
    )
}
