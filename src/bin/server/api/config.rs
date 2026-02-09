//! API handler for frontend configuration

use crate::state::AppState;
use axum::{extract::State, response::IntoResponse};
use std::sync::Arc;

/// Return frontend configuration (WebSocket URL, ICE servers, etc.)
pub async fn config_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let config = &state.config;

    let force_relay = std::env::var("FORCE_RELAY")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    // In relay-only mode, skip STUN servers (they leak Docker IPs in VPN/NAT setups)
    let mut ice_servers = if force_relay {
        vec![]
    } else {
        vec![serde_json::json!({ "urls": "stun:stun.l.google.com:19302" })]
    };

    if !config.turn_server.is_empty() {
        // Provide both UDP and TCP TURN URLs for maximum compatibility
        // Server (webrtc-rs) uses UDP, Windows browsers may need TCP
        let turn_url = &config.turn_server;
        let mut turn_urls = vec![turn_url.clone()];

        // If no transport specified, add TCP variant for browsers behind strict firewalls
        if !turn_url.contains("?transport=") {
            turn_urls.push(format!("{}?transport=tcp", turn_url));
        }

        ice_servers.push(serde_json::json!({
            "urls": turn_urls,
            "username": config.turn_username,
            "credential": config.turn_password
        }));
    }

    let ice_transport_policy = if force_relay { "relay" } else { "all" };

    let response = serde_json::json!({
        "wsUrl": config.ws_url,
        "iceServers": ice_servers,
        "iceTransportPolicy": ice_transport_policy,
        "speakerColors": [
            "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
            "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98"
        ],
        "fabEnabled": state.fab_enabled,
        "fabUrl": state.fab_url.as_deref().unwrap_or(""),
        "fabSendType": state.fab_send_type
    });

    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        response.to_string()
    )
}
