//! WebRTC and WebSocket handlers

pub mod audio;

use crate::state::{AppState, ClientConnection};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use parakeet_rs::TranscriptionSession;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio::time::interval;
use uuid::Uuid;
use webrtc::{
    ice_transport::{
        ice_candidate::RTCIceCandidateInit, ice_credential_type::RTCIceCredentialType,
        ice_server::RTCIceServer,
    },
    peer_connection::{
        configuration::RTCConfiguration, policy::ice_transport_policy::RTCIceTransportPolicy,
        sdp::session_description::RTCSessionDescription, RTCPeerConnection,
    },
    track::track_local::TrackLocal,
};

/// WebSocket upgrade handler
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(session_id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, session_id, state))
}

/// Handle WebSocket connection
pub async fn handle_socket(socket: WebSocket, session_id: String, state: Arc<AppState>) {
    let client_id = Uuid::new_v4().to_string();
    let count = state.client_count.fetch_add(1, Ordering::SeqCst) + 1;
    eprintln!(
        "[WebRTC] Client {} connecting to session {} (total: {})",
        &client_id[..8],
        session_id,
        count
    );

    // Get the session
    let session = match state.session_manager.get_session(&session_id).await {
        Some(s) => s,
        None => {
            eprintln!("[WebRTC] Session {} not found", session_id);
            state.client_count.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    // Add client to session
    session.add_client();

    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Channel for ICE candidates
    let (ice_tx, mut ice_rx) = mpsc::channel::<String>(100);

    // Subscribe to session broadcasts
    let mut subtitle_rx = session.subscribe_subtitles();
    let mut status_rx = session.subscribe_status();

    // Create peer connection
    let turn_server = std::env::var("TURN_SERVER").unwrap_or_default();
    let turn_username = std::env::var("TURN_USERNAME").unwrap_or_default();
    let turn_password = std::env::var("TURN_PASSWORD").unwrap_or_default();
    let force_relay = std::env::var("FORCE_RELAY")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    // In relay-only mode, skip STUN servers (they leak Docker IPs in VPN/NAT setups)
    let mut ice_servers = if force_relay {
        vec![]
    } else {
        vec![RTCIceServer {
            urls: vec!["stun:stun.l.google.com:19302".to_owned()],
            ..Default::default()
        }]
    };

    if !turn_server.is_empty() {
        eprintln!("[WebRTC] Configuring TURN: {} (user: {})", turn_server, turn_username);
        ice_servers.push(RTCIceServer {
            urls: vec![turn_server.clone()],
            username: turn_username,
            credential: turn_password,
            credential_type: RTCIceCredentialType::Password,
        });
    } else {
        eprintln!("[WebRTC] No TURN server configured");
    }

    // Use relay-only policy for VPN/Transit Gateway/complex NAT setups
    let ice_transport_policy = if force_relay {
        eprintln!("[WebRTC] FORCE_RELAY enabled - using TURN relay only");
        RTCIceTransportPolicy::Relay
    } else {
        RTCIceTransportPolicy::All
    };

    let config = RTCConfiguration {
        ice_servers,
        ice_transport_policy,
        ..Default::default()
    };

    let peer_connection = match state.api.new_peer_connection(config).await {
        Ok(pc) => Arc::new(pc),
        Err(e) => {
            eprintln!("[WebRTC] Failed to create peer connection: {}", e);
            session.remove_client();
            state.client_count.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    // Get audio track for this session
    let audio_track = {
        let audio_states = state.session_audio.read().await;
        audio_states.get(&session_id).map(|s| s.audio_track.clone())
    };

    // Add audio track if available
    if let Some(track) = audio_track {
        let rtp_sender = match peer_connection
            .add_track(track as Arc<dyn TrackLocal + Send + Sync>)
            .await
        {
            Ok(sender) => sender,
            Err(e) => {
                eprintln!("[WebRTC] Failed to add audio track: {}", e);
                session.remove_client();
                state.client_count.fetch_sub(1, Ordering::SeqCst);
                return;
            }
        };

        // Read RTCP packets
        tokio::spawn(async move {
            let mut rtcp_buf = vec![0u8; 1500];
            while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
        });
    }

    // Handle ICE candidates
    let ice_tx_clone = ice_tx.clone();
    let client_id_ice = client_id.clone();
    peer_connection.on_ice_candidate(Box::new(move |candidate| {
        let ice_tx = ice_tx_clone.clone();
        let cid = client_id_ice.clone();
        Box::pin(async move {
            if let Some(candidate) = candidate {
                eprintln!(
                    "[WebRTC] {} ICE candidate: {:?}",
                    &cid[..8],
                    candidate.to_json()
                );
                if let Ok(json) = candidate.to_json() {
                    let msg = serde_json::json!({
                        "type": "ice-candidate",
                        "candidate": json
                    });
                    ice_tx.send(msg.to_string()).await.ok();
                }
            }
        })
    }));

    // Handle ICE gathering state changes
    let client_id_gather = client_id.clone();
    peer_connection.on_ice_gathering_state_change(Box::new(move |state| {
        eprintln!(
            "[WebRTC] {} ICE gathering state: {:?}",
            &client_id_gather[..8],
            state
        );
        Box::pin(async {})
    }));

    // Handle ICE connection state changes
    let client_id_ice_conn = client_id.clone();
    peer_connection.on_ice_connection_state_change(Box::new(move |state| {
        eprintln!(
            "[WebRTC] Client {} ICE connection state: {:?}",
            &client_id_ice_conn[..8],
            state
        );
        Box::pin(async {})
    }));

    // Handle connection state changes
    let client_id_clone = client_id.clone();
    peer_connection.on_peer_connection_state_change(Box::new(move |state| {
        eprintln!(
            "[WebRTC] Client {} peer connection state: {:?}",
            &client_id_clone[..8],
            state
        );
        Box::pin(async {})
    }));

    // Store client
    {
        let mut clients = state.clients.lock().await;
        clients.insert(
            client_id.clone(),
            ClientConnection {
                id: client_id.clone(),
                session_id: session_id.clone(),
                peer_connection: peer_connection.clone(),
                ice_tx,
            },
        );
    }

    // Send welcome message with session info
    let session_info = session.info().await;
    let welcome = serde_json::json!({
        "type": "welcome",
        "client_id": &client_id[..8],
        "session": session_info
    });
    if ws_sender
        .send(Message::Text(welcome.to_string()))
        .await
        .is_err()
    {
        cleanup_client(&state, &client_id, &session).await;
        return;
    }

    // Send cached last subtitle to late-joining client
    if let Some(last_subtitle) = session.get_last_subtitle() {
        let preview: String = last_subtitle.chars().take(80).collect();
        eprintln!(
            "[WebRTC] Sending cached subtitle to late-joining client: {}...",
            preview
        );
        if ws_sender
            .send(Message::Text(last_subtitle))
            .await
            .is_err()
        {
            cleanup_client(&state, &client_id, &session).await;
            return;
        }
    }

    // Create ping interval to keep WebSocket alive
    let mut ping_interval = interval(Duration::from_secs(10));

    // Main message loop
    loop {
        tokio::select! {
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = handle_client_message(
                            &text,
                            &peer_connection,
                            &mut ws_sender,
                        ).await {
                            eprintln!("[WebRTC] Error handling message: {}", e);
                        }
                    }
                    Some(Ok(Message::Pong(_))) => {
                        // Client responded to ping, connection is alive
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(e)) => {
                        eprintln!("[WebRTC] WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }

            Some(ice_msg) = ice_rx.recv() => {
                if ws_sender.send(Message::Text(ice_msg)).await.is_err() {
                    break;
                }
            }

            msg = subtitle_rx.recv() => {
                match msg {
                    Ok(json) => {
                        // Truncate safely for UTF-8 strings
                        let preview: String = json.chars().take(80).collect();
                        eprintln!("[WebRTC] Sending subtitle to client: {}...", preview);
                        if ws_sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("[WebRTC] Lagged {} subtitle messages", n);
                    }
                    Err(_) => break,
                }
            }

            msg = status_rx.recv() => {
                match msg {
                    Ok(json) => {
                        if ws_sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }

            _ = ping_interval.tick() => {
                // Send ping to keep connection alive
                if ws_sender.send(Message::Ping(vec![])).await.is_err() {
                    eprintln!("[WebRTC] Failed to send ping, connection closed");
                    break;
                }
            }
        }
    }

    cleanup_client(&state, &client_id, &session).await;
}

/// Handle incoming WebSocket messages (signaling)
async fn handle_client_message(
    text: &str,
    peer_connection: &Arc<RTCPeerConnection>,
    ws_sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: serde_json::Value = serde_json::from_str(text)?;

    match msg["type"].as_str() {
        Some("ready") => {
            let offer = peer_connection.create_offer(None).await?;
            peer_connection.set_local_description(offer.clone()).await?;

            // Log the SDP to see negotiated codec details
            if let Some(audio_line) = offer.sdp.lines().find(|l| l.starts_with("a=rtpmap:") && l.contains("opus")) {
                eprintln!("[WebRTC] SDP audio codec: {}", audio_line);
            }

            let offer_msg = serde_json::json!({
                "type": "offer",
                "sdp": offer.sdp
            });
            ws_sender
                .send(Message::Text(offer_msg.to_string()))
                .await?;
            eprintln!("[WebRTC] Sent offer to client");
        }
        Some("answer") => {
            if let Some(sdp) = msg["sdp"].as_str() {
                let answer = RTCSessionDescription::answer(sdp.to_owned())?;
                peer_connection.set_remote_description(answer).await?;
                eprintln!("[WebRTC] Set remote description (answer)");
            }
        }
        Some("ice-candidate") => {
            if let Some(candidate) = msg.get("candidate") {
                let ice_candidate: RTCIceCandidateInit = serde_json::from_value(candidate.clone())?;
                peer_connection.add_ice_candidate(ice_candidate).await?;
            }
        }
        _ => {}
    }

    Ok(())
}

/// Clean up client connection
async fn cleanup_client(
    state: &Arc<AppState>,
    client_id: &str,
    session: &Arc<TranscriptionSession>,
) {
    let mut clients = state.clients.lock().await;
    if let Some(client) = clients.remove(client_id) {
        client.peer_connection.close().await.ok();
    }
    session.remove_client();
    let count = state.client_count.fetch_sub(1, Ordering::SeqCst) - 1;
    eprintln!(
        "[WebRTC] Client {} disconnected (total: {})",
        &client_id[..8],
        count
    );
}
