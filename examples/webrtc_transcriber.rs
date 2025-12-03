/*
WebRTC Real-Time Transcription Server with Speaker Diarization

Uses WebRTC for ultra-low-latency audio streaming (~100-300ms latency).
Reads PCM audio from stdin, encodes to Opus, streams via WebRTC.
Subtitles are sent via WebSocket signaling channel.

Usage:
  ffmpeg -i input.wav -f s16le -ar 16000 -ac 1 - | \
    cargo run --release --example webrtc_transcriber --features sortformer -- \
      --tdt-model ./tdt

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                   webrtc_transcriber                         │
  ├─────────────────────────────────────────────────────────────┤
  │  stdin (PCM) ──► Transcriber ──► Subtitles ──► WebSocket    │
  │       │                                                      │
  │       └──► Opus Encoder ──► WebRTC Track ──► Browser        │
  └─────────────────────────────────────────────────────────────┘

WebSocket Signaling Protocol:
  Client -> Server:
    - {"type":"ready"} - Client ready to receive offer
    - {"type":"answer","sdp":"..."} - SDP answer
    - {"type":"ice-candidate","candidate":{...}} - ICE candidate

  Server -> Client:
    - {"type":"offer","sdp":"..."} - SDP offer with audio track
    - {"type":"ice-candidate","candidate":{...}} - ICE candidate
    - {"type":"subtitle","text":"...","speaker":0,"start":1.0,"end":2.0}
    - {"type":"status","buffer_time":15.0,"total_duration":30.0}
*/

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use bytes::Bytes;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, Mutex};
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;
use webrtc::{
    api::{
        interceptor_registry::register_default_interceptors,
        media_engine::{MediaEngine, MIME_TYPE_OPUS},
        APIBuilder,
    },
    ice::network_type::NetworkType,
    ice_transport::{
        ice_candidate::RTCIceCandidateInit,
        ice_credential_type::RTCIceCredentialType,
        ice_server::RTCIceServer,
    },
    interceptor::registry::Registry,
    peer_connection::{
        configuration::RTCConfiguration,
        peer_connection_state::RTCPeerConnectionState,
        policy::ice_transport_policy::RTCIceTransportPolicy,
        sdp::session_description::RTCSessionDescription,
        RTCPeerConnection,
    },
    rtp_transceiver::rtp_codec::RTCRtpCodecCapability,
    track::track_local::{
        track_local_static_rtp::TrackLocalStaticRTP, TrackLocal, TrackLocalWriter,
    },
};

#[derive(Parser)]
#[command(name = "webrtc_transcriber")]
#[command(about = "WebRTC server for ultra-low-latency real-time transcription")]
struct Args {
    /// HTTP/WebSocket server port
    /// Can also be set via PORT environment variable
    #[arg(long, env = "PORT", default_value = "8080")]
    port: u16,

    /// Path to TDT model directory
    /// Can also be set via TDT_MODEL_PATH environment variable
    #[arg(long, env = "TDT_MODEL_PATH", default_value = ".")]
    tdt_model: String,

    /// Path to diarization model (ONNX)
    /// Can also be set via DIAR_MODEL_PATH environment variable
    #[arg(long, env = "DIAR_MODEL_PATH", default_value = "diar_streaming_sortformer_4spk-v2.onnx")]
    diar_model: String,

    /// Ring buffer size in seconds
    #[arg(long, default_value = "15")]
    buffer: f32,

    /// Process interval in seconds
    #[arg(long, default_value = "2")]
    interval: f32,

    /// Confirm threshold in seconds
    #[arg(long, default_value = "3")]
    confirm: f32,

    /// Use low-latency mode
    #[arg(long)]
    low_latency: bool,

    /// Use ultra-low-latency mode
    #[arg(long)]
    ultra_low_latency: bool,

    /// Use extreme-low-latency mode (fastest, may reduce accuracy)
    #[arg(long)]
    extreme_low_latency: bool,

    /// Use pause-based confirmation for better quality with low latency
    /// Confirms tokens at natural speech pauses rather than fixed time threshold
    #[arg(long)]
    pause_based: bool,

    /// Use speedy mode - optimized pause-based with lower latency
    /// Good balance of latency and quality
    /// Can also be set via SPEEDY_MODE=1 environment variable
    #[arg(long, env = "SPEEDY_MODE")]
    speedy: bool,

    /// Use lookahead mode - best quality transcription
    /// Uses sliding window of pause segments for future context
    /// Each segment is transcribed with knowledge of subsequent segments
    /// Can also be set via LOOKAHEAD_MODE=1 environment variable
    #[arg(long, env = "LOOKAHEAD_MODE")]
    lookahead: bool,

    /// Number of segments to look ahead (default: 2)
    /// Only used in lookahead mode
    #[arg(long, default_value = "2")]
    lookahead_segments: usize,

    /// Path to frontend directory
    /// Can also be set via FRONTEND_PATH environment variable
    #[arg(long, env = "FRONTEND_PATH", default_value = "./frontend")]
    frontend: PathBuf,

    /// Public IP address for WebRTC ICE candidates (for WSL2/Docker)
    /// If not specified, tries to auto-detect the host IP
    /// Can also be set via PUBLIC_IP environment variable
    #[arg(long, env = "PUBLIC_IP")]
    public_ip: Option<String>,
}

/// Client connection with WebRTC peer connection
struct ClientConnection {
    id: String,
    peer_connection: Arc<RTCPeerConnection>,
    ice_tx: mpsc::Sender<String>,
}

/// Runtime configuration for frontend
#[derive(Clone)]
struct RuntimeConfig {
    /// WebSocket URL for signaling
    ws_url: String,
    /// TURN server URL
    turn_server: String,
    /// TURN username
    turn_username: String,
    /// TURN password
    turn_password: String,
}

/// Shared application state
struct AppState {
    /// Subtitle broadcast channel
    subtitle_tx: broadcast::Sender<String>,
    /// Status broadcast channel
    status_tx: broadcast::Sender<String>,
    /// Audio track for broadcasting
    audio_track: Arc<TrackLocalStaticRTP>,
    /// WebRTC API
    api: webrtc::api::API,
    /// Connected clients
    clients: Mutex<HashMap<String, ClientConnection>>,
    /// Client count
    client_count: AtomicU64,
    /// Runtime configuration for frontend
    config: RuntimeConfig,
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("Error: This example requires the 'sortformer' feature.");
    eprintln!("Run with: cargo run --release --example webrtc_transcriber --features sortformer");
    std::process::exit(1);
}

#[cfg(feature = "sortformer")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Determine latency mode
    // lookahead: best quality using sliding window of pause segments (includes pause-based)
    // speedy: pause-based with faster settings, best balance of latency and quality
    // pause-based: uses silence detection for confirmation, good quality with low latency
    // extreme-low-latency: 5s buffer, 0.5s interval, 0.8s confirm = ~1.3s expected latency
    // ultra-low-latency: 8s buffer, 1.0s interval, 1.5s confirm = ~2.5s expected latency
    // low-latency: 10s buffer, 1.5s interval, 2.0s confirm = ~3.5s expected latency
    let (buffer, interval, confirm, pause_based, lookahead_mode, lookahead_segs, mode_name) = if args.lookahead {
        // Lookahead mode: best quality with future context
        (20.0, 0.3, 1.5, true, true, args.lookahead_segments, "lookahead")
    } else if args.speedy {
        // Speedy mode: faster pause-based with lower latency
        (8.0, 0.2, 0.4, true, false, 2, "speedy")
    } else if args.pause_based {
        // Pause-based mode: use silence detection for smarter confirmation
        (10.0, 0.3, 0.5, true, false, 2, "pause-based")
    } else if args.extreme_low_latency {
        (5.0, 0.5, 0.8, false, false, 2, "extreme-low-latency")
    } else if args.ultra_low_latency {
        (8.0, 1.0, 1.5, false, false, 2, "ultra-low-latency")
    } else if args.low_latency {
        (10.0, 1.5, 2.0, false, false, 2, "low-latency")
    } else {
        (args.buffer, args.interval, args.confirm, false, false, 2, "default")
    };

    eprintln!("===========================================");
    eprintln!("  WebRTC Real-Time Transcription Server");
    eprintln!("===========================================");
    eprintln!("Port: {}", args.port);
    eprintln!("TDT Model: {}", args.tdt_model);
    eprintln!("Diarization Model: {}", args.diar_model);
    eprintln!(
        "Mode: {} ({:.1}s buffer, {:.1}s interval, {:.1}s confirm{})",
        mode_name, buffer, interval, confirm,
        if pause_based { ", pause-based" } else { "" }
    );
    if lookahead_mode {
        eprintln!("Lookahead: enabled ({} segments ahead)", lookahead_segs);
        eprintln!("Confirmation: sliding window with future context");
        eprintln!("Expected transcription latency: ~1-3s (depends on speech pauses, best quality)");
    } else if pause_based {
        eprintln!("Confirmation: pause-based (better quality with natural pauses)");
        eprintln!("Expected transcription latency: ~0.3-1.5s (depends on speech pauses)");
    } else {
        eprintln!("Expected transcription latency: ~{:.1}s", confirm + interval);
    }
    eprintln!("Expected audio latency: ~100-300ms (WebRTC)");
    eprintln!("Frontend: {}", args.frontend.display());
    eprintln!("===========================================");
    eprintln!();

    // Create WebRTC API with Opus codec
    let mut media_engine = MediaEngine::default();
    media_engine.register_default_codecs()?;

    let mut registry = Registry::new();
    registry = register_default_interceptors(registry, &mut media_engine)?;

    // Detect host IP for WSL2/Docker environments
    // In WSL2, we need to use the actual WSL2 IP that Windows can reach
    let detected_ip = std::process::Command::new("hostname")
        .arg("-I")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .and_then(|s| s.split_whitespace().next().map(String::from))
        });

    // Determine which IP to use for NAT 1-to-1 mapping
    // Priority: --public-ip > auto-detected WSL2 IP > fallback to 127.0.0.1
    let nat_ip = args
        .public_ip
        .clone()
        .or(detected_ip.clone())
        .unwrap_or_else(|| "127.0.0.1".to_owned());

    eprintln!("WebRTC NAT 1:1 IP: {} (detected: {:?}, custom: {:?})",
              nat_ip, detected_ip, args.public_ip);

    // Configure settings for WSL2/Docker environment
    // Use single IP for NAT 1:1 mapping (multiple IPs cause validation errors)
    let mut setting_engine = webrtc::api::setting_engine::SettingEngine::default();
    setting_engine.set_nat_1to1_ips(
        vec![nat_ip],
        webrtc::ice_transport::ice_candidate_type::RTCIceCandidateType::Host,
    );

    // Enable both UDP and TCP candidates for better connectivity
    // TCP is useful when UDP is blocked (e.g., WSL2/Docker environments)
    setting_engine.set_network_types(vec![
        NetworkType::Udp4,
        NetworkType::Udp6,
        NetworkType::Tcp4,
        NetworkType::Tcp6,
    ]);

    let api = APIBuilder::new()
        .with_media_engine(media_engine)
        .with_interceptor_registry(registry)
        .with_setting_engine(setting_engine)
        .build();

    // Create audio track for broadcasting
    let audio_track = Arc::new(TrackLocalStaticRTP::new(
        RTCRtpCodecCapability {
            mime_type: MIME_TYPE_OPUS.to_owned(),
            clock_rate: 48000, // Opus uses 48kHz
            channels: 1,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".to_owned(),
            rtcp_feedback: vec![],
        },
        "audio".to_owned(),
        "webrtc-transcriber".to_owned(),
    ));

    // Create broadcast channels
    let (subtitle_tx, _) = broadcast::channel::<String>(1000);
    let (status_tx, _) = broadcast::channel::<String>(100);

    // Read TURN configuration from environment
    let turn_server = std::env::var("TURN_SERVER").unwrap_or_default();
    let turn_username = std::env::var("TURN_USERNAME").unwrap_or_default();
    let turn_password = std::env::var("TURN_PASSWORD").unwrap_or_default();

    // Build WebSocket URL for browser connection
    // Priority: WS_HOST env var > PUBLIC_IP > detected IP > localhost
    let ws_host = std::env::var("WS_HOST")
        .ok()
        .or_else(|| args.public_ip.clone())
        .or_else(|| detected_ip.clone())
        .unwrap_or_else(|| "localhost".to_owned());

    let ws_url = format!("ws://{}:{}/ws", ws_host, args.port);

    let runtime_config = RuntimeConfig {
        ws_url: ws_url.clone(),
        turn_server: turn_server.clone(),
        turn_username: turn_username.clone(),
        turn_password: turn_password.clone(),
    };

    eprintln!("Frontend config: ws_url={}, turn_server={}", ws_url, if turn_server.is_empty() { "(none)" } else { &turn_server });

    let state = Arc::new(AppState {
        subtitle_tx: subtitle_tx.clone(),
        status_tx: status_tx.clone(),
        audio_track: audio_track.clone(),
        api,
        clients: Mutex::new(HashMap::new()),
        client_count: AtomicU64::new(0),
        config: runtime_config,
    });

    // Running flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));

    // Spawn transcription and audio encoding thread
    let sub_tx = subtitle_tx.clone();
    let stat_tx = status_tx.clone();
    let audio_track_clone = audio_track.clone();
    let run = running.clone();
    let tdt_model = args.tdt_model.clone();
    let diar_model = args.diar_model.clone();
    let is_extreme_mode = args.extreme_low_latency;

    let is_pause_based = pause_based;
    let is_lookahead = lookahead_mode;
    let lookahead_count = lookahead_segs;
    std::thread::spawn(move || {
        run_transcription_and_audio(
            &tdt_model,
            &diar_model,
            buffer,
            interval,
            confirm,
            is_pause_based,
            is_lookahead,
            lookahead_count,
            sub_tx,
            stat_tx,
            audio_track_clone,
            run,
            is_extreme_mode,
        )
    });

    // Build router
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "OK" }))
        .route("/api/config", get(config_handler))
        .fallback_service(ServeDir::new(&args.frontend))
        .layer(CorsLayer::permissive())
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    eprintln!("Server listening on http://{}", addr);
    eprintln!("WebSocket signaling: ws://{}/ws", addr);
    eprintln!("Frontend: http://{}", addr);
    eprintln!();
    eprintln!("Waiting for stdin audio (pipe ffmpeg output here)...");
    eprintln!();

    // Graceful shutdown
    let run_shutdown = running.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\nShutting down...");
        run_shutdown.store(false, Ordering::SeqCst);
    });

    axum::serve(listener, app).await?;

    Ok(())
}

/// WebSocket upgrade handler
#[cfg(feature = "sortformer")]
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Configuration endpoint for frontend
/// Returns runtime configuration as JSON
#[cfg(feature = "sortformer")]
async fn config_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let config = &state.config;

    // Build ICE servers array
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
        "audio": {
            "sampleRate": 16000,
            "channels": 1,
            "bufferSize": 4096
        },
        "subtitles": {
            "maxSegments": 1000,
            "autoScroll": true,
            "showTimestamps": true
        },
        "speakerColors": [
            "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
            "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98"
        ],
        "reconnect": {
            "enabled": true,
            "delay": 2000,
            "maxDelay": 30000,
            "maxAttempts": 10
        }
    });

    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        response.to_string()
    )
}

/// Handle WebSocket connection for signaling
#[cfg(feature = "sortformer")]
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let client_id = Uuid::new_v4().to_string();
    let count = state.client_count.fetch_add(1, Ordering::SeqCst) + 1;
    eprintln!("[WebRTC] Client {} connected (total: {})", &client_id[..8], count);

    let (mut ws_sender, mut ws_receiver) = socket.split();

    // Channel for sending ICE candidates to client
    let (ice_tx, mut ice_rx) = mpsc::channel::<String>(100);

    // Subscribe to broadcasts
    let mut subtitle_rx = state.subtitle_tx.subscribe();
    let mut status_rx = state.status_tx.subscribe();

    // Create peer connection with TURN server support
    let turn_server = std::env::var("TURN_SERVER").unwrap_or_default();
    let turn_username = std::env::var("TURN_USERNAME").unwrap_or_default();
    let turn_password = std::env::var("TURN_PASSWORD").unwrap_or_default();

    let mut ice_servers = vec![RTCIceServer {
        urls: vec!["stun:stun.l.google.com:19302".to_owned()],
        ..Default::default()
    }];

    // Check if TURN server is configured before moving the value
    let has_turn_server = !turn_server.is_empty();

    // Add TURN server if configured
    if has_turn_server {
        eprintln!("[WebRTC] Using TURN server: {}", turn_server);
        ice_servers.push(RTCIceServer {
            urls: vec![turn_server],
            username: turn_username,
            credential: turn_password,
            credential_type: RTCIceCredentialType::Password,
        });
    }

    // Use All mode to allow direct connections with the public IP
    // Note: webrtc-rs may not fully support TURNS relay-only mode
    let ice_transport_policy = RTCIceTransportPolicy::All;
    if has_turn_server {
        eprintln!("[WebRTC] TURN server configured, using All ICE transport policy");
    }

    let config = RTCConfiguration {
        ice_servers,
        ice_transport_policy,
        ..Default::default()
    };

    let peer_connection = match state.api.new_peer_connection(config).await {
        Ok(pc) => Arc::new(pc),
        Err(e) => {
            eprintln!("[WebRTC] Failed to create peer connection: {}", e);
            state.client_count.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    // Add audio track to peer connection
    let rtp_sender = match peer_connection
        .add_track(state.audio_track.clone() as Arc<dyn TrackLocal + Send + Sync>)
        .await
    {
        Ok(sender) => sender,
        Err(e) => {
            eprintln!("[WebRTC] Failed to add audio track: {}", e);
            state.client_count.fetch_sub(1, Ordering::SeqCst);
            return;
        }
    };

    // Read RTCP packets (required for WebRTC)
    tokio::spawn(async move {
        let mut rtcp_buf = vec![0u8; 1500];
        while let Ok((_, _)) = rtp_sender.read(&mut rtcp_buf).await {}
    });

    // Handle ICE candidates
    let ice_tx_clone = ice_tx.clone();
    let client_id_ice = client_id.clone();
    peer_connection.on_ice_candidate(Box::new(move |candidate| {
        let ice_tx = ice_tx_clone.clone();
        let cid = client_id_ice.clone();
        Box::pin(async move {
            if let Some(candidate) = candidate {
                eprintln!("[WebRTC] {} ICE candidate: {:?}", &cid[..8], candidate.to_json());
                if let Ok(json) = candidate.to_json() {
                    let msg = serde_json::json!({
                        "type": "ice-candidate",
                        "candidate": json
                    });
                    if let Err(e) = ice_tx.send(msg.to_string()).await {
                        eprintln!("[WebRTC] Failed to send ICE candidate: {}", e);
                    }
                }
            } else {
                eprintln!("[WebRTC] {} ICE gathering complete", &cid[..8]);
            }
        })
    }));

    // Handle connection state changes
    let client_id_clone = client_id.clone();
    peer_connection.on_peer_connection_state_change(Box::new(move |state| {
        eprintln!("[WebRTC] Client {} state: {:?}", &client_id_clone[..8], state);
        Box::pin(async {})
    }));

    // Store client
    {
        let mut clients = state.clients.lock().await;
        clients.insert(
            client_id.clone(),
            ClientConnection {
                id: client_id.clone(),
                peer_connection: peer_connection.clone(),
                ice_tx,
            },
        );
    }

    // Send welcome message
    let welcome = serde_json::json!({
        "type": "welcome",
        "message": "Connected to WebRTC transcription server",
        "client_id": &client_id[..8]
    });
    if ws_sender.send(Message::Text(welcome.to_string())).await.is_err() {
        cleanup_client(&state, &client_id).await;
        return;
    }

    // Main message loop
    loop {
        tokio::select! {
            // Handle client messages
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
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Err(_)) => break,
                    _ => {}
                }
            }

            // Forward ICE candidates
            Some(ice_msg) = ice_rx.recv() => {
                if ws_sender.send(Message::Text(ice_msg)).await.is_err() {
                    break;
                }
            }

            // Forward subtitles
            msg = subtitle_rx.recv() => {
                match msg {
                    Ok(json) => {
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

            // Forward status
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
        }
    }

    cleanup_client(&state, &client_id).await;
}

/// Handle incoming client signaling messages
#[cfg(feature = "sortformer")]
async fn handle_client_message(
    text: &str,
    peer_connection: &Arc<RTCPeerConnection>,
    ws_sender: &mut futures_util::stream::SplitSink<WebSocket, Message>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: serde_json::Value = serde_json::from_str(text)?;

    match msg["type"].as_str() {
        Some("ready") => {
            // Client is ready, create and send offer
            let offer = peer_connection.create_offer(None).await?;
            peer_connection.set_local_description(offer.clone()).await?;

            let offer_msg = serde_json::json!({
                "type": "offer",
                "sdp": offer.sdp
            });
            ws_sender.send(Message::Text(offer_msg.to_string())).await?;
            eprintln!("[WebRTC] Sent offer to client");
        }
        Some("answer") => {
            // Client sent answer
            if let Some(sdp) = msg["sdp"].as_str() {
                let answer = RTCSessionDescription::answer(sdp.to_owned())?;
                peer_connection.set_remote_description(answer).await?;
                eprintln!("[WebRTC] Set remote description (answer)");
            }
        }
        Some("ice-candidate") => {
            // Client sent ICE candidate
            if let Some(candidate) = msg.get("candidate") {
                eprintln!("[WebRTC] Received ICE candidate from client: {:?}", candidate);
                let ice_candidate: RTCIceCandidateInit = serde_json::from_value(candidate.clone())?;
                peer_connection.add_ice_candidate(ice_candidate).await?;
                eprintln!("[WebRTC] Added client ICE candidate successfully");
            }
        }
        _ => {}
    }

    Ok(())
}

/// Cleanup client on disconnect
#[cfg(feature = "sortformer")]
async fn cleanup_client(state: &Arc<AppState>, client_id: &str) {
    let mut clients = state.clients.lock().await;
    if let Some(client) = clients.remove(client_id) {
        client.peer_connection.close().await.ok();
    }
    let count = state.client_count.fetch_sub(1, Ordering::SeqCst) - 1;
    eprintln!("[WebRTC] Client {} disconnected (total: {})", &client_id[..8], count);
}

/// Opus encoder wrapper
#[cfg(feature = "sortformer")]
struct OpusEncoder {
    encoder: audiopus::coder::Encoder,
    /// Resampling buffer (16kHz -> 48kHz)
    resample_buffer: Vec<i16>,
    /// Frame size in samples at 48kHz (20ms = 960 samples)
    frame_size: usize,
    /// Output buffer
    output_buffer: Vec<u8>,
    /// RTP sequence number
    sequence: u16,
    /// RTP timestamp
    timestamp: u32,
    /// SSRC
    ssrc: u32,
}

#[cfg(feature = "sortformer")]
impl OpusEncoder {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = audiopus::coder::Encoder::new(
            audiopus::SampleRate::Hz48000,
            audiopus::Channels::Mono,
            audiopus::Application::Voip, // Low latency
        )?;

        Ok(Self {
            encoder,
            resample_buffer: Vec::with_capacity(960 * 2), // 20ms at 48kHz
            frame_size: 960, // 20ms at 48kHz
            output_buffer: vec![0u8; 1500],
            sequence: 0,
            timestamp: 0,
            ssrc: rand::random(),
        })
    }

    /// Encode PCM samples (16kHz f32) to Opus RTP packets
    /// Returns RTP packets ready to send
    fn encode(&mut self, samples_16k: &[f32]) -> Vec<Bytes> {
        let mut packets = Vec::new();

        // Convert f32 to i16 and resample 16kHz -> 48kHz (3x)
        for sample in samples_16k {
            let s16 = (*sample * 32767.0) as i16;
            // Simple 3x upsampling (linear interpolation would be better)
            self.resample_buffer.push(s16);
            self.resample_buffer.push(s16);
            self.resample_buffer.push(s16);
        }

        // Encode complete frames (20ms = 960 samples at 48kHz)
        while self.resample_buffer.len() >= self.frame_size {
            let frame: Vec<i16> = self.resample_buffer.drain(..self.frame_size).collect();

            // Encode to Opus
            match self.encoder.encode(&frame, &mut self.output_buffer[12..]) {
                Ok(len) => {
                    if len > 0 {
                        // Build RTP packet
                        let packet = self.build_rtp_packet(len);
                        packets.push(packet);
                    }
                }
                Err(e) => {
                    eprintln!("[Opus] Encode error: {}", e);
                }
            }
        }

        packets
    }

    fn build_rtp_packet(&mut self, payload_len: usize) -> Bytes {
        let mut packet = vec![0u8; 12 + payload_len];

        // RTP header (12 bytes)
        packet[0] = 0x80; // Version 2, no padding, no extension, no CSRC
        packet[1] = 111;  // Payload type for Opus

        // Sequence number
        packet[2] = (self.sequence >> 8) as u8;
        packet[3] = self.sequence as u8;
        self.sequence = self.sequence.wrapping_add(1);

        // Timestamp
        packet[4] = (self.timestamp >> 24) as u8;
        packet[5] = (self.timestamp >> 16) as u8;
        packet[6] = (self.timestamp >> 8) as u8;
        packet[7] = self.timestamp as u8;
        self.timestamp = self.timestamp.wrapping_add(self.frame_size as u32);

        // SSRC
        packet[8] = (self.ssrc >> 24) as u8;
        packet[9] = (self.ssrc >> 16) as u8;
        packet[10] = (self.ssrc >> 8) as u8;
        packet[11] = self.ssrc as u8;

        // Copy payload
        packet[12..12 + payload_len].copy_from_slice(&self.output_buffer[12..12 + payload_len]);

        Bytes::from(packet)
    }
}

#[cfg(feature = "sortformer")]
fn run_transcription_and_audio(
    tdt_model: &str,
    diar_model: &str,
    buffer_secs: f32,
    interval_secs: f32,
    confirm_secs: f32,
    pause_based_mode: bool,
    lookahead_mode: bool,
    lookahead_segments: usize,
    subtitle_tx: broadcast::Sender<String>,
    status_tx: broadcast::Sender<String>,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    extreme_mode: bool,
) {
    use parakeet_rs::{ExecutionConfig, RealtimeTDTConfig, RealtimeTDTDiarized};
    use std::sync::mpsc as std_mpsc;
    use std::time::{Duration, Instant};

    // Initialize Opus encoder
    let mut opus_encoder = match OpusEncoder::new() {
        Ok(enc) => enc,
        Err(e) => {
            eprintln!("[Opus] Failed to create encoder: {}", e);
            let error_msg = serde_json::json!({
                "type": "error",
                "message": format!("Failed to initialize Opus encoder: {}", e)
            });
            subtitle_tx.send(error_msg.to_string()).ok();
            return;
        }
    };

    // Initialize transcriber
    let config = RealtimeTDTConfig {
        buffer_size_secs: buffer_secs,
        process_interval_secs: interval_secs,
        confirm_threshold_secs: confirm_secs,
        pause_based_confirm: pause_based_mode,
        pause_threshold_secs: if lookahead_mode { 0.4 } else if pause_based_mode { 0.35 } else { 0.4 },
        silence_energy_threshold: if lookahead_mode { 0.01 } else if pause_based_mode { 0.008 } else { 0.01 },
        lookahead_mode,
        lookahead_segments,
    };

    // Use ExecutionConfig::from_env() for GPU detection via USE_GPU env var
    let exec_config = ExecutionConfig::from_env();

    let transcriber = match RealtimeTDTDiarized::new(tdt_model, diar_model, Some(exec_config), Some(config)) {
        Ok(t) => t,
        Err(e) => {
            let error_msg = serde_json::json!({
                "type": "error",
                "message": format!("Failed to initialize transcriber: {}", e)
            });
            subtitle_tx.send(error_msg.to_string()).ok();
            return;
        }
    };

    // In extreme mode, reduce batch size for faster response at cost of more overhead
    let batch_mode_label = if extreme_mode { "extreme (50ms batches)" } else { "normal (100ms batches)" };
    eprintln!("[Transcriber] Initialized with Opus encoder (separate threads, {} mode)", batch_mode_label);

    // Create channel to send audio samples to transcription thread
    // Use unbounded channel - we never want to drop audio samples for transcription
    let (audio_tx, audio_rx) = std_mpsc::channel::<Vec<f32>>();

    // Spawn transcription thread (non-blocking for audio streaming)
    let sub_tx = subtitle_tx.clone();
    let stat_tx = status_tx.clone();
    let run_transcriber = running.clone();
    let transcriber_handle = std::thread::spawn(move || {
        run_transcription_thread(transcriber, audio_rx, sub_tx, stat_tx, run_transcriber);
    });

    let mut stdin = io::stdin().lock();
    let sample_rate = 16000;
    // Read 20ms chunks to match Opus frame size (320 samples at 16kHz)
    let chunk_samples = 320;
    let bytes_per_chunk = chunk_samples * 2;

    let mut byte_buffer = vec![0u8; bytes_per_chunk];
    let mut total_samples: usize = 0;

    // Buffer for batching samples to transcription
    // In extreme mode: 50ms = 800 samples (faster but more overhead)
    // In normal mode: 100ms = 1600 samples (balanced)
    let transcription_batch_size = if extreme_mode { 800 } else { 1600 };
    let mut transcription_buffer: Vec<f32> = Vec::with_capacity(transcription_batch_size);

    // RTP pacing: track when streaming started and count packets sent
    // Each Opus frame is 20ms, so we need to send packets at 20ms intervals
    let stream_start_time = Instant::now();
    let packet_duration = Duration::from_millis(20);
    let mut packets_sent: u64 = 0;

    // Create tokio runtime for async audio track writing
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    while running.load(Ordering::SeqCst) {
        // Read PCM s16le from stdin
        match stdin.read_exact(&mut byte_buffer) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                eprintln!("[Audio] End of stdin input");
                break;
            }
            Err(e) => {
                eprintln!("[Audio] Error reading stdin: {}", e);
                break;
            }
        }

        // Convert bytes to f32 samples
        let samples: Vec<f32> = byte_buffer
            .chunks(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        total_samples += samples.len();

        // Buffer samples for transcription (batch to reduce overhead)
        transcription_buffer.extend_from_slice(&samples);
        if transcription_buffer.len() >= transcription_batch_size {
            // Send batch to transcription thread (never drop)
            let batch = std::mem::replace(&mut transcription_buffer, Vec::with_capacity(transcription_batch_size));
            if audio_tx.send(batch).is_err() {
                eprintln!("[Audio] Transcription thread disconnected");
                break;
            }
        }

        // Encode to Opus and send via WebRTC with proper pacing
        let rtp_packets = opus_encoder.encode(&samples);
        for packet in &rtp_packets {
            // Calculate when this packet should be sent based on packet count
            let target_send_time = stream_start_time + packet_duration * (packets_sent as u32);
            let now = Instant::now();

            // If we're ahead of schedule, sleep until the target time
            if now < target_send_time {
                std::thread::sleep(target_send_time - now);
            }

            let track = audio_track.clone();
            let pkt = packet.clone();
            rt.block_on(async {
                match track.write(&pkt).await {
                    Ok(n) => {
                        // Log occasionally (every ~1 second)
                        if total_samples % (sample_rate * 1) < chunk_samples {
                            eprintln!("[RTP] Wrote {} bytes, seq={}, pacing_drift={}ms",
                                     n, opus_encoder.sequence,
                                     (Instant::now() - target_send_time).as_millis() as i64);
                        }
                    }
                    Err(e) => {
                        // Ignore write errors when no clients connected
                        let err_str = e.to_string();
                        if !err_str.contains("no receiver") && !err_str.contains("ErrClosedPipe") {
                            eprintln!("[WebRTC] Write error: {}", e);
                        }
                    }
                }
            });

            packets_sent += 1;
        }
    }

    // Send any remaining samples
    if !transcription_buffer.is_empty() {
        audio_tx.send(transcription_buffer).ok();
    }

    // Signal end of audio and wait for transcription to finish
    drop(audio_tx);
    eprintln!("[Audio] Waiting for transcription to finalize...");
    transcriber_handle.join().ok();

    let end_msg = serde_json::json!({
        "type": "end",
        "total_duration": total_samples as f32 / sample_rate as f32
    });
    subtitle_tx.send(end_msg.to_string()).ok();

    eprintln!(
        "[Audio] Complete. Total duration: {:.2}s",
        total_samples as f32 / sample_rate as f32
    );
}

/// Separate thread for transcription to avoid blocking audio streaming
#[cfg(feature = "sortformer")]
fn run_transcription_thread(
    mut transcriber: parakeet_rs::RealtimeTDTDiarized,
    audio_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    subtitle_tx: broadcast::Sender<String>,
    status_tx: broadcast::Sender<String>,
    running: Arc<AtomicBool>,
) {
    use std::time::Instant;

    let sample_rate = 16000;
    let mut total_samples: usize = 0;
    let mut last_status_time = Instant::now();

    // Process audio samples as they arrive
    while running.load(Ordering::SeqCst) {
        // First, wait for at least one batch
        let first_batch = match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(batch) => batch,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };

        // Collect the first batch
        let mut all_samples = first_batch;

        // Drain any additional pending batches to reduce latency
        while let Ok(batch) = audio_rx.try_recv() {
            all_samples.extend(batch);
        }

        total_samples += all_samples.len();
        let current_time = total_samples as f32 / sample_rate as f32;

        // Process all collected samples through transcriber
        match transcriber.push_audio(&all_samples) {
            Ok(result) => {
                for segment in &result.segments {
                    let subtitle_msg = serde_json::json!({
                        "type": "subtitle",
                        "text": segment.text,
                        "speaker": segment.speaker,
                        "start": segment.start_time,
                        "end": segment.end_time,
                        "is_final": segment.is_final
                    });
                    subtitle_tx.send(subtitle_msg.to_string()).ok();

                    eprintln!(
                        "[Speaker {}] {} [{:.2}s-{:.2}s]",
                        segment.speaker.map(|s| s.to_string()).unwrap_or("?".to_string()),
                        segment.text,
                        segment.start_time,
                        segment.end_time
                    );
                }

                if last_status_time.elapsed().as_secs() >= 1 {
                    let status_msg = serde_json::json!({
                        "type": "status",
                        "buffer_time": result.buffer_time,
                        "total_duration": current_time
                    });
                    status_tx.send(status_msg.to_string()).ok();
                    last_status_time = Instant::now();
                }
            }
            Err(e) => {
                eprintln!("[Transcriber] Error: {}", e);
            }
        }
    }

    // Drain any remaining batches before finalizing
    while let Ok(batch) = audio_rx.try_recv() {
        total_samples += batch.len();
        transcriber.push_audio(&batch).ok();
    }

    // Finalize transcription
    eprintln!("[Transcriber] Finalizing...");
    match transcriber.finalize() {
        Ok(result) => {
            for segment in &result.segments {
                let subtitle_msg = serde_json::json!({
                    "type": "subtitle",
                    "text": segment.text,
                    "speaker": segment.speaker,
                    "start": segment.start_time,
                    "end": segment.end_time,
                    "is_final": true
                });
                subtitle_tx.send(subtitle_msg.to_string()).ok();

                eprintln!(
                    "[Speaker {}] {} [{:.2}s-{:.2}s]",
                    segment.speaker.map(|s| s.to_string()).unwrap_or("?".to_string()),
                    segment.text,
                    segment.start_time,
                    segment.end_time
                );
            }
        }
        Err(e) => {
            eprintln!("[Transcriber] Finalization error: {}", e);
        }
    }

    eprintln!("[Transcriber] Thread complete");
}

// Random number generator for SSRC
mod rand {
    pub fn random<T: Default + From<u8>>() -> T
    where
        T: std::ops::BitOr<Output = T> + std::ops::Shl<usize, Output = T>,
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        // Simple pseudo-random from time
        T::from((nanos & 0xFF) as u8)
            | (T::from(((nanos >> 8) & 0xFF) as u8) << 8)
            | (T::from(((nanos >> 16) & 0xFF) as u8) << 16)
            | (T::from(((nanos >> 24) & 0xFF) as u8) << 24)
    }
}
