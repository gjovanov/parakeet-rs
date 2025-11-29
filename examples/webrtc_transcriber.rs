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
    ice_transport::{
        ice_candidate::RTCIceCandidateInit,
        ice_credential_type::RTCIceCredentialType,
        ice_server::RTCIceServer,
    },
    interceptor::registry::Registry,
    peer_connection::{
        configuration::RTCConfiguration,
        peer_connection_state::RTCPeerConnectionState,
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
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Path to TDT model directory
    #[arg(long, default_value = ".")]
    tdt_model: String,

    /// Path to diarization model (ONNX)
    #[arg(long, default_value = "diar_streaming_sortformer_4spk-v2.onnx")]
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

    /// Path to frontend directory
    #[arg(long, default_value = "./frontend")]
    frontend: PathBuf,

    /// Public IP address for WebRTC ICE candidates (for WSL2/Docker)
    /// If not specified, tries to auto-detect the host IP
    #[arg(long)]
    public_ip: Option<String>,
}

/// Client connection with WebRTC peer connection
struct ClientConnection {
    id: String,
    peer_connection: Arc<RTCPeerConnection>,
    ice_tx: mpsc::Sender<String>,
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
    let (buffer, interval, confirm, mode_name) = if args.ultra_low_latency {
        (8.0, 1.0, 1.5, "ultra-low-latency")
    } else if args.low_latency {
        (10.0, 1.5, 2.0, "low-latency")
    } else {
        (args.buffer, args.interval, args.confirm, "default")
    };

    eprintln!("===========================================");
    eprintln!("  WebRTC Real-Time Transcription Server");
    eprintln!("===========================================");
    eprintln!("Port: {}", args.port);
    eprintln!("TDT Model: {}", args.tdt_model);
    eprintln!("Diarization Model: {}", args.diar_model);
    eprintln!(
        "Mode: {} ({:.1}s buffer, {:.1}s interval, {:.1}s confirm)",
        mode_name, buffer, interval, confirm
    );
    eprintln!("Expected transcription latency: ~{:.1}s", confirm + interval);
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

    let state = Arc::new(AppState {
        subtitle_tx: subtitle_tx.clone(),
        status_tx: status_tx.clone(),
        audio_track: audio_track.clone(),
        api,
        clients: Mutex::new(HashMap::new()),
        client_count: AtomicU64::new(0),
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

    std::thread::spawn(move || {
        run_transcription_and_audio(
            &tdt_model,
            &diar_model,
            buffer,
            interval,
            confirm,
            sub_tx,
            stat_tx,
            audio_track_clone,
            run,
        )
    });

    // Build router
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "OK" }))
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

    // Add TURN server if configured
    if !turn_server.is_empty() {
        eprintln!("[WebRTC] Using TURN server: {}", turn_server);
        ice_servers.push(RTCIceServer {
            urls: vec![turn_server],
            username: turn_username,
            credential: turn_password,
            credential_type: RTCIceCredentialType::Password,
        });
    }

    let config = RTCConfiguration {
        ice_servers,
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
    subtitle_tx: broadcast::Sender<String>,
    status_tx: broadcast::Sender<String>,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
) {
    use parakeet_rs::{RealtimeTDTConfig, RealtimeTDTDiarized};
    use std::time::Instant;

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
    };

    let mut transcriber = match RealtimeTDTDiarized::new(tdt_model, diar_model, None, Some(config)) {
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

    eprintln!("[Transcriber] Initialized with Opus encoder");

    let mut stdin = io::stdin().lock();
    let sample_rate = 16000;
    // Read 20ms chunks to match Opus frame size (320 samples at 16kHz)
    let chunk_samples = 320;
    let bytes_per_chunk = chunk_samples * 2;

    let mut byte_buffer = vec![0u8; bytes_per_chunk];
    let mut total_samples: usize = 0;
    let mut last_status_time = Instant::now();

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
                eprintln!("[Transcriber] End of stdin input");
                break;
            }
            Err(e) => {
                eprintln!("[Transcriber] Error reading stdin: {}", e);
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
        let current_time = total_samples as f32 / sample_rate as f32;

        // Encode to Opus and send via WebRTC
        let rtp_packets = opus_encoder.encode(&samples);
        for packet in &rtp_packets {
            let track = audio_track.clone();
            let pkt = packet.clone();
            rt.block_on(async {
                match track.write(&pkt).await {
                    Ok(n) => {
                        // Log occasionally (every ~1 second)
                        if total_samples % (sample_rate * 1) < chunk_samples {
                            eprintln!("[RTP] Wrote {} bytes, seq={}", n, opus_encoder.sequence);
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
        }

        // Process through transcriber
        match transcriber.push_audio(&samples) {
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

    let end_msg = serde_json::json!({
        "type": "end",
        "total_duration": total_samples as f32 / sample_rate as f32
    });
    subtitle_tx.send(end_msg.to_string()).ok();

    eprintln!(
        "[Transcriber] Complete. Total duration: {:.2}s",
        total_samples as f32 / sample_rate as f32
    );
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
