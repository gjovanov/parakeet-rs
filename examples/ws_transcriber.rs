/*
High-Performance WebSocket Transcription Server with Speaker Diarization

Uses axum (tokio team's web framework) for optimal WebSocket performance.
Reads PCM audio from stdin (piped from ffmpeg), processes through RealtimeTDTDiarized,
and broadcasts audio chunks + subtitle segments to all connected WebSocket clients.

Also serves the frontend static files for convenience.

Usage:
  # From audio file (IMPORTANT: -re flag for real-time pacing!)
  ffmpeg -re -i input.wav -f s16le -ar 16000 -ac 1 - | \
    cargo run --release --example ws_transcriber --features sortformer -- \
      --tdt-model ./tdt --broadcast-audio

  # Low latency mode (~3.5s transcription delay)
  ffmpeg -re -i input.wav -f s16le -ar 16000 -ac 1 - | \
    cargo run --release --example ws_transcriber --features sortformer -- \
      --tdt-model ./tdt --low-latency --broadcast-audio

  # Ultra low latency mode (~2.5s delay, may reduce quality)
  ffmpeg -re -i input.wav -f s16le -ar 16000 -ac 1 - | \
    cargo run --release --example ws_transcriber --features sortformer -- \
      --tdt-model ./tdt --ultra-low-latency --broadcast-audio

  # From microphone (Linux/PulseAudio) - no -re needed, already real-time
  ffmpeg -f pulse -i default -f s16le -ar 16000 -ac 1 - | \
    cargo run --release --example ws_transcriber --features sortformer -- --tdt-model ./tdt

  Note: The -re flag is CRITICAL when streaming from files. It makes ffmpeg read
  at the native frame rate instead of as fast as possible. Without it, the WebSocket
  will be overwhelmed with data causing buffer issues in the client.

WebSocket Protocol:
  Server -> Client messages (JSON):
    - {"type":"audio","data":"<base64 PCM>","timestamp":1.23}
    - {"type":"subtitle","text":"Hello","speaker":0,"start":1.0,"end":2.0,"is_final":true}
    - {"type":"status","buffer_time":15.0,"total_duration":30.0}
    - {"type":"error","message":"..."}

Frontend: http://localhost:8080 (serves ./frontend/)
WebSocket: ws://localhost:8080/ws
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
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use std::io::{self, Read};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::{cors::CorsLayer, services::ServeDir};

#[derive(Parser)]
#[command(name = "ws_transcriber")]
#[command(about = "High-performance WebSocket server for real-time transcription")]
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

    /// Use low-latency mode (~3.5s latency instead of ~5s)
    #[arg(long)]
    low_latency: bool,

    /// Use ultra-low-latency mode (~2.5s latency, may reduce quality)
    #[arg(long)]
    ultra_low_latency: bool,

    /// Broadcast audio to clients (for playback)
    #[arg(long)]
    broadcast_audio: bool,

    /// Audio broadcast chunk size in ms (larger = less overhead, more latency)
    #[arg(long, default_value = "250")]
    audio_chunk_ms: u64,

    /// Path to frontend directory (for static file serving)
    #[arg(long, default_value = "./frontend")]
    frontend: PathBuf,
}

/// Shared application state
struct AppState {
    subtitle_tx: broadcast::Sender<String>,
    audio_tx: broadcast::Sender<String>,
    status_tx: broadcast::Sender<String>,
    client_count: AtomicU64,
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("Error: This example requires the 'sortformer' feature.");
    eprintln!("Run with: cargo run --release --example ws_transcriber --features sortformer");
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
    eprintln!("  Real-Time Transcription Server (axum)");
    eprintln!("===========================================");
    eprintln!("Port: {}", args.port);
    eprintln!("TDT Model: {}", args.tdt_model);
    eprintln!("Diarization Model: {}", args.diar_model);
    eprintln!("Mode: {} ({:.1}s buffer, {:.1}s interval, {:.1}s confirm)",
        mode_name, buffer, interval, confirm);
    eprintln!("Expected latency: ~{:.1}s", confirm + interval);
    eprintln!("Broadcast audio: {}", args.broadcast_audio);
    eprintln!("Frontend: {}", args.frontend.display());
    eprintln!("===========================================");
    eprintln!();

    // Create broadcast channels
    let (subtitle_tx, _) = broadcast::channel::<String>(1000);
    let (audio_tx, _) = broadcast::channel::<String>(1000);
    let (status_tx, _) = broadcast::channel::<String>(100);

    let state = Arc::new(AppState {
        subtitle_tx: subtitle_tx.clone(),
        audio_tx: audio_tx.clone(),
        status_tx: status_tx.clone(),
        client_count: AtomicU64::new(0),
    });

    // Running flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));

    // Spawn transcription thread (reads from stdin)
    let sub_tx = subtitle_tx.clone();
    let aud_tx = audio_tx.clone();
    let stat_tx = status_tx.clone();
    let run = running.clone();
    let broadcast_audio = args.broadcast_audio;
    let audio_chunk_ms = args.audio_chunk_ms;
    let tdt_model = args.tdt_model.clone();
    let diar_model = args.diar_model.clone();

    std::thread::spawn(move || {
        run_transcription(
            &tdt_model,
            &diar_model,
            buffer,
            interval,
            confirm,
            broadcast_audio,
            audio_chunk_ms,
            sub_tx,
            aud_tx,
            stat_tx,
            run,
        )
    });

    // Build router
    let app = Router::new()
        // WebSocket endpoint
        .route("/ws", get(ws_handler))
        // Health check
        .route("/health", get(|| async { "OK" }))
        // Serve frontend static files
        .fallback_service(ServeDir::new(&args.frontend))
        // CORS for development
        .layer(CorsLayer::permissive())
        // Shared state
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    eprintln!("Server listening on http://{}", addr);
    eprintln!("WebSocket endpoint: ws://{}/ws", addr);
    eprintln!("Frontend: http://{}", addr);
    eprintln!();
    eprintln!("Waiting for stdin audio (pipe ffmpeg output here)...");
    eprintln!();

    // Graceful shutdown on Ctrl+C
    let run_shutdown = running.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        eprintln!("\nShutting down...");
        run_shutdown.store(false, Ordering::SeqCst);
    });

    // Serve
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

/// Handle individual WebSocket connection
#[cfg(feature = "sortformer")]
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let count = state.client_count.fetch_add(1, Ordering::SeqCst) + 1;
    eprintln!("[WebSocket] Client connected (total: {})", count);

    let (mut sender, mut receiver) = socket.split();

    // Subscribe to broadcast channels
    let mut subtitle_rx = state.subtitle_tx.subscribe();
    let mut audio_rx = state.audio_tx.subscribe();
    let mut status_rx = state.status_tx.subscribe();

    // Send welcome message
    let welcome = serde_json::json!({
        "type": "welcome",
        "message": "Connected to transcription server"
    });
    if sender.send(Message::Text(welcome.to_string())).await.is_err() {
        state.client_count.fetch_sub(1, Ordering::SeqCst);
        return;
    }

    // Message forwarding loop
    loop {
        tokio::select! {
            // Forward subtitles
            msg = subtitle_rx.recv() => {
                match msg {
                    Ok(json) => {
                        if sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("[WebSocket] Lagged {} subtitle messages", n);
                    }
                    Err(_) => break,
                }
            }

            // Forward audio
            msg = audio_rx.recv() => {
                match msg {
                    Ok(json) => {
                        if sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("[WebSocket] Lagged {} audio messages", n);
                    }
                    Err(_) => break,
                }
            }

            // Forward status
            msg = status_rx.recv() => {
                match msg {
                    Ok(json) => {
                        if sender.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(_) => break,
                }
            }

            // Handle client messages (ping/pong, close)
            msg = receiver.next() => {
                match msg {
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(Message::Ping(data))) => {
                        if sender.send(Message::Pong(data)).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }

    let count = state.client_count.fetch_sub(1, Ordering::SeqCst) - 1;
    eprintln!("[WebSocket] Client disconnected (total: {})", count);
}

/// Audio chunk with timing info for paced delivery
struct TimedAudioChunk {
    data: Vec<u8>,
    timestamp: f32,
    target_send_time: std::time::Instant,
}

#[cfg(feature = "sortformer")]
fn run_transcription(
    tdt_model: &str,
    diar_model: &str,
    buffer_secs: f32,
    interval_secs: f32,
    confirm_secs: f32,
    broadcast_audio: bool,
    audio_chunk_ms: u64,
    subtitle_tx: broadcast::Sender<String>,
    audio_tx: broadcast::Sender<String>,
    status_tx: broadcast::Sender<String>,
    running: Arc<AtomicBool>,
) {
    use base64::Engine;
    use parakeet_rs::{RealtimeTDTConfig, RealtimeTDTDiarized};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

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

    eprintln!("[Transcriber] Initialized, reading from stdin...");

    let sample_rate = 16000;
    let audio_chunk_samples = (audio_chunk_ms as usize * sample_rate / 1000) as usize;
    let bytes_per_chunk = audio_chunk_samples * 2; // 2 bytes per i16 sample
    let chunk_duration = Duration::from_millis(audio_chunk_ms);

    // Create channel for paced audio delivery
    let (audio_queue_tx, audio_queue_rx) = mpsc::sync_channel::<TimedAudioChunk>(100);

    // Spawn paced audio sender thread
    let audio_tx_clone = audio_tx.clone();
    let running_clone = running.clone();
    let audio_sender = std::thread::spawn(move || {
        while running_clone.load(Ordering::SeqCst) {
            match audio_queue_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(chunk) => {
                    // Wait until target send time
                    let now = Instant::now();
                    if chunk.target_send_time > now {
                        std::thread::sleep(chunk.target_send_time - now);
                    }

                    // Send audio
                    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&chunk.data);
                    let audio_msg = serde_json::json!({
                        "type": "audio",
                        "data": audio_b64,
                        "timestamp": chunk.timestamp
                    });
                    audio_tx_clone.send(audio_msg.to_string()).ok();
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        eprintln!("[Audio Sender] Thread exiting");
    });

    let mut stdin = io::stdin().lock();
    let mut byte_buffer = vec![0u8; bytes_per_chunk];
    let mut total_samples: usize = 0;
    let mut last_status_time = Instant::now();
    let stream_start_time = Instant::now();

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

        // Queue audio for paced delivery (if enabled)
        if broadcast_audio {
            // Calculate when this chunk should be sent based on its position in the stream
            // Add a small initial delay (500ms) to allow buffer to build
            let chunk_count = total_samples / audio_chunk_samples;
            let target_send_time = stream_start_time
                + Duration::from_millis(500)  // Initial buffer delay
                + chunk_duration * chunk_count as u32;

            let timed_chunk = TimedAudioChunk {
                data: byte_buffer.clone(),
                timestamp: current_time,
                target_send_time,
            };

            // Non-blocking send - if queue is full, we're behind and should catch up
            if audio_queue_tx.try_send(timed_chunk).is_err() {
                eprintln!("[Warning] Audio queue full, dropping chunk");
            }
        }

        // Process through transcriber
        match transcriber.push_audio(&samples) {
            Ok(result) => {
                // Broadcast new segments
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

                // Send status update every second
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
                let error_msg = serde_json::json!({
                    "type": "error",
                    "message": format!("Transcription error: {}", e)
                });
                subtitle_tx.send(error_msg.to_string()).ok();
            }
        }
    }

    // Finalize
    eprintln!("[Transcriber] Finalizing...");
    drop(audio_queue_tx); // Signal audio sender to stop
    audio_sender.join().ok();

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

    // Send end message
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
