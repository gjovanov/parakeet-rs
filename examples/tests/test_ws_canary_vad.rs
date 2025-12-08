//! WebSocket test for Canary VAD transcription
//!
//! Tests the full server pipeline via WebSocket:
//! 1. Connect to server
//! 2. Configure Canary with pause-based VAD mode
//! 3. Stream audio
//! 4. Receive transcripts
//!
//! Run with: cargo run --release --example test_ws_canary_vad

use std::process::Command;
use std::time::Duration;
use tungstenite::{connect, Message};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== WebSocket Canary VAD Test ===\n");

    // Extract audio
    println!("Extracting audio...");
    let output = Command::new("ffmpeg")
        .args([
            "-i", "./media/broadcast.wav",
            "-ss", "60",
            "-t", "15",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "-"
        ])
        .output()?;

    if !output.status.success() {
        eprintln!("FFmpeg error: {}", String::from_utf8_lossy(&output.stderr));
        return Err("FFmpeg failed".into());
    }

    let audio_bytes = output.stdout;
    println!("Audio: {} bytes ({:.2}s)\n", audio_bytes.len(), audio_bytes.len() as f32 / 32000.0);

    // Connect to WebSocket
    println!("Connecting to ws://localhost:8082/ws...");
    let (mut socket, _response) = connect("ws://localhost:8082/ws")?;
    println!("Connected!\n");

    // Read initial messages
    println!("Reading initial messages...");
    for _ in 0..3 {
        if let Ok(msg) = socket.read() {
            if let Message::Text(text) = msg {
                if let Ok(data) = serde_json::from_str::<Value>(&text) {
                    println!("  Received: {}", data.get("type").unwrap_or(&json!("unknown")));
                }
            }
        }
    }

    // Send config for Canary with pause-based mode
    let config = json!({
        "type": "config",
        "config": {
            "model": "canary-1b",
            "mode": "pause_based",
            "language": "en"
        }
    });
    println!("\nSending config: {:?}", config);
    socket.send(Message::Text(config.to_string()))?;

    // Read config response
    if let Ok(Message::Text(text)) = socket.read() {
        println!("Config response: {}", text);
    }

    // Send start command
    let start_msg = json!({"type": "start"});
    println!("\nSending start...");
    socket.send(Message::Text(start_msg.to_string()))?;

    // Read start response
    if let Ok(Message::Text(text)) = socket.read() {
        println!("Start response: {}", text);
    }

    // Stream audio in chunks
    println!("\nStreaming audio...");
    let chunk_size = 3200; // 100ms at 16kHz, 16-bit
    let mut chunks_sent = 0;
    let mut transcripts: Vec<String> = Vec::new();

    // Set socket to non-blocking for reading
    socket.get_mut().set_nonblocking(true)?;

    for chunk in audio_bytes.chunks(chunk_size) {
        socket.send(Message::Binary(chunk.to_vec()))?;
        chunks_sent += 1;

        // Try to read any incoming messages
        loop {
            match socket.read() {
                Ok(Message::Text(text)) => {
                    if let Ok(data) = serde_json::from_str::<Value>(&text) {
                        if data.get("type") == Some(&json!("transcript")) {
                            if let Some(segments) = data.get("segments").and_then(|s| s.as_array()) {
                                for seg in segments {
                                    if let Some(text) = seg.get("text").and_then(|t| t.as_str()) {
                                        if !text.is_empty() {
                                            let start = seg.get("start_time").and_then(|s| s.as_f64()).unwrap_or(0.0);
                                            println!("\n  [TRANSCRIPT] {:.2}s: \"{}\"", start, text);
                                            transcripts.push(text.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(tungstenite::Error::Io(ref e)) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    break;
                }
                Err(_) => break,
                _ => {}
            }
        }

        if chunks_sent % 10 == 0 {
            print!("\r  Sent {}ms of audio...", chunks_sent * 100);
        }

        std::thread::sleep(Duration::from_millis(50));
    }

    println!("\n\nFinished streaming {} chunks", chunks_sent);

    // Set back to blocking for final messages
    socket.get_mut().set_nonblocking(false)?;
    socket.get_mut().set_read_timeout(Some(Duration::from_secs(5)))?;

    // Send stop
    let stop_msg = json!({"type": "stop"});
    println!("Sending stop...");
    socket.send(Message::Text(stop_msg.to_string()))?;

    // Collect remaining transcripts
    println!("Collecting final transcripts...");
    loop {
        match socket.read() {
            Ok(Message::Text(text)) => {
                if let Ok(data) = serde_json::from_str::<Value>(&text) {
                    if data.get("type") == Some(&json!("transcript")) {
                        if let Some(segments) = data.get("segments").and_then(|s| s.as_array()) {
                            for seg in segments {
                                if let Some(text) = seg.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        let start = seg.get("start_time").and_then(|s| s.as_f64()).unwrap_or(0.0);
                                        println!("  [FINAL] {:.2}s: \"{}\"", start, text);
                                        transcripts.push(text.to_string());
                                    }
                                }
                            }
                        }
                    } else if data.get("type") == Some(&json!("stopped")) {
                        println!("  Received stopped");
                        break;
                    }
                }
            }
            Err(_) => break,
            _ => {}
        }
    }

    println!("\n=== Results ===");
    println!("Total transcripts: {}", transcripts.len());

    if !transcripts.is_empty() {
        println!("\nFull transcript:");
        println!("{}", transcripts.join(" "));
        println!("\n✓ SUCCESS: Canary VAD WebSocket transcription working!");
        Ok(())
    } else {
        println!("\n✗ FAILURE: No transcripts received!");
        Err("No transcripts".into())
    }
}
