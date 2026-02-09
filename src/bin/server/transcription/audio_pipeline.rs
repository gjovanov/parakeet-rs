//! Audio pipeline: FFmpeg spawn, PCM reading, noise cancellation, Opus encoding, RTP sending

use super::AudioSource;
use crate::webrtc_handlers::audio::OpusEncoder;
use parakeet_rs::noise_cancellation::{create_noise_canceller, NoiseCancellationType};
use parakeet_rs::{SessionState, TranscriptionSession};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Instant;
use webrtc::track::track_local::{track_local_static_rtp::TrackLocalStaticRTP, TrackLocalWriter};

/// Spawn an FFmpeg process for the given audio source
pub fn spawn_ffmpeg(source: &AudioSource) -> Result<std::process::Child, std::io::Error> {
    use std::process::{Command, Stdio};

    match source {
        AudioSource::File(path) => {
            Command::new("ffmpeg")
                .args([
                    "-re",
                    "-i",
                    path.to_str().unwrap_or(""),
                    "-f",
                    "s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-loglevel",
                    "error",
                    "-",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
        }
        AudioSource::Srt(url) => {
            Command::new("ffmpeg")
                .args([
                    "-hide_banner",
                    "-nostats",
                    "-loglevel",
                    "error",
                    "-analyzeduration",
                    "1M",
                    "-fflags",
                    "+nobuffer+genpts+igndts+discardcorrupt",
                    "-flags",
                    "low_delay",
                    "-protocol_whitelist",
                    "file,udp,rtp,srt",
                    "-i",
                    url,
                    "-map",
                    "a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-acodec",
                    "pcm_s16le",
                    "-f",
                    "s16le",
                    "-",
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        }
    }
}

/// Run the audio pipeline: read PCM from FFmpeg, apply noise cancellation,
/// encode to Opus, send via RTP, and feed samples to the transcription channel.
pub fn run_audio_pipeline(
    session: Arc<TranscriptionSession>,
    audio_source: &AudioSource,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    ffmpeg_pid: Arc<AtomicU32>,
    audio_tx: std_mpsc::SyncSender<Vec<f32>>,
    is_srt: bool,
    duration_secs: f32,
) {
    use std::io::Read;

    // Initialize Opus encoder
    let mut opus_encoder = match OpusEncoder::new() {
        Ok(enc) => enc,
        Err(e) => {
            eprintln!("[Session {}] Failed to create Opus encoder: {}", session.id, e);
            return;
        }
    };

    // Initialize noise canceller if enabled
    let noise_type = NoiseCancellationType::from_str(&session.noise_cancellation);
    let mut noise_canceller = create_noise_canceller(noise_type, None);
    if noise_canceller.is_some() {
        eprintln!(
            "[Session {}] Noise cancellation enabled: {}",
            session.id, session.noise_cancellation
        );
    }

    // Spawn initial FFmpeg process
    let mut ffmpeg = match spawn_ffmpeg(audio_source) {
        Ok(child) => {
            ffmpeg_pid.store(child.id(), Ordering::SeqCst);
            eprintln!(
                "[Session {}] ffmpeg spawned with pid {}",
                session.id, child.id()
            );
            child
        }
        Err(e) => {
            eprintln!("[Session {}] Failed to spawn ffmpeg: {}", session.id, e);
            let _ = session.status_tx.send(
                serde_json::json!({
                    "type": "error",
                    "message": format!("Failed to spawn ffmpeg: {}", e)
                })
                .to_string(),
            );
            return;
        }
    };

    eprintln!(
        "[Session {}] Started ffmpeg ({})",
        session.id,
        if is_srt { "SRT low-latency" } else { "real-time pacing" }
    );

    let mut stdout = ffmpeg.stdout.take().expect("Failed to get ffmpeg stdout");

    // Create tokio runtime for async WebRTC writes
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Read 20ms chunks (320 samples at 16kHz = 640 bytes)
    let chunk_samples = 320;
    let bytes_per_chunk = chunk_samples * 2;
    let mut byte_buffer = vec![0u8; bytes_per_chunk];
    let mut total_samples: usize = 0;
    let mut last_status_time = Instant::now();

    // SRT reconnection state
    let mut reconnect_attempts = 0u32;
    const MAX_RECONNECT_ATTEMPTS: u32 = 10;
    const INITIAL_RECONNECT_DELAY_MS: u64 = 1000;
    const MAX_RECONNECT_DELAY_MS: u64 = 30000;

    // Outer loop for SRT reconnection
    'outer: loop {
        // Audio streaming loop
        while running.load(Ordering::SeqCst) {
            match stdout.read_exact(&mut byte_buffer) {
                Ok(_) => {
                    reconnect_attempts = 0;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    if is_srt {
                        eprintln!("[Session {}] SRT stream disconnected", session.id);
                        let _ = ffmpeg.kill();
                        let _ = ffmpeg.wait();

                        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS {
                            eprintln!(
                                "[Session {}] Max reconnection attempts ({}) reached, stopping",
                                session.id, MAX_RECONNECT_ATTEMPTS
                            );
                            break 'outer;
                        }

                        reconnect_attempts += 1;
                        let delay_ms = std::cmp::min(
                            INITIAL_RECONNECT_DELAY_MS * (1 << (reconnect_attempts - 1)),
                            MAX_RECONNECT_DELAY_MS,
                        );

                        eprintln!(
                            "[Session {}] Reconnecting in {}ms (attempt {}/{})",
                            session.id, delay_ms, reconnect_attempts, MAX_RECONNECT_ATTEMPTS
                        );

                        let status_msg = serde_json::json!({
                            "type": "reconnecting",
                            "attempt": reconnect_attempts,
                            "max_attempts": MAX_RECONNECT_ATTEMPTS,
                            "delay_ms": delay_ms
                        });
                        session.status_tx.send(status_msg.to_string()).ok();

                        std::thread::sleep(std::time::Duration::from_millis(delay_ms));

                        if !running.load(Ordering::SeqCst) {
                            break 'outer;
                        }

                        match spawn_ffmpeg(audio_source) {
                            Ok(mut child) => {
                                ffmpeg_pid.store(child.id(), Ordering::SeqCst);
                                eprintln!(
                                    "[Session {}] ffmpeg reconnected with pid {}",
                                    session.id, child.id()
                                );
                                ffmpeg = child;
                                stdout = ffmpeg.stdout.take().expect("Failed to get ffmpeg stdout");

                                let status_msg = serde_json::json!({ "type": "reconnected" });
                                session.status_tx.send(status_msg.to_string()).ok();

                                continue 'outer;
                            }
                            Err(e) => {
                                eprintln!(
                                    "[Session {}] Failed to reconnect ffmpeg: {}",
                                    session.id, e
                                );
                                continue 'outer;
                            }
                        }
                    } else {
                        eprintln!("[Session {}] End of audio stream", session.id);
                        break 'outer;
                    }
                }
                Err(e) => {
                    eprintln!("[Session {}] Error reading from ffmpeg: {}", session.id, e);
                    if is_srt {
                        let _ = ffmpeg.kill();
                        let _ = ffmpeg.wait();
                        continue 'outer;
                    }
                    break 'outer;
                }
            }

            // Convert bytes to f32 samples
            let raw_samples: Vec<f32> = byte_buffer
                .chunks(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();

            // Apply noise cancellation if enabled
            let samples = if let Some(ref mut nc) = noise_canceller {
                nc.process(&raw_samples)
            } else {
                raw_samples
            };

            if samples.is_empty() {
                continue;
            }

            total_samples += samples.len();
            let current_time = total_samples as f32 / 16000.0;

            // Send samples to transcription thread (non-blocking)
            if let Err(std_mpsc::TrySendError::Full(_)) = audio_tx.try_send(samples.clone()) {
                if total_samples % (16000 * 30) < 320 {
                    eprintln!(
                        "[Session {}] WARNING: Transcription buffer full at {:.1}s - CPU cannot keep up.",
                        session.id, current_time
                    );
                }
            }

            // Encode to Opus and send via WebRTC
            let rtp_packets = opus_encoder.encode(&samples);
            static RTP_PACKETS_SENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static RTP_NO_RECEIVER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            for packet in &rtp_packets {
                let track = audio_track.clone();
                let pkt = packet.clone();
                rt.block_on(async {
                    match track.write(&pkt).await {
                        Ok(_) => {
                            RTP_PACKETS_SENT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(e) => {
                            let err_str = e.to_string();
                            if err_str.contains("no receiver") || err_str.contains("ErrClosedPipe") {
                                RTP_NO_RECEIVER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            } else {
                                eprintln!("[Session {}] WebRTC write error: {}", session.id, e);
                            }
                        }
                    }
                });
            }
            // Log RTP stats periodically
            if last_status_time.elapsed().as_secs() >= 1 && total_samples % (16000 * 5) < 320 {
                let sent = RTP_PACKETS_SENT.load(std::sync::atomic::Ordering::Relaxed);
                let no_recv = RTP_NO_RECEIVER.load(std::sync::atomic::Ordering::Relaxed);
                eprintln!("[Session {}] RTP stats: {} packets sent, {} no-receiver errors", session.id, sent, no_recv);
            }

            // Update progress every second
            if last_status_time.elapsed().as_secs() >= 1 {
                rt.block_on(session.set_progress(current_time));

                let status_msg = serde_json::json!({
                    "type": "status",
                    "progress_secs": current_time,
                    "total_duration": duration_secs,
                    "is_live": is_srt
                });
                session.status_tx.send(status_msg.to_string()).ok();
                last_status_time = Instant::now();
            }
        }

        break 'outer;
    }

    // Cleanup
    drop(audio_tx);
    let _ = ffmpeg.kill();
    let _ = ffmpeg.wait();
}
