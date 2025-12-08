//! Transcription logic for sessions

use crate::webrtc_handlers::audio::OpusEncoder;
use parakeet_rs::{SessionState, TranscriptionSession};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use webrtc::track::track_local::{track_local_static_rtp::TrackLocalStaticRTP, TrackLocalWriter};

/// Run transcription for a session
pub fn run_session_transcription(
    session: Arc<TranscriptionSession>,
    wav_path: PathBuf,
    model_path: PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    model_id: String,
    language: String,
    ffmpeg_pid: Arc<AtomicU32>,
) {
    use parakeet_rs::streaming_transcriber::StreamingTranscriber;
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::sync::mpsc as std_mpsc;
    use std::time::Instant;

    eprintln!(
        "[Session {}] Starting transcription for {}",
        session.id,
        wav_path.display()
    );

    // Get duration using ffprobe
    let duration_secs = get_audio_duration(&wav_path).unwrap_or(0.0);
    eprintln!("[Session {}] Total duration: {:.2}s", session.id, duration_secs);

    // Check if this is Canary or TDT model
    let is_canary = model_id == "canary-1b";

    // Channel to send audio samples to transcription thread
    let (audio_tx, audio_rx) = std_mpsc::sync_channel::<Vec<f32>>(50000);

    // Get the mode for transcription config
    let mode = session.mode.clone();
    eprintln!("[Session {}] Using transcription mode: {}", session.id, mode);

    // Check if this is a VAD mode
    let is_vad_mode = mode.starts_with("vad_");
    let vad_base_mode = if is_vad_mode {
        mode.strip_prefix("vad_").unwrap_or("speedy").to_string()
    } else {
        mode.clone()
    };

    // Get VAD model path from env
    let vad_model_path =
        std::env::var("VAD_MODEL_PATH").unwrap_or_else(|_| "silero_vad.onnx".to_string());

    // Spawn transcription thread
    let transcription_session = session.clone();
    let transcription_running = running.clone();
    let transcription_thread = std::thread::spawn(move || {
        // Create the appropriate transcriber based on model type and mode
        let mut transcriber: Box<dyn StreamingTranscriber> = if is_vad_mode {
            // VAD-triggered transcription
            if is_canary {
                // VAD + Canary
                use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};

                let config = RealtimeCanaryVadConfig::buffered(language.clone());

                eprintln!(
                    "[Session {}] Creating VAD+Canary transcriber from {:?} (language: {}, vad_mode: {})",
                    transcription_session.id, model_path, language, &vad_base_mode
                );

                match RealtimeCanaryVad::new(
                    &model_path,
                    &vad_model_path,
                    Some(exec_config),
                    Some(config),
                ) {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!(
                            "[Session {}] Failed to create VAD+Canary transcriber: {}",
                            transcription_session.id, e
                        );
                        return;
                    }
                }
            } else {
                // VAD + TDT with diarization
                use parakeet_rs::realtime_tdt_vad::{RealtimeTdtVad, RealtimeTdtVadConfig};
                use parakeet_rs::vad::VadConfig;

                let diar_path = match diar_path {
                    Some(p) => p,
                    None => {
                        eprintln!(
                            "[Session {}] No diarization model configured for TDT",
                            transcription_session.id
                        );
                        return;
                    }
                };

                let vad_config = VadConfig::from_mode(&vad_base_mode);
                let config = RealtimeTdtVadConfig {
                    vad: vad_config,
                    enable_diarization: true,
                };

                eprintln!(
                    "[Session {}] Creating VAD+TDT transcriber from {:?} (vad_mode: {})",
                    transcription_session.id, model_path, &vad_base_mode
                );

                match RealtimeTdtVad::new(
                    &model_path,
                    Some(&diar_path),
                    &vad_model_path,
                    Some(exec_config),
                    Some(config),
                ) {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!(
                            "[Session {}] Failed to create VAD+TDT transcriber: {}",
                            transcription_session.id, e
                        );
                        return;
                    }
                }
            }
        } else if is_canary {
            // Use Canary for multilingual transcription (non-VAD)
            use parakeet_rs::realtime_canary::RealtimeCanary;

            let canary_config = create_canary_config(&mode, language.clone());

            eprintln!(
                "[Session {}] Creating Canary transcriber from {:?} (language: {}, mode: {})",
                transcription_session.id, model_path, language, mode
            );

            match RealtimeCanary::new(&model_path, Some(exec_config), Some(canary_config)) {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!(
                        "[Session {}] Failed to create Canary transcriber: {}",
                        transcription_session.id, e
                    );
                    return;
                }
            }
        } else {
            // Use TDT with diarization (non-VAD)
            use parakeet_rs::RealtimeTDTDiarized;

            let diar_path = match diar_path {
                Some(p) => p,
                None => {
                    eprintln!(
                        "[Session {}] No diarization model configured for TDT",
                        transcription_session.id
                    );
                    return;
                }
            };

            let config = create_transcription_config(&mode);

            eprintln!(
                "[Session {}] Creating TDT transcriber from {:?}",
                transcription_session.id, model_path
            );

            match RealtimeTDTDiarized::new(&model_path, &diar_path, Some(exec_config), Some(config))
            {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!(
                        "[Session {}] Failed to create TDT transcriber: {}",
                        transcription_session.id, e
                    );
                    return;
                }
            }
        };

        let model_type = match (is_vad_mode, is_canary) {
            (true, true) => "VAD+Canary",
            (true, false) => "VAD+TDT",
            (false, true) => "Canary",
            (false, false) => "TDT",
        };
        eprintln!(
            "[Session {}] Transcription thread started ({})",
            transcription_session.id, model_type
        );

        // Process audio samples as they arrive
        let mut chunks_processed = 0u64;
        while transcription_running.load(Ordering::SeqCst) {
            // Wait for at least one batch
            let first_batch = match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(batch) => batch,
                Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
            };

            // Collect the first batch
            let mut all_samples = first_batch;

            // Drain any additional pending batches
            while let Ok(batch) = audio_rx.try_recv() {
                all_samples.extend(batch);
            }

            chunks_processed += 1;

            // Process all collected samples
            match transcriber.push_audio(&all_samples) {
                Ok(result) => {
                    emit_streaming_segments(&transcription_session, &result.segments);
                }
                Err(e) => {
                    if chunks_processed % 100 == 0 {
                        eprintln!(
                            "[Session {}] push_audio error (batch {}): {}",
                            transcription_session.id, chunks_processed, e
                        );
                    }
                }
            }

            // Check if we should stop
            if !transcription_running.load(Ordering::SeqCst) {
                eprintln!(
                    "[Session {}] Stop requested, exiting transcription loop",
                    transcription_session.id
                );
                break;
            }
        }

        // Finalize if not stopped
        if transcription_running.load(Ordering::SeqCst) {
            // Drain remaining batches
            let mut remaining_samples = Vec::new();
            while let Ok(batch) = audio_rx.try_recv() {
                remaining_samples.extend(batch);
            }
            if !remaining_samples.is_empty() {
                transcriber.push_audio(&remaining_samples).ok();
            }

            eprintln!(
                "[Session {}] Finalizing transcription...",
                transcription_session.id
            );
            match transcriber.finalize() {
                Ok(result) => {
                    emit_streaming_segments(&transcription_session, &result.segments);
                }
                Err(e) => {
                    eprintln!(
                        "[Session {}] Finalization error: {}",
                        transcription_session.id, e
                    );
                }
            }
        } else {
            eprintln!(
                "[Session {}] Skipping finalization (session stopped)",
                transcription_session.id
            );
        }

        eprintln!(
            "[Session {}] Transcription thread finished",
            transcription_session.id
        );
    });

    // Initialize Opus encoder
    let mut opus_encoder = match OpusEncoder::new() {
        Ok(enc) => enc,
        Err(e) => {
            eprintln!(
                "[Session {}] Failed to create Opus encoder: {}",
                session.id, e
            );
            return;
        }
    };

    // Spawn ffmpeg with -re for real-time pacing
    let mut ffmpeg = match Command::new("ffmpeg")
        .args([
            "-re",
            "-i",
            wav_path.to_str().unwrap_or(""),
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
    {
        Ok(child) => {
            ffmpeg_pid.store(child.id(), Ordering::SeqCst);
            eprintln!(
                "[Session {}] ffmpeg spawned with pid {}",
                session.id,
                child.id()
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
        "[Session {}] Started ffmpeg with real-time pacing",
        session.id
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

    // Audio streaming loop
    while running.load(Ordering::SeqCst) {
        // Read PCM from ffmpeg stdout
        match stdout.read_exact(&mut byte_buffer) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                eprintln!("[Session {}] End of audio stream", session.id);
                break;
            }
            Err(e) => {
                eprintln!("[Session {}] Error reading from ffmpeg: {}", session.id, e);
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
        for packet in &rtp_packets {
            let track = audio_track.clone();
            let pkt = packet.clone();
            rt.block_on(async {
                if let Err(e) = track.write(&pkt).await {
                    let err_str = e.to_string();
                    if !err_str.contains("no receiver") && !err_str.contains("ErrClosedPipe") {
                        eprintln!("[Session {}] WebRTC write error: {}", session.id, e);
                    }
                }
            });
        }

        // Update progress every second
        if last_status_time.elapsed().as_secs() >= 1 {
            rt.block_on(session.set_progress(current_time));

            let status_msg = serde_json::json!({
                "type": "status",
                "progress_secs": current_time,
                "total_duration": duration_secs
            });
            session.status_tx.send(status_msg.to_string()).ok();
            last_status_time = Instant::now();
        }
    }

    // Cleanup
    drop(audio_tx);
    let _ = ffmpeg.kill();
    let _ = ffmpeg.wait();

    // Wait for transcription to finish
    transcription_thread.join().ok();

    // Mark session as completed
    rt.block_on(session.set_state(SessionState::Completed));
    session.stop();

    let end_msg = serde_json::json!({
        "type": "end",
        "total_duration": duration_secs
    });
    session.status_tx.send(end_msg.to_string()).ok();

    eprintln!(
        "[Session {}] Complete. Duration: {:.2}s",
        session.id, duration_secs
    );
}

/// Create RealtimeCanaryConfig based on latency mode
fn create_canary_config(
    mode: &str,
    language: String,
) -> parakeet_rs::realtime_canary::RealtimeCanaryConfig {
    use parakeet_rs::realtime_canary::RealtimeCanaryConfig;

    match mode {
        "speedy" => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
        },
        "pause_based" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
        },
        "low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 1.5,
            process_interval_secs: 1.5,
            language,
        },
        "ultra_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 6.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
        },
        "extreme_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 4.0,
            min_audio_secs: 0.5,
            process_interval_secs: 0.5,
            language,
        },
        "lookahead" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
        },
        _ => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
        },
    }
}

/// Create RealtimeTDTConfig based on latency mode
fn create_transcription_config(mode: &str) -> parakeet_rs::RealtimeTDTConfig {
    use parakeet_rs::RealtimeTDTConfig;

    match mode {
        "speedy" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.4,
            pause_based_confirm: true,
            pause_threshold_secs: 0.35,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "pause_based" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 1.5,
            confirm_threshold_secs: 2.0,
            pause_based_confirm: false,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "ultra_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 1.0,
            confirm_threshold_secs: 1.5,
            pause_based_confirm: false,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "extreme_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 5.0,
            process_interval_secs: 0.5,
            confirm_threshold_secs: 0.8,
            pause_based_confirm: false,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "lookahead" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            lookahead_mode: true,
            lookahead_segments: 2,
        },
        _ => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.4,
            pause_based_confirm: true,
            pause_threshold_secs: 0.35,
            silence_energy_threshold: 0.008,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
    }
}

/// Helper to emit transcript segments from StreamingTranscriber
fn emit_streaming_segments(
    session: &TranscriptionSession,
    segments: &[parakeet_rs::streaming_transcriber::TranscriptionSegment],
) {
    if !segments.is_empty() {
        eprintln!(
            "[Session {}] emit_streaming_segments: {} segment(s)",
            session.id,
            segments.len()
        );
    }

    for segment in segments {
        let subtitle_msg = serde_json::json!({
            "type": "subtitle",
            "text": segment.text,
            "speaker": segment.speaker,
            "start": segment.start_time,
            "end": segment.end_time,
            "is_final": segment.is_final
        });

        let receiver_count = session.subtitle_tx.receiver_count();
        let subtitle_str = subtitle_msg.to_string();

        // Cache the last subtitle for late-joining clients
        session.set_last_subtitle(subtitle_str.clone());

        let _ = session.subtitle_tx.send(subtitle_str);

        // Log all segments
        let partial_marker = if segment.is_final { "FINAL" } else { "partial" };
        eprintln!(
            "[Session {} | {} | Speaker {}] \"{}\" [{:.2}s-{:.2}s] (receivers: {})",
            session.id,
            partial_marker,
            segment
                .speaker
                .map(|s| s.to_string())
                .unwrap_or_else(|| "?".to_string()),
            segment.text,
            segment.start_time,
            segment.end_time,
            receiver_count
        );
    }
}

/// Get audio duration using ffprobe
fn get_audio_duration(path: &std::path::Path) -> Option<f32> {
    use std::process::Command;

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str()?,
        ])
        .output()
        .ok()?;

    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()
}
