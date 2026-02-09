//! Transcription logic for sessions

mod audio_pipeline;
mod configs;
mod emitters;
mod factory;
mod vod;

use crate::api::sessions::{ParallelConfig, PauseConfig};
use parakeet_rs::growing_text::GrowingTextMerger;
use parakeet_rs::{SessionState, TranscriptionSession};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use webrtc::track::track_local::track_local_static_rtp::TrackLocalStaticRTP;

/// Audio source for transcription
#[derive(Debug, Clone)]
pub enum AudioSource {
    /// File on disk
    File(PathBuf),
    /// SRT stream URL
    Srt(String),
}

impl AudioSource {
    pub fn is_srt(&self) -> bool {
        matches!(self, AudioSource::Srt(_))
    }

    pub fn display(&self) -> String {
        match self {
            AudioSource::File(path) => path.display().to_string(),
            AudioSource::Srt(url) => url.clone(),
        }
    }
}

/// Run transcription for a session
pub fn run_session_transcription(
    session: Arc<TranscriptionSession>,
    audio_source: AudioSource,
    model_path: PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    model_id: String,
    language: String,
    ffmpeg_pid: Arc<AtomicU32>,
    parallel_config: Option<ParallelConfig>,
    pause_config: Option<PauseConfig>,
    sentence_completion: String,
) {
    use std::sync::mpsc as std_mpsc;

    // Check if this is VoD mode - handle separately
    if session.mode == "vod" {
        vod::run_vod_transcription(
            session,
            audio_source,
            model_path,
            exec_config,
            model_id,
            language,
        );
        return;
    }

    let is_srt = audio_source.is_srt();

    eprintln!(
        "[Session {}] Starting transcription for {} ({})",
        session.id,
        audio_source.display(),
        if is_srt { "SRT stream" } else { "file" }
    );

    // Get duration using ffprobe (only for files)
    let duration_secs = match &audio_source {
        AudioSource::File(path) => vod::get_audio_duration(path).unwrap_or(0.0),
        AudioSource::Srt(_) => 0.0,
    };

    if !is_srt {
        eprintln!("[Session {}] Total duration: {:.2}s", session.id, duration_secs);
    } else {
        eprintln!("[Session {}] Live stream (duration unknown)", session.id);
    }

    let is_canary = model_id == "canary-1b" || model_id == "canary-180m-flash";
    let is_canary_flash = model_id == "canary-180m-flash";

    // Channel to send audio samples to transcription thread
    let (audio_tx, audio_rx) = std_mpsc::sync_channel::<Vec<f32>>(50000);

    let mode = session.mode.clone();
    eprintln!("[Session {}] Using transcription mode: {}", session.id, mode);

    let is_vad_mode = mode.starts_with("vad_");
    let vad_base_mode = if is_vad_mode {
        mode.strip_prefix("vad_").unwrap_or("speedy").to_string()
    } else {
        mode.clone()
    };

    let vad_model_path =
        std::env::var("VAD_MODEL_PATH").unwrap_or_else(|_| "silero_vad.onnx".to_string());

    // Spawn transcription thread with panic catching
    let transcription_session = session.clone();
    let transcription_running = running.clone();
    let sentence_completion_clone = sentence_completion.clone();
    let transcription_thread = std::thread::spawn(move || {
        let session_id = transcription_session.id.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_transcription_inner(
                transcription_session.clone(),
                audio_rx,
                model_path,
                diar_path,
                exec_config,
                is_canary,
                is_canary_flash,
                is_vad_mode,
                mode,
                vad_base_mode,
                vad_model_path,
                language,
                transcription_running,
                parallel_config,
                pause_config,
                sentence_completion_clone,
            );
        }));

        if let Err(panic_err) = result {
            let panic_msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_err.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            eprintln!(
                "[Session {}] PANIC in transcription thread: {}",
                session_id, panic_msg
            );
        }
    });

    // Run the audio pipeline (FFmpeg → PCM → noise cancel → Opus → RTP)
    audio_pipeline::run_audio_pipeline(
        session.clone(),
        &audio_source,
        audio_track,
        running.clone(),
        ffmpeg_pid,
        audio_tx,
        is_srt,
        duration_secs,
    );

    // Wait for transcription to finish with timeout
    eprintln!("[Session {}] Waiting for transcription thread to finish...", session.id);
    let join_timeout = std::time::Duration::from_secs(5);
    let join_start = std::time::Instant::now();

    loop {
        if transcription_thread.is_finished() {
            transcription_thread.join().ok();
            eprintln!("[Session {}] Transcription thread joined successfully", session.id);
            break;
        }
        if join_start.elapsed() > join_timeout {
            eprintln!(
                "[Session {}] Transcription thread did not finish within {}s - abandoning (may continue in background)",
                session.id, join_timeout.as_secs()
            );
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Mark session as completed (file) or stopped (SRT stream)
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let final_state = if is_srt {
        SessionState::Stopped
    } else {
        SessionState::Completed
    };
    rt.block_on(session.set_state(final_state));
    session.stop();

    let end_msg = serde_json::json!({
        "type": "end",
        "total_duration": duration_secs,
        "is_live": is_srt
    });
    session.status_tx.send(end_msg.to_string()).ok();

    if is_srt {
        eprintln!("[Session {}] Stopped.", session.id);
    } else {
        eprintln!("[Session {}] Complete. Duration: {:.2}s", session.id, duration_secs);
    }
}

/// Inner transcription logic running on a dedicated thread
fn run_transcription_inner(
    transcription_session: Arc<TranscriptionSession>,
    audio_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    model_path: PathBuf,
    diar_path: Option<PathBuf>,
    exec_config: parakeet_rs::ExecutionConfig,
    is_canary: bool,
    is_canary_flash: bool,
    is_vad_mode: bool,
    mode: String,
    vad_base_mode: String,
    vad_model_path: String,
    language: String,
    transcription_running: Arc<AtomicBool>,
    parallel_config: Option<ParallelConfig>,
    pause_config: Option<PauseConfig>,
    sentence_completion: String,
) {
    use parakeet_rs::sentence_buffer::{SentenceBuffer, SentenceBufferMode};
    use parakeet_rs::streaming_transcriber::StreamingTranscriber;
    use std::sync::mpsc as std_mpsc;

    let is_parallel_mode = mode == "parallel";
    let is_pause_parallel_mode = mode == "pause_parallel";

    // Create the appropriate transcriber
    let mut transcriber: Box<dyn StreamingTranscriber> = match factory::create_transcriber(
        factory::TranscriberParams {
            session_id: transcription_session.id.clone(),
            model_path,
            diar_path,
            exec_config,
            is_canary,
            is_canary_flash,
            is_vad_mode,
            mode: mode.clone(),
            vad_base_mode,
            vad_model_path,
            language,
            parallel_config,
            pause_config,
        },
    ) {
        Some(t) => t,
        None => return,
    };

    let model_type = factory::model_type_name(
        is_pause_parallel_mode,
        is_parallel_mode,
        is_vad_mode,
        is_canary_flash,
        is_canary,
    );
    eprintln!(
        "[Session {}] Transcription thread started ({})",
        transcription_session.id, model_type
    );

    // Create sentence buffer based on completion mode
    let buffer_mode = SentenceBufferMode::from_str(&sentence_completion);
    let mut sentence_buffer = SentenceBuffer::with_mode(buffer_mode);
    eprintln!(
        "[Session {}] Sentence buffer mode: {}",
        transcription_session.id, buffer_mode.as_str()
    );

    // Create growing text merger for incremental transcript display
    let mut growing_merger = GrowingTextMerger::new();
    eprintln!(
        "[Session {}] Growing text merger enabled",
        transcription_session.id
    );

    // Process audio samples as they arrive
    let mut chunks_processed = 0u64;
    while transcription_running.load(Ordering::SeqCst) {
        let first_batch = match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(batch) => batch,
            Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("[Session {}] Audio channel disconnected", transcription_session.id);
                break;
            }
        };

        let mut all_samples = first_batch;
        while let Ok(batch) = audio_rx.try_recv() {
            all_samples.extend(batch);
        }

        chunks_processed += 1;

        let inference_start = std::time::Instant::now();
        match transcriber.push_audio(&all_samples) {
            Ok(mut result) => {
                let inference_time = inference_start.elapsed();
                let inference_time_ms = inference_time.as_millis() as u32;

                for segment in &mut result.segments {
                    if segment.inference_time_ms.is_none() {
                        segment.inference_time_ms = Some(inference_time_ms);
                    }
                }

                // Hallucination rejection: skip results from extremely slow inference (>10s)
                if inference_time.as_secs() > 10 && !is_parallel_mode && !is_pause_parallel_mode {
                    eprintln!(
                        "[Session {}] HALLUCINATION GUARD: inference took {:.1}s, skipping result",
                        transcription_session.id,
                        inference_time.as_secs_f32(),
                    );
                    continue;
                }

                if inference_time.as_secs() > 5 && !is_parallel_mode && !is_pause_parallel_mode {
                    eprintln!(
                        "[Session {}] SLOW inference: {:.1}s for {} samples",
                        transcription_session.id,
                        inference_time.as_secs_f32(),
                        all_samples.len()
                    );
                }

                // Apply hallucination truncation
                for segment in &mut result.segments {
                    segment.text = emitters::truncate_hallucination_text(&segment.text);
                }

                for segment in result.segments {
                    if segment.text.trim().is_empty() {
                        continue;
                    }

                    let prev_finalized = growing_merger.get_finalized_sentences().len();
                    let growing_result = growing_merger.push(&segment.text, segment.is_final);
                    let new_finalized = growing_merger.get_finalized_sentences().len();

                    // Emit FINAL for each newly finalized sentence
                    for i in prev_finalized..new_finalized {
                        let fs = &growing_merger.get_finalized_sentences()[i];
                        let trimmed = fs.text.trim();
                        if trimmed.len() <= 2 || trimmed.chars().all(|c| !c.is_alphanumeric()) {
                            continue;
                        }
                        let final_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                            text: fs.text.clone(),
                            start_time: segment.start_time,
                            end_time: segment.end_time,
                            speaker: segment.speaker,
                            confidence: None,
                            is_final: true,
                            inference_time_ms: segment.inference_time_ms,
                        };
                        emitters::emit_final_subtitle(&transcription_session, &final_segment, &growing_result);
                    }

                    emitters::emit_partial_subtitle(&transcription_session, &segment, &growing_result);
                }
            }
            Err(e) => {
                eprintln!(
                    "[Session {}] push_audio error (batch {}): {}",
                    transcription_session.id, chunks_processed, e
                );
            }
        }

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
                for segment in result.segments {
                    if let Some(merged) = sentence_buffer.push(segment) {
                        emitters::emit_streaming_segments(&transcription_session, &[merged], &mut growing_merger);
                    }
                }
                if let Some(flushed) = sentence_buffer.flush() {
                    emitters::emit_streaming_segments(&transcription_session, &[flushed], &mut growing_merger);
                }
            }
            Err(e) => {
                eprintln!(
                    "[Session {}] Finalization error: {}",
                    transcription_session.id, e
                );
                if let Some(flushed) = sentence_buffer.flush() {
                    emitters::emit_streaming_segments(&transcription_session, &[flushed], &mut growing_merger);
                }
            }
        }
    } else {
        eprintln!(
            "[Session {}] Skipping finalization (session stopped)",
            transcription_session.id
        );
        if let Some(flushed) = sentence_buffer.flush() {
            emitters::emit_streaming_segments(&transcription_session, &[flushed], &mut growing_merger);
        }
    }

    // Flush growing merger
    if let Some(flushed_text) = growing_merger.flush() {
        eprintln!(
            "[Session {}] Flushing growing merger: \"{}\"",
            transcription_session.id,
            &flushed_text.chars().take(80).collect::<String>()
        );
        let growing_result = growing_merger.push("", false);
        let flush_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
            text: flushed_text,
            start_time: 0.0,
            end_time: 0.0,
            speaker: None,
            confidence: None,
            is_final: true,
            inference_time_ms: None,
        };
        emitters::emit_final_subtitle(&transcription_session, &flush_segment, &growing_result);
    }

    eprintln!(
        "[Session {}] Transcription thread finished",
        transcription_session.id
    );
}
