//! Transcription logic for sessions

use crate::api::sessions::{ParallelConfig, PauseConfig};
use crate::webrtc_handlers::audio::OpusEncoder;
use parakeet_rs::noise_cancellation::{create_noise_canceller, NoiseCancellationType};
use parakeet_rs::growing_text::GrowingTextMerger;
use parakeet_rs::{SessionState, TranscriptionSession, VodConfig, VodSegment, VodTranscriberCanary, VodTranscriberTDT};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use webrtc::track::track_local::{track_local_static_rtp::TrackLocalStaticRTP, TrackLocalWriter};

/// Audio source for transcription
#[derive(Debug, Clone)]
pub enum AudioSource {
    /// File on disk
    File(PathBuf),
    /// SRT stream URL
    Srt(String),
}

impl AudioSource {
    /// Check if this is an SRT stream
    pub fn is_srt(&self) -> bool {
        matches!(self, AudioSource::Srt(_))
    }

    /// Get display name for logging
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
    use parakeet_rs::streaming_transcriber::StreamingTranscriber;
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::sync::mpsc as std_mpsc;
    use std::time::Instant;

    // Check if this is VoD mode - handle separately
    if session.mode == "vod" {
        run_vod_transcription(
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
        AudioSource::File(path) => get_audio_duration(path).unwrap_or(0.0),
        AudioSource::Srt(_) => 0.0, // Unknown for live streams
    };

    if !is_srt {
        eprintln!("[Session {}] Total duration: {:.2}s", session.id, duration_secs);
    } else {
        eprintln!("[Session {}] Live stream (duration unknown)", session.id);
    }

    // Check if this is Canary or TDT model
    let is_canary = model_id == "canary-1b" || model_id == "canary-180m-flash";
    let is_canary_flash = model_id == "canary-180m-flash";

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

    // Spawn transcription thread with panic catching
    let transcription_session = session.clone();
    let transcription_running = running.clone();
    let sentence_completion_clone = sentence_completion.clone();
    let transcription_thread = std::thread::spawn(move || {
        // Wrap entire transcription in catch_unwind to prevent server crash
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
            // Session will be marked as stopped by cleanup
        }
    });

    // Inner function containing actual transcription logic
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
        parallel_config: Option<super::api::sessions::ParallelConfig>,
        pause_config: Option<super::api::sessions::PauseConfig>,
        sentence_completion: String,
    ) {
        use parakeet_rs::streaming_transcriber::StreamingTranscriber;
        use parakeet_rs::sentence_buffer::{SentenceBuffer, SentenceBufferMode};
        use std::sync::mpsc as std_mpsc;

        // Check if parallel mode
        let is_parallel_mode = mode == "parallel";
        let is_pause_parallel_mode = mode == "pause_parallel";

        // Create the appropriate transcriber based on model type and mode
        let mut transcriber: Box<dyn StreamingTranscriber> = if is_pause_parallel_mode {
            // Pause-based parallel mode - supports both Canary and TDT
            let num_threads = match &parallel_config {
                Some(cfg) => cfg.num_threads,
                None => if is_canary { 8 } else { 4 }, // TDT is faster, needs fewer threads
            };

            if is_canary {
                use parakeet_rs::pause_parallel_canary::{PauseParallelCanary, PauseParallelConfig};

                // Get pause config values or use defaults (0.5s for better sentence boundaries)
                let pause_threshold_secs = pause_config.as_ref()
                    .map(|p| p.pause_threshold_ms as f32 / 1000.0)
                    .unwrap_or(0.5);
                let silence_energy = pause_config.as_ref()
                    .map(|p| p.silence_energy_threshold)
                    .unwrap_or(0.008);
                let max_segment_secs = pause_config.as_ref()
                    .map(|p| p.max_segment_secs)
                    .unwrap_or(6.0);  // Increased for longer sentences
                let context_buffer = pause_config.as_ref()
                    .map(|p| p.context_buffer_secs)
                    .unwrap_or(2.0);  // Add context for better boundaries

                let config = PauseParallelConfig {
                    num_threads,
                    language: language.clone(),
                    intra_threads: 1,
                    pause_threshold_secs,
                    silence_energy_threshold: silence_energy,
                    max_segment_duration_secs: max_segment_secs,
                    context_buffer_secs: context_buffer,
                };

                eprintln!(
                    "[Session {}] Creating PauseParallelCanary transcriber with {} threads, {}ms pause, {}s context (diar: {:?})",
                    transcription_session.id, config.num_threads, (config.pause_threshold_secs * 1000.0) as u32, config.context_buffer_secs, diar_path
                );

                // Use diarization constructor if available and diar_path is provided
                #[cfg(feature = "sortformer")]
                let result = PauseParallelCanary::new_with_diarization(
                    &model_path,
                    diar_path.as_ref(),
                    Some(exec_config),
                    Some(config),
                );
                #[cfg(not(feature = "sortformer"))]
                let result = PauseParallelCanary::new(&model_path, Some(exec_config), Some(config));

                match result {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!(
                            "[Session {}] Failed to create PauseParallelCanary transcriber: {}",
                            transcription_session.id, e
                        );
                        return;
                    }
                }
            } else {
                use parakeet_rs::pause_parallel_tdt::{PauseParallelTDT, PauseParallelTDTConfig};

                // Get pause config values or use defaults (0.5s for better sentence boundaries)
                let pause_threshold_secs = pause_config.as_ref()
                    .map(|p| p.pause_threshold_ms as f32 / 1000.0)
                    .unwrap_or(0.5);
                let silence_energy = pause_config.as_ref()
                    .map(|p| p.silence_energy_threshold)
                    .unwrap_or(0.008);
                let max_segment_secs = pause_config.as_ref()
                    .map(|p| p.max_segment_secs)
                    .unwrap_or(6.0);  // Increased for longer sentences
                let context_buffer = pause_config.as_ref()
                    .map(|p| p.context_buffer_secs)
                    .unwrap_or(2.0);  // Add context for better boundaries

                let config = PauseParallelTDTConfig {
                    num_threads,
                    intra_threads: 2,
                    pause_threshold_secs,
                    silence_energy_threshold: silence_energy,
                    max_segment_duration_secs: max_segment_secs,
                    context_buffer_secs: context_buffer,
                };

                eprintln!(
                    "[Session {}] Creating PauseParallelTDT transcriber with {} threads, {}ms pause, {}s context (diar: {:?})",
                    transcription_session.id, config.num_threads, (config.pause_threshold_secs * 1000.0) as u32, config.context_buffer_secs, diar_path
                );

                // Use diarization constructor if available and diar_path is provided
                #[cfg(feature = "sortformer")]
                let result = if diar_path.is_some() {
                    PauseParallelTDT::new_with_diarization(
                        &model_path,
                        diar_path.as_ref(),
                        Some(exec_config),
                        Some(config),
                    )
                } else {
                    PauseParallelTDT::new(&model_path, Some(exec_config), Some(config))
                };

                #[cfg(not(feature = "sortformer"))]
                let result = PauseParallelTDT::new(&model_path, Some(exec_config), Some(config));

                match result {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!(
                            "[Session {}] Failed to create PauseParallelTDT transcriber: {}",
                            transcription_session.id, e
                        );
                        return;
                    }
                }
            }
        } else if is_parallel_mode {
            // Parallel sliding window mode - supports both Canary and TDT
            let (num_threads, buffer_size) = match &parallel_config {
                Some(cfg) => (cfg.num_threads, cfg.buffer_size_secs),
                None => if is_canary { (8, 6) } else { (4, 6) }, // TDT is faster, needs fewer threads
            };

            if is_canary {
                use parakeet_rs::parallel_canary::{ParallelCanary, ParallelCanaryConfig};

                let config = ParallelCanaryConfig {
                    num_threads,
                    buffer_size_chunks: buffer_size,
                    chunk_duration_secs: 1.0,
                    language: language.clone(),
                    intra_threads: 1,
                };

                eprintln!(
                    "[Session {}] Creating ParallelCanary transcriber with {} threads, {}s buffer (diar: {:?})",
                    transcription_session.id, config.num_threads, config.buffer_size_chunks, diar_path
                );

                // Use diarization constructor if available and diar_path is provided
                #[cfg(feature = "sortformer")]
                let result = ParallelCanary::new_with_diarization(
                    &model_path,
                    diar_path.as_ref(),
                    Some(exec_config),
                    Some(config),
                );
                #[cfg(not(feature = "sortformer"))]
                let result = ParallelCanary::new(&model_path, Some(exec_config), Some(config));

                match result {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!(
                            "[Session {}] Failed to create ParallelCanary transcriber: {}",
                            transcription_session.id, e
                        );
                        return;
                    }
                }
            } else {
                use parakeet_rs::parallel_tdt::{ParallelTDT, ParallelTDTConfig};

                let config = ParallelTDTConfig {
                    num_threads,
                    buffer_size_chunks: buffer_size,
                    chunk_duration_secs: 1.0,
                    intra_threads: 2,
                };

                eprintln!(
                    "[Session {}] Creating ParallelTDT transcriber with {} threads, {}s buffer (diar: {:?})",
                    transcription_session.id, config.num_threads, config.buffer_size_chunks, diar_path
                );

                // Use diarization constructor if available and diar_path is provided
                #[cfg(feature = "sortformer")]
                let result = if diar_path.is_some() {
                    ParallelTDT::new_with_diarization(
                        &model_path,
                        diar_path.as_ref(),
                        Some(exec_config),
                        Some(config),
                    )
                } else {
                    ParallelTDT::new(&model_path, Some(exec_config), Some(config))
                };

                #[cfg(not(feature = "sortformer"))]
                let result = ParallelTDT::new(&model_path, Some(exec_config), Some(config));

                match result {
                    Ok(t) => Box::new(t),
                    Err(e) => {
                        eprintln!(
                            "[Session {}] Failed to create ParallelTDT transcriber: {}",
                            transcription_session.id, e
                        );
                        return;
                    }
                }
            }
        } else if is_vad_mode {
            // VAD-triggered transcription
            if is_canary {
                // VAD + Canary with optional diarization
                use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};

                // Choose config based on VAD sub-mode
                let config = if vad_base_mode == "sliding_window" {
                    RealtimeCanaryVadConfig::sliding_window(language.clone())
                } else {
                    RealtimeCanaryVadConfig::buffered(language.clone())
                };

                eprintln!(
                    "[Session {}] Creating VAD+Canary transcriber from {:?} (language: {}, vad_mode: {}, diar: {:?})",
                    transcription_session.id, model_path, language, &vad_base_mode, diar_path
                );

                #[cfg(feature = "sortformer")]
                let result = RealtimeCanaryVad::new(
                    &model_path,
                    diar_path.as_ref(),
                    &vad_model_path,
                    Some(exec_config),
                    Some(config),
                );
                #[cfg(not(feature = "sortformer"))]
                let result = RealtimeCanaryVad::new(
                    &model_path,
                    &vad_model_path,
                    Some(exec_config),
                    Some(config),
                );
                match result {
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
                // VAD + TDT with diarization (requires sortformer feature)
                #[cfg(feature = "sortformer")]
                {
                    use parakeet_rs::realtime_tdt_vad::{RealtimeTdtVad, RealtimeTdtVadConfig};

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

                    // Choose config based on VAD sub-mode
                    let config = if vad_base_mode == "sliding_window" {
                        RealtimeTdtVadConfig::sliding_window()
                    } else {
                        RealtimeTdtVadConfig::from_mode(&vad_base_mode)
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
                #[cfg(not(feature = "sortformer"))]
                {
                    eprintln!(
                        "[Session {}] VAD+TDT mode requires sortformer feature",
                        transcription_session.id
                    );
                    return;
                }
            }
        } else if is_canary_flash {
            // Use Canary Flash for fast multilingual transcription with KV cache
            use parakeet_rs::realtime_canary_flash::{RealtimeCanaryFlash, RealtimeCanaryFlashConfig};

            let flash_config = RealtimeCanaryFlashConfig {
                buffer_size_secs: 8.0,
                min_audio_secs: 1.0,
                process_interval_secs: 0.5,
                language: language.clone(),
            };

            eprintln!(
                "[Session {}] Creating Canary Flash transcriber from {:?} (language: {}, mode: {}, diar: {:?})",
                transcription_session.id, model_path, language, mode, diar_path
            );

            // Use diarization constructor if available and diar_path is provided
            #[cfg(feature = "sortformer")]
            let result = if diar_path.is_some() {
                RealtimeCanaryFlash::new_with_diarization(
                    &model_path,
                    diar_path.as_ref(),
                    Some(exec_config),
                    Some(flash_config),
                )
            } else {
                RealtimeCanaryFlash::new(&model_path, Some(exec_config), Some(flash_config))
            };

            #[cfg(not(feature = "sortformer"))]
            let result = RealtimeCanaryFlash::new(&model_path, Some(exec_config), Some(flash_config));

            match result {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!(
                        "[Session {}] Failed to create Canary Flash transcriber: {}",
                        transcription_session.id, e
                    );
                    return;
                }
            }
        } else if is_canary {
            // Use Canary for multilingual transcription (non-VAD)
            use parakeet_rs::realtime_canary::RealtimeCanary;

            let canary_config = create_canary_config(&mode, language.clone());

            eprintln!(
                "[Session {}] Creating Canary transcriber from {:?} (language: {}, mode: {}, diar: {:?})",
                transcription_session.id, model_path, language, mode, diar_path
            );

            // Use diarization constructor if available and diar_path is provided
            #[cfg(feature = "sortformer")]
            let result = if diar_path.is_some() {
                RealtimeCanary::new_with_diarization(
                    &model_path,
                    diar_path.as_ref(),
                    Some(exec_config),
                    Some(canary_config),
                )
            } else {
                RealtimeCanary::new(&model_path, Some(exec_config), Some(canary_config))
            };

            #[cfg(not(feature = "sortformer"))]
            let result = RealtimeCanary::new(&model_path, Some(exec_config), Some(canary_config));

            match result {
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
            // Use TDT with diarization (non-VAD) - requires sortformer feature
            #[cfg(feature = "sortformer")]
            {
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

                let config = create_transcription_config(&mode, pause_config.as_ref());

                eprintln!(
                    "[Session {}] Creating TDT transcriber from {:?} (pause: {}ms)",
                    transcription_session.id, model_path,
                    (config.pause_threshold_secs * 1000.0) as u32
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
            }
            #[cfg(not(feature = "sortformer"))]
            {
                // Use ParallelTDT with 1 thread as fallback when sortformer is not available
                use parakeet_rs::parallel_tdt::{ParallelTDT, ParallelTDTConfig};

                let config = ParallelTDTConfig {
                    num_threads: 1,
                    buffer_size_chunks: 6,
                    chunk_duration_secs: 1.0,
                    intra_threads: 4,
                };

                eprintln!(
                    "[Session {}] Creating TDT transcriber from {:?} (single-thread fallback, no diarization)",
                    transcription_session.id, model_path
                );

                match ParallelTDT::new(&model_path, Some(exec_config), Some(config))
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
            }
        };

        let model_type = if is_pause_parallel_mode {
            if is_canary { "PauseParallelCanary" } else { "PauseParallelTDT" }
        } else if is_parallel_mode {
            if is_canary { "ParallelCanary" } else { "ParallelTDT" }
        } else {
            match (is_vad_mode, is_canary_flash, is_canary) {
                (true, _, true) => "VAD+Canary",
                (true, _, false) => "VAD+TDT",
                (false, true, _) => "CanaryFlash",
                (false, false, true) => "Canary",
                (false, false, false) => "TDT",
            }
        };
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
            // Wait for at least one batch
            let first_batch = match audio_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(batch) => batch,
                Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                    eprintln!("[Session {}] Audio channel disconnected", transcription_session.id);
                    break;
                }
            };

            // Collect the first batch
            let mut all_samples = first_batch;

            // Drain any additional pending batches
            while let Ok(batch) = audio_rx.try_recv() {
                all_samples.extend(batch);
            }

            chunks_processed += 1;

            // Process all collected samples
            let inference_start = std::time::Instant::now();
            match transcriber.push_audio(&all_samples) {
                Ok(mut result) => {
                    let inference_time = inference_start.elapsed();
                    let inference_time_ms = inference_time.as_millis() as u32;

                    // Set inference time on segments that don't already have it
                    // (Parallel modes set inference_time_ms internally from worker threads)
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

                    // Log slow inference calls (> 5 seconds) for non-parallel modes
                    if inference_time.as_secs() > 5 && !is_parallel_mode && !is_pause_parallel_mode {
                        eprintln!(
                            "[Session {}] SLOW inference: {:.1}s for {} samples",
                            transcription_session.id,
                            inference_time.as_secs_f32(),
                            all_samples.len()
                        );
                    }

                    // Apply hallucination truncation to each segment's text
                    for segment in &mut result.segments {
                        segment.text = truncate_hallucination_text(&segment.text);
                    }

                    // Process segments through sentence buffer
                    for segment in result.segments {
                        // Skip empty segments (truncated away entirely)
                        if segment.text.trim().is_empty() {
                            continue;
                        }
                        if let Some(merged) = sentence_buffer.push(segment) {
                            emit_streaming_segments(&transcription_session, &[merged], &mut growing_merger);
                        }
                    }
                }
                Err(e) => {
                    // Always log errors
                    eprintln!(
                        "[Session {}] push_audio error (batch {}): {}",
                        transcription_session.id, chunks_processed, e
                    );
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
                    // Process final segments through buffer
                    for segment in result.segments {
                        if let Some(merged) = sentence_buffer.push(segment) {
                            emit_streaming_segments(&transcription_session, &[merged], &mut growing_merger);
                        }
                    }
                    // Flush any remaining buffered content
                    if let Some(flushed) = sentence_buffer.flush() {
                        emit_streaming_segments(&transcription_session, &[flushed], &mut growing_merger);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[Session {}] Finalization error: {}",
                        transcription_session.id, e
                    );
                    // Still flush buffer on error
                    if let Some(flushed) = sentence_buffer.flush() {
                        emit_streaming_segments(&transcription_session, &[flushed], &mut growing_merger);
                    }
                }
            }
        } else {
            eprintln!(
                "[Session {}] Skipping finalization (session stopped)",
                transcription_session.id
            );
            // Flush buffer even when stopped to emit any pending content
            if let Some(flushed) = sentence_buffer.flush() {
                emit_streaming_segments(&transcription_session, &[flushed], &mut growing_merger);
            }
        }

        eprintln!(
            "[Session {}] Transcription thread finished",
            transcription_session.id
        );
    } // end of run_transcription_inner

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

    // Initialize noise canceller if enabled
    let noise_type = NoiseCancellationType::from_str(&session.noise_cancellation);
    let mut noise_canceller = create_noise_canceller(noise_type, None);
    if noise_canceller.is_some() {
        eprintln!(
            "[Session {}] Noise cancellation enabled: {}",
            session.id, session.noise_cancellation
        );
    }

    // Build FFmpeg command based on source type
    let spawn_ffmpeg = |source: &AudioSource| -> Result<std::process::Child, std::io::Error> {
        match source {
            AudioSource::File(path) => {
                // File playback with real-time pacing
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
                // SRT stream with low-latency settings
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
    };

    // Spawn initial FFmpeg process
    let mut ffmpeg = match spawn_ffmpeg(&audio_source) {
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
            // Read PCM from ffmpeg stdout
            match stdout.read_exact(&mut byte_buffer) {
                Ok(_) => {
                    // Reset reconnect counter on successful read
                    reconnect_attempts = 0;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                    if is_srt {
                        // SRT stream disconnected - attempt reconnection
                        eprintln!("[Session {}] SRT stream disconnected", session.id);

                        // Kill current ffmpeg process
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

                        // Send reconnecting status to clients
                        let status_msg = serde_json::json!({
                            "type": "reconnecting",
                            "attempt": reconnect_attempts,
                            "max_attempts": MAX_RECONNECT_ATTEMPTS,
                            "delay_ms": delay_ms
                        });
                        session.status_tx.send(status_msg.to_string()).ok();

                        // Wait before reconnecting
                        std::thread::sleep(std::time::Duration::from_millis(delay_ms));

                        // Check if we should still be running
                        if !running.load(Ordering::SeqCst) {
                            break 'outer;
                        }

                        // Spawn new ffmpeg process
                        match spawn_ffmpeg(&audio_source) {
                            Ok(mut child) => {
                                ffmpeg_pid.store(child.id(), Ordering::SeqCst);
                                eprintln!(
                                    "[Session {}] ffmpeg reconnected with pid {}",
                                    session.id, child.id()
                                );
                                ffmpeg = child;
                                stdout = ffmpeg.stdout.take().expect("Failed to get ffmpeg stdout");

                                // Send reconnected status
                                let status_msg = serde_json::json!({
                                    "type": "reconnected"
                                });
                                session.status_tx.send(status_msg.to_string()).ok();

                                continue 'outer; // Restart inner loop
                            }
                            Err(e) => {
                                eprintln!(
                                    "[Session {}] Failed to reconnect ffmpeg: {}",
                                    session.id, e
                                );
                                // Continue trying
                                continue 'outer;
                            }
                        }
                    } else {
                        // File stream ended normally
                        eprintln!("[Session {}] End of audio stream", session.id);
                        break 'outer;
                    }
                }
                Err(e) => {
                    eprintln!("[Session {}] Error reading from ffmpeg: {}", session.id, e);
                    if is_srt {
                        // Try to reconnect on errors for SRT
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

            // Skip empty output (noise canceller may buffer internally)
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
            // Log RTP stats every 5 seconds
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

        // If running flag is false, exit the outer loop
        break 'outer;
    }

    // Cleanup
    drop(audio_tx);
    let _ = ffmpeg.kill();
    let _ = ffmpeg.wait();

    // Wait for transcription to finish with timeout
    // If transcription is stuck in a long ONNX inference call, we don't want to wait forever
    eprintln!("[Session {}] Waiting for transcription thread to finish...", session.id);
    let join_timeout = std::time::Duration::from_secs(5);
    let join_start = std::time::Instant::now();

    // Use a loop with short sleeps to allow checking timeout
    // Note: std::thread::JoinHandle doesn't support timeout directly,
    // so we'll use try_recv pattern with the thread
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
            // The thread will continue running but we won't wait for it
            // This prevents the cleanup from blocking indefinitely
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Mark session as completed (file) or stopped (SRT stream)
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
        eprintln!(
            "[Session {}] Stopped. Elapsed: {:.2}s",
            session.id, total_samples as f32 / 16000.0
        );
    } else {
        eprintln!(
            "[Session {}] Complete. Duration: {:.2}s",
            session.id, duration_secs
        );
    }
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
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        "pause_based" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        "low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 1.5,
            process_interval_secs: 1.5,
            language,
            pause_based_confirm: false,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        "ultra_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 6.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.5,
            silence_energy_threshold: 0.008,
        },
        "extreme_low_latency" => RealtimeCanaryConfig {
            buffer_size_secs: 4.0,
            min_audio_secs: 0.5,
            process_interval_secs: 0.5,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.4,
            silence_energy_threshold: 0.008,
        },
        "lookahead" => RealtimeCanaryConfig {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
        _ => RealtimeCanaryConfig {
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            process_interval_secs: 1.0,
            language,
            pause_based_confirm: true,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
        },
    }
}

/// Create RealtimeTDTConfig based on latency mode with optional pause config override
fn create_transcription_config(
    mode: &str,
    pause_config: Option<&super::api::sessions::PauseConfig>,
) -> parakeet_rs::RealtimeTDTConfig {
    use parakeet_rs::RealtimeTDTConfig;

    // Get pause config values or use mode defaults (0.6s for better sentence boundaries)
    let (pause_threshold, silence_energy) = match pause_config {
        Some(pc) => (
            pc.pause_threshold_ms as f32 / 1000.0,
            pc.silence_energy_threshold,
        ),
        None => match mode {
            "speedy" => (0.6, 0.008),  // 600ms for complete sentences
            _ => (0.5, 0.008),
        },
    };

    match mode {
        "speedy" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.5,  // Increased for stability
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "pause_based" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 1.5,
            confirm_threshold_secs: 2.0,
            pause_based_confirm: false,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "ultra_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 1.0,
            confirm_threshold_secs: 1.5,
            pause_based_confirm: false,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "extreme_low_latency" => RealtimeTDTConfig {
            buffer_size_secs: 5.0,
            process_interval_secs: 0.5,
            confirm_threshold_secs: 0.8,
            pause_based_confirm: false,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
        "lookahead" => RealtimeTDTConfig {
            buffer_size_secs: 10.0,
            process_interval_secs: 0.3,
            confirm_threshold_secs: 0.5,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: true,
            lookahead_segments: 2,
        },
        _ => RealtimeTDTConfig {
            buffer_size_secs: 8.0,
            process_interval_secs: 0.2,
            confirm_threshold_secs: 0.4,
            pause_based_confirm: true,
            pause_threshold_secs: pause_threshold,
            silence_energy_threshold: silence_energy,
            lookahead_mode: false,
            lookahead_segments: 2,
        },
    }
}

/// Helper to emit transcript segments from StreamingTranscriber with growing text support
fn emit_streaming_segments(
    session: &TranscriptionSession,
    segments: &[parakeet_rs::streaming_transcriber::TranscriptionSegment],
    growing_merger: &mut GrowingTextMerger,
) {
    if !segments.is_empty() {
        eprintln!(
            "[Session {}] emit_streaming_segments: {} segment(s)",
            session.id,
            segments.len()
        );
    }

    for segment in segments {
        // Process through growing text merger for incremental display
        let growing_result = growing_merger.push(&segment.text, segment.is_final);

        // Emit delta as primary text when available and non-empty,
        // falling back to segment.text. Keep segment.text as raw_text for clients that need it.
        let primary_text = if !growing_result.delta.is_empty() {
            &growing_result.delta
        } else {
            &segment.text
        };

        let subtitle_msg = serde_json::json!({
            "type": "subtitle",
            "text": primary_text,
            "raw_text": segment.text,
            "growing_text": growing_result.current_sentence,
            "full_transcript": growing_result.buffer,
            "delta": growing_result.delta,
            "tail_changed": growing_result.tail_changed,
            "speaker": segment.speaker,
            "start": segment.start_time,
            "end": segment.end_time,
            "is_final": segment.is_final,
            "inference_time_ms": segment.inference_time_ms
        });

        let receiver_count = session.subtitle_tx.receiver_count();
        let subtitle_str = subtitle_msg.to_string();

        // Cache the last subtitle for late-joining clients
        session.set_last_subtitle(subtitle_str.clone());

        let _ = session.subtitle_tx.send(subtitle_str);

        // Log with growing text info
        let partial_marker = if segment.is_final { "FINAL" } else { "partial" };
        let tail_marker = if growing_result.tail_changed { " [TAIL CHANGED]" } else { "" };
        eprintln!(
            "[Session {} | {} | Speaker {}] \"{}\" [{:.2}s-{:.2}s] (inference: {}ms, receivers: {}){}",
            session.id,
            partial_marker,
            segment
                .speaker
                .map(|s| s.to_string())
                .unwrap_or_else(|| "?".to_string()),
            segment.text,
            segment.start_time,
            segment.end_time,
            segment.inference_time_ms.unwrap_or(0),
            receiver_count,
            tail_marker
        );
    }
}

/// Truncate text at first detected hallucination (3+ consecutive repeated words
/// or 3+ repeated 2-3 word phrases) for the emit pipeline
fn truncate_hallucination_text(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return text.to_string();
    }

    // Check for 3+ consecutive identical words
    let mut consecutive_count = 1;
    for i in 1..words.len() {
        if words[i].to_lowercase() == words[i - 1].to_lowercase() && words[i].len() > 1 {
            consecutive_count += 1;
            if consecutive_count >= 3 {
                let truncate_at = i - consecutive_count + 1;
                if truncate_at > 0 {
                    return words[..truncate_at].join(" ");
                }
                return String::new();
            }
        } else {
            consecutive_count = 1;
        }
    }

    // Check for repeated phrases (2-3 word patterns)
    for pattern_len in 2..=3 {
        if words.len() < pattern_len * 3 {
            continue;
        }
        for i in 0..=(words.len() - pattern_len * 3) {
            let pattern: Vec<&str> = words[i..i + pattern_len].to_vec();
            let mut pattern_count = 1;
            let mut j = i + pattern_len;
            while j + pattern_len <= words.len() {
                let candidate: Vec<&str> = words[j..j + pattern_len].to_vec();
                if candidate
                    .iter()
                    .zip(pattern.iter())
                    .all(|(a, b)| a.to_lowercase() == b.to_lowercase())
                {
                    pattern_count += 1;
                    if pattern_count >= 3 {
                        if i > 0 {
                            return words[..i].join(" ");
                        }
                        return String::new();
                    }
                    j += pattern_len;
                } else {
                    break;
                }
            }
        }
    }

    text.to_string()
}

/// Run VoD batch transcription
fn run_vod_transcription(
    session: Arc<TranscriptionSession>,
    audio_source: AudioSource,
    model_path: PathBuf,
    exec_config: parakeet_rs::ExecutionConfig,
    model_id: String,
    language: String,
) {
    use std::fs;

    // VoD only supports file sources
    let wav_path = match &audio_source {
        AudioSource::File(path) => path.clone(),
        AudioSource::Srt(_) => {
            eprintln!("[Session {}] VoD mode does not support SRT streams", session.id);
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(session.set_state(SessionState::Stopped));
            return;
        }
    };

    eprintln!(
        "[Session {}] Starting VoD transcription for {}",
        session.id,
        wav_path.display()
    );

    // Set state to running
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(session.set_state(SessionState::Running));

    // Check if Canary or TDT model
    let is_canary = model_id == "canary-1b" || model_id == "canary-180m-flash";

    // Determine config based on model type:
    // - Use 1 worker to avoid GPU OOM when processing concurrently
    // - 3-min chunks fit comfortably in GPU VRAM
    let (num_workers, chunk_duration, overlap_duration) = if is_canary {
        (1, 180.0, 15.0)  // 3 min chunks, 15s overlap, 1 worker (GPU sequential)
    } else {
        (1, 180.0, 15.0)  // 3 min chunks, 15s overlap (TDT encoder limit ~2500 frames ≈ 3.3 min)
    };

    // Create VoD config
    let vod_config = VodConfig {
        chunk_duration_secs: chunk_duration,
        overlap_duration_secs: overlap_duration,
        num_workers,
        dedup_threshold: 0.8,
        language: language.clone(),
    };

    // Progress callback
    let session_for_progress = session.clone();
    let progress_callback = Box::new(move |progress: parakeet_rs::VodProgress| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(session_for_progress.set_vod_progress(
            progress.total_chunks,
            progress.completed_chunks,
        ));

        // Send progress via status channel
        let progress_msg = serde_json::json!({
            "type": "vod_progress",
            "total_chunks": progress.total_chunks,
            "completed_chunks": progress.completed_chunks,
            "current_chunk": progress.current_chunk,
            "percent": progress.percent,
        });
        let _ = session_for_progress.status_tx.send(progress_msg.to_string());
    });

    // Segment callback for real-time subtitle emission with growing text
    let session_for_segments = session.clone();
    let growing_merger = std::sync::Arc::new(std::sync::Mutex::new(GrowingTextMerger::new()));
    let segment_callback: parakeet_rs::SegmentCallback = Box::new(move |segments: &[VodSegment], chunk_index: usize, is_final_chunk: bool| {
        let mut merger = growing_merger.lock().unwrap();

        for (seg_idx, segment) in segments.iter().enumerate() {
            // Determine if this segment should be treated as final
            // In VoD, each sentence-level segment is essentially final
            let is_final = true;

            // Process through growing text merger
            let growing_result = merger.push(&segment.text, is_final);

            // Create subtitle message matching streaming format
            let subtitle_msg = serde_json::json!({
                "type": "subtitle",
                "text": segment.text,
                "growing_text": growing_result.current_sentence,
                "full_transcript": growing_result.buffer,
                "delta": growing_result.delta,
                "tail_changed": growing_result.tail_changed,
                "speaker": segment.speaker,
                "start": segment.start,
                "end": segment.end,
                "is_final": is_final,
                "vod_chunk": chunk_index,
                "vod_segment": seg_idx,
            });

            let subtitle_str = subtitle_msg.to_string();

            // Cache the last subtitle for late-joining clients
            session_for_segments.set_last_subtitle(subtitle_str.clone());

            // Send to subtitle channel
            let _ = session_for_segments.subtitle_tx.send(subtitle_str);
        }

        eprintln!(
            "[Session {}] VoD chunk {} emitted {} segments via subtitle channel{}",
            session_for_segments.id,
            chunk_index,
            segments.len(),
            if is_final_chunk { " (final)" } else { "" }
        );
    });

    // Run transcription with segment callback
    let result = if is_canary {
        eprintln!("[Session {}] Using Canary model for VoD", session.id);
        match VodTranscriberCanary::new(&model_path, vod_config, Some(exec_config)) {
            Ok(transcriber) => transcriber.transcribe_file_with_segments(&wav_path, &session.id, Some(progress_callback), Some(segment_callback)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create Canary VoD transcriber: {}", session.id, e);
                rt.block_on(session.set_state(SessionState::Stopped));
                return;
            }
        }
    } else {
        eprintln!("[Session {}] Using TDT model for VoD", session.id);
        match VodTranscriberTDT::new(&model_path, vod_config, Some(exec_config)) {
            Ok(transcriber) => transcriber.transcribe_file_with_segments(&wav_path, &session.id, Some(progress_callback), Some(segment_callback)),
            Err(e) => {
                eprintln!("[Session {}] Failed to create TDT VoD transcriber: {}", session.id, e);
                rt.block_on(session.set_state(SessionState::Stopped));
                return;
            }
        }
    };

    match result {
        Ok(transcript) => {
            // Generate transcript file path: same as wav file with .transcript.json suffix
            // e.g., broadcast.wav => broadcast.transcript.json
            let transcript_path = {
                let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("transcript");
                wav_path.with_file_name(format!("{}.transcript.json", stem))
            };

            // Save transcript
            match serde_json::to_string_pretty(&transcript) {
                Ok(json) => {
                    if let Err(e) = fs::write(&transcript_path, &json) {
                        eprintln!("[Session {}] Failed to write transcript: {}", session.id, e);
                    } else {
                        eprintln!(
                            "[Session {}] Transcript saved to {}",
                            session.id,
                            transcript_path.display()
                        );

                        // Set transcript path in session
                        rt.block_on(session.set_transcript_path(transcript_path.clone()));
                    }
                }
                Err(e) => {
                    eprintln!("[Session {}] Failed to serialize transcript: {}", session.id, e);
                }
            }

            // Send completion message
            let complete_msg = serde_json::json!({
                "type": "vod_complete",
                "transcript_available": true,
                "duration_secs": transcript.duration_secs,
                "segment_count": transcript.segments.len(),
            });
            let _ = session.status_tx.send(complete_msg.to_string());

            eprintln!(
                "[Session {}] VoD transcription completed: {} segments, {:.1}s duration",
                session.id,
                transcript.segments.len(),
                transcript.duration_secs
            );

            // Set state to completed
            rt.block_on(session.set_state(SessionState::Completed));
        }
        Err(e) => {
            eprintln!("[Session {}] VoD transcription failed: {}", session.id, e);

            let error_msg = serde_json::json!({
                "type": "error",
                "message": format!("Transcription failed: {}", e),
            });
            let _ = session.status_tx.send(error_msg.to_string());

            rt.block_on(session.set_state(SessionState::Stopped));
        }
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
