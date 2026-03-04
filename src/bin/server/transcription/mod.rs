//! Transcription logic for sessions

mod audio_pipeline;
mod configs;
mod emitters;
mod factory;
mod vod;

use crate::api::sessions::{GrowingSegmentsConfig, ParallelConfig, PauseConfig};
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

/// Run an audio-only session (no transcription, just stream audio via WebRTC)
pub fn run_audio_only_session(
    session: Arc<TranscriptionSession>,
    audio_source: AudioSource,
    audio_track: Arc<TrackLocalStaticRTP>,
    running: Arc<AtomicBool>,
    ffmpeg_pid: Arc<AtomicU32>,
) {
    let is_srt = audio_source.is_srt();

    eprintln!(
        "[Session {}] Starting audio-only session for {} ({})",
        session.id,
        audio_source.display(),
        if is_srt { "SRT stream" } else { "file" }
    );

    let duration_secs = match &audio_source {
        AudioSource::File(path) => vod::get_audio_duration(path).unwrap_or(0.0),
        AudioSource::Srt(_) => 0.0,
    };

    // Create a dummy channel — immediately drop the receiver so audio_pipeline
    // silently discards samples (TrySendError::Disconnected is ignored)
    let (audio_tx, _audio_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(1);

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
        eprintln!("[Session {}] Audio-only session stopped.", session.id);
    } else {
        eprintln!("[Session {}] Audio-only session complete. Duration: {:.2}s", session.id, duration_secs);
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
    growing_segments_config: Option<GrowingSegmentsConfig>,
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

    let is_canary_qwen = model_id == "canary-qwen-2b";
    let is_canary = model_id == "canary-1b" || model_id == "canary-180m-flash" || is_canary_qwen;
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
                is_canary_qwen,
                is_vad_mode,
                mode,
                vad_base_mode,
                vad_model_path,
                language,
                transcription_running,
                parallel_config,
                pause_config,
                growing_segments_config,
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
    is_canary_qwen: bool,
    is_vad_mode: bool,
    mode: String,
    vad_base_mode: String,
    vad_model_path: String,
    language: String,
    transcription_running: Arc<AtomicBool>,
    parallel_config: Option<ParallelConfig>,
    pause_config: Option<PauseConfig>,
    growing_segments_config: Option<GrowingSegmentsConfig>,
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
            is_canary_qwen,
            is_vad_mode,
            mode: mode.clone(),
            vad_base_mode,
            vad_model_path,
            language,
            parallel_config,
            pause_config,
            growing_segments_config,
        },
    ) {
        Some(t) => t,
        None => return,
    };

    let model_type = factory::model_type_name(
        is_pause_parallel_mode,
        is_parallel_mode,
        is_vad_mode,
        is_canary_qwen,
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

    // For canary growing_segments: accumulate confirmed words instead of pushing raw full-buffer text
    let is_growing_canary = mode == "growing_segments" && is_canary && !is_canary_flash;
    let mut confirmed_accumulator = String::new();
    let mut finalized_word_count: usize = 0;
    let mut recent_finals: Vec<String> = Vec::new(); // Track recent FINALs for dedup
    let mut last_growing_sentence = String::new(); // Track previous growing_text for sentence promotion

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

                // Hallucination rejection: skip results from extremely slow inference
                // Large models (canary-qwen 2.5B) legitimately need 30-60s on CPU,
                // so use a higher threshold for them.
                let hallucination_threshold_secs = if is_canary_qwen { 120 } else { 10 };
                let slow_threshold_secs = if is_canary_qwen { 60 } else { 5 };
                if inference_time.as_secs() > hallucination_threshold_secs
                    && !is_parallel_mode
                    && !is_pause_parallel_mode
                {
                    eprintln!(
                        "[Session {}] HALLUCINATION GUARD: inference took {:.1}s, skipping result",
                        transcription_session.id,
                        inference_time.as_secs_f32(),
                    );
                    continue;
                }

                if inference_time.as_secs() > slow_threshold_secs
                    && !is_parallel_mode
                    && !is_pause_parallel_mode
                {
                    eprintln!(
                        "[Session {}] SLOW inference: {:.1}s for {} samples",
                        transcription_session.id,
                        inference_time.as_secs_f32(),
                        all_samples.len()
                    );
                }

                // Apply hallucination truncation and text normalization
                for segment in &mut result.segments {
                    segment.text = emitters::truncate_hallucination_text(&segment.text);
                    segment.text = emitters::normalize_text(&segment.text);
                }

                if is_growing_canary {
                    // Canary growing_segments: accumulate confirmed words, push full growing
                    // text (confirmed + unstable) to the merger once per inference pass.
                    // Filter out fragment FINALs (< 4 words) to avoid noisy short emissions.

                    // Phase 1: Accumulate confirmed words, capture unstable tail
                    let mut unstable_text = String::new();
                    let mut last_segment: Option<&parakeet_rs::streaming_transcriber::TranscriptionSegment> = None;
                    for segment in &result.segments {
                        let text = segment.text.trim();
                        if text.is_empty() {
                            continue;
                        }
                        last_segment = Some(segment);
                        if segment.is_final {
                            if !confirmed_accumulator.is_empty() {
                                confirmed_accumulator.push(' ');
                            }
                            confirmed_accumulator.push_str(text);
                        } else {
                            unstable_text = text.to_string();
                        }
                    }

                    // Need at least one non-empty segment to proceed
                    let Some(ref_segment) = last_segment else {
                        continue;
                    };

                    // Phase 2: Build full growing text from non-finalized confirmed + unstable
                    let acc_words: Vec<&str> = confirmed_accumulator.split_whitespace().collect();
                    let remaining: String = if finalized_word_count < acc_words.len() {
                        acc_words[finalized_word_count..].join(" ")
                    } else {
                        String::new()
                    };
                    let full_text = match (remaining.is_empty(), unstable_text.is_empty()) {
                        (true, true) => String::new(),
                        (true, false) => unstable_text,
                        (false, true) => remaining,
                        (false, false) => format!("{} {}", remaining, unstable_text),
                    };

                    if !full_text.is_empty() {
                        // Phase 3: Push once to growing_merger (as partial — let merger detect sentence boundaries)
                        let prev_finalized = growing_merger.get_finalized_sentences().len();
                        let growing_result = growing_merger.push(&full_text, false);
                        let new_finalized = growing_merger.get_finalized_sentences().len();

                        // Phase 4: Collect FINALs for newly finalized sentences
                        // - Always advance finalized_word_count
                        // - Strip trailing echo fragments (< 3 words after sentence boundary)
                        // - Skip FINALs that overlap with recent emissions (stopword-filtered)
                        // - Suppress fragment FINALs (< 4 words AND < 15 chars)
                        // - Merge consecutive short FINALs (< 5 words) into one
                        let mut collected_finals: Vec<String> = Vec::new();

                        // Stopwords for content-word echo dedup (German + English)
                        let stopwords: std::collections::HashSet<&str> = [
                            "der", "die", "das", "und", "ist", "in", "von", "zu", "mit",
                            "für", "auf", "ein", "eine", "es", "sie", "er", "wir", "ich",
                            "nicht", "auch", "den", "dem", "des", "am", "im", "an", "um",
                            "nach", "bei", "aus", "wie", "oder", "aber", "noch", "wird",
                            "hat", "sind", "war", "dass", "sich", "nur", "so", "vor",
                            "the", "a", "an", "and", "is", "in", "of", "to", "with",
                            "for", "on", "it", "he", "she", "we", "i", "not", "also",
                        ].iter().copied().collect();

                        for i in prev_finalized..new_finalized {
                            let fs = &growing_merger.get_finalized_sentences()[i];
                            let trimmed = fs.text.trim();
                            if trimmed.len() <= 2 || trimmed.chars().all(|c| !c.is_alphanumeric()) {
                                finalized_word_count += fs.text.split_whitespace().count();
                                continue;
                            }
                            finalized_word_count += fs.text.split_whitespace().count();

                            // Strip trailing echo fragments: split by sentence boundaries,
                            // remove trailing parts with < 3 words (echoes from sliding window)
                            let cleaned = {
                                let parts: Vec<&str> = trimmed
                                    .split_inclusive(|c: char| c == '.' || c == '!' || c == '?')
                                    .collect();
                                let count_words = |s: &str| -> usize {
                                    s.trim().split_whitespace()
                                        .filter(|w| w.len() > 1 || w.chars().any(|c| c.is_alphanumeric()))
                                        .count()
                                };
                                let mut end = parts.len();
                                while end > 1 && count_words(parts[end - 1]) < 3 {
                                    end -= 1;
                                }
                                parts[..end].join("").trim().to_string()
                            };

                            // After stripping, re-check minimum length
                            let clean_words = cleaned.split_whitespace().count();
                            if clean_words < 4 && cleaned.len() < 15 {
                                continue;
                            }

                            // Echo dedup: check overlap with recent FINALs (last 8)
                            // Uses stopword-filtered content words for more accurate overlap
                            let cleaned_lower = cleaned.to_lowercase();
                            let curr_all_words: std::collections::HashSet<String> = cleaned_lower
                                .split_whitespace()
                                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                                .filter(|w| !w.is_empty())
                                .collect();
                            let curr_content: std::collections::HashSet<&String> = curr_all_words
                                .iter()
                                .filter(|w| !stopwords.contains(w.as_str()))
                                .collect();

                            let is_echo = recent_finals.iter().rev().take(15).any(|prev| {
                                let prev_lower = prev.to_lowercase();
                                // Exact duplicate
                                if cleaned_lower == prev_lower { return true; }
                                // Substring check for short FINALs
                                if curr_all_words.len() < 4 && prev_lower.contains(&cleaned_lower) { return true; }
                                if curr_all_words.len() < 4 && cleaned_lower.contains(&prev_lower) { return true; }
                                // Content-word overlap (stopword-filtered, threshold 0.65)
                                if curr_content.len() >= 2 {
                                    let prev_all: std::collections::HashSet<String> = prev_lower
                                        .split_whitespace()
                                        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                                        .filter(|w| !w.is_empty())
                                        .collect();
                                    let prev_content: std::collections::HashSet<&String> = prev_all
                                        .iter()
                                        .filter(|w| !stopwords.contains(w.as_str()))
                                        .collect();
                                    if prev_content.is_empty() { return false; }
                                    let common = curr_content.iter()
                                        .filter(|w| prev_content.contains(*w))
                                        .count();
                                    let overlap_curr = common as f64 / curr_content.len() as f64;
                                    let overlap_prev = common as f64 / prev_content.len() as f64;
                                    if overlap_curr > 0.50 || overlap_prev > 0.50 { return true; }
                                }
                                false
                            });

                            if is_echo {
                                continue;
                            }

                            collected_finals.push(cleaned);
                        }

                        // Phase 4a: Merge consecutive short FINALs (< 5 words) into one
                        let mut merged_finals: Vec<String> = Vec::new();
                        for text in collected_finals {
                            let wc = text.split_whitespace().count();
                            if wc < 5 {
                                // Short FINAL — try merging with previous
                                if let Some(last) = merged_finals.last_mut() {
                                    let combined_wc = last.split_whitespace().count() + wc;
                                    if combined_wc <= 15 {
                                        last.push(' ');
                                        last.push_str(&text);
                                        continue;
                                    }
                                }
                            }
                            merged_finals.push(text);
                        }

                        // Emit merged FINALs
                        for cleaned in merged_finals {
                            let normalized = emitters::normalize_text(&cleaned);

                            recent_finals.push(normalized.clone());
                            if recent_finals.len() > 20 {
                                recent_finals.remove(0);
                            }

                            let final_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                                text: normalized,
                                start_time: ref_segment.start_time,
                                end_time: ref_segment.end_time,
                                speaker: ref_segment.speaker,
                                confidence: None,
                                is_final: true,
                                inference_time_ms: ref_segment.inference_time_ms,
                            };
                            emitters::emit_final_subtitle(&transcription_session, &final_segment, &growing_result);
                            // Reset after emitting a real FINAL (prevents double-promotion)
                            last_growing_sentence.clear();
                        }

                        // Phase 4b: Promote unconfirmed sentences
                        // If the growing sentence transitioned (previous ended with .!? and
                        // current is different) but no FINAL was emitted, emit the previous
                        // sentence as a promoted FINAL.
                        if prev_finalized == new_finalized && !last_growing_sentence.is_empty() {
                            let prev_trimmed = last_growing_sentence.trim();
                            let curr_sentence = growing_result.current_sentence.trim();
                            let prev_ends_sentence = prev_trimmed.ends_with('.')
                                || prev_trimmed.ends_with('!')
                                || prev_trimmed.ends_with('?');

                            if prev_ends_sentence && !curr_sentence.is_empty() {
                                // Check word overlap — low overlap means sentence transitioned
                                let prev_words: std::collections::HashSet<&str> =
                                    prev_trimmed.split_whitespace().collect();
                                let curr_words: std::collections::HashSet<&str> =
                                    curr_sentence.split_whitespace().collect();
                                let common = prev_words.intersection(&curr_words).count();
                                let overlap = if prev_words.is_empty() { 1.0 }
                                    else { common as f64 / prev_words.len() as f64 };

                                if overlap < 0.15 && prev_words.len() >= 8 {
                                    // Transition detected — check not already in recent_finals (dedup)
                                    let prev_lower = prev_trimmed.to_lowercase();
                                    let already_emitted = recent_finals.iter().rev().take(15).any(|f| {
                                        let f_lower = f.to_lowercase();
                                        if f_lower == prev_lower { return true; }
                                        // Bag-of-words check
                                        let f_words: std::collections::HashSet<String> = f_lower
                                            .split_whitespace()
                                            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                                            .filter(|w| !w.is_empty())
                                            .collect();
                                        let p_words: std::collections::HashSet<String> = prev_lower
                                            .split_whitespace()
                                            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                                            .filter(|w| !w.is_empty())
                                            .collect();
                                        if f_words.is_empty() || p_words.is_empty() { return false; }
                                        let c = f_words.intersection(&p_words).count();
                                        c as f64 / p_words.len() as f64 > 0.5
                                            || c as f64 / f_words.len() as f64 > 0.5
                                    });

                                    if !already_emitted {
                                        let promoted_text = emitters::normalize_text(prev_trimmed);
                                        eprintln!(
                                            "[Session {} | PROMOTED | Speaker {}] \"{}\"",
                                            transcription_session.id,
                                            ref_segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
                                            &promoted_text.chars().take(80).collect::<String>(),
                                        );

                                        recent_finals.push(promoted_text.clone());
                                        if recent_finals.len() > 20 {
                                            recent_finals.remove(0);
                                        }

                                        let final_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                                            text: promoted_text,
                                            start_time: ref_segment.start_time,
                                            end_time: ref_segment.end_time,
                                            speaker: ref_segment.speaker,
                                            confidence: None,
                                            is_final: true,
                                            inference_time_ms: ref_segment.inference_time_ms,
                                        };
                                        emitters::emit_final_subtitle(
                                            &transcription_session, &final_segment, &growing_result
                                        );
                                    }
                                }
                            }
                        }

                        // Update tracking
                        last_growing_sentence = growing_result.current_sentence.clone();

                        // Phase 5: Emit PARTIAL with full growing text (suppress if stale)
                        let growing_text = &growing_result.current_sentence;
                        let is_stale = emitters::is_stale_partial(growing_text, &recent_finals);

                        if !is_stale {
                            let display_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                                text: full_text,
                                start_time: ref_segment.start_time,
                                end_time: ref_segment.end_time,
                                speaker: ref_segment.speaker,
                                confidence: None,
                                is_final: false,
                                inference_time_ms: ref_segment.inference_time_ms,
                            };
                            emitters::emit_partial_subtitle(
                                &transcription_session, &display_segment, &growing_result
                            );
                        }
                    }
                } else {
                    // Standard path: TDT and non-growing canary modes
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
