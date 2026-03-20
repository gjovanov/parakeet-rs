//! Transcription logic for sessions

mod audio_pipeline;
mod configs;
mod emitters;
mod factory;

use crate::api::sessions::{GrowingSegmentsConfig, PauseConfig};
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
        AudioSource::File(_path) => 0.0,
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
    pause_config: Option<PauseConfig>,
    growing_segments_config: Option<GrowingSegmentsConfig>,
    sentence_completion: String,
) {
    use std::sync::mpsc as std_mpsc;

    let is_srt = audio_source.is_srt();

    eprintln!(
        "[Session {}] Starting transcription for {} ({})",
        session.id,
        audio_source.display(),
        if is_srt { "SRT stream" } else { "file" }
    );

    // Get duration using ffprobe (only for files)
    let duration_secs = match &audio_source {
        AudioSource::File(_path) => 0.0,
        AudioSource::Srt(_) => 0.0,
    };

    if !is_srt {
        eprintln!("[Session {}] Total duration: {:.2}s", session.id, duration_secs);
    } else {
        eprintln!("[Session {}] Live stream (duration unknown)", session.id);
    }

    let is_canary = model_id == "canary-1b";

    // Channel to send audio samples to transcription thread
    let (audio_tx, audio_rx) = std_mpsc::sync_channel::<Vec<f32>>(50000);

    let mode = session.mode.clone();
    eprintln!("[Session {}] Using transcription mode: {}", session.id, mode);

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
                mode,
                language,
                transcription_running,
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
    mode: String,
    language: String,
    transcription_running: Arc<AtomicBool>,
    pause_config: Option<PauseConfig>,
    growing_segments_config: Option<GrowingSegmentsConfig>,
    sentence_completion: String,
) {
    use parakeet_rs::sentence_buffer::{SentenceBuffer, SentenceBufferMode};
    use parakeet_rs::streaming_transcriber::StreamingTranscriber;
    use std::sync::mpsc as std_mpsc;

    // Extract growing segments tuning parameters before config is moved to factory
    let gs_echo_dedup_threshold = growing_segments_config.as_ref()
        .and_then(|gs| gs.echo_dedup_threshold).unwrap_or(0.50);
    let gs_echo_dedup_window = growing_segments_config.as_ref()
        .and_then(|gs| gs.echo_dedup_window).unwrap_or(15);
    let gs_min_final_words = growing_segments_config.as_ref()
        .and_then(|gs| gs.min_final_words).unwrap_or(4);
    let gs_promotion_enabled = growing_segments_config.as_ref()
        .and_then(|gs| gs.promotion_enabled).unwrap_or(true);
    let gs_promotion_min_words = growing_segments_config.as_ref()
        .and_then(|gs| gs.promotion_min_words).unwrap_or(8);

    // Create the appropriate transcriber
    let mut transcriber: Box<dyn StreamingTranscriber> = match factory::create_transcriber(
        factory::TranscriberParams {
            session_id: transcription_session.id.clone(),
            model_path,
            diar_path,
            exec_config,
            is_canary,
            mode: mode.clone(),
            language: language.clone(),
            pause_config,
            growing_segments_config,
        },
    ) {
        Some(t) => t,
        None => return,
    };

    let model_type = factory::model_type_name(is_canary);
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
    let is_growing_canary = mode == "growing_segments" && is_canary;
    if is_growing_canary {
        eprintln!(
            "[Session {}] Growing canary pipeline: echo_dedup={:.2}/{}, min_final_words={}, promotion={}/{}words",
            transcription_session.id,
            gs_echo_dedup_threshold, gs_echo_dedup_window, gs_min_final_words,
            if gs_promotion_enabled { "on" } else { "off" }, gs_promotion_min_words,
        );
    }

    let mut confirmed_accumulator = String::new();
    let mut finalized_word_count: usize = 0;
    let mut recent_finals: std::collections::VecDeque<String> = std::collections::VecDeque::new();
    let mut last_growing_sentence = String::new(); // Track previous growing_text for sentence promotion
    let mut pending_fragment = String::new(); // Buffer short fragments for merge with next FINAL
    // Cross-cycle emission buffer: (text, timestamp, start_time, end_time, speaker, inference_ms)
    let mut pending_emissions: std::collections::VecDeque<(String, std::time::Instant, f32, f32, Option<usize>, Option<u32>)> = std::collections::VecDeque::new();

    // Stopwords for content-word echo dedup (German + English) — built once, reused
    let stopwords: std::collections::HashSet<&str> = [
        "der", "die", "das", "und", "ist", "in", "von", "zu", "mit",
        "für", "auf", "ein", "eine", "es", "sie", "er", "wir", "ich",
        "nicht", "auch", "den", "dem", "des", "am", "im", "an", "um",
        "nach", "bei", "aus", "wie", "oder", "aber", "noch", "wird",
        "hat", "sind", "war", "dass", "sich", "nur", "so", "vor",
        "mich", "mir", "dich", "dir", "uns", "ihr", "ihm",
        "kann", "soll", "muss", "will", "darf", "mag",
        "durch", "gegen", "über", "unter", "zwischen", "ohne",
        "the", "a", "an", "and", "is", "in", "of", "to", "with",
        "for", "on", "it", "he", "she", "we", "i", "not", "also",
    ].iter().copied().collect();

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
                let hallucination_threshold_secs: u64 = 10;
                let slow_threshold_secs: u64 = 5;
                if inference_time.as_secs() > hallucination_threshold_secs
                {
                    eprintln!(
                        "[Session {}] HALLUCINATION GUARD: inference took {:.1}s, skipping result",
                        transcription_session.id,
                        inference_time.as_secs_f32(),
                    );
                    continue;
                }

                if inference_time.as_secs() > slow_threshold_secs
                {
                    eprintln!(
                        "[Session {}] SLOW inference: {:.1}s for {} samples",
                        transcription_session.id,
                        inference_time.as_secs_f32(),
                        all_samples.len()
                    );
                }

                // Apply hallucination truncation and text normalization
                let normalize_start = std::time::Instant::now();
                for segment in &mut result.segments {
                    segment.text = emitters::truncate_hallucination_text(&segment.text);
                    segment.text = emitters::normalize_text(&segment.text);
                }
                let normalize_ms = normalize_start.elapsed().as_millis();

                if is_growing_canary {
                    // Canary growing_segments: process model output into text for the merger.
                    // Two paths: WordConfirmer (word-level) or standard (sentence-level).

                    // Extract all text and ref_segment from result
                    let phase1_start = std::time::Instant::now();
                    let mut all_text = String::new();
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
                        if !all_text.is_empty() { all_text.push(' '); }
                        all_text.push_str(text);
                    }

                    // Need at least one non-empty segment to proceed
                    let Some(ref_segment) = last_segment else {
                        continue;
                    };

                    let phase1_ms = phase1_start.elapsed().as_millis();

                    // Phase 2: Build full_text for the merger
                    // remaining confirmed + unstable tail
                    let phase2_start = std::time::Instant::now();
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

                    let phase2_ms = phase2_start.elapsed().as_millis();

                    if !full_text.is_empty() {
                        // Phase 3: Push once to growing_merger
                        // If a pause was detected, push as is_final=true to force finalization
                        let phase3_start = std::time::Instant::now();
                        let pause_detected = transcriber.take_pause_detected();
                        let prev_finalized = growing_merger.get_finalized_sentences().len();
                        let growing_result = growing_merger.push(&full_text, pause_detected);
                        let mut new_finalized = growing_merger.get_finalized_sentences().len();

                        // If pause detected but merger didn't finalize (e.g., text doesn't end
                        // with punctuation), force-flush the working buffer
                        if pause_detected && new_finalized == prev_finalized {
                            if let Some(flushed) = growing_merger.flush() {
                                if flushed.split_whitespace().count() >= 3 {
                                    // Re-read finalized count after flush
                                    new_finalized = growing_merger.get_finalized_sentences().len();
                                }
                            }
                        }
                        let phase3_ms = phase3_start.elapsed().as_millis();

                        // Phase 4: Collect FINALs for newly finalized sentences
                        // - Always advance finalized_word_count
                        // - Strip trailing echo fragments (< 3 words after sentence boundary)
                        // - Skip FINALs that overlap with recent emissions (stopword-filtered)
                        // - Suppress fragment FINALs (< 4 words AND < 15 chars)
                        // - Merge consecutive short FINALs (< 5 words) into one
                        let phase4_start = std::time::Instant::now();
                        let mut collected_finals: Vec<String> = Vec::new();

                        for i in prev_finalized..new_finalized {
                            let fs = &growing_merger.get_finalized_sentences()[i];
                            let trimmed = fs.text.trim();
                            if trimmed.len() <= 2 || trimmed.chars().all(|c| !c.is_alphanumeric()) {
                                finalized_word_count += fs.text.split_whitespace().count();
                                continue;
                            }
                            finalized_word_count += fs.text.split_whitespace().count();

                            // Strip trailing echo fragments: split by sentence boundaries,
                            // remove trailing parts with < 3 words UNLESS they end with .!?
                            // (short complete sentences like "Genau." should be preserved)
                            let cleaned = {
                                let parts: Vec<&str> = trimmed
                                    .split_inclusive(|c: char| c == '.' || c == '!' || c == '?')
                                    .collect();
                                let count_words = |s: &str| -> usize {
                                    s.trim().split_whitespace()
                                        .filter(|w| w.len() > 1 || w.chars().any(|c| c.is_alphanumeric()))
                                        .count()
                                };
                                let ends_with_terminator = |s: &str| -> bool {
                                    let t = s.trim();
                                    t.ends_with('.') || t.ends_with('!') || t.ends_with('?')
                                };
                                let mut end = parts.len();
                                while end > 1
                                    && count_words(parts[end - 1]) < 3
                                    && !ends_with_terminator(parts[end - 1])
                                {
                                    end -= 1;
                                }
                                parts[..end].join("").trim().to_string()
                            };

                            // After stripping, re-check minimum length
                            let clean_words = cleaned.split_whitespace().count();
                            if clean_words < gs_min_final_words && cleaned.len() < 15 {
                                continue;
                            }

                            // Echo dedup: check overlap with recent FINALs
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

                            let is_echo = recent_finals.iter().rev().take(gs_echo_dedup_window).any(|prev| {
                                let prev_lower = prev.to_lowercase();
                                // Exact duplicate
                                if cleaned_lower == prev_lower { return true; }
                                // Substring check for short FINALs
                                if curr_all_words.len() < gs_min_final_words && prev_lower.contains(&cleaned_lower) { return true; }
                                if curr_all_words.len() < gs_min_final_words && cleaned_lower.contains(&prev_lower) { return true; }
                                // Content-word overlap (stopword-filtered, conjunctive threshold)
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
                                    // Asymmetric dedup: if either side has strong overlap (>0.65)
                                    // AND both sides have at least minimal overlap (>0.30),
                                    // it's an echo. This catches long-vs-short FINAL pairs.
                                    let high_threshold = gs_echo_dedup_threshold.max(0.50);
                                    let low_threshold = gs_echo_dedup_threshold * 0.60; // ~0.30 for default 0.50
                                    if (overlap_curr > high_threshold || overlap_prev > high_threshold)
                                        && overlap_curr > low_threshold
                                        && overlap_prev > low_threshold
                                    { return true; }
                                }
                                false
                            });

                            if is_echo {
                                continue;
                            }

                            collected_finals.push(cleaned);
                        }

                        // Phase 4a-i: Intra-FINAL echo detection
                        // If a FINAL contains consecutive sentences with >60% word overlap,
                        // keep only the longer one (sliding window produces two wordings of same content)
                        let collected_finals: Vec<String> = collected_finals.into_iter().map(|text| {
                            let sentences: Vec<&str> = text
                                .split_inclusive(|c: char| c == '.' || c == '!' || c == '?')
                                .collect();
                            if sentences.len() < 2 {
                                return text;
                            }
                            let mut kept: Vec<&str> = vec![sentences[0]];
                            for i in 1..sentences.len() {
                                let prev_words: std::collections::HashSet<&str> = kept.last().unwrap()
                                    .split_whitespace()
                                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                                    .filter(|w| !w.is_empty())
                                    .collect();
                                let curr_words: std::collections::HashSet<&str> = sentences[i]
                                    .split_whitespace()
                                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
                                    .filter(|w| !w.is_empty())
                                    .collect();
                                if prev_words.len() >= 3 && curr_words.len() >= 3 {
                                    let common = prev_words.intersection(&curr_words).count();
                                    let overlap = common as f64 / prev_words.len().min(curr_words.len()) as f64;
                                    if overlap > 0.60 {
                                        // Duplicate sentence — keep the longer one
                                        if curr_words.len() > prev_words.len() {
                                            *kept.last_mut().unwrap() = sentences[i];
                                        }
                                        continue; // skip adding this sentence
                                    }
                                }
                                kept.push(sentences[i]);
                            }
                            kept.join("").trim().to_string()
                        }).filter(|t| !t.is_empty()).collect();

                        // Phase 4a-ii: Strip leading punctuation/fragments
                        // FINALs like ". Tante Bertuti am 15." have punctuation leaking from previous segment
                        let collected_finals: Vec<String> = collected_finals.into_iter().map(|text| {
                            text.trim_start_matches(|c: char| c == '.' || c == ',' || c == ';' || c == ':' || c == ' ')
                                .trim()
                                .to_string()
                        }).filter(|t| !t.is_empty() && t.split_whitespace().count() >= 2).collect();

                        // Phase 4a-iii: Merge consecutive short FINALs (< 8 words) into one
                        let mut merged_finals: Vec<String> = Vec::new();
                        for text in collected_finals {
                            let wc = text.split_whitespace().count();
                            if wc < 8 {
                                // Short FINAL — try merging with previous
                                if let Some(last) = merged_finals.last_mut() {
                                    let combined_wc = last.split_whitespace().count() + wc;
                                    if combined_wc <= 25 {
                                        last.push(' ');
                                        last.push_str(&text);
                                        continue;
                                    }
                                }
                            }
                            merged_finals.push(text);
                        }

                        // Phase 4a-iv: Buffer fragment FINALs (< min_final_words) for merge with next cycle
                        // Instead of emitting tiny fragments standalone, hold them and prepend to next FINAL
                        let mut final_to_emit: Vec<String> = Vec::new();
                        for text in merged_finals {
                            let wc = text.split_whitespace().count();
                            if wc < gs_min_final_words {
                                // Buffer this fragment
                                pending_fragment.push_str(if pending_fragment.is_empty() { "" } else { " " });
                                pending_fragment.push_str(&text);
                            } else {
                                // Prepend any buffered fragment
                                if !pending_fragment.is_empty() {
                                    let mut combined = std::mem::take(&mut pending_fragment);
                                    combined.push(' ');
                                    combined.push_str(&text);
                                    final_to_emit.push(combined);
                                } else {
                                    final_to_emit.push(text);
                                }
                            }
                        }

                        let phase4_dedup_ms = phase4_start.elapsed().as_millis();

                        // Phase 4a-v: Cross-cycle FINAL buffering
                        // Add new FINALs to pending_emissions buffer with timestamps.
                        // FINALs are held for 2s to allow merging with adjacent-cycle fragments.
                        let now = std::time::Instant::now();
                        for cleaned in final_to_emit {
                            let normalized = emitters::normalize_text(&cleaned);
                            // Try to merge with the most recent pending emission
                            let merged = if let Some((ref mut prev_text, ref prev_time, ..)) = pending_emissions.back_mut() {
                                let prev_wc = prev_text.split_whitespace().count();
                                let curr_wc = normalized.split_whitespace().count();
                                // Merge if both are short and recent
                                if (prev_wc < 10 || curr_wc < 10) && prev_wc + curr_wc <= 30
                                    && now.duration_since(*prev_time).as_millis() < 2000
                                {
                                    prev_text.push(' ');
                                    prev_text.push_str(&normalized);
                                    true
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                            if !merged {
                                pending_emissions.push_back((normalized, now, ref_segment.start_time, ref_segment.end_time, ref_segment.speaker, ref_segment.inference_time_ms));
                            }
                        }

                        // Emit any pending FINALs that are older than 2 seconds
                        let emit_start = std::time::Instant::now();
                        let emission_delay = std::time::Duration::from_secs(2);
                        while let Some((_text, time, _start_t, _end_t, _speaker, _inf_ms)) = pending_emissions.front() {
                            if now.duration_since(*time) < emission_delay {
                                break; // Not old enough yet
                            }
                            let (text, _time, start_t, end_t, speaker, inf_ms) = pending_emissions.pop_front().unwrap();

                            recent_finals.push_back(text.clone());
                            if recent_finals.len() > 20 {
                                recent_finals.pop_front();
                            }

                            let final_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                                text: text.clone(),
                                raw_text: None,
                                start_time: start_t,
                                end_time: end_t,
                                speaker,
                                confidence: None,
                                is_final: true,
                                inference_time_ms: inf_ms,
                            };
                            emitters::emit_final_subtitle(&transcription_session, &final_segment, &growing_result);

                            last_growing_sentence.clear();
                        }
                        let emit_ms = emit_start.elapsed().as_millis();

                        // Phase 4b: Promote unconfirmed sentences
                        // If the growing sentence transitioned (previous ended with .!? and
                        // current is different) but no FINAL was emitted, emit the previous
                        // sentence as a promoted FINAL.
                        if gs_promotion_enabled && prev_finalized == new_finalized && !last_growing_sentence.is_empty() {
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

                                if overlap < 0.15 && prev_words.len() >= gs_promotion_min_words {
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
                                        let display_text = emitters::normalize_text(prev_trimmed);

                                        eprintln!(
                                            "[Session {} | PROMOTED | Speaker {}] \"{}\"",
                                            transcription_session.id,
                                            ref_segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
                                            &display_text.chars().take(80).collect::<String>(),
                                        );

                                        recent_finals.push_back(display_text.clone());
                                        if recent_finals.len() > 20 {
                                            recent_finals.pop_front();
                                        }

                                        let final_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                                            text: display_text,
                                            raw_text: None,
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

                        // Log step-by-step timing
                        let total_pipeline_ms = inference_start.elapsed().as_millis();
                        eprintln!(
                            "[Session {}] [Timing] inference={}ms normalize={}ms P1={}ms P2={}ms P3_merger={}ms P4_dedup={}ms emit={}ms total={}ms",
                            transcription_session.id,
                            inference_time_ms,
                            normalize_ms,
                            phase1_ms,
                            phase2_ms,
                            phase3_ms,
                            phase4_dedup_ms,
                            emit_ms,
                            total_pipeline_ms,
                        );

                        // Phase 5: Emit PARTIAL with full growing text (suppress if stale)
                        let partial_text = full_text;
                        let recent_finals_vec: Vec<String> = recent_finals.iter().cloned().collect();
                        let is_stale = emitters::is_stale_partial(&partial_text, &recent_finals_vec);

                        if !is_stale {
                            let display_segment = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                                text: partial_text,
                                raw_text: None,
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
                                raw_text: None,
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

    // Flush any remaining pending emissions
    if is_growing_canary && !pending_emissions.is_empty() {
        let growing_result = growing_merger.push("", false);
        for (text, _time, start_t, end_t, speaker, inf_ms) in pending_emissions.drain(..) {
            recent_finals.push_back(text.clone());
            if recent_finals.len() > 20 { recent_finals.pop_front(); }
            let seg = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                text, raw_text: None, start_time: start_t, end_time: end_t,
                speaker, confidence: None, is_final: true, inference_time_ms: inf_ms,
            };
            emitters::emit_final_subtitle(&transcription_session, &seg, &growing_result);
        }
        // Also flush pending fragment
        if !pending_fragment.is_empty() {
            let text = std::mem::take(&mut pending_fragment);
            let seg = parakeet_rs::streaming_transcriber::TranscriptionSegment {
                text, raw_text: None, start_time: 0.0, end_time: 0.0,
                speaker: None, confidence: None, is_final: true, inference_time_ms: None,
            };
            emitters::emit_final_subtitle(&transcription_session, &seg, &growing_result);
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
            raw_text: None,
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
