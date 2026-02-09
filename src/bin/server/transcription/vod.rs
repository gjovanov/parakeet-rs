//! VoD (Video on Demand) batch transcription

use super::AudioSource;
use parakeet_rs::growing_text::GrowingTextMerger;
use parakeet_rs::{SessionState, TranscriptionSession, VodConfig, VodSegment, VodTranscriberCanary, VodTranscriberTDT};
use std::path::PathBuf;
use std::sync::Arc;

/// Run VoD batch transcription
pub fn run_vod_transcription(
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
        (1, 180.0, 15.0)
    } else {
        (1, 180.0, 15.0)
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
            let is_final = true;
            let growing_result = merger.push(&segment.text, is_final);

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
            session_for_segments.set_last_subtitle(subtitle_str.clone());
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
            let transcript_path = {
                let stem = wav_path.file_stem().and_then(|s| s.to_str()).unwrap_or("transcript");
                wav_path.with_file_name(format!("{}.transcript.json", stem))
            };

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
                        rt.block_on(session.set_transcript_path(transcript_path.clone()));
                    }
                }
                Err(e) => {
                    eprintln!("[Session {}] Failed to serialize transcript: {}", session.id, e);
                }
            }

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
pub fn get_audio_duration(path: &std::path::Path) -> Option<f32> {
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
