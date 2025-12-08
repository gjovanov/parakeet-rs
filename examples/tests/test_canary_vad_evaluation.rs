//! Canary VAD Pause-Based Evaluation Test
//!
//! This test evaluates the Canary VAD pause-based transcription against reference transcripts.
//! Goal: Each segment should be approximately one sentence (max 1-2 commas).
//!
//! Test files:
//! - media/VOD_2025_1203_1900_ORF_OOE_broadcast.wav + transcript.corrected.json
//! - media/VOD_2025_1203_1900_ORF_S_broadcast.wav + transcript.corrected.json
//!
//! Run with: cargo run --release --example test_canary_vad_evaluation

use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};
use parakeet_rs::vad::VadConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::process::Command;

/// Word with timestamp from reference transcript
#[derive(Debug, Deserialize, Clone)]
struct Word {
    word: String,
    start: f64,
    end: f64,
    #[allow(dead_code)]
    probability: f64,
}

/// Segment from reference transcript
#[derive(Debug, Deserialize)]
struct ReferenceSegment {
    text: String,
    words: Vec<Word>,
}

/// Full reference transcript
#[derive(Debug, Deserialize)]
struct ReferenceTranscript {
    text: String,
    segments: Vec<ReferenceSegment>,
}

/// A produced segment from our transcriber
#[derive(Debug, Clone, Serialize)]
struct ProducedSegment {
    text: String,
    start_time: f32,
    end_time: f32,
    comma_count: usize,
    word_count: usize,
}

/// Test result for a single audio file
#[derive(Debug, Serialize)]
struct TestResult {
    audio_file: String,
    total_segments: usize,
    avg_segment_duration_secs: f32,
    avg_commas_per_segment: f32,
    segments_with_0_commas: usize,
    segments_with_1_comma: usize,
    segments_with_2_commas: usize,
    segments_with_3plus_commas: usize,
    total_audio_duration_secs: f32,
    segments: Vec<ProducedSegment>,
}

/// Evaluation metrics
#[derive(Debug, Serialize)]
struct EvaluationMetrics {
    comma_distribution_score: f32,  // Higher is better (more segments with 0-2 commas)
    avg_segment_duration: f32,
    segment_duration_stddev: f32,
    total_segments: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Canary VAD Pause-Based Evaluation Test ===\n");

    // Check required files
    let canary_path = "./canary";
    let vad_path = "./silero_vad.onnx";

    if !std::path::Path::new(canary_path).exists() {
        eprintln!("ERROR: Canary model not found at {}", canary_path);
        return Err("Missing Canary model".into());
    }
    if !std::path::Path::new(vad_path).exists() {
        eprintln!("ERROR: VAD model not found at {}", vad_path);
        return Err("Missing VAD model".into());
    }

    // Test files
    let test_files = vec![
        (
            "./media/VOD_2025_1203_1900_ORF_OOE_broadcast.wav",
            "./media/VOD_2025_1203_1900_ORF_OOE_transcript.corrected.json",
        ),
        (
            "./media/VOD_2025_1203_1900_ORF_S_broadcast.wav",
            "./media/VOD_2025_1203_1900_ORF_S_transcript.corrected.json",
        ),
    ];

    let mut all_results: Vec<TestResult> = Vec::new();

    for (audio_path, transcript_path) in &test_files {
        if !std::path::Path::new(audio_path).exists() {
            eprintln!("WARNING: Audio file not found: {}", audio_path);
            continue;
        }
        if !std::path::Path::new(transcript_path).exists() {
            eprintln!("WARNING: Transcript file not found: {}", transcript_path);
            continue;
        }

        println!("\n{}", "=".repeat(60));
        println!("Testing: {}", audio_path);
        println!("{}\n", "=".repeat(60));

        // Load reference transcript for comparison
        let transcript_json = fs::read_to_string(transcript_path)?;
        let reference: ReferenceTranscript = serde_json::from_str(&transcript_json)?;
        println!("Reference transcript loaded:");
        println!("  - Total text length: {} chars", reference.text.len());
        println!("  - Segments: {}", reference.segments.len());

        // Get reference sentences (split by period for comparison)
        let reference_sentences: Vec<&str> = reference.text
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        println!("  - Estimated sentences (by period): {}", reference_sentences.len());

        // Create VAD config - using pause_based mode
        let vad_config = VadConfig::pause_based();
        println!("\nVAD Config (pause_based):");
        println!("  speech_threshold: {}", vad_config.speech_threshold);
        println!("  silence_trigger_ms: {}", vad_config.silence_trigger_ms);
        println!("  max_speech_secs: {}", vad_config.max_speech_secs);
        println!("  speech_pad_start_ms: {}", vad_config.speech_pad_start_ms);
        println!("  speech_pad_end_ms: {}", vad_config.speech_pad_end_ms);
        println!("  max_pauses: {}", vad_config.max_pauses);

        let config = RealtimeCanaryVadConfig {
            vad: vad_config,
            language: "de".to_string(),
            min_buffer_duration: 0.0,  // Immediate mode - transcribe each VAD segment
            max_buffer_duration: 15.0,
            long_pause_threshold: 1.5,
        };

        // Create the transcriber
        println!("\nLoading models...");
        let mut transcriber = RealtimeCanaryVad::new(
            canary_path,
            vad_path,
            None,
            Some(config),
        )?;
        println!("Models loaded successfully!");

        // Load audio using ffmpeg
        println!("\nLoading audio from {}...", audio_path);
        let output = Command::new("ffmpeg")
            .args([
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "f32le",
                "-"
            ])
            .output()?;

        if !output.status.success() {
            eprintln!("FFmpeg error: {}", String::from_utf8_lossy(&output.stderr));
            return Err("FFmpeg failed".into());
        }

        // Convert bytes to f32 samples
        let bytes = output.stdout;
        let samples: Vec<f32> = bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let total_duration = samples.len() as f32 / 16000.0;
        println!("Audio loaded: {} samples ({:.2}s)", samples.len(), total_duration);

        // Stream audio in chunks
        println!("\n=== Processing audio through Canary VAD ===\n");

        let chunk_size = 1600; // 100ms chunks at 16kHz
        let mut produced_segments: Vec<ProducedSegment> = Vec::new();

        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            let result = transcriber.push_audio(chunk)?;

            // Print progress every 5 seconds
            if i % 50 == 0 {
                print!("\rProcessed: {:.1}s / {:.1}s, VAD state: {}, segments: {}     ",
                       result.total_duration, total_duration,
                       transcriber.vad_state(), produced_segments.len());
            }

            // Collect transcription results
            for segment in &result.segments {
                let comma_count = segment.text.matches(',').count();
                let word_count = segment.text.split_whitespace().count();

                produced_segments.push(ProducedSegment {
                    text: segment.text.clone(),
                    start_time: segment.start_time,
                    end_time: segment.end_time,
                    comma_count,
                    word_count,
                });

                println!("\n[SEGMENT {}] {:.2}s - {:.2}s (commas: {}, words: {})",
                         produced_segments.len(),
                         segment.start_time, segment.end_time,
                         comma_count, word_count);
                println!("  \"{}\"", segment.text);
            }
        }

        // Finalize
        println!("\n\nFinalizing...");
        let final_result = transcriber.finalize()?;
        for segment in &final_result.segments {
            let comma_count = segment.text.matches(',').count();
            let word_count = segment.text.split_whitespace().count();

            produced_segments.push(ProducedSegment {
                text: segment.text.clone(),
                start_time: segment.start_time,
                end_time: segment.end_time,
                comma_count,
                word_count,
            });

            println!("\n[FINAL SEGMENT {}] {:.2}s - {:.2}s (commas: {}, words: {})",
                     produced_segments.len(),
                     segment.start_time, segment.end_time,
                     comma_count, word_count);
            println!("  \"{}\"", segment.text);
        }

        // Calculate statistics
        let total_segments = produced_segments.len();
        let avg_duration = if total_segments > 0 {
            produced_segments.iter()
                .map(|s| s.end_time - s.start_time)
                .sum::<f32>() / total_segments as f32
        } else {
            0.0
        };

        let avg_commas = if total_segments > 0 {
            produced_segments.iter()
                .map(|s| s.comma_count as f32)
                .sum::<f32>() / total_segments as f32
        } else {
            0.0
        };

        let segments_0_commas = produced_segments.iter().filter(|s| s.comma_count == 0).count();
        let segments_1_comma = produced_segments.iter().filter(|s| s.comma_count == 1).count();
        let segments_2_commas = produced_segments.iter().filter(|s| s.comma_count == 2).count();
        let segments_3plus = produced_segments.iter().filter(|s| s.comma_count >= 3).count();

        let result = TestResult {
            audio_file: audio_path.to_string(),
            total_segments,
            avg_segment_duration_secs: avg_duration,
            avg_commas_per_segment: avg_commas,
            segments_with_0_commas: segments_0_commas,
            segments_with_1_comma: segments_1_comma,
            segments_with_2_commas: segments_2_commas,
            segments_with_3plus_commas: segments_3plus,
            total_audio_duration_secs: total_duration,
            segments: produced_segments,
        };

        // Print summary
        println!("\n\n{}", "=".repeat(60));
        println!("RESULTS: {}", audio_path);
        println!("{}", "=".repeat(60));
        println!("Total segments produced: {}", result.total_segments);
        println!("Average segment duration: {:.2}s", result.avg_segment_duration_secs);
        println!("Average commas per segment: {:.2}", result.avg_commas_per_segment);
        println!("\nComma distribution:");
        println!("  0 commas: {} segments ({:.1}%)",
                 result.segments_with_0_commas,
                 100.0 * result.segments_with_0_commas as f32 / total_segments.max(1) as f32);
        println!("  1 comma:  {} segments ({:.1}%)",
                 result.segments_with_1_comma,
                 100.0 * result.segments_with_1_comma as f32 / total_segments.max(1) as f32);
        println!("  2 commas: {} segments ({:.1}%)",
                 result.segments_with_2_commas,
                 100.0 * result.segments_with_2_commas as f32 / total_segments.max(1) as f32);
        println!("  3+ commas: {} segments ({:.1}%) <- GOAL: minimize this",
                 result.segments_with_3plus_commas,
                 100.0 * result.segments_with_3plus_commas as f32 / total_segments.max(1) as f32);

        // Calculate score (higher is better)
        let good_segments = segments_0_commas + segments_1_comma + segments_2_commas;
        let score = 100.0 * good_segments as f32 / total_segments.max(1) as f32;
        println!("\nSCORE (segments with 0-2 commas): {:.1}%", score);

        if score >= 80.0 {
            println!("  ✓ PASS: Most segments are sentence-level");
        } else if score >= 60.0 {
            println!("  ~ PARTIAL: Some segments need tuning");
        } else {
            println!("  ✗ FAIL: Too many multi-sentence segments");
        }

        all_results.push(result);
    }

    // Overall summary
    println!("\n\n{}", "=".repeat(60));
    println!("OVERALL SUMMARY");
    println!("{}", "=".repeat(60));

    let total_good = all_results.iter()
        .map(|r| r.segments_with_0_commas + r.segments_with_1_comma + r.segments_with_2_commas)
        .sum::<usize>();
    let total_all = all_results.iter()
        .map(|r| r.total_segments)
        .sum::<usize>();
    let overall_score = 100.0 * total_good as f32 / total_all.max(1) as f32;

    println!("Total segments across all files: {}", total_all);
    println!("Segments with 0-2 commas: {} ({:.1}%)", total_good, overall_score);
    println!("Segments with 3+ commas: {} ({:.1}%)",
             total_all - total_good,
             100.0 - overall_score);

    let overall_avg_commas: f32 = all_results.iter()
        .map(|r| r.avg_commas_per_segment * r.total_segments as f32)
        .sum::<f32>() / total_all.max(1) as f32;
    println!("Overall average commas per segment: {:.2}", overall_avg_commas);

    // Save results to JSON
    let results_path = "./media/vad_evaluation_results.json";
    let results_json = serde_json::to_string_pretty(&all_results)?;
    fs::write(results_path, &results_json)?;
    println!("\nDetailed results saved to: {}", results_path);

    // Final verdict
    println!("\n{}", "=".repeat(60));
    if overall_score >= 80.0 && overall_avg_commas <= 2.0 {
        println!("OVERALL: ✓ PASS - VAD pause_based is working well for sentence-level segmentation");
    } else {
        println!("OVERALL: ✗ NEEDS TUNING - Adjust VAD parameters for better sentence boundaries");
        println!("\nSuggestions:");
        if overall_avg_commas > 2.0 {
            println!("  - Decrease silence_trigger_ms to segment on shorter pauses");
            println!("  - Decrease max_pauses to trigger transcription earlier");
            println!("  - Decrease max_speech_secs to force shorter segments");
        }
    }

    Ok(())
}
