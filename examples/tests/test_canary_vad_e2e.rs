//! End-to-end integration test for Canary VAD transcription
//!
//! This test verifies the full pipeline:
//! 1. Load real audio file
//! 2. Stream audio through VAD segmenter
//! 3. Get transcription results from Canary model
//!
//! Run with: cargo run --release --example test_canary_vad_e2e

use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Canary VAD End-to-End Integration Test ===\n");

    // Check required files
    let canary_path = "./canary";
    let vad_path = "./silero_vad.onnx";
    let audio_path = "./media/broadcast.wav";

    if !std::path::Path::new(canary_path).exists() {
        eprintln!("ERROR: Canary model not found at {}", canary_path);
        return Err("Missing Canary model".into());
    }
    if !std::path::Path::new(vad_path).exists() {
        eprintln!("ERROR: VAD model not found at {}", vad_path);
        return Err("Missing VAD model".into());
    }
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("ERROR: Audio file not found at {}", audio_path);
        return Err("Missing audio file".into());
    }

    println!("Loading models...");
    println!("  Canary: {}", canary_path);
    println!("  VAD: {}", vad_path);
    println!();

    // Use buffered mode for better transcription quality
    // This accumulates 2-3 seconds of speech before transcribing
    let config = RealtimeCanaryVadConfig::buffered("de".to_string());
    println!("Buffered Mode Config:");
    println!("  min_buffer_duration: {:.1}s", config.min_buffer_duration);
    println!("  max_buffer_duration: {:.1}s", config.max_buffer_duration);
    println!("  long_pause_threshold: {:.1}s", config.long_pause_threshold);
    println!();

    // Create the transcriber
    let mut transcriber = RealtimeCanaryVad::new(
        canary_path,
        vad_path,
        None,
        Some(config),
    )?;

    println!("Models loaded successfully!\n");

    // Extract 30 seconds of audio
    println!("Extracting audio from {}...", audio_path);
    let output = Command::new("ffmpeg")
        .args([
            "-i", audio_path,
            "-ss", "60",  // Start at 60s into the file
            "-t", "30",   // 30 seconds
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

    println!("Audio loaded: {} samples ({:.2}s)\n", samples.len(), samples.len() as f32 / 16000.0);

    // Check amplitude WITHOUT normalizing - same as server does
    let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    println!("Max amplitude: {:.4} (NOT normalizing - simulating server behavior)", max_val);

    // Stream audio in chunks (simulating real-time) - NO NORMALIZATION
    println!("\n=== Streaming audio through VAD + Canary (unnormalized like server) ===\n");

    let chunk_size = 1600; // 100ms chunks at 16kHz
    let mut total_transcripts = 0;
    let mut all_text = String::new();

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let result = transcriber.push_audio(chunk)?;

        // Print status every second
        if i % 10 == 0 {
            print!("\rProcessed: {:.1}s, VAD state: {}, speaking: {}     ",
                   result.total_duration, transcriber.vad_state(), result.is_speaking);
        }

        // Print transcription results
        for segment in &result.segments {
            println!("\n\n[TRANSCRIPTION] {:.2}s - {:.2}s:", segment.start_time, segment.end_time);
            println!("  \"{}\"", segment.text);
            all_text.push_str(&segment.text);
            all_text.push(' ');
            total_transcripts += 1;
        }
    }

    // Finalize any remaining audio
    println!("\n\nFinalizing...");
    let final_result = transcriber.finalize()?;
    for segment in &final_result.segments {
        println!("\n[FINAL TRANSCRIPTION] {:.2}s - {:.2}s:", segment.start_time, segment.end_time);
        println!("  \"{}\"", segment.text);
        all_text.push_str(&segment.text);
        all_text.push(' ');
        total_transcripts += 1;
    }

    println!("\n\n=== Test Results ===");
    println!("Total transcription segments: {}", total_transcripts);
    println!("Total duration processed: {:.2}s", transcriber.total_duration());

    if total_transcripts > 0 {
        println!("\n✓ SUCCESS: Canary VAD transcription is working!");
        println!("\nFull transcript:");
        println!("{}", all_text.trim());
    } else {
        println!("\n✗ FAILURE: No transcriptions produced!");
        println!("  This could indicate a problem with VAD or Canary model.");
        return Err("No transcriptions produced".into());
    }

    Ok(())
}
