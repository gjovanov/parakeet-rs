/*
Quasi-realtime transcription with high-quality TDT model and speaker diarization.

This example demonstrates near-realtime transcription using the ParakeetTDT model
with overlapping chunks. Unlike true streaming, this approach has ~5-10 second
latency but produces much better quality than the EOU streaming model.

Model files required:
- TDT model: encoder-model.onnx, decoder_joint-model.onnx, vocab.txt in a directory
- Diarization: diar_streaming_sortformer_4spk-v2.onnx

Usage:
  # Default (10s chunks, 2s overlap)
  cargo run --release --example realtime_tdt --features sortformer -- audio.wav

  # Low latency mode (5s chunks)
  cargo run --release --example realtime_tdt --features sortformer -- --low-latency audio.wav

  # High quality mode (15s chunks)
  cargo run --release --example realtime_tdt --features sortformer -- --high-quality audio.wav

  # Custom settings
  cargo run --release --example realtime_tdt --features sortformer -- --chunk 8 --overlap 2 audio.wav

  # With GPU acceleration
  cargo run --release --example realtime_tdt --features "sortformer,cuda" -- audio.wav
*/

use clap::Parser;
use std::io::{self, Write};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "realtime_tdt")]
#[command(about = "Quasi-realtime transcription with TDT model")]
struct Args {
    /// Input audio file (WAV, 16kHz)
    audio_file: String,

    /// Path to TDT model directory
    #[arg(long, default_value = ".")]
    tdt_model: String,

    /// Path to diarization model (ONNX)
    #[arg(long, default_value = "diar_streaming_sortformer_4spk-v2.onnx")]
    diar_model: String,

    /// Chunk size in seconds
    #[arg(long, default_value = "10")]
    chunk: f32,

    /// Overlap between chunks in seconds
    #[arg(long, default_value = "2")]
    overlap: f32,

    /// Use low-latency preset (5s chunks)
    #[arg(long)]
    low_latency: bool,

    /// Use high-quality preset (15s chunks)
    #[arg(long)]
    high_quality: bool,

    /// Simulate real-time processing speed
    #[arg(long)]
    simulate_realtime: bool,

    /// Show timing information
    #[arg(long)]
    timing: bool,
}

#[cfg(not(feature = "sortformer"))]
fn main() {
    eprintln!("Error: This example requires the 'sortformer' feature.");
    eprintln!("Please run with: cargo run --example realtime_tdt --features sortformer <audio.wav>");
    std::process::exit(1);
}

#[cfg(feature = "sortformer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Determine config based on presets
    let config = if args.low_latency {
        println!("Using low-latency mode (5s chunks, 1s overlap)");
        parakeet_rs::RealtimeTDTConfig::low_latency()
    } else if args.high_quality {
        println!("Using high-quality mode (15s chunks, 3s overlap)");
        parakeet_rs::RealtimeTDTConfig::high_quality()
    } else {
        parakeet_rs::RealtimeTDTConfig {
            chunk_size_secs: args.chunk,
            overlap_secs: args.overlap,
            min_buffer_secs: args.chunk,
            emit_partials: true,
        }
    };

    println!("Config: {:.1}s chunks, {:.1}s overlap", config.chunk_size_secs, config.overlap_secs);
    println!("Expected latency: ~{:.1}s", config.chunk_size_secs + 1.0);
    println!();

    // Load audio
    println!("Loading audio: {}", args.audio_file);
    let mut reader = hound::WavReader::open(&args.audio_file)?;
    let spec = reader.spec();

    let mut audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    // Convert to mono if needed
    if spec.channels == 2 {
        audio = audio
            .chunks(2)
            .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect();
    }

    let duration = audio.len() as f32 / spec.sample_rate as f32;
    println!(
        "Audio: {:.1}s @ {} Hz, {} samples",
        duration,
        spec.sample_rate,
        audio.len()
    );
    println!();

    // Create transcriber
    println!("Loading TDT model from: {}", args.tdt_model);
    println!("Loading diarization model: {}", args.diar_model);

    let mut transcriber = parakeet_rs::RealtimeTDTDiarized::new(
        &args.tdt_model,
        &args.diar_model,
        None,
        Some(config.clone()),
    )?;

    println!();
    println!("Starting transcription...");
    println!("{}", "=".repeat(60));

    let start = Instant::now();

    // Simulate streaming by feeding audio in small chunks
    let feed_chunk_ms = 100; // Feed 100ms at a time
    let feed_chunk_samples = (feed_chunk_ms as f32 * spec.sample_rate as f32 / 1000.0) as usize;

    let mut last_text_len = 0;

    for chunk in audio.chunks(feed_chunk_samples) {
        // Simulate real-time if requested
        if args.simulate_realtime {
            let target_time = std::time::Duration::from_millis(
                (chunk.len() as f32 / spec.sample_rate as f32 * 1000.0) as u64
            );
            std::thread::sleep(target_time);
        }

        // Process chunk
        let result = transcriber.push_audio(chunk)?;

        // Display new segments
        for segment in &result.segments {
            let timing = if args.timing {
                format!(" [{:.2}s-{:.2}s]", segment.start_time, segment.end_time)
            } else {
                String::new()
            };

            println!(
                "[{}] {}{}",
                segment.speaker_display(),
                segment.text,
                timing
            );
        }

        // Show partial progress
        if result.full_text.len() > last_text_len + 50 {
            eprint!("\rBuffer: {:.1}s, Text: {} chars",
                result.buffer_time, result.full_text.len());
            io::stderr().flush()?;
            last_text_len = result.full_text.len();
        }
    }

    // Finalize
    eprint!("\r\x1b[K"); // Clear progress line
    let final_result = transcriber.finalize()?;

    for segment in &final_result.segments {
        let timing = if args.timing {
            format!(" [{:.2}s-{:.2}s]", segment.start_time, segment.end_time)
        } else {
            String::new()
        };

        println!(
            "[{}] {}{}",
            segment.speaker_display(),
            segment.text,
            timing
        );
    }

    println!("\n{}", "=".repeat(60));

    let elapsed = start.elapsed();
    let rtf = elapsed.as_secs_f32() / duration;

    println!("\nFull transcription:");
    println!("{}", final_result.full_text);

    println!("\n{}", "=".repeat(60));
    println!(
        "Completed in {:.2}s (audio: {:.1}s, RTF: {:.2}x)",
        elapsed.as_secs_f32(),
        duration,
        rtf
    );

    Ok(())
}
