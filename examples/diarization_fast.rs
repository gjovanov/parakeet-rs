/*
Fast Speaker Diarization with Parallel Transcription

Optimized for long audio files (30-60+ minutes):
- Sortformer v2: Processes full audio in streaming mode (handles any length)
- Parakeet-TDT: Chunks audio into ~5 min segments, processes in parallel
- GPU acceleration: Use --features cuda or --features tensorrt for 5-10x speedup

Usage:
cargo run --release --example diarization_fast --features sortformer audio.wav
cargo run --release --example diarization_fast --features "sortformer,cuda" audio.wav

Options (via environment variables):
- CHUNK_DURATION_SECS: Chunk size in seconds (default: 300 = 5 minutes)
- CHUNK_OVERLAP_SECS: Overlap between chunks (default: 2 seconds)
- PARALLEL_WORKERS: Number of parallel TDT instances (default: num_cpus, use 1 for low memory)
- INTRA_THREADS: Threads per ONNX session (default: 4)
*/

#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use parakeet_rs::sortformer::{DiarizationConfig, Sortformer};
#[cfg(feature = "sortformer")]
use parakeet_rs::{ExecutionConfig, ExecutionProvider, TimestampMode, Transcriber};
#[cfg(feature = "sortformer")]
use rayon::prelude::*;
#[cfg(feature = "sortformer")]
use std::env;
#[cfg(feature = "sortformer")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "sortformer")]
use std::sync::Arc;
#[cfg(feature = "sortformer")]
use std::time::Instant;

#[cfg(feature = "sortformer")]
use parakeet_rs::TimedToken;

#[cfg(feature = "sortformer")]
fn get_env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "sortformer")]
fn get_execution_config() -> ExecutionConfig {
    let intra_threads = get_env_or("INTRA_THREADS", 4);

    let provider = {
        #[cfg(feature = "tensorrt")]
        {
            ExecutionProvider::TensorRT
        }
        #[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
        {
            ExecutionProvider::Cuda
        }
        #[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
        {
            ExecutionProvider::Cpu
        }
    };

    ExecutionConfig::new()
        .with_execution_provider(provider)
        .with_intra_threads(intra_threads)
}

#[cfg(feature = "sortformer")]
struct AudioChunk {
    samples: Vec<f32>,
    start_time: f32,
    chunk_index: usize,
}

#[cfg(feature = "sortformer")]
fn chunk_audio(
    audio: &[f32],
    sample_rate: u32,
    chunk_duration_secs: f32,
    overlap_secs: f32,
) -> Vec<AudioChunk> {
    let samples_per_chunk = (chunk_duration_secs * sample_rate as f32) as usize;
    let overlap_samples = (overlap_secs * sample_rate as f32) as usize;
    let step_samples = samples_per_chunk - overlap_samples;

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < audio.len() {
        let end = (start + samples_per_chunk).min(audio.len());
        let samples = audio[start..end].to_vec();
        let start_time = start as f32 / sample_rate as f32;

        chunks.push(AudioChunk {
            samples,
            start_time,
            chunk_index,
        });

        if end >= audio.len() {
            break;
        }

        start += step_samples;
        chunk_index += 1;
    }

    chunks
}

#[cfg(feature = "sortformer")]
fn merge_transcription_results(
    mut chunk_results: Vec<(usize, f32, Vec<TimedToken>)>,
    overlap_secs: f32,
) -> Vec<TimedToken> {
    // Sort by chunk index
    chunk_results.sort_by_key(|(idx, _, _)| *idx);

    let mut merged: Vec<TimedToken> = Vec::new();

    for (chunk_idx, time_offset, tokens) in chunk_results {
        for mut token in tokens {
            // Adjust timestamps to absolute time
            token.start += time_offset;
            token.end += time_offset;

            // For overlapping regions, skip tokens that overlap with already processed ones
            if chunk_idx > 0 && !merged.is_empty() {
                let last_end = merged.last().map(|t| t.end).unwrap_or(0.0);
                // Skip tokens that start before the overlap threshold
                if token.start < last_end - overlap_secs * 0.5 {
                    continue;
                }
                // If token overlaps partially, adjust or skip based on content similarity
                if token.start < last_end {
                    // Check if this looks like a duplicate (similar timing)
                    if let Some(last) = merged.last() {
                        if (token.start - last.start).abs() < 0.5
                            && token.text.trim() == last.text.trim()
                        {
                            continue; // Skip duplicate
                        }
                    }
                }
            }

            merged.push(token);
        }
    }

    merged
}

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!(
            "Run with: cargo run --release --example diarization_fast --features sortformer <audio.wav>"
        );
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let start_time = Instant::now();
        let args: Vec<String> = env::args().collect();
        let audio_path = args.get(1).expect(
            "Usage: cargo run --release --example diarization_fast --features sortformer <audio.wav>",
        );

        // Configuration
        let chunk_duration_secs: f32 = get_env_or("CHUNK_DURATION_SECS", 300.0); // 5 minutes
        let chunk_overlap_secs: f32 = get_env_or("CHUNK_OVERLAP_SECS", 2.0);
        let parallel_workers: usize = get_env_or("PARALLEL_WORKERS", num_cpus::get().min(4));
        let exec_config = get_execution_config();

        println!("{}", "=".repeat(80));
        println!("FAST DIARIZATION MODE");
        println!(
            "• Chunk size: {:.0}s, Overlap: {:.1}s, Workers: {}",
            chunk_duration_secs, chunk_overlap_secs, parallel_workers
        );
        println!("• Execution: {:?}", exec_config.execution_provider);
        println!("{}", "=".repeat(80));

        // Step 1: Load audio
        println!("\nStep 1/4: Loading audio...");
        let mut reader = hound::WavReader::open(audio_path)?;
        let spec = reader.spec();

        let audio: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|s| s as f32 / 32768.0))
                .collect::<Result<Vec<_>, _>>()?,
        };

        // Convert stereo to mono if needed
        let audio: Vec<f32> = if spec.channels == 2 {
            audio
                .chunks(2)
                .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
                .collect()
        } else {
            audio
        };

        let duration = audio.len() as f32 / spec.sample_rate as f32;
        println!(
            "Loaded {:.1}s of audio ({} samples @ {} Hz)",
            duration,
            audio.len(),
            spec.sample_rate
        );

        // Step 2: Diarization (full audio, streaming)
        println!("\nStep 2/4: Running Sortformer diarization (streaming)...");
        let diar_start = Instant::now();

        let mut sortformer = Sortformer::with_config(
            "diar_streaming_sortformer_4spk-v2.onnx",
            Some(exec_config.clone()),
            DiarizationConfig::callhome(),
        )?;

        let speaker_segments = sortformer.diarize(audio.clone(), spec.sample_rate, 1)?;
        println!(
            "Found {} speaker segments in {:.2}s",
            speaker_segments.len(),
            diar_start.elapsed().as_secs_f32()
        );

        // Step 3: Chunk audio and transcribe in parallel
        println!("\nStep 3/4: Transcribing chunks in parallel...");
        let trans_start = Instant::now();

        let chunks = chunk_audio(&audio, spec.sample_rate, chunk_duration_secs, chunk_overlap_secs);
        let total_chunks = chunks.len();
        println!("Split into {} chunks", total_chunks);

        // Progress counter
        let completed = Arc::new(AtomicUsize::new(0));

        // Set rayon thread pool size
        rayon::ThreadPoolBuilder::new()
            .num_threads(parallel_workers)
            .build_global()
            .ok(); // Ignore if already initialized

        // Process chunks in parallel
        let chunk_results: Vec<_> = chunks
            .into_par_iter()
            .map(|chunk| {
                // Each thread creates its own TDT instance
                let mut parakeet =
                    parakeet_rs::ParakeetTDT::from_pretrained("./tdt", Some(exec_config.clone()))
                        .expect("Failed to load TDT model");

                let result = parakeet
                    .transcribe_samples(
                        chunk.samples,
                        spec.sample_rate,
                        1, // mono
                        Some(TimestampMode::Sentences),
                    )
                    .ok();

                let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                eprint!("\r  Progress: {}/{} chunks", done, total_chunks);

                result.map(|r| (chunk.chunk_index, chunk.start_time, r.tokens))
            })
            .collect();

        eprintln!(); // New line after progress

        // Merge results
        let valid_results: Vec<_> = chunk_results.into_iter().flatten().collect();
        let all_tokens = merge_transcription_results(valid_results, chunk_overlap_secs);

        println!(
            "Transcribed {} sentences in {:.2}s",
            all_tokens.len(),
            trans_start.elapsed().as_secs_f32()
        );

        // Step 4: Attribute speakers
        println!("\nStep 4/4: Attributing speakers...\n");

        for token in &all_tokens {
            // Find speaker with maximum overlap
            let speaker = speaker_segments
                .iter()
                .filter_map(|s| {
                    let overlap_start = token.start.max(s.start);
                    let overlap_end = token.end.min(s.end);
                    let overlap = (overlap_end - overlap_start).max(0.0);
                    if overlap > 0.0 {
                        Some((s.speaker_id, overlap))
                    } else {
                        None
                    }
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(id, _)| format!("Speaker {}", id))
                .unwrap_or_else(|| "UNKNOWN".to_string());

            println!(
                "[{:.2}s - {:.2}s] {}: {}",
                token.start, token.end, speaker, token.text
            );
        }

        println!("\n{}", "=".repeat(80));
        let elapsed = start_time.elapsed();
        let realtime_factor = duration / elapsed.as_secs_f32();
        println!(
            "\n✓ Completed in {:.2}s ({:.2}x realtime)",
            elapsed.as_secs_f32(),
            realtime_factor
        );
        println!("• Audio duration: {:.1}s", duration);
        println!("• Processing speed: {:.1}s per minute of audio", elapsed.as_secs_f32() / (duration / 60.0));

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
