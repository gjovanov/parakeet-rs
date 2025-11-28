/*
Real-Time Streaming Transcription with Speaker Diarization

This example demonstrates low-latency (~4-5 second) streaming transcription
with speaker attribution using ParakeetEOU and Sortformer.

Requirements:
- ParakeetEOU model files in ./eou/ directory:
  - encoder.onnx
  - decoder_joint.onnx
  - tokenizer.json
- Sortformer model: diar_streaming_sortformer_4spk-v2.onnx

Usage:
# From microphone (default audio device)
cargo run --release --example realtime_streaming --features sortformer

# From audio file (simulated real-time)
cargo run --release --example realtime_streaming --features sortformer -- --file audio.wav

# With GPU acceleration
cargo run --release --example realtime_streaming --features "sortformer,cuda"

# Show help
cargo run --release --example realtime_streaming --features sortformer -- --help
*/

use clap::Parser;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "sortformer")]
use hound;
#[cfg(feature = "sortformer")]
use parakeet_rs::{
    realtime::{RealtimeCallback, RealtimeConfig, RealtimeResult, SpeakerUpdate},
    SimplifiedRealtimeTranscriber, Speaker,
};

#[derive(Parser, Debug)]
#[command(name = "realtime_streaming")]
#[command(about = "Real-time streaming transcription with speaker diarization")]
struct Args {
    /// Audio file to process (simulates real-time input)
    /// If not specified, reads from default microphone
    #[arg(short, long)]
    file: Option<String>,

    /// Path to ASR model directory
    #[arg(long, default_value = "./eou")]
    asr_model: String,

    /// Path to diarization model
    #[arg(long, default_value = "diar_streaming_sortformer_4spk-v2.onnx")]
    diar_model: String,

    /// Chunk size in milliseconds (160ms recommended for stability)
    #[arg(long, default_value = "160")]
    chunk_ms: u64,

    /// Output format: text, json
    #[arg(long, default_value = "text")]
    format: String,

    /// Show timing information
    #[arg(long)]
    timing: bool,
}

#[cfg(feature = "sortformer")]
struct ConsoleCallback {
    show_timing: bool,
    start_time: Instant,
    json_format: bool,
}

#[cfg(feature = "sortformer")]
impl ConsoleCallback {
    fn new(show_timing: bool, json_format: bool) -> Self {
        Self {
            show_timing,
            start_time: Instant::now(),
            json_format,
        }
    }
}

#[cfg(feature = "sortformer")]
impl RealtimeCallback for ConsoleCallback {
    fn on_partial(&self, result: RealtimeResult) {
        if result.text.is_empty() {
            return;
        }

        if self.json_format {
            let json = serde_json::json!({
                "type": "partial",
                "text": result.text,
                "speaker": result.speaker_display(),
                "start": result.start_time,
                "end": result.end_time,
                "confidence": result.confidence,
                "utterance_id": result.utterance_id
            });
            println!("{}", json);
        } else {
            let timing = if self.show_timing {
                format!(
                    " [{:.2}s-{:.2}s, latency: {:.0}ms]",
                    result.start_time,
                    result.end_time,
                    self.start_time.elapsed().as_millis() as f32 - result.end_time * 1000.0
                )
            } else {
                String::new()
            };

            print!(
                "\r\x1b[K[{}] {}{}",
                result.speaker_display(),
                result.text,
                timing
            );
            io::stdout().flush().unwrap();
        }
    }

    fn on_final(&self, result: RealtimeResult) {
        if result.text.is_empty() {
            return;
        }

        if self.json_format {
            let json = serde_json::json!({
                "type": "final",
                "text": result.text,
                "speaker": result.speaker_display(),
                "start": result.start_time,
                "end": result.end_time,
                "confidence": result.confidence,
                "utterance_id": result.utterance_id
            });
            println!("{}", json);
        } else {
            let timing = if self.show_timing {
                format!(
                    " [{:.2}s-{:.2}s, conf: {:.0}%]",
                    result.start_time,
                    result.end_time,
                    result.confidence * 100.0
                )
            } else {
                String::new()
            };

            println!(
                "\r\x1b[K[{}] {}{}",
                result.speaker_display(),
                result.text,
                timing
            );
        }
    }

    fn on_speaker_update(&self, update: SpeakerUpdate) {
        if self.json_format {
            let json = serde_json::json!({
                "type": "speaker_update",
                "time_range": update.time_range,
                "old_speaker": format!("{:?}", update.old_speaker),
                "new_speaker": format!("{:?}", update.new_speaker),
                "utterance_id": update.utterance_id
            });
            println!("{}", json);
        } else if self.show_timing {
            eprintln!(
                "  [Speaker change at {:.2}s-{:.2}s: {} -> {}]",
                update.time_range.0,
                update.time_range.1,
                match update.old_speaker {
                    Speaker::Unknown => "?".to_string(),
                    Speaker::Id(id) => id.to_string(),
                },
                match update.new_speaker {
                    Speaker::Unknown => "?".to_string(),
                    Speaker::Id(id) => id.to_string(),
                }
            );
        }
    }
}

#[cfg(feature = "sortformer")]
fn process_file(args: &Args, audio_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Processing file: {}", audio_path);
    println!("ASR model: {}", args.asr_model);
    println!("Diarization model: {}", args.diar_model);
    println!();

    // Load audio
    let mut reader = hound::WavReader::open(audio_path)?;
    let spec = reader.spec();

    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    // Convert to mono if stereo
    let audio: Vec<f32> = if spec.channels == 2 {
        audio
            .chunks(2)
            .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        audio
    };

    let duration = audio.len() as f32 / spec.sample_rate as f32;
    println!(
        "Audio: {:.1}s @ {} Hz, {} samples",
        duration,
        spec.sample_rate,
        audio.len()
    );
    println!();

    // Create config
    let config = RealtimeConfig::new()
        .asr_model_path(&args.asr_model)
        .diarization_model_path(&args.diar_model)
        .emit_partials(true);

    // Create transcriber
    let mut transcriber = SimplifiedRealtimeTranscriber::new(config)?;

    // Process in real-time chunks
    let chunk_samples = (args.chunk_ms as f32 * spec.sample_rate as f32 / 1000.0) as usize;

    let start = Instant::now();
    let mut last_text = String::new();

    println!("Starting transcription (chunk size: {}ms)...", args.chunk_ms);
    println!("{}", "=".repeat(60));

    for (i, chunk) in audio.chunks(chunk_samples).enumerate() {

        // Simulate real-time by waiting (skip for faster testing)
        // Uncomment for true real-time simulation:
        // let target_time = Duration::from_millis(i as u64 * args.chunk_ms);
        // let elapsed = start.elapsed();
        // if target_time > elapsed {
        //     std::thread::sleep(target_time - elapsed);
        // }

        // Process chunk
        match transcriber.process(chunk) {
            Ok(result) => {
                if !result.text.is_empty() && result.text != last_text {
                    let timing = if args.timing {
                        format!(
                            " [{:.2}s-{:.2}s]",
                            result.start_time, result.end_time
                        )
                    } else {
                        String::new()
                    };

                    if result.is_final {
                        print!("\r\x1b[K"); // Clear progress line
                        println!(
                            "[{}] {}{}",
                            result.speaker_display(),
                            result.text,
                            timing
                        );
                        last_text.clear();
                    } else {
                        print!(
                            "\r\x1b[K[{}] {}{}",
                            result.speaker_display(),
                            result.text,
                            timing
                        );
                        io::stdout().flush()?;
                        last_text = result.text.clone();
                    }
                }
            }
            Err(e) => eprintln!("\nError at chunk {}: {}", i, e),
        }
    }

    print!("\r\x1b[K"); // Clear any partial output

    // Finalize
    let final_result = transcriber.finalize();
    if !final_result.text.is_empty() {
        println!(
            "\n[{}] {}",
            final_result.speaker_display(),
            final_result.text
        );
    }

    println!("\n{}", "=".repeat(60));
    let elapsed = start.elapsed();
    let rtf = elapsed.as_secs_f32() / duration;
    println!(
        "Completed in {:.2}s (RTF: {:.2}x, {:.1}x realtime)",
        elapsed.as_secs_f32(),
        rtf,
        1.0 / rtf
    );

    Ok(())
}

#[cfg(feature = "sortformer")]
fn process_microphone(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    println!("Real-time microphone transcription");
    println!("ASR model: {}", args.asr_model);
    println!("Diarization model: {}", args.diar_model);
    println!();

    // Set up audio capture
    let host = cpal::default_host();
    let device = match host.default_input_device() {
        Some(d) => d,
        None => {
            eprintln!("Error: No audio input device available.");
            eprintln!();
            eprintln!("If running in WSL, microphone access is not available by default.");
            eprintln!("Use file mode instead:");
            eprintln!("  cargo run --release --example realtime_streaming --features sortformer -- --file <audio.wav>");
            return Err("No input device available".into());
        }
    };

    println!("Using input device: {}", device.name()?);

    let supported_config = device.default_input_config()?;
    let sample_rate = supported_config.sample_rate().0;
    let channels = supported_config.channels() as usize;

    println!("Sample rate: {} Hz, Channels: {}", sample_rate, channels);
    println!();

    // Create config
    let config = RealtimeConfig::new()
        .asr_model_path(&args.asr_model)
        .diarization_model_path(&args.diar_model)
        .emit_partials(true);

    // Create transcriber
    let transcriber = Arc::new(std::sync::Mutex::new(SimplifiedRealtimeTranscriber::new(
        config,
    )?));

    // Set up Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    ctrlc::set_handler(move || {
        running_clone.store(false, Ordering::SeqCst);
    })?;

    // Resampling buffer (if needed)
    let needs_resample = sample_rate != 16000;
    if needs_resample {
        println!("Note: Resampling from {} Hz to 16000 Hz", sample_rate);
    }

    // Audio callback
    let transcriber_clone = transcriber.clone();
    let running_audio = running.clone();
    let json_format = args.format == "json";
    let show_timing = args.timing;

    let mut last_text = String::new();
    let start_time = Instant::now();

    let stream = device.build_input_stream(
        &supported_config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !running_audio.load(Ordering::SeqCst) {
                return;
            }

            // Convert to mono
            let mono: Vec<f32> = if channels > 1 {
                data.chunks(channels)
                    .map(|c| c.iter().sum::<f32>() / channels as f32)
                    .collect()
            } else {
                data.to_vec()
            };

            // Simple resampling (if needed)
            let samples: Vec<f32> = if needs_resample {
                let ratio = 16000.0 / sample_rate as f32;
                let new_len = (mono.len() as f32 * ratio) as usize;
                (0..new_len)
                    .map(|i| {
                        let src_idx = (i as f32 / ratio) as usize;
                        mono.get(src_idx).copied().unwrap_or(0.0)
                    })
                    .collect()
            } else {
                mono
            };

            // Process
            if let Ok(mut t) = transcriber_clone.lock() {
                if let Ok(result) = t.process(&samples) {
                    if !result.text.is_empty() && result.text != last_text {
                        if json_format {
                            let json = serde_json::json!({
                                "type": if result.is_final { "final" } else { "partial" },
                                "text": result.text,
                                "speaker": result.speaker_display(),
                                "start": result.start_time,
                                "end": result.end_time,
                            });
                            println!("{}", json);
                        } else {
                            let timing = if show_timing {
                                format!(
                                    " [{:.2}s-{:.2}s, latency: {:.0}ms]",
                                    result.start_time,
                                    result.end_time,
                                    start_time.elapsed().as_millis() as f32
                                        - result.end_time * 1000.0
                                )
                            } else {
                                String::new()
                            };

                            if result.is_final {
                                println!(
                                    "\r\x1b[K[{}] {}{}",
                                    result.speaker_display(),
                                    result.text,
                                    timing
                                );
                            } else {
                                print!(
                                    "\r\x1b[K[{}] {}{}",
                                    result.speaker_display(),
                                    result.text,
                                    timing
                                );
                                let _ = io::stdout().flush();
                            }
                        }
                        last_text = result.text;
                    }
                }
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    stream.play()?;

    println!("Listening... Press Ctrl+C to stop.");
    println!("{}", "=".repeat(60));

    // Wait for Ctrl+C
    while running.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));
    }

    println!("\n{}", "=".repeat(60));
    println!("Stopped.");

    // Final result
    if let Ok(mut t) = transcriber.lock() {
        let final_result = t.finalize();
        if !final_result.text.is_empty() {
            println!(
                "Final: [{}] {}",
                final_result.speaker_display(),
                final_result.text
            );
        }
        println!("Total duration: {:.2}s", t.total_duration());
    }

    Ok(())
}

#[allow(unreachable_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "sortformer"))]
    {
        eprintln!("Error: This example requires the 'sortformer' feature.");
        eprintln!("Run with: cargo run --release --example realtime_streaming --features sortformer");
        return Err("sortformer feature not enabled".into());
    }

    #[cfg(feature = "sortformer")]
    {
        let args = Args::parse();

        if let Some(ref file) = args.file {
            process_file(&args, file)?;
        } else {
            process_microphone(&args)?;
        }

        Ok(())
    }

    #[cfg(not(feature = "sortformer"))]
    unreachable!()
}
