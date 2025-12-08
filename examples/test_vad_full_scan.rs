use parakeet_rs::vad::{SileroVad, VAD_SAMPLE_RATE};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Full scan of broadcast.wav for speech...\n");

    // Extract 60 seconds of audio from different parts
    let starts = vec![0, 60, 120, 180, 300, 600, 900];

    for start_sec in starts {
        // Extract audio segment
        let output = Command::new("ffmpeg")
            .args([
                "-i", "./media/broadcast.wav",
                "-ss", &start_sec.to_string(),
                "-t", "30",
                "-ar", "16000",
                "-ac", "1",
                "-f", "f32le",
                "-"
            ])
            .output()?;

        if !output.status.success() || output.stdout.is_empty() {
            println!("Segment at {}s: no audio available", start_sec);
            continue;
        }

        // Convert bytes to f32 samples
        let bytes = output.stdout;
        let samples: Vec<f32> = bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

        // Normalize audio
        let normalized: Vec<f32> = samples.iter().map(|&s| s / max_val.max(0.001)).collect();

        // Test VAD
        let mut vad = SileroVad::new("./silero_vad.onnx", None)?;

        let chunk_size = 512;
        let mut speech_chunks = 0;
        let mut total_chunks = 0;
        let mut high_prob_count = 0;

        for chunk in normalized.chunks(chunk_size) {
            let prob = vad.process(chunk)?;
            total_chunks += 1;

            if prob > 0.5 {
                speech_chunks += 1;
            }
            if prob > 0.3 {
                high_prob_count += 1;
            }
        }

        let speech_pct = 100.0 * speech_chunks as f32 / total_chunks as f32;
        let high_pct = 100.0 * high_prob_count as f32 / total_chunks as f32;

        println!("Segment at {:4}s: max_amp={:.3}, speech={:3}/{} ({:.1}%), >0.3={:.1}%",
                 start_sec, max_val, speech_chunks, total_chunks, speech_pct, high_pct);
    }

    Ok(())
}
