use parakeet_rs::vad::{SileroVad, VAD_SAMPLE_RATE};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Debugging Silero VAD...");
    
    // Extract first 5 seconds
    let output = Command::new("ffmpeg")
        .args(["-i", "./media/broadcast.wav", "-t", "5", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .output()?;
    
    let bytes = output.stdout;
    let samples: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    // Check audio stats
    let max_val = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let mean_val = samples.iter().map(|&x| x.abs()).sum::<f32>() / samples.len() as f32;
    println!("Audio stats: max={:.4}, mean={:.6}, samples={}", max_val, mean_val, samples.len());
    
    // Test VAD
    let mut vad = SileroVad::new("./silero_vad.onnx", None)?;
    
    let chunk_size = 512;
    println!("\nFirst 20 chunks:");
    for (i, chunk) in samples.chunks(chunk_size).take(20).enumerate() {
        let prob = vad.process(chunk)?;
        let chunk_max = chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("Chunk {:2}: prob={:.4}, chunk_max={:.4}, t={:.3}s", 
                 i, prob, chunk_max, i as f32 * 0.032);
    }
    
    // Check what the output tensor looks like
    println!("\nChecking with normalized audio...");
    
    // Normalize audio
    let normalized: Vec<f32> = samples.iter().map(|&s| s / max_val).collect();
    
    let mut vad2 = SileroVad::new("./silero_vad.onnx", None)?;
    println!("\nFirst 20 chunks (normalized):");
    for (i, chunk) in normalized.chunks(chunk_size).take(20).enumerate() {
        let prob = vad2.process(chunk)?;
        println!("Chunk {:2}: prob={:.4}", i, prob);
    }
    
    Ok(())
}
