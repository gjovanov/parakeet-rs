use parakeet_rs::vad::{SileroVad, VadConfig, VadSegmenter, VAD_SAMPLE_RATE};
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Silero VAD with real audio...");
    
    // Extract first 10 seconds of audio using ffmpeg
    let output = Command::new("ffmpeg")
        .args(["-i", "./media/broadcast.wav", "-t", "10", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .output()?;
    
    if !output.status.success() {
        eprintln!("FFmpeg stderr: {}", String::from_utf8_lossy(&output.stderr));
        return Err("FFmpeg failed".into());
    }
    
    // Convert bytes to f32 samples
    let bytes = output.stdout;
    let samples: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    println!("Extracted {} samples ({:.2}s)", samples.len(), samples.len() as f32 / VAD_SAMPLE_RATE as f32);
    
    // Test VAD with real audio
    let mut vad = SileroVad::new("./silero_vad.onnx", None)?;
    println!("VAD loaded");
    
    let chunk_size = 512;
    let mut speech_chunks = 0;
    let mut total_chunks = 0;
    
    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let prob = vad.process(chunk)?;
        total_chunks += 1;
        
        if prob > 0.5 {
            speech_chunks += 1;
            if speech_chunks <= 10 || i % 50 == 0 {
                println!("Chunk {} ({:.2}s): speech prob = {:.3}", i, i as f32 * 0.032, prob);
            }
        }
    }
    
    println!("\nSummary: {} speech chunks out of {} total ({:.1}%)", 
             speech_chunks, total_chunks, 
             100.0 * speech_chunks as f32 / total_chunks as f32);
    
    // Test VadSegmenter
    println!("\n=== Testing VadSegmenter ===");
    let config = VadConfig::pause_based();
    println!("Config: threshold={}, silence_trigger={}ms, min_speech={}ms", 
             config.speech_threshold, config.silence_trigger_ms, config.min_speech_ms);
    
    let mut segmenter = VadSegmenter::new("./silero_vad.onnx", config, None)?;
    
    // Push audio in chunks to simulate streaming
    let chunk_duration_samples = VAD_SAMPLE_RATE / 10; // 100ms chunks
    let mut total_segments = 0;
    
    for (i, chunk) in samples.chunks(chunk_duration_samples).enumerate() {
        let segments = segmenter.push_audio(chunk)?;
        for seg in &segments {
            total_segments += 1;
            println!("Segment {}: {:.2}s - {:.2}s ({} samples, {:.2}s)", 
                     total_segments, seg.start_time, seg.end_time, 
                     seg.samples.len(), seg.samples.len() as f32 / VAD_SAMPLE_RATE as f32);
        }
        
        // Print state occasionally
        if i % 20 == 0 {
            println!("  t={:.1}s state={} is_speaking={}", 
                     segmenter.current_time(), segmenter.state_name(), segmenter.is_speaking());
        }
    }
    
    // Finalize
    if let Some(seg) = segmenter.finalize()? {
        total_segments += 1;
        println!("Final segment {}: {:.2}s - {:.2}s ({} samples)", 
                 total_segments, seg.start_time, seg.end_time, seg.samples.len());
    }
    
    println!("\nTotal segments: {}", total_segments);
    
    Ok(())
}
