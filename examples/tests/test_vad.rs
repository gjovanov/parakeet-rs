use parakeet_rs::vad::{SileroVad, VadConfig, VadSegmenter, VAD_SAMPLE_RATE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Silero VAD...");
    
    // Create VAD instance
    let mut vad = SileroVad::new("./silero_vad.onnx", None)?;
    println!("VAD loaded successfully");
    
    // Generate test audio: 1 second of sine wave (simulating speech)
    let sample_rate = VAD_SAMPLE_RATE;
    let duration_samples = sample_rate;
    let freq = 440.0; // A4 note
    
    let samples: Vec<f32> = (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();
    
    println!("Generated {} samples of test audio", samples.len());
    
    // Process in chunks
    let chunk_size = 512;
    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let prob = vad.process(chunk)?;
        if i < 5 || prob > 0.3 {
            println!("Chunk {}: probability = {:.4}", i, prob);
        }
    }
    
    println!("\nVAD processing completed successfully!");
    
    // Test VadSegmenter
    println!("\nTesting VadSegmenter...");
    let config = VadConfig::pause_based();
    let mut segmenter = VadSegmenter::new("./silero_vad.onnx", config, None)?;
    println!("Segmenter created");
    
    // Push audio
    let segments = segmenter.push_audio(&samples)?;
    println!("After push_audio: {} segments, is_speaking={}", segments.len(), segmenter.is_speaking());
    
    // Try finalize
    if let Some(seg) = segmenter.finalize()? {
        println!("Finalized segment: {:.2}s - {:.2}s ({} samples)", 
                 seg.start_time, seg.end_time, seg.samples.len());
    } else {
        println!("No segment from finalize (may need more speech)");
    }
    
    Ok(())
}
