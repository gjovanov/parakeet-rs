//! Quick 3-minute VAD test for quality evaluation
use parakeet_rs::realtime_canary_vad::{RealtimeCanaryVad, RealtimeCanaryVadConfig};
use parakeet_rs::vad::VadConfig;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 3-Minute Canary VAD Quality Test ===\n");

    let audio_path = "./media/VOD_2025_1203_1900_ORF_OOE_broadcast.wav";
    
    // Load audio (first 3 minutes = 180 seconds)
    println!("Loading first 3 minutes of audio...");
    let output = Command::new("ffmpeg")
        .args(["-i", audio_path, "-t", "180", "-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .output()?;

    if !output.status.success() {
        eprintln!("FFmpeg error: {}", String::from_utf8_lossy(&output.stderr));
        return Err("FFmpeg failed".into());
    }

    let bytes = output.stdout;
    let samples: Vec<f32> = bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    let total_duration = samples.len() as f32 / 16000.0;
    println!("Loaded {:.1}s of audio ({} samples)\n", total_duration, samples.len());

    // Create VAD config with current pause_based settings
    let vad_config = VadConfig::pause_based();
    println!("VAD Config:");
    println!("  speech_threshold: {}", vad_config.speech_threshold);
    println!("  silence_trigger_ms: {}", vad_config.silence_trigger_ms);
    println!("  max_speech_secs: {}", vad_config.max_speech_secs);
    println!("  speech_pad_end_ms: {}", vad_config.speech_pad_end_ms);
    println!("  max_pauses: {}", vad_config.max_pauses);

    let config = RealtimeCanaryVadConfig {
        vad: vad_config,
        language: "de".to_string(),
        min_buffer_duration: 0.0,  // Immediate mode
        max_buffer_duration: 15.0,
        long_pause_threshold: 1.5,
    };

    println!("\nLoading models...");
    let mut transcriber = RealtimeCanaryVad::new(
        "./canary", "./silero_vad.onnx", None, Some(config)
    )?;
    println!("Models loaded!\n");

    println!("=== Processing ===\n");

    let chunk_size = 1600; // 100ms
    let mut segments: Vec<(f32, f32, String, usize)> = Vec::new(); // (start, end, text, commas)

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let result = transcriber.push_audio(chunk)?;
        
        // Progress every 10s
        if i % 100 == 0 {
            print!("\rProgress: {:.0}s / {:.0}s, segments: {}    ", 
                   result.total_duration, total_duration, segments.len());
        }

        for seg in &result.segments {
            let commas = seg.text.matches(',').count();
            let words = seg.text.split_whitespace().count();
            segments.push((seg.start_time, seg.end_time, seg.text.clone(), commas));
            
            println!("\n[SEG {}] {:.2}s-{:.2}s ({:.1}s) | {} words, {} commas",
                     segments.len(), seg.start_time, seg.end_time,
                     seg.end_time - seg.start_time, words, commas);
            println!("  \"{}\"", seg.text);
        }
    }

    // Finalize
    let final_result = transcriber.finalize()?;
    for seg in &final_result.segments {
        let commas = seg.text.matches(',').count();
        let words = seg.text.split_whitespace().count();
        segments.push((seg.start_time, seg.end_time, seg.text.clone(), commas));
        println!("\n[FINAL {}] {:.2}s-{:.2}s | {} words, {} commas",
                 segments.len(), seg.start_time, seg.end_time, words, commas);
        println!("  \"{}\"", seg.text);
    }

    // Statistics
    println!("\n\n=== RESULTS ===\n");
    println!("Total segments: {}", segments.len());
    
    let total_seg_duration: f32 = segments.iter().map(|(s, e, _, _)| e - s).sum();
    let avg_duration = total_seg_duration / segments.len() as f32;
    println!("Average segment duration: {:.2}s", avg_duration);

    let avg_commas: f32 = segments.iter().map(|(_, _, _, c)| *c as f32).sum::<f32>() / segments.len() as f32;
    println!("Average commas per segment: {:.2}", avg_commas);

    let c0 = segments.iter().filter(|(_, _, _, c)| *c == 0).count();
    let c1 = segments.iter().filter(|(_, _, _, c)| *c == 1).count();
    let c2 = segments.iter().filter(|(_, _, _, c)| *c == 2).count();
    let c3plus = segments.iter().filter(|(_, _, _, c)| *c >= 3).count();
    
    println!("\nComma distribution:");
    println!("  0 commas: {} ({:.0}%)", c0, 100.0 * c0 as f32 / segments.len() as f32);
    println!("  1 comma:  {} ({:.0}%)", c1, 100.0 * c1 as f32 / segments.len() as f32);
    println!("  2 commas: {} ({:.0}%)", c2, 100.0 * c2 as f32 / segments.len() as f32);
    println!("  3+ commas: {} ({:.0}%) <- should minimize", c3plus, 100.0 * c3plus as f32 / segments.len() as f32);

    // Check for short segments (orphans)
    let short_segs = segments.iter().filter(|(s, e, _, _)| e - s < 1.0).count();
    println!("\nShort segments (<1s): {} ({:.0}%) <- should minimize", 
             short_segs, 100.0 * short_segs as f32 / segments.len() as f32);

    // Quality score
    let good_segs = c0 + c1 + c2;
    let score = 100.0 * good_segs as f32 / segments.len() as f32;
    println!("\n=== QUALITY SCORE: {:.0}% (segments with 0-2 commas) ===", score);

    Ok(())
}
