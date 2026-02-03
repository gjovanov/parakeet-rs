//! Test: Canary speedy mode growing sentence output vs reference transcript
//!
//! Simulates the webrtc_transcriber pipeline for a canary speedy session:
//! RealtimeCanary -> SentenceBuffer -> GrowingTextMerger
//!
//! Run with: cargo run --release --example test_canary_speedy_growing
//!
//! Requires: ./canary/ model directory, ./media/broadcast_1.wav

use parakeet_rs::growing_text::GrowingTextMerger;
use parakeet_rs::realtime_canary::{RealtimeCanary, RealtimeCanaryConfig};
use parakeet_rs::sentence_buffer::{SentenceBuffer, SentenceBufferMode};
use parakeet_rs::streaming_transcriber::StreamingTranscriber;
use std::process::Command;

/// Safely truncate a UTF-8 string to approximately `max_chars` characters
fn truncate_utf8(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars - 3).collect();
    format!("{}...", truncated)
}

/// Truncate hallucinated repetitions (mirrors webrtc_transcriber pipeline)
fn truncate_hallucination_text(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 6 {
        return text.to_string();
    }

    // Check for 3+ consecutive repeated words
    let mut repeat_count = 1;
    for i in 1..words.len() {
        if words[i].to_lowercase() == words[i - 1].to_lowercase() {
            repeat_count += 1;
            if repeat_count >= 3 {
                // Truncate before the repetition
                let truncate_at = i + 1 - repeat_count;
                return words[..truncate_at].join(" ");
            }
        } else {
            repeat_count = 1;
        }
    }

    // Check for repeated 2-3 word phrases
    for phrase_len in 2..=3 {
        if words.len() < phrase_len * 3 {
            continue;
        }
        for start in 0..words.len().saturating_sub(phrase_len * 3 - 1) {
            let phrase: Vec<&str> = words[start..start + phrase_len].to_vec();
            let mut consecutive = 1;
            let mut pos = start + phrase_len;
            while pos + phrase_len <= words.len() {
                let next: Vec<&str> = words[pos..pos + phrase_len].to_vec();
                let matches = phrase
                    .iter()
                    .zip(next.iter())
                    .all(|(a, b)| a.to_lowercase() == b.to_lowercase());
                if matches {
                    consecutive += 1;
                    pos += phrase_len;
                } else {
                    break;
                }
            }
            if consecutive >= 3 {
                return words[..start + phrase_len].join(" ");
            }
        }
    }

    text.to_string()
}

/// Compute word-level similarity between two texts (Jaccard-like + order-aware)
fn word_similarity(generated: &str, reference: &str) -> (f64, usize, usize, usize) {
    let gen_words: Vec<String> = generated
        .split_whitespace()
        .map(|w: &str| w.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let ref_words: Vec<String> = reference
        .split_whitespace()
        .map(|w: &str| w.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string())
        .filter(|w| !w.is_empty())
        .collect();

    if ref_words.is_empty() {
        return (0.0, 0, gen_words.len(), 0);
    }

    // LCS-based similarity (order-aware)
    let m = gen_words.len();
    let n = ref_words.len();
    let mut dp = vec![vec![0u32; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            if gen_words[i - 1] == ref_words[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }
    let lcs_len = dp[m][n] as usize;
    let max_len = m.max(n);
    let similarity = if max_len > 0 {
        lcs_len as f64 / max_len as f64
    } else {
        0.0
    };

    (similarity, lcs_len, m, n)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let audio_path = args.get(1).map(|s| s.as_str()).unwrap_or("./media/broadcast_1.wav");
    let transcript_path = args.get(2).map(|s| s.as_str()).unwrap_or("./media/broadcast_1.transcript.txt");

    println!("=== Canary Speedy Mode: Growing Sentence Test ===");
    println!("Audio: {}", audio_path);
    println!("Transcript: {}\n", transcript_path);

    // Check required files
    let canary_path = "./canary";

    if !std::path::Path::new(canary_path).exists() {
        eprintln!("ERROR: Canary model not found at {}", canary_path);
        return Err("Missing Canary model. Set CANARY_MODEL_PATH or place model at ./canary/".into());
    }
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("ERROR: Audio file not found at {}", audio_path);
        return Err("Missing audio file".into());
    }
    if !std::path::Path::new(transcript_path).exists() {
        eprintln!("ERROR: Reference transcript not found at {}", transcript_path);
        return Err("Missing reference transcript".into());
    }

    // Load reference transcript
    let reference_text = std::fs::read_to_string(transcript_path)?;
    let reference_text = reference_text.trim();
    println!("Reference transcript: {} chars, {} words",
        reference_text.len(),
        reference_text.split_whitespace().count()
    );

    // Create speedy config (matching webrtc_transcriber create_canary_config("speedy", ...))
    let config = RealtimeCanaryConfig {
        buffer_size_secs: 8.0,
        min_audio_secs: 1.0,
        process_interval_secs: 0.5,
        language: "de".to_string(),
        pause_based_confirm: true,
        pause_threshold_secs: 0.6,
        silence_energy_threshold: 0.008,
    };

    println!("\nSpeedy Config:");
    println!("  buffer_size: {:.1}s", config.buffer_size_secs);
    println!("  process_interval: {:.1}s", config.process_interval_secs);
    println!("  min_audio: {:.1}s", config.min_audio_secs);
    println!("  pause_based_confirm: {}", config.pause_based_confirm);
    println!("  pause_threshold: {:.2}s", config.pause_threshold_secs);

    // Create pipeline components
    println!("\nLoading Canary model from {}...", canary_path);
    let canary = RealtimeCanary::new(canary_path, None, Some(config))?;
    // Use the StreamingTranscriber trait (same as webrtc_transcriber pipeline)
    let mut transcriber: Box<dyn StreamingTranscriber> = Box::new(canary);
    let mut sentence_buffer = SentenceBuffer::with_mode(SentenceBufferMode::Minimal);
    let mut growing_merger = GrowingTextMerger::new();
    println!("Model loaded.\n");

    // Extract first 5 minutes of audio via ffmpeg
    println!("Extracting audio from {} (first 5 min)...", audio_path);
    let output = Command::new("ffmpeg")
        .args([
            "-i", audio_path,
            "-t", "300",
            "-ar", "16000",
            "-ac", "1",
            "-f", "f32le",
            "-",
        ])
        .output()?;

    if !output.status.success() {
        eprintln!("FFmpeg error: {}", String::from_utf8_lossy(&output.stderr));
        return Err("FFmpeg failed".into());
    }

    let bytes = output.stdout;
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let duration_secs = samples.len() as f32 / 16000.0;
    println!("Audio: {} samples ({:.1}s)\n", samples.len(), duration_secs);

    // Stream audio through the pipeline
    println!("=== Streaming through Canary speedy pipeline ===\n");

    let chunk_size = 1600; // 100ms at 16kHz
    let mut segment_count = 0u64;
    let mut growing_snapshots: Vec<(f32, String)> = Vec::new();
    let mut last_growing_text = String::new();

    let start_time = std::time::Instant::now();

    for (i, chunk) in samples.chunks(chunk_size).enumerate() {
        let time_pos = (i * chunk_size) as f32 / 16000.0;

        match transcriber.push_audio(chunk) {
            Ok(result) => {
                for mut segment in result.segments.into_iter() {
                    // Apply hallucination truncation (mirrors webrtc_transcriber pipeline)
                    segment.text = truncate_hallucination_text(&segment.text);

                    // Skip empty segments
                    if segment.text.trim().is_empty() {
                        continue;
                    }

                    // Process through sentence buffer
                    if let Some(merged) = sentence_buffer.push(segment) {
                        segment_count += 1;

                        // Process through growing text merger
                        let growing_result = growing_merger.push(&merged.text, merged.is_final);

                        // Track growing text snapshots
                        if growing_result.buffer != last_growing_text {
                            last_growing_text = growing_result.buffer.clone();
                            growing_snapshots.push((time_pos, last_growing_text.clone()));
                        }

                        // Log progress
                        let partial_marker = if merged.is_final { "FINAL" } else { "partial" };
                        let tail_marker = if growing_result.tail_changed { " [TAIL]" } else { "" };
                        let delta_display = truncate_utf8(&growing_result.delta, 60);
                        let text_display = truncate_utf8(&merged.text, 80);
                        println!(
                            "[{:.1}s | {} | {}{}] delta=\"{}\"",
                            time_pos,
                            partial_marker,
                            delta_display,
                            tail_marker,
                            text_display,
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("[{:.1}s] Error: {}", time_pos, e);
            }
        }

        // Progress indicator every 10 seconds
        if i % 160 == 0 && i > 0 {
            eprint!("\rProgress: {:.0}s / {:.0}s ({:.0}%)   ",
                time_pos, duration_secs, time_pos / duration_secs * 100.0);
        }
    }

    // End-of-stream: finalize transcriber, flush sentence buffer, flush growing merger
    match transcriber.finalize() {
        Ok(result) => {
            for mut segment in result.segments {
                segment.text = truncate_hallucination_text(&segment.text);
                if segment.text.trim().is_empty() { continue; }
                if let Some(merged) = sentence_buffer.push(segment) {
                    segment_count += 1;
                    let growing_result = growing_merger.push(&merged.text, merged.is_final);
                    if growing_result.buffer != last_growing_text {
                        last_growing_text = growing_result.buffer.clone();
                    }
                }
            }
        }
        Err(e) => eprintln!("Finalize error: {}", e),
    }
    // Flush sentence buffer
    if let Some(flushed) = sentence_buffer.flush() {
        let growing_result = growing_merger.push(&flushed.text, flushed.is_final);
        if growing_result.buffer != last_growing_text {
            last_growing_text = growing_result.buffer.clone();
        }
    }
    let elapsed = start_time.elapsed();
    let rtf = elapsed.as_secs_f32() / duration_secs;
    println!("\n\n=== Pipeline Complete ===");
    println!("Processed: {:.1}s audio in {:.1}s (RTF: {:.2}x)", duration_secs, elapsed.as_secs_f32(), rtf);
    println!("Segments emitted: {}", segment_count);
    println!("Growing text snapshots: {}", growing_snapshots.len());

    // Get final growing text (full buffer = finalized + working)
    let final_growing = if !last_growing_text.is_empty() {
        last_growing_text.clone()
    } else {
        "(empty)".to_string()
    };

    // Get finalized-only text (primary quality metric)
    let finalized_sentences = growing_merger.get_finalized_sentences();
    let finalized_text = if finalized_sentences.is_empty() {
        // If nothing was finalized, fall back to the full buffer
        final_growing.clone()
    } else {
        finalized_sentences.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ")
    };

    // Print outputs
    println!("\n=== Finalized Text (primary metric) ===");
    println!("{}", finalized_text);
    println!("\n=== Full Growing Text (debug) ===");
    println!("{}", final_growing);

    // === Primary metric: Finalized text vs reference ===
    println!("\n=== Finalized Text vs Reference Transcript (PRIMARY) ===");
    let (fin_similarity, fin_lcs_len, fin_gen_words, fin_ref_words) = word_similarity(&finalized_text, reference_text);
    println!("Finalized words: {}", fin_gen_words);
    println!("Reference words: {}", fin_ref_words);
    println!("LCS matching words: {}", fin_lcs_len);
    println!("Word-order similarity (LCS): {:.1}%", fin_similarity * 100.0);

    let fin_gen_word_set: std::collections::HashSet<String> = finalized_text
        .split_whitespace()
        .map(|w: &str| w.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string())
        .collect();
    let fin_ref_word_set: std::collections::HashSet<String> = reference_text
        .split_whitespace()
        .map(|w: &str| w.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string())
        .collect();
    let fin_common = fin_gen_word_set.intersection(&fin_ref_word_set).count();
    let fin_recall = if !fin_ref_word_set.is_empty() {
        fin_common as f64 / fin_ref_word_set.len() as f64
    } else {
        0.0
    };
    let fin_precision = if !fin_gen_word_set.is_empty() {
        fin_common as f64 / fin_gen_word_set.len() as f64
    } else {
        0.0
    };
    println!("Finalized word recall: {:.1}% ({}/{})", fin_recall * 100.0, fin_common, fin_ref_word_set.len());
    println!("Finalized word precision: {:.1}% ({}/{})", fin_precision * 100.0, fin_common, fin_gen_word_set.len());

    // === Secondary metric: Full buffer vs reference (debug) ===
    println!("\n=== Full Buffer vs Reference Transcript (SECONDARY/DEBUG) ===");
    let (similarity, lcs_len, gen_words, ref_words) = word_similarity(&final_growing, reference_text);
    println!("Generated words: {}", gen_words);
    println!("Reference words: {}", ref_words);
    println!("LCS matching words: {}", lcs_len);
    println!("Word-order similarity (LCS): {:.1}%", similarity * 100.0);

    // Word recall (what fraction of reference words appear in generated text)
    let gen_word_set: std::collections::HashSet<String> = final_growing
        .split_whitespace()
        .map(|w: &str| w.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string())
        .collect();
    let ref_word_set: std::collections::HashSet<String> = reference_text
        .split_whitespace()
        .map(|w: &str| w.to_lowercase().trim_matches(|c: char| c.is_ascii_punctuation()).to_string())
        .collect();
    let common_words = gen_word_set.intersection(&ref_word_set).count();
    let recall = if !ref_word_set.is_empty() {
        common_words as f64 / ref_word_set.len() as f64
    } else {
        0.0
    };
    let precision = if !gen_word_set.is_empty() {
        common_words as f64 / gen_word_set.len() as f64
    } else {
        0.0
    };
    println!("Unique word recall: {:.1}% ({}/{})", recall * 100.0, common_words, ref_word_set.len());
    println!("Unique word precision: {:.1}% ({}/{})", precision * 100.0, common_words, gen_word_set.len());

    // Check for repetition issues in finalized output
    let words: Vec<&str> = finalized_text.split_whitespace().collect();
    let mut max_repeat = 1;
    let mut current_repeat = 1;
    for i in 1..words.len() {
        if words[i].to_lowercase() == words[i - 1].to_lowercase() {
            current_repeat += 1;
            max_repeat = max_repeat.max(current_repeat);
        } else {
            current_repeat = 1;
        }
    }
    println!("\nMax consecutive word repeat: {}", max_repeat);
    if max_repeat >= 3 {
        println!("WARNING: Repetition hallucination detected in output!");
    } else {
        println!("OK: No repetition hallucinations in output.");
    }

    // Check that start_time advances (not frozen)
    if growing_snapshots.len() > 5 {
        let first_snap_time = growing_snapshots[0].0;
        let mid_snap_time = growing_snapshots[growing_snapshots.len() / 2].0;
        let last_snap_time = growing_snapshots.last().unwrap().0;
        println!("\nTimestamp progression:");
        println!("  First snapshot at: {:.1}s", first_snap_time);
        println!("  Mid snapshot at:   {:.1}s", mid_snap_time);
        println!("  Last snapshot at:  {:.1}s", last_snap_time);
        if last_snap_time > first_snap_time + 10.0 {
            println!("OK: Timestamps advance throughout the stream.");
        } else {
            println!("WARNING: Timestamps may be frozen.");
        }
    }

    // Print growing text evolution (first 10 and last 5 snapshots)
    if !growing_snapshots.is_empty() {
        println!("\n=== Growing Text Evolution (first 10 snapshots) ===");
        for (i, (t, text)) in growing_snapshots.iter().take(10).enumerate() {
            let display = if text.chars().count() > 120 {
                let tail: String = text.chars().rev().take(117).collect::<Vec<_>>().into_iter().rev().collect();
                format!("...{}", tail)
            } else {
                text.clone()
            };
            println!("[{}] {:.1}s: \"{}\"", i, t, display);
        }
        if growing_snapshots.len() > 15 {
            println!("\n=== Growing Text Evolution (last 5 snapshots) ===");
            for (i, (t, text)) in growing_snapshots.iter().rev().take(5).rev().enumerate() {
                let idx = growing_snapshots.len() - 5 + i;
                let display = if text.chars().count() > 120 {
                    let tail: String = text.chars().rev().take(117).collect::<Vec<_>>().into_iter().rev().collect();
                    format!("...{}", tail)
                } else {
                    text.clone()
                };
                println!("[{}] {:.1}s: \"{}\"", idx, t, display);
            }
        }
    }

    println!("\n=== Test Complete ===");
    Ok(())
}
