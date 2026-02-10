//! Integration tests for transcription quality across modes and models.
//!
//! These tests require a running parakeet-server and ONNX models.
//! Run with: cargo test --test integration_transcription -- --ignored
//!
//! The tests:
//! 1. Create a session with a specific mode + model
//! 2. Upload a German audio fixture
//! 3. Start the session and collect transcription via WebSocket
//! 4. Compare against reference transcripts using WER, CER, and key-phrase metrics
//!
//! Set PARAKEET_TEST_URL to override the server URL (default: http://localhost:3000)

use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Compute Word Error Rate between reference and hypothesis.
/// WER = (S + D + I) / N, where N = words in reference
fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let ref_words = normalize_words(reference);
    let hyp_words = normalize_words(hypothesis);

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    let distance = levenshtein_words(&ref_words, &hyp_words);
    distance as f32 / ref_words.len() as f32
}

/// Compute Character Error Rate between reference and hypothesis.
fn char_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let ref_chars: Vec<char> = normalize_text(reference).chars().collect();
    let hyp_chars: Vec<char> = normalize_text(hypothesis).chars().collect();

    if ref_chars.is_empty() {
        return if hyp_chars.is_empty() { 0.0 } else { 1.0 };
    }

    let distance = levenshtein_chars(&ref_chars, &hyp_chars);
    distance as f32 / ref_chars.len() as f32
}

/// Check what fraction of key phrases appear in the hypothesis.
fn key_phrase_recall(hypothesis: &str, key_phrases: &[String]) -> f32 {
    if key_phrases.is_empty() {
        return 1.0;
    }
    let hyp_lower = hypothesis.to_lowercase();
    let found = key_phrases
        .iter()
        .filter(|p| hyp_lower.contains(&p.to_lowercase()))
        .count();
    found as f32 / key_phrases.len() as f32
}

fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_words(text: &str) -> Vec<String> {
    normalize_text(text)
        .split_whitespace()
        .map(|w| w.to_string())
        .collect()
}

fn levenshtein_words(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = std::cmp::min(
                std::cmp::min(prev[j] + 1, curr[j - 1] + 1),
                prev[j - 1] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

fn levenshtein_chars(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = std::cmp::min(
                std::cmp::min(prev[j] + 1, curr[j - 1] + 1),
                prev[j - 1] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// Test fixture definitions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Deserialize)]
struct Reference {
    file: String,
    language: String,
    reference: String,
    key_phrases: Vec<String>,
    duration_secs: f32,
}

fn load_references() -> HashMap<String, Reference> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("references.json");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&content).expect("Failed to parse references.json")
}

fn server_url() -> String {
    std::env::var("PARAKEET_TEST_URL").unwrap_or_else(|_| "http://localhost:3000".to_string())
}

fn fixture_path(filename: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(filename)
}

// ---------------------------------------------------------------------------
// Server interaction helpers
// ---------------------------------------------------------------------------

/// Create a session, upload audio, start, wait for completion, collect transcript.
async fn run_transcription_session(
    model: &str,
    mode: &str,
    fixture: &Reference,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let base = server_url();

    // 1. Create session
    let create_body = serde_json::json!({
        "model": model,
        "mode": mode,
        "language": fixture.language,
        "sentence_completion": "off"
    });
    let resp = client
        .post(&format!("{}/api/sessions", base))
        .json(&create_body)
        .send()
        .await
        .map_err(|e| format!("Create session failed: {}", e))?;
    let session: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Parse create response: {}", e))?;
    let session_id = session["id"]
        .as_str()
        .ok_or("No session id")?
        .to_string();

    // 2. Upload audio file
    let audio_path = fixture_path(&fixture.file);
    let audio_bytes = std::fs::read(&audio_path)
        .map_err(|e| format!("Read audio file {}: {}", audio_path.display(), e))?;
    let file_part = reqwest::multipart::Part::bytes(audio_bytes)
        .file_name(fixture.file.clone())
        .mime_str("audio/wav")
        .unwrap();
    let form = reqwest::multipart::Form::new().part("file", file_part);
    client
        .post(&format!("{}/api/sessions/{}/upload", base, session_id))
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Upload failed: {}", e))?;

    // 3. Connect WebSocket for subtitles BEFORE starting
    let ws_url = format!(
        "{}/api/sessions/{}/ws",
        base.replace("http", "ws"),
        session_id
    );
    let (mut ws, _) = tokio_tungstenite::connect_async(&ws_url)
        .await
        .map_err(|e| format!("WebSocket connect: {}", e))?;

    // 4. Start session
    client
        .post(&format!("{}/api/sessions/{}/start", base, session_id))
        .send()
        .await
        .map_err(|e| format!("Start failed: {}", e))?;

    // 5. Collect transcription messages
    use futures_util::StreamExt;
    let mut finals: Vec<String> = Vec::new();
    let mut last_growing = String::new();
    let timeout = Duration::from_secs((fixture.duration_secs as u64) + 30);
    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        let msg = tokio::time::timeout_at(deadline, ws.next()).await;
        match msg {
            Ok(Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text)))) => {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                    match parsed["type"].as_str() {
                        Some("subtitle") => {
                            if parsed["is_final"].as_bool() == Some(true) {
                                if let Some(t) = parsed["text"].as_str() {
                                    finals.push(t.to_string());
                                }
                            }
                            if let Some(g) = parsed["growing_text"].as_str() {
                                if !g.is_empty() {
                                    last_growing = g.to_string();
                                }
                            }
                            if let Some(ft) = parsed["full_transcript"].as_str() {
                                if !ft.is_empty() {
                                    last_growing = ft.to_string();
                                }
                            }
                        }
                        Some("end") | Some("vod_complete") => break,
                        Some("error") => {
                            return Err(format!(
                                "Server error: {}",
                                parsed["message"].as_str().unwrap_or("unknown")
                            ));
                        }
                        _ => {}
                    }
                }
            }
            Ok(Some(Ok(_))) => {} // binary/ping/pong
            Ok(Some(Err(e))) => return Err(format!("WebSocket error: {}", e)),
            Ok(None) => break,
            Err(_) => return Err("Timeout waiting for transcription".to_string()),
        }
    }

    // 6. Stop session (cleanup)
    let _ = client
        .post(&format!("{}/api/sessions/{}/stop", base, session_id))
        .send()
        .await;

    // Build full transcript from finals, or fall back to growing text
    let transcript = if !finals.is_empty() {
        finals.join(" ")
    } else {
        last_growing
    };

    Ok(transcript)
}

// ---------------------------------------------------------------------------
// Quality report
// ---------------------------------------------------------------------------

struct QualityReport {
    model: String,
    mode: String,
    fixture_name: String,
    transcript: String,
    reference: String,
    wer: f32,
    cer: f32,
    key_phrase_recall: f32,
}

impl std::fmt::Display for QualityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "  Model: {}, Mode: {}, Fixture: {}", self.model, self.mode, self.fixture_name)?;
        writeln!(f, "  WER: {:.1}%  CER: {:.1}%  Key Phrase Recall: {:.1}%",
            self.wer * 100.0, self.cer * 100.0, self.key_phrase_recall * 100.0)?;
        writeln!(f, "  Reference:  {}", &self.reference.chars().take(100).collect::<String>())?;
        writeln!(f, "  Transcript: {}", &self.transcript.chars().take(100).collect::<String>())
    }
}

// ---------------------------------------------------------------------------
// Metric unit tests (always run)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod metric_tests {
    use super::*;

    #[test]
    fn test_wer_identical() {
        assert_eq!(word_error_rate("hello world", "hello world"), 0.0);
    }

    #[test]
    fn test_wer_completely_wrong() {
        let wer = word_error_rate("hello world", "foo bar baz");
        assert!(wer > 0.9, "WER should be high for completely different text: {}", wer);
    }

    #[test]
    fn test_wer_one_substitution() {
        // "hello world" → "hello earth" = 1 sub / 2 words = 0.5
        let wer = word_error_rate("hello world", "hello earth");
        assert!((wer - 0.5).abs() < 0.01, "WER should be 0.5, got {}", wer);
    }

    #[test]
    fn test_wer_empty_reference() {
        assert_eq!(word_error_rate("", ""), 0.0);
        assert_eq!(word_error_rate("", "some text"), 1.0);
    }

    #[test]
    fn test_wer_case_insensitive() {
        assert_eq!(word_error_rate("Hello World", "hello world"), 0.0);
    }

    #[test]
    fn test_wer_ignores_punctuation() {
        assert_eq!(word_error_rate("Hello, world!", "hello world"), 0.0);
    }

    #[test]
    fn test_cer_identical() {
        assert_eq!(char_error_rate("hello", "hello"), 0.0);
    }

    #[test]
    fn test_cer_one_char_diff() {
        let cer = char_error_rate("hello", "hallo");
        assert!(cer > 0.0 && cer < 0.5);
    }

    #[test]
    fn test_key_phrase_recall_all_found() {
        let phrases = vec!["Wien".to_string(), "Salzburg".to_string()];
        assert_eq!(key_phrase_recall("Wetter in Wien und Salzburg", &phrases), 1.0);
    }

    #[test]
    fn test_key_phrase_recall_none_found() {
        let phrases = vec!["Berlin".to_string(), "Hamburg".to_string()];
        assert_eq!(key_phrase_recall("Wetter in Wien und Salzburg", &phrases), 0.0);
    }

    #[test]
    fn test_key_phrase_recall_partial() {
        let phrases = vec!["Wien".to_string(), "Berlin".to_string()];
        assert_eq!(key_phrase_recall("Wetter in Wien", &phrases), 0.5);
    }

    #[test]
    fn test_key_phrase_case_insensitive() {
        let phrases = vec!["WIEN".to_string()];
        assert_eq!(key_phrase_recall("Wetter in wien", &phrases), 1.0);
    }

    #[test]
    fn test_normalize_text() {
        assert_eq!(normalize_text("Hello, World!  How  Are You?"), "hello world how are you");
    }

    #[test]
    fn test_normalize_text_german() {
        assert_eq!(normalize_text("Österreich: 25°C!"), "österreich 25c");
    }

    #[test]
    fn test_levenshtein_words_identical() {
        let a = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(levenshtein_words(&a, &a), 0);
    }

    #[test]
    fn test_levenshtein_words_insertion() {
        let a = vec!["hello".to_string()];
        let b = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(levenshtein_words(&a, &b), 1);
    }

    #[test]
    fn test_wer_german_realistic() {
        // Simulate a realistic transcription with minor errors
        let reference = "Die Temperatur beträgt heute fünfundzwanzig Grad Celsius";
        let hypothesis = "Die Temperatur betragt heute fünfundzwanzig Grad Celsius";
        let wer = word_error_rate(reference, hypothesis);
        // 1 word wrong out of 7 ≈ 14%
        assert!(wer < 0.2, "WER should be low for minor errors: {}", wer);
    }
}

// ---------------------------------------------------------------------------
// Integration tests (require running server — #[ignore])
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration {
    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_canary_speedy_de_short() {
        let refs = load_references();
        let fixture = refs.get("de_short").expect("de_short fixture not found");
        let transcript = run_transcription_session("canary-1b", "speedy", fixture)
            .await
            .expect("Transcription failed");

        let wer = word_error_rate(&fixture.reference, &transcript);
        let cer = char_error_rate(&fixture.reference, &transcript);
        let recall = key_phrase_recall(&transcript, &fixture.key_phrases);

        let report = QualityReport {
            model: "canary-1b".to_string(),
            mode: "speedy".to_string(),
            fixture_name: "de_short".to_string(),
            transcript: transcript.clone(),
            reference: fixture.reference.clone(),
            wer,
            cer,
            key_phrase_recall: recall,
        };
        eprintln!("\n{}", report);

        // Relaxed thresholds for TTS-generated audio (not natural speech)
        assert!(recall >= 0.3, "Key phrase recall too low: {:.1}%", recall * 100.0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_canary_speedy_de_medium() {
        let refs = load_references();
        let fixture = refs.get("de_medium").expect("de_medium fixture not found");
        let transcript = run_transcription_session("canary-1b", "speedy", fixture)
            .await
            .expect("Transcription failed");

        let wer = word_error_rate(&fixture.reference, &transcript);
        let recall = key_phrase_recall(&transcript, &fixture.key_phrases);

        eprintln!("\n  [canary-1b/speedy/de_medium] WER: {:.1}% Recall: {:.1}%",
            wer * 100.0, recall * 100.0);
        eprintln!("  Transcript: {}", &transcript.chars().take(120).collect::<String>());

        assert!(recall >= 0.3, "Key phrase recall too low: {:.1}%", recall * 100.0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_canary_pause_based_de_long() {
        let refs = load_references();
        let fixture = refs.get("de_long").expect("de_long fixture not found");
        let transcript = run_transcription_session("canary-1b", "pause_based", fixture)
            .await
            .expect("Transcription failed");

        let wer = word_error_rate(&fixture.reference, &transcript);
        let recall = key_phrase_recall(&transcript, &fixture.key_phrases);

        eprintln!("\n  [canary-1b/pause_based/de_long] WER: {:.1}% Recall: {:.1}%",
            wer * 100.0, recall * 100.0);
        eprintln!("  Transcript: {}", &transcript.chars().take(120).collect::<String>());

        assert!(recall >= 0.3, "Key phrase recall too low: {:.1}%", recall * 100.0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_canary_vod_de_news() {
        let refs = load_references();
        let fixture = refs.get("de_news").expect("de_news fixture not found");
        let transcript = run_transcription_session("canary-1b", "vod", fixture)
            .await
            .expect("Transcription failed");

        let wer = word_error_rate(&fixture.reference, &transcript);
        let cer = char_error_rate(&fixture.reference, &transcript);
        let recall = key_phrase_recall(&transcript, &fixture.key_phrases);

        eprintln!("\n  [canary-1b/vod/de_news] WER: {:.1}% CER: {:.1}% Recall: {:.1}%",
            wer * 100.0, cer * 100.0, recall * 100.0);
        eprintln!("  Transcript: {}", &transcript.chars().take(120).collect::<String>());

        // VoD should be more accurate than streaming
        assert!(recall >= 0.3, "Key phrase recall too low: {:.1}%", recall * 100.0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_canary_low_latency_de_medium() {
        let refs = load_references();
        let fixture = refs.get("de_medium").expect("de_medium fixture not found");
        let transcript = run_transcription_session("canary-1b", "low_latency", fixture)
            .await
            .expect("Transcription failed");

        let recall = key_phrase_recall(&transcript, &fixture.key_phrases);

        eprintln!("\n  [canary-1b/low_latency/de_medium] Recall: {:.1}%", recall * 100.0);
        eprintln!("  Transcript: {}", &transcript.chars().take(120).collect::<String>());

        assert!(recall >= 0.2, "Key phrase recall too low: {:.1}%", recall * 100.0);
    }

    /// A/B test: KV cache (O(n)) vs full decode (O(n²)) quality comparison.
    /// Loads the Canary 1B model directly (no server) and compares both decode
    /// paths against reference transcripts. The KV cache path uses decoder_mems
    /// (the intended usage), while the full path re-processes all tokens with
    /// empty mems each step. The cached path should have equal or better quality.
    #[tokio::test]
    #[ignore]
    async fn test_kv_cache_quality() {
        let refs = load_references();
        let model_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("canary");
        if !model_dir.exists() {
            eprintln!("Skipping: canary model not found at {}", model_dir.display());
            return;
        }

        let config = parakeet_rs::canary::CanaryConfig {
            language: "de".to_string(),
            ..Default::default()
        };
        let mut model = parakeet_rs::canary::CanaryModel::from_pretrained(
            &model_dir, None, Some(config),
        ).expect("Failed to load Canary model");

        // Use de_short — other fixtures may not work well in direct model mode
        // (server applies additional preprocessing like VAD/resampling)
        let fixture_names = ["de_short"];

        for name in &fixture_names {
            let fixture = refs.get(*name).expect(&format!("{} fixture not found", name));
            let audio_path = fixture_path(&fixture.file);
            let audio_bytes = std::fs::read(&audio_path).expect("Read audio file");

            // Decode WAV to f32 samples
            let cursor = std::io::Cursor::new(audio_bytes);
            let reader = hound::WavReader::new(cursor).expect("Parse WAV");
            let spec = reader.spec();
            let samples: Vec<f32> = if spec.bits_per_sample == 16 {
                reader.into_samples::<i16>()
                    .map(|s| s.unwrap() as f32 / 32768.0)
                    .collect()
            } else {
                reader.into_samples::<f32>()
                    .map(|s| s.unwrap())
                    .collect()
            };

            // Transcribe with KV cache (default path)
            let t0 = std::time::Instant::now();
            let cached_result = model.transcribe(&samples).expect("transcribe() failed");
            let cached_time = t0.elapsed();

            // Transcribe with full decode (O(n²) path)
            let t1 = std::time::Instant::now();
            let full_result = model.transcribe_full(&samples).expect("transcribe_full() failed");
            let full_time = t1.elapsed();

            let cached_wer = word_error_rate(&fixture.reference, &cached_result);
            let full_wer = word_error_rate(&fixture.reference, &full_result);
            let cached_recall = key_phrase_recall(&cached_result, &fixture.key_phrases);
            let full_recall = key_phrase_recall(&full_result, &fixture.key_phrases);

            eprintln!("\n  [KV Cache A/B] Fixture: {}", name);
            eprintln!("    Reference: {}", &fixture.reference.chars().take(80).collect::<String>());
            eprintln!("    Cached ({:?}): WER={:.1}% Recall={:.0}% \"{}\"",
                cached_time, cached_wer * 100.0, cached_recall * 100.0,
                &cached_result.chars().take(80).collect::<String>());
            eprintln!("    Full   ({:?}): WER={:.1}% Recall={:.0}% \"{}\"",
                full_time, full_wer * 100.0, full_recall * 100.0,
                &full_result.chars().take(80).collect::<String>());
            eprintln!("    Speedup: {:.1}x",
                full_time.as_secs_f64() / cached_time.as_secs_f64().max(0.001));

            // Cached path should produce reasonable quality (WER < 50%)
            assert!(
                cached_wer < 0.50,
                "KV cache WER too high: {:.1}% for fixture {}.\n  Result: {}\n  Reference: {}",
                cached_wer * 100.0, name, cached_result, fixture.reference
            );

            // Cached path should have at least some key phrase recall
            assert!(
                cached_recall >= 0.3,
                "KV cache recall too low: {:.0}% for fixture {}",
                cached_recall * 100.0, name
            );

            // Cached path quality should be equal or better than full path
            // (KV cache is the intended model usage; empty mems is a workaround)
            assert!(
                cached_wer <= full_wer + 0.10,
                "KV cache should not be significantly worse than full decode.\n  Cached WER: {:.1}% vs Full WER: {:.1}% for {}",
                cached_wer * 100.0, full_wer * 100.0, name
            );
        }
    }

    /// Comprehensive quality report across all modes and fixtures
    #[tokio::test]
    #[ignore]
    async fn test_full_quality_report() {
        let refs = load_references();
        let modes = ["speedy", "pause_based", "low_latency", "vod"];
        let fixtures = ["de_short", "de_medium", "de_long", "de_news"];

        eprintln!("\n{}", "=".repeat(72));
        eprintln!("TRANSCRIPTION QUALITY REPORT");
        eprintln!("{}\n", "=".repeat(72));

        let mut reports: Vec<QualityReport> = Vec::new();

        for mode in &modes {
            for fixture_name in &fixtures {
                let fixture = match refs.get(*fixture_name) {
                    Some(f) => f,
                    None => continue,
                };

                eprintln!("  Running: canary-1b / {} / {} ...", mode, fixture_name);

                match run_transcription_session("canary-1b", mode, fixture).await {
                    Ok(transcript) => {
                        let wer = word_error_rate(&fixture.reference, &transcript);
                        let cer = char_error_rate(&fixture.reference, &transcript);
                        let recall = key_phrase_recall(&transcript, &fixture.key_phrases);

                        let report = QualityReport {
                            model: "canary-1b".to_string(),
                            mode: mode.to_string(),
                            fixture_name: fixture_name.to_string(),
                            transcript,
                            reference: fixture.reference.clone(),
                            wer,
                            cer,
                            key_phrase_recall: recall,
                        };
                        eprintln!("{}", report);
                        reports.push(report);
                    }
                    Err(e) => {
                        eprintln!("  FAILED: {}\n", e);
                    }
                }
            }
        }

        // Summary
        if !reports.is_empty() {
            let avg_wer: f32 = reports.iter().map(|r| r.wer).sum::<f32>() / reports.len() as f32;
            let avg_cer: f32 = reports.iter().map(|r| r.cer).sum::<f32>() / reports.len() as f32;
            let avg_recall: f32 =
                reports.iter().map(|r| r.key_phrase_recall).sum::<f32>() / reports.len() as f32;

            eprintln!("\n{}", "=".repeat(72));
            eprintln!("SUMMARY ({} tests)", reports.len());
            eprintln!("  Average WER: {:.1}%", avg_wer * 100.0);
            eprintln!("  Average CER: {:.1}%", avg_cer * 100.0);
            eprintln!("  Average Key Phrase Recall: {:.1}%", avg_recall * 100.0);
            eprintln!("{}\n", "=".repeat(72));
        }
    }
}
