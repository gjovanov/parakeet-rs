//! Subtitle emission functions for partial, final, and streaming segments

use parakeet_rs::growing_text::{GrowingTextMerger, GrowingTextResult};
use parakeet_rs::streaming_transcriber::TranscriptionSegment;
use parakeet_rs::TranscriptionSession;

/// Emit a partial subtitle for live display.
/// Called for every segment to keep the live subtitle area updated with growing text.
pub fn emit_partial_subtitle(
    session: &TranscriptionSession,
    segment: &TranscriptionSegment,
    growing_result: &GrowingTextResult,
) {
    // Normalize growing text outputs (the merger can re-introduce spacing issues
    // when joining tokens from different segments, e.g. "EU" + "-Verfahren" → "EU -Verfahren")
    let growing_text = normalize_text(&growing_result.current_sentence);
    let full_transcript = normalize_text(&growing_result.buffer);
    let delta = normalize_text(&growing_result.delta);

    let subtitle_msg = serde_json::json!({
        "type": "subtitle",
        "text": segment.text,
        "raw_text": segment.raw_text.as_deref().unwrap_or(&segment.text),
        "growing_text": growing_text,
        "full_transcript": full_transcript,
        "delta": delta,
        "tail_changed": growing_result.tail_changed,
        "speaker": segment.speaker,
        "start": segment.start_time,
        "end": segment.end_time,
        "is_final": false,
        "inference_time_ms": segment.inference_time_ms
    });

    let receiver_count = session.subtitle_tx.receiver_count();
    let subtitle_str = subtitle_msg.to_string();
    session.set_last_subtitle(subtitle_str.clone());
    let _ = session.subtitle_tx.send(subtitle_str);

    let tail_marker = if growing_result.tail_changed { " [TAIL CHANGED]" } else { "" };
    eprintln!(
        "[Session {} | partial | Speaker {}] \"{}\" [{:.2}s-{:.2}s] (inference: {}ms, receivers: {}){}",
        session.id,
        segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
        &segment.text.chars().take(80).collect::<String>(),
        segment.start_time,
        segment.end_time,
        segment.inference_time_ms.unwrap_or(0),
        receiver_count,
        tail_marker
    );
}

/// Emit a final subtitle for the transcript.
/// Called when the growing_merger finalizes a sentence.
pub fn emit_final_subtitle(
    session: &TranscriptionSession,
    merged: &TranscriptionSegment,
    growing_result: &GrowingTextResult,
) {
    // Normalize the finalized text (growing merger can re-introduce spacing issues
    // when joining tokens from different segments)
    let final_text = normalize_text(&merged.text);

    // For FINAL messages, use the finalized text as growing_text so the live
    // subtitle display shows the completed sentence (not the working buffer
    // which may be just "." right after finalization).
    let growing_text = if growing_result.current_sentence.trim().len() <= 2 {
        final_text.clone()
    } else {
        normalize_text(&growing_result.current_sentence)
    };
    let full_transcript = normalize_text(&growing_result.buffer);

    let subtitle_msg = serde_json::json!({
        "type": "subtitle",
        "text": final_text,
        "raw_text": merged.raw_text.as_deref().unwrap_or(&merged.text),
        "growing_text": growing_text,
        "full_transcript": full_transcript,
        "delta": "",
        "tail_changed": false,
        "speaker": merged.speaker,
        "start": merged.start_time,
        "end": merged.end_time,
        "is_final": true,
        "inference_time_ms": merged.inference_time_ms
    });

    let receiver_count = session.subtitle_tx.receiver_count();
    let subtitle_str = subtitle_msg.to_string();
    session.set_last_subtitle(subtitle_str.clone());
    let _ = session.subtitle_tx.send(subtitle_str);

    eprintln!(
        "[Session {} | FINAL | Speaker {}] \"{}\" [{:.2}s-{:.2}s] (inference: {}ms, receivers: {})",
        session.id,
        merged.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
        &merged.text.chars().take(80).collect::<String>(),
        merged.start_time,
        merged.end_time,
        merged.inference_time_ms.unwrap_or(0),
        receiver_count,
    );
}

/// Helper to emit transcript segments from StreamingTranscriber with growing text support.
/// Used for finalize/flush paths where sentence_buffer emits remaining content.
pub fn emit_streaming_segments(
    session: &TranscriptionSession,
    segments: &[TranscriptionSegment],
    growing_merger: &mut GrowingTextMerger,
) {
    for segment in segments {
        let growing_result = growing_merger.push(&segment.text, segment.is_final);

        // Normalize all text outputs
        let normalized_text = normalize_text(&segment.text);
        let growing_text = normalize_text(&growing_result.current_sentence);
        let full_transcript = normalize_text(&growing_result.buffer);
        let delta = normalize_text(&growing_result.delta);

        let subtitle_msg = serde_json::json!({
            "type": "subtitle",
            "text": normalized_text,
            "raw_text": segment.raw_text.as_deref().unwrap_or(&segment.text),
            "growing_text": growing_text,
            "full_transcript": full_transcript,
            "delta": delta,
            "tail_changed": growing_result.tail_changed,
            "speaker": segment.speaker,
            "start": segment.start_time,
            "end": segment.end_time,
            "is_final": segment.is_final,
            "inference_time_ms": segment.inference_time_ms
        });

        let subtitle_str = subtitle_msg.to_string();
        session.set_last_subtitle(subtitle_str.clone());
        let _ = session.subtitle_tx.send(subtitle_str);

        let partial_marker = if segment.is_final { "FINAL" } else { "partial" };
        eprintln!(
            "[Session {} | {} | Speaker {}] \"{}\" [{:.2}s-{:.2}s] (inference: {}ms, receivers: {})",
            session.id,
            partial_marker,
            segment.speaker.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string()),
            &segment.text.chars().take(80).collect::<String>(),
            segment.start_time,
            segment.end_time,
            segment.inference_time_ms.unwrap_or(0),
            session.subtitle_tx.receiver_count(),
        );
    }
}

/// Truncate text at first detected hallucination (3+ consecutive repeated words
/// or 3+ repeated 2-3 word phrases) for the emit pipeline
pub fn truncate_hallucination_text(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return text.to_string();
    }

    // Check for 3+ consecutive identical words
    let mut consecutive_count = 1;
    for i in 1..words.len() {
        if words[i].to_lowercase() == words[i - 1].to_lowercase() && words[i].len() > 1 {
            consecutive_count += 1;
            if consecutive_count >= 3 {
                let truncate_at = i + 1 - consecutive_count;
                if truncate_at > 0 {
                    return words[..truncate_at].join(" ");
                }
                return String::new();
            }
        } else {
            consecutive_count = 1;
        }
    }

    // Check for repeated phrases (2-3 word patterns)
    for pattern_len in 2..=3 {
        if words.len() < pattern_len * 3 {
            continue;
        }
        for i in 0..=(words.len() - pattern_len * 3) {
            let pattern: Vec<&str> = words[i..i + pattern_len].to_vec();
            let mut pattern_count = 1;
            let mut j = i + pattern_len;
            while j + pattern_len <= words.len() {
                let candidate: Vec<&str> = words[j..j + pattern_len].to_vec();
                if candidate
                    .iter()
                    .zip(pattern.iter())
                    .all(|(a, b)| a.to_lowercase() == b.to_lowercase())
                {
                    pattern_count += 1;
                    if pattern_count >= 3 {
                        if i > 0 {
                            return words[..i].join(" ");
                        }
                        return String::new();
                    }
                    j += pattern_len;
                } else {
                    break;
                }
            }
        }
    }

    text.to_string()
}

/// Normalize text spacing issues from ASR model output.
///
/// Fixes:
/// 1. Missing space between letter and digit: "am5" → "am 5"
/// 2. Missing space before uppercase start of compound: "kammerÖsterreich" → "kammer Österreich"
/// 3. Spurious " -" hyphen: "EU -Kommission" → "EU-Kommission"
/// 4. Spurious space before comma/period in numbers: "1 ,3" → "1,3"
pub fn normalize_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    // Phase 1: Fix " -X" → "-X" (spurious space before hyphen-word)
    // e.g. "EU -Kommission" → "EU-Kommission", "Kartell -Verfahren" → "Kartell-Verfahren"
    let mut result = String::with_capacity(text.len() + 32);
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        // Pattern: "<letter> -<letter>" → "<letter>-<letter>"
        if i + 2 < len && chars[i] == ' ' && chars[i + 1] == '-' && chars[i + 2].is_alphabetic() {
            // Check preceding char is alphanumeric
            if i > 0 && chars[i - 1].is_alphanumeric() {
                result.push('-');
                i += 2; // skip space and hyphen, next iteration picks up the letter
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    // Phase 2: Fix "digit ,digit" or "digit .digit" → "digit,digit" / "digit.digit"
    // e.g. "1 ,3" → "1,3", "25 .000" → "25.000"
    let input = result;
    result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    i = 0;

    while i < len {
        if i + 2 < len
            && chars[i] == ' '
            && (chars[i + 1] == ',' || chars[i + 1] == '.')
            && chars[i + 2].is_ascii_digit()
        {
            // Check preceding char is a digit
            if i > 0 && chars[i - 1].is_ascii_digit() {
                // Skip the space, push the punct directly
                result.push(chars[i + 1]);
                i += 2;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }

    // Phase 3: Insert space between lowercase letter and digit
    // e.g. "am5" → "am 5", "ab2027" → "ab 2027"
    // But NOT inside words like "h264" or after uppercase (acronyms)
    let input = result;
    result = String::with_capacity(input.len() + 16);
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();

    for i in 0..len {
        result.push(chars[i]);
        if i + 1 < len && chars[i].is_alphabetic() && chars[i + 1].is_ascii_digit() {
            // Insert space between letter and digit
            // But skip if the letter is uppercase and previous is also uppercase (acronym like "A4")
            let is_lowercase = chars[i].is_lowercase();
            if is_lowercase {
                result.push(' ');
            }
        }
    }

    // Phase 4: Insert space before uppercase Ö, Ä, Ü when preceded by lowercase letter
    // e.g. "kammerÖsterreich" → "kammer Österreich", "inÖsterreich" → "in Österreich"
    let input = result;
    result = String::with_capacity(input.len() + 16);
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();

    for i in 0..len {
        if i > 0
            && (chars[i] == 'Ö' || chars[i] == 'Ä' || chars[i] == 'Ü')
            && chars[i - 1].is_lowercase()
        {
            result.push(' ');
        }
        result.push(chars[i]);
    }

    result
}

/// Check if a PARTIAL's growing text overlaps significantly with recently
/// emitted FINALs. Used to suppress stale PARTIALs that echo confirmed text.
pub fn is_stale_partial(growing_text: &str, recent_finals: &[String]) -> bool {
    let trimmed = growing_text.trim();
    if trimmed.is_empty() {
        return true;
    }

    let partial_words: std::collections::HashSet<&str> = trimmed
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();

    if partial_words.is_empty() {
        return true;
    }

    // Check against last 5 FINALs
    for prev in recent_finals.iter().rev().take(5) {
        let prev_words: std::collections::HashSet<&str> = prev
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if prev_words.is_empty() {
            continue;
        }

        let common = partial_words
            .iter()
            .filter(|w| prev_words.contains(*w))
            .count();
        let overlap = common as f64 / partial_words.len() as f64;

        if overlap >= 0.6 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // truncate_hallucination_text
    // ========================================================================

    #[test]
    fn test_no_hallucination() {
        let text = "Dies ist ein normaler Satz ohne Wiederholungen.";
        assert_eq!(truncate_hallucination_text(text), text);
    }

    #[test]
    fn test_short_text_passthrough() {
        assert_eq!(truncate_hallucination_text("one two three"), "one two three");
        assert_eq!(truncate_hallucination_text("a b"), "a b");
        assert_eq!(truncate_hallucination_text(""), "");
    }

    #[test]
    fn test_consecutive_word_repetition() {
        let text = "Das ist ist ist ein Problem";
        let result = truncate_hallucination_text(text);
        // Should truncate before the 3rd "ist"
        assert!(!result.contains("ist ist ist"), "Got: {}", result);
        assert!(result.contains("Das"));
    }

    #[test]
    fn test_consecutive_word_at_start() {
        let text = "und und und das ist";
        let result = truncate_hallucination_text(text);
        // Repetition starts at index 0, should return empty
        assert!(result.is_empty() || !result.contains("und und und"), "Got: {}", result);
    }

    #[test]
    fn test_two_word_phrase_repetition() {
        let text = "guten morgen guten morgen guten morgen wie geht es";
        let result = truncate_hallucination_text(text);
        let count = result.matches("guten morgen").count();
        assert!(count < 3, "Should truncate repeated 2-word phrase, got: {}", result);
    }

    #[test]
    fn test_three_word_phrase_repetition() {
        let text = "das ist gut das ist gut das ist gut weiter";
        let result = truncate_hallucination_text(text);
        let count = result.matches("das ist gut").count();
        assert!(count < 3, "Should truncate repeated 3-word phrase, got: {}", result);
    }

    #[test]
    fn test_phrase_repetition_with_prefix() {
        let text = "Also dann guten morgen guten morgen guten morgen";
        let result = truncate_hallucination_text(text);
        assert!(result.contains("Also"), "Prefix should be preserved, got: {}", result);
    }

    #[test]
    fn test_case_insensitive_repetition() {
        let text = "Test TEST test test normal words here";
        let result = truncate_hallucination_text(text);
        // Case insensitive: Test/TEST/test/test → 4 consecutive
        assert!(!result.to_lowercase().contains("test test test"), "Got: {}", result);
    }

    #[test]
    fn test_single_char_words_ignored() {
        // Single-char words should not trigger repetition (len > 1 check)
        let text = "a a a a normal words here";
        let result = truncate_hallucination_text(text);
        assert_eq!(result, text, "Single-char words should not trigger truncation");
    }

    #[test]
    fn test_two_repetitions_ok() {
        // Only 2 repetitions should NOT trigger truncation
        let text = "das ist das ist weiter geht es";
        let result = truncate_hallucination_text(text);
        assert_eq!(result, text, "Two repetitions should be fine");
    }

    // ========================================================================
    // normalize_text
    // ========================================================================

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize_text(""), "");
    }

    #[test]
    fn test_normalize_no_changes() {
        let text = "Dies ist ein normaler Satz.";
        assert_eq!(normalize_text(text), text);
    }

    #[test]
    fn test_normalize_letter_digit_spacing() {
        assert_eq!(normalize_text("am5."), "am 5.");
        assert_eq!(normalize_text("am17."), "am 17.");
        assert_eq!(normalize_text("ab2027"), "ab 2027");
        assert_eq!(normalize_text("um1,3 Millionen"), "um 1,3 Millionen");
        assert_eq!(normalize_text("fast1,4 Milliarden"), "fast 1,4 Milliarden");
        assert_eq!(normalize_text("Pertutti am5."), "Pertutti am 5.");
    }

    #[test]
    fn test_normalize_hyphen_spacing() {
        assert_eq!(normalize_text("EU -Kommission"), "EU-Kommission");
        assert_eq!(normalize_text("Kartell -Verfahren"), "Kartell-Verfahren");
        assert_eq!(normalize_text("EU -Verfahren gegen"), "EU-Verfahren gegen");
        assert_eq!(normalize_text("All -in -One"), "All-in-One");
    }

    #[test]
    fn test_normalize_comma_period_spacing() {
        assert_eq!(normalize_text("1 ,3 Millionen"), "1,3 Millionen");
        assert_eq!(normalize_text("25 .000 Fahrzeuge"), "25.000 Fahrzeuge");
        assert_eq!(normalize_text("um1 ,3"), "um 1,3");
    }

    #[test]
    fn test_normalize_umlaut_spacing() {
        assert_eq!(normalize_text("kammerÖsterreich"), "kammer Österreich");
        assert_eq!(normalize_text("inÖsterreich"), "in Österreich");
        assert_eq!(normalize_text("gegenÖsterreich"), "gegen Österreich");
        assert_eq!(normalize_text("UVP inÖsterreich"), "UVP in Österreich");
        assert_eq!(normalize_text("spartÖsterreich"), "spart Österreich");
    }

    #[test]
    fn test_normalize_preserves_normal_uppercase() {
        // Don't insert space before regular uppercase in middle of sentence
        let text = "Hallo Welt";
        assert_eq!(normalize_text(text), text);
    }

    #[test]
    fn test_normalize_combined_issues() {
        assert_eq!(
            normalize_text("EU -Kommission leitet Kartell -Verfahren gegen Red Bull ein"),
            "EU-Kommission leitet Kartell-Verfahren gegen Red Bull ein"
        );
        assert_eq!(
            normalize_text("WirtschaftskammerÖsterreich"),
            "Wirtschaftskammer Österreich"
        );
        assert_eq!(
            normalize_text("Paket um1 ,3 Millionen"),
            "Paket um 1,3 Millionen"
        );
    }

    #[test]
    fn test_normalize_digit_preserves_acronyms() {
        // Uppercase letter + digit should NOT get space (e.g. "A4", "B2")
        assert_eq!(normalize_text("A4 Papier"), "A4 Papier");
        assert_eq!(normalize_text("ORF2"), "ORF2");
    }
}
