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
    let subtitle_msg = serde_json::json!({
        "type": "subtitle",
        "text": segment.text,
        "raw_text": segment.text,
        "growing_text": growing_result.current_sentence,
        "full_transcript": growing_result.buffer,
        "delta": growing_result.delta,
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
    // For FINAL messages, use the finalized text as growing_text so the live
    // subtitle display shows the completed sentence (not the working buffer
    // which may be just "." right after finalization).
    let growing_text = if growing_result.current_sentence.trim().len() <= 2 {
        &merged.text
    } else {
        &growing_result.current_sentence
    };

    let subtitle_msg = serde_json::json!({
        "type": "subtitle",
        "text": merged.text,
        "raw_text": merged.text,
        "growing_text": growing_text,
        "full_transcript": growing_result.buffer,
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

        let subtitle_msg = serde_json::json!({
            "type": "subtitle",
            "text": segment.text,
            "raw_text": segment.text,
            "growing_text": growing_result.current_sentence,
            "full_transcript": growing_result.buffer,
            "delta": growing_result.delta,
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
}
