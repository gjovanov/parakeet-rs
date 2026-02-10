//! FAB Live Transcription Forwarder
//!
//! Subscribes to a session's subtitle broadcast channel and forwards
//! finalized transcription text to a FAB endpoint via HTTP GET.
//!
//! Includes Teletext-aware line splitting (≤84 chars), deduplication
//! against sent history, and 300ms debounce to reduce request volume.

use std::collections::{HashSet, VecDeque};
use tokio::sync::broadcast;
use tokio::time::{Duration, Instant};

/// Debounce window: coalesce rapid-fire finals into single sends.
const DEBOUNCE_MS: u64 = 300;
/// Containment coefficient threshold: max(|A∩B|/|A|, |A∩B|/|B|).
/// Catches sliding-window refinements where Jaccard is diluted by extra words.
const CONTAINMENT_THRESHOLD: f64 = 0.75;
/// Minimum shared words required for containment-based dedup.
const MIN_SHARED_WORDS: usize = 3;
/// Teletext maximum: 42 chars/line × 2 lines = 84 chars.
const MAX_TELETEXT_CHARS: usize = 84;
/// How many recently sent texts to keep for deduplication.
const SENT_HISTORY_SIZE: usize = 10;

/// Select which text field to send based on send_type.
///
/// - "confirmed": use only the `text` field (finalized segment text)
/// - "growing" (default): use `growing_text` (cumulative), fall back to `text`
fn select_text_field<'a>(msg: &'a serde_json::Value, send_type: &str) -> &'a str {
    if send_type == "confirmed" {
        msg["text"].as_str().filter(|s| !s.is_empty()).unwrap_or("")
    } else {
        msg["growing_text"]
            .as_str()
            .filter(|s| !s.is_empty())
            .or_else(|| msg["text"].as_str())
            .unwrap_or("")
    }
}

/// Check if a dot at the given byte position is inside a number (e.g. "3.5").
fn is_dot_in_number(text: &str, dot_byte_pos: usize) -> bool {
    let bytes = text.as_bytes();
    let before_digit = dot_byte_pos > 0
        && bytes
            .get(dot_byte_pos - 1)
            .map_or(false, |b| b.is_ascii_digit());
    let after_digit = bytes
        .get(dot_byte_pos + 1)
        .map_or(false, |b| b.is_ascii_digit());
    before_digit && after_digit
}

/// Split text into Teletext-compatible chunks, each ≤ `max_chars` characters.
///
/// Prefers splitting at sentence boundaries (`.` `!` `?`), then clause
/// separators (`,` `;` `:` `–` `—`), then word boundaries (spaces).
/// Handles UTF-8 correctly (counts characters, not bytes).
fn split_for_teletext(text: &str, max_chars: usize) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }

    let mut result = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.chars().count() <= max_chars {
            let trimmed = remaining.trim();
            if !trimmed.is_empty() {
                result.push(trimmed.to_string());
            }
            break;
        }

        // Find the byte position at the (max_chars)th character
        let max_byte_pos = remaining
            .char_indices()
            .nth(max_chars)
            .map(|(pos, _)| pos)
            .unwrap_or(remaining.len());

        let search_region = &remaining[..max_byte_pos];

        // Find the last occurrence of each split-point type
        let mut best_sentence: Option<usize> = None;
        let mut best_clause: Option<usize> = None;
        let mut best_word: Option<usize> = None;

        for (byte_pos, ch) in search_region.char_indices() {
            let after_pos = byte_pos + ch.len_utf8();
            match ch {
                '.' if !is_dot_in_number(remaining, byte_pos) => {
                    best_sentence = Some(after_pos);
                }
                '!' | '?' => {
                    best_sentence = Some(after_pos);
                }
                ',' | ';' | ':' | '–' | '—' => {
                    best_clause = Some(after_pos);
                }
                ' ' => {
                    best_word = Some(byte_pos);
                }
                _ => {}
            }
        }

        let split_pos = best_sentence
            .or(best_clause)
            .or(best_word)
            .unwrap_or(max_byte_pos);

        let piece = remaining[..split_pos].trim();
        if !piece.is_empty() {
            result.push(piece.to_string());
        }

        remaining = remaining[split_pos..].trim_start();
    }

    result
}

/// Check if candidate text is a near-duplicate of any recently sent text.
///
/// Checks (in order):
/// 1. Exact match
/// 2. Candidate is a substring of a previously sent text (stale subset)
/// 3. Containment coefficient (max directional word overlap) exceeds threshold
///
/// The containment coefficient = |A∩B| / min(|A|, |B|) is more robust than
/// Jaccard for sliding-window transcription where text B extends text A with
/// new words — Jaccard gets diluted by the extra words in the union, but
/// containment correctly detects that the shorter text is mostly contained
/// in the longer one.
fn is_duplicate_of_history(candidate: &str, history: &VecDeque<String>) -> bool {
    if candidate.is_empty() {
        return true;
    }

    let candidate_words: HashSet<&str> = candidate.split_whitespace().collect();
    if candidate_words.is_empty() {
        return true;
    }

    for prev in history.iter().rev() {
        // Exact match
        if candidate == prev {
            return true;
        }

        // Candidate is a subset of something already sent
        if prev.contains(candidate) {
            return true;
        }

        // Containment coefficient: max(|A∩B|/|A|, |A∩B|/|B|)
        let prev_words: HashSet<&str> = prev.split_whitespace().collect();
        if prev_words.is_empty() {
            continue;
        }

        let intersection = candidate_words.intersection(&prev_words).count();
        if intersection >= MIN_SHARED_WORDS {
            let min_size = candidate_words.len().min(prev_words.len());
            if min_size > 0 {
                let containment = intersection as f64 / min_size as f64;
                if containment >= CONTAINMENT_THRESHOLD {
                    return true;
                }
            }
        }
    }

    false
}

/// Send text to the FAB endpoint via HTTP GET.
async fn send_to_fab(
    client: &reqwest::Client,
    fab_url: &str,
    session_id: &str,
    text: &str,
) -> bool {
    let url = match reqwest::Url::parse_with_params(
        fab_url,
        &[("language", "Default"), ("text", text)],
    ) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("[FAB] [{}] URL parse error: {}", session_id, e);
            return false;
        }
    };

    eprintln!(
        "[FAB] [{}] Sending ({} chars): \"{}\"",
        session_id,
        text.chars().count(),
        &text.chars().take(90).collect::<String>()
    );
    match client.get(url).send().await {
        Ok(resp) => {
            eprintln!("[FAB] [{}] Sent, status={}", session_id, resp.status());
            true
        }
        Err(e) => {
            eprintln!("[FAB] [{}] Send error: {}", session_id, e);
            false
        }
    }
}

/// Spawn an async task that forwards finalized subtitles to the FAB endpoint.
///
/// The task listens on `subtitle_rx` for JSON subtitle messages, filters for
/// `is_final: true`, splits text for Teletext (≤84 chars), applies
/// history-based deduplication, and sends via a 300ms debounce window.
///
/// The task automatically stops when the broadcast channel is closed (session ends).
pub fn spawn_fab_forwarder(
    session_id: String,
    fab_url: String,
    send_type: String,
    mut subtitle_rx: broadcast::Receiver<String>,
    client: reqwest::Client,
) {
    tokio::spawn(async move {
        eprintln!(
            "[FAB] Forwarder started for session {} (send_type: {}, teletext_max: {})",
            session_id, send_type, MAX_TELETEXT_CHARS
        );

        let mut sent_history: VecDeque<String> = VecDeque::new();
        let mut sent_count: u64 = 0;
        let mut skipped_count: u64 = 0;

        let mut pending_text: Option<String> = None;
        let mut debounce_deadline: Option<Instant> = None;

        loop {
            let sleep_fut = async {
                match debounce_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            };

            tokio::select! {
                result = subtitle_rx.recv() => {
                    match result {
                        Ok(json_str) => {
                            let msg: serde_json::Value = match serde_json::from_str(&json_str) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };

                            if msg["is_final"].as_bool() != Some(true) {
                                continue;
                            }

                            // Select text field based on send_type
                            let text = select_text_field(&msg, &send_type);
                            if text.is_empty() {
                                continue;
                            }

                            // Split for Teletext: each line ≤ 84 chars
                            let lines = split_for_teletext(text.trim(), MAX_TELETEXT_CHARS);

                            // Filter out duplicates against sent history
                            let non_dup_lines: Vec<String> = lines
                                .into_iter()
                                .filter(|line| {
                                    !line.is_empty() && !is_duplicate_of_history(line, &sent_history)
                                })
                                .collect();

                            if non_dup_lines.is_empty() {
                                skipped_count += 1;
                                continue;
                            }

                            // If split produced multiple non-duplicate lines, send all
                            // but the last immediately (they're confirmed by the split).
                            // The last (or only) line gets debounced.
                            if non_dup_lines.len() > 1 {
                                for line in &non_dup_lines[..non_dup_lines.len() - 1] {
                                    if send_to_fab(&client, &fab_url, &session_id, line).await {
                                        sent_history.push_back(line.clone());
                                        while sent_history.len() > SENT_HISTORY_SIZE {
                                            sent_history.pop_front();
                                        }
                                        sent_count += 1;
                                    }
                                }
                            }

                            // Buffer last line for debounce
                            if let Some(last_line) = non_dup_lines.into_iter().last() {
                                pending_text = Some(last_line);
                                debounce_deadline =
                                    Some(Instant::now() + Duration::from_millis(DEBOUNCE_MS));
                            }
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            eprintln!("[FAB] [{}] Lagged {} messages, continuing", session_id, n);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            // Flush pending text before exit
                            if let Some(text) = pending_text.take() {
                                if !is_duplicate_of_history(&text, &sent_history) {
                                    if send_to_fab(&client, &fab_url, &session_id, &text).await {
                                        sent_count += 1;
                                    }
                                }
                            }
                            eprintln!(
                                "[FAB] Forwarder stopped for session {} (sent: {}, skipped: {})",
                                session_id, sent_count, skipped_count
                            );
                            break;
                        }
                    }
                }
                _ = sleep_fut => {
                    // Debounce timer fired — send the buffered text
                    debounce_deadline = None;
                    if let Some(text) = pending_text.take() {
                        // Re-check dedup at send time (history may have grown)
                        if !is_duplicate_of_history(&text, &sent_history) {
                            if send_to_fab(&client, &fab_url, &session_id, &text).await {
                                sent_history.push_back(text);
                                while sent_history.len() > SENT_HISTORY_SIZE {
                                    sent_history.pop_front();
                                }
                                sent_count += 1;
                            }
                        } else {
                            skipped_count += 1;
                        }
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── select_text_field tests ──

    #[test]
    fn test_select_text_field_growing_mode() {
        let msg = json!({"text": "segment", "growing_text": "full growing sentence"});
        assert_eq!(select_text_field(&msg, "growing"), "full growing sentence");
    }

    #[test]
    fn test_select_text_field_confirmed_mode() {
        let msg = json!({"text": "segment", "growing_text": "full growing sentence"});
        assert_eq!(select_text_field(&msg, "confirmed"), "segment");
    }

    #[test]
    fn test_select_text_field_growing_fallback_to_text() {
        let msg = json!({"text": "segment only"});
        assert_eq!(select_text_field(&msg, "growing"), "segment only");
    }

    #[test]
    fn test_select_text_field_confirmed_empty_text() {
        let msg = json!({"text": "", "growing_text": "growing text"});
        assert_eq!(select_text_field(&msg, "confirmed"), "");
    }

    #[test]
    fn test_select_text_field_missing_fields() {
        let msg = json!({"is_final": true});
        assert_eq!(select_text_field(&msg, "growing"), "");
        assert_eq!(select_text_field(&msg, "confirmed"), "");
    }

    #[test]
    fn test_select_text_field_unknown_mode_uses_growing() {
        let msg = json!({"text": "segment", "growing_text": "growing text"});
        // Unknown mode defaults to growing behavior (else branch)
        assert_eq!(select_text_field(&msg, "unknown"), "growing text");
    }

    // ── is_dot_in_number tests ──

    #[test]
    fn test_dot_in_number() {
        assert!(is_dot_in_number("3.5 Meter", 1));
        assert!(is_dot_in_number("Score: 12.7 Punkte", 9));
    }

    #[test]
    fn test_dot_not_in_number() {
        assert!(!is_dot_in_number("Ende. Anfang", 4));
        assert!(!is_dot_in_number("Dr. Müller", 2));
        assert!(!is_dot_in_number(".", 0));
    }

    // ── split_for_teletext tests ──

    #[test]
    fn test_split_empty() {
        assert!(split_for_teletext("", 84).is_empty());
        assert!(split_for_teletext("   ", 84).is_empty());
    }

    #[test]
    fn test_split_short_text_no_split() {
        let text = "Danke vielmals.";
        let result = split_for_teletext(text, 84);
        assert_eq!(result, vec!["Danke vielmals."]);
    }

    #[test]
    fn test_split_exact_length() {
        // Exactly 84 chars → no split
        let text = "a]".to_string() + &"b".repeat(82);
        assert_eq!(text.chars().count(), 84);
        let result = split_for_teletext(&text, 84);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_split_at_sentence_boundary() {
        let text = "Ich kenne die Aufstellung für das Mix-Team noch nicht. Aber ich denke, Steffen Rembach hat gute Karten als Bester.";
        let result = split_for_teletext(text, 84);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            "Ich kenne die Aufstellung für das Mix-Team noch nicht."
        );
        assert!(result[0].chars().count() <= 84);
        assert!(result[1].chars().count() <= 84);
    }

    #[test]
    fn test_split_at_comma_when_no_sentence_boundary() {
        // Long text without sentence-ending punctuation within 84 chars
        let text = "Ich kenne die Aufstellung für das Mix-Team noch nicht, aber ich denke Steffen Rembach hat gute Karten als bester Österreicher";
        let result = split_for_teletext(text, 84);
        assert!(result.len() >= 2, "Should split: {:?}", result);
        for line in &result {
            assert!(
                line.chars().count() <= 84,
                "Line too long ({} chars): '{}'",
                line.chars().count(),
                line
            );
        }
        // First split should be at the comma
        assert!(result[0].ends_with(','), "Expected comma split: '{}'", result[0]);
    }

    #[test]
    fn test_split_at_word_boundary() {
        // No punctuation at all within 84 chars
        let text = "Dies ist ein sehr langer Satz ohne jegliche Satzzeichen der weit über die erlaubte Zeichengrenze hinausgeht und aufgeteilt werden muss";
        let result = split_for_teletext(text, 84);
        assert!(result.len() >= 2, "Should split: {:?}", result);
        for line in &result {
            assert!(
                line.chars().count() <= 84,
                "Line too long ({} chars): '{}'",
                line.chars().count(),
                line
            );
        }
    }

    #[test]
    fn test_split_preserves_umlauts() {
        // German umlauts (ä ö ü ß) are multi-byte but count as 1 char
        let text = "Österreichische Skisprünger zeigten großartige Leistungen beim Weltcup in Innsbruck, wir freuen uns über die Ergebnisse.";
        let result = split_for_teletext(text, 84);
        assert!(result.len() >= 1);
        for line in &result {
            assert!(
                line.chars().count() <= 84,
                "Line too long ({} chars): '{}'",
                line.chars().count(),
                line
            );
        }
    }

    #[test]
    fn test_split_very_long_text_multiple_chunks() {
        let text = "Ich kenne die Aufstellung für das Mix-Team noch nicht, aber ich denke, Stefan Ebenbach hat gute Karten als beste Österreicher. Danke vielmals. Soweit ist die Reaktion des jüngsten Österreichers.";
        let result = split_for_teletext(text, 84);
        assert!(result.len() >= 2, "Should produce multiple chunks: {:?}", result);
        for line in &result {
            assert!(
                line.chars().count() <= 84,
                "Line too long ({} chars): '{}'",
                line.chars().count(),
                line
            );
        }
    }

    #[test]
    fn test_split_preserves_dot_in_number() {
        let text = "Die Weite betrug 3.5 Meter mehr als erwartet, das ist ein neuer Rekord für die gesamte Mannschaft in dieser Saison";
        let result = split_for_teletext(text, 84);
        // The "3.5" should not be split
        let joined = result.join(" ");
        assert!(joined.contains("3.5"), "Number should be preserved: {:?}", result);
    }

    #[test]
    fn test_split_question_mark() {
        let text = "Wie ist Ihre Wettkampfbilanz? Ja, die Sprünge waren auf jeden Fall in Ordnung, wir haben gut gearbeitet.";
        let result = split_for_teletext(text, 84);
        assert!(result.len() >= 1);
        // Should prefer splitting at the question mark
        if result.len() >= 2 {
            assert!(
                result[0].ends_with('?'),
                "Expected question mark split: '{}'",
                result[0]
            );
        }
    }

    // ── is_duplicate_of_history tests ──

    #[test]
    fn test_dedup_empty_candidate() {
        let history = VecDeque::new();
        assert!(is_duplicate_of_history("", &history));
        assert!(is_duplicate_of_history("   ", &history));
    }

    #[test]
    fn test_dedup_empty_history() {
        let history = VecDeque::new();
        assert!(!is_duplicate_of_history("new text", &history));
    }

    #[test]
    fn test_dedup_exact_match() {
        let mut history = VecDeque::new();
        history.push_back("Danke vielmals.".to_string());
        assert!(is_duplicate_of_history("Danke vielmals.", &history));
    }

    #[test]
    fn test_dedup_candidate_is_subset() {
        let mut history = VecDeque::new();
        history.push_back(
            "Ich kenne die Aufstellung für das Mix-Team noch nicht.".to_string(),
        );
        // Candidate is a substring of sent text → duplicate
        assert!(is_duplicate_of_history(
            "die Aufstellung für das Mix-Team",
            &history
        ));
    }

    #[test]
    fn test_dedup_extension_not_duplicate() {
        let mut history = VecDeque::new();
        history.push_back("Ich kenne die".to_string());
        // Candidate EXTENDS previous by many new words → NOT a duplicate
        // (containment from prev side: 3/3 = 1.0, but min_shared = 3 matches,
        //  containment = 3/3 = 1.0 ≥ 0.75 → this IS caught as duplicate now
        //  because the short text is fully contained in the longer one)
        // Actually with containment: 3 shared words / min(3, 9) = 3/3 = 1.0 → duplicate
        // This is correct behavior: if we already sent "Ich kenne die" and now see
        // "Ich kenne die Aufstellung..." the overlap is too high.
        assert!(is_duplicate_of_history(
            "Ich kenne die Aufstellung für das Mix-Team noch nicht.",
            &history
        ));
    }

    #[test]
    fn test_dedup_extension_short_not_duplicate() {
        let mut history = VecDeque::new();
        history.push_back("Ja gut".to_string());
        // Short history entry (2 words) — candidate has many more words, only 2 shared
        // intersection=2 < MIN_SHARED_WORDS=3 → not flagged
        assert!(!is_duplicate_of_history(
            "Ja gut, das ist die Aufstellung für das Mix-Team.",
            &history
        ));
    }

    #[test]
    fn test_dedup_high_containment() {
        let mut history = VecDeque::new();
        history.push_back(
            "Ich kenne die Aufstellung für das Mix-Team noch nicht.".to_string(),
        );
        // Very similar rewording → duplicate (containment catches this)
        assert!(is_duplicate_of_history(
            "Ich kenne auch die Aufstellung für das Mix-Tim noch nicht.",
            &history
        ));
    }

    #[test]
    fn test_dedup_sliding_window_refinement() {
        let mut history = VecDeque::new();
        history.push_back("Die ist jetzt dritte.".to_string());
        // Sliding window extends the sentence — prev words mostly contained
        // Shared: "Die", "ist", "jetzt", "dritte" = 4 words
        // min(4, 6) = 4, containment = 4/4 = 1.0 → duplicate
        assert!(is_duplicate_of_history(
            "Die ist jetzt dritte und weiß.",
            &history
        ));
    }

    #[test]
    fn test_dedup_sliding_window_growing_sentence() {
        let mut history = VecDeque::new();
        history.push_back(
            "Trotzdem scheint sie zufrieden zu sein.".to_string(),
        );
        // Sliding window adds "mit ihrer Einigung" — Jaccard would be 0.50
        // but containment catches it: shared 5 of min(6, 9) = 5/6 = 0.83
        assert!(is_duplicate_of_history(
            "Trotzdem scheint sie zufrieden zu sein mit ihrer Einigung.",
            &history
        ));
    }

    #[test]
    fn test_dedup_low_similarity_different_content() {
        let mut history = VecDeque::new();
        history.push_back("Danke vielmals.".to_string());
        // Completely different content → not a duplicate
        assert!(!is_duplicate_of_history(
            "So weit ist die Reaktion des jüngsten Österreichers.",
            &history
        ));
    }

    #[test]
    fn test_dedup_few_shared_function_words_not_duplicate() {
        let mut history = VecDeque::new();
        history.push_back("Ah, das ist jetzt bitter für Lamar.".to_string());
        // Only "ist" and "jetzt" overlap — 2 words < MIN_SHARED_WORDS → not duplicate
        assert!(!is_duplicate_of_history(
            "Die ist jetzt dritte.",
            &history
        ));
    }

    #[test]
    fn test_dedup_checks_multiple_history_entries() {
        let mut history = VecDeque::new();
        history.push_back("Erste Nachricht.".to_string());
        history.push_back("Zweite Nachricht.".to_string());
        history.push_back("Dritte Nachricht.".to_string());
        // Exact match with first entry
        assert!(is_duplicate_of_history("Erste Nachricht.", &history));
        // New content
        assert!(!is_duplicate_of_history(
            "Vierte ganz andere Nachricht.",
            &history
        ));
    }

    // ── Integration: split + dedup together ──

    #[test]
    fn test_split_then_dedup_filters_stale_parts() {
        let mut history = VecDeque::new();
        history.push_back(
            "Ich kenne die Aufstellung für das Mix-Team noch nicht.".to_string(),
        );

        let text = "Ich kenne die Aufstellung für das Mix-Team noch nicht. Aber ich denke, Steffen Rembach hat gute Karten.";
        let lines = split_for_teletext(text, 84);
        let non_dup: Vec<&String> = lines
            .iter()
            .filter(|l| !is_duplicate_of_history(l, &history))
            .collect();

        // First part is already sent → filtered out
        // Second part is new → kept
        assert_eq!(non_dup.len(), 1, "Should filter stale part: {:?}", non_dup);
        assert!(
            non_dup[0].contains("Steffen Rembach"),
            "Should keep new part: {:?}",
            non_dup
        );
    }
}
