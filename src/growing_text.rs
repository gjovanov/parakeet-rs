//! Growing text merger for incremental transcription display
//!
//! Implements a tail-overwrite algorithm that merges partial transcriptions
//! into a growing buffer, showing text building up word-by-word until finalized.
//!
//! Works with both TDT (Parakeet) and Canary models via the unified
//! TranscriptionSegment interface.

use serde::Serialize;

/// Configuration for the growing text merger
#[derive(Debug, Clone)]
pub struct GrowingTextConfig {
    /// How many tokens back to search for anchor point (default: 80)
    pub search_back_tokens: usize,
    /// Maximum anchor match length (default: 40)
    pub max_match_tokens: usize,
    /// Minimum per-token similarity threshold (default: 0.78)
    pub min_each_sim: f32,
    /// Minimum average similarity threshold (default: 0.90)
    pub min_avg_sim: f32,
    /// Number of recent sentences to keep in working list (default: 5)
    pub working_sentences: usize,
}

impl Default for GrowingTextConfig {
    fn default() -> Self {
        Self {
            search_back_tokens: 80,
            max_match_tokens: 40,
            min_each_sim: 0.78,
            min_avg_sim: 0.90,
            working_sentences: 5,
        }
    }
}

/// A finalized sentence stored in the total list
#[derive(Debug, Clone)]
pub struct FinalizedSentence {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
}

/// Result of pushing new text to the merger
#[derive(Debug, Clone, Serialize)]
pub struct GrowingTextResult {
    /// The full growing buffer (finalized + working)
    pub buffer: String,
    /// Just the current working sentence (for live display)
    pub current_sentence: String,
    /// New text added since last update (for client optimization)
    pub delta: String,
    /// Whether the tail was modified (vs pure append)
    pub tail_changed: bool,
    /// Number of finalized sentences
    pub finalized_count: usize,
}

/// Growing text merger that maintains a consistent, incrementally growing transcript
pub struct GrowingTextMerger {
    config: GrowingTextConfig,

    // Total list - all finalized sentences (cold storage)
    finalized_sentences: Vec<FinalizedSentence>,

    // Working list - recent text being actively matched (hot)
    working_buffer: String,
    working_tokens: Vec<String>,

    // Previous state for delta calculation
    previous_buffer: String,
}

impl GrowingTextMerger {
    /// Create a new merger with default configuration
    pub fn new() -> Self {
        Self::with_config(GrowingTextConfig::default())
    }

    /// Create a new merger with custom configuration
    pub fn with_config(config: GrowingTextConfig) -> Self {
        Self {
            config,
            finalized_sentences: Vec::new(),
            working_buffer: String::new(),
            working_tokens: Vec::new(),
            previous_buffer: String::new(),
        }
    }

    /// Process new transcription text and return what to emit
    ///
    /// The algorithm:
    /// 1. Tokenize new text
    /// 2. Find best anchor point in working buffer tail
    /// 3. Merge: keep buffer up to anchor, append new text from anchor
    /// 4. If is_final and ends with sentence terminator, finalize the sentence
    pub fn push(&mut self, text: &str, is_final: bool) -> GrowingTextResult {
        let text = text.trim();
        if text.is_empty() {
            return self.current_result(false);
        }

        // Tokenize new text
        let new_tokens: Vec<&str> = text.split_whitespace().collect();
        if new_tokens.is_empty() {
            return self.current_result(false);
        }

        // Find best anchor point
        let (anchor_idx, match_len, _score) = self.find_best_anchor(&new_tokens);

        let tail_changed;

        if anchor_idx == self.working_tokens.len() || match_len == 0 {
            // No anchor found - check for divergence (complete restart detection)
            // If the new text's first 5 tokens share < 30% similarity with the
            // last 5 working tokens, this looks like a complete restart
            if self.working_tokens.len() >= 5 && new_tokens.len() >= 5 {
                let tail_start = self.working_tokens.len() - 5;
                let tail_slice = &self.working_tokens[tail_start..];
                let new_head: Vec<String> = new_tokens[..5].iter().map(|s| s.to_string()).collect();
                let matching = tail_slice.iter().zip(new_head.iter())
                    .filter(|(a, b)| Self::token_similarity(a, b) > 0.7)
                    .count();
                let similarity = matching as f32 / 5.0;
                if similarity < 0.3 {
                    eprintln!(
                        "[GrowingTextMerger] Divergence detected (similarity {:.0}%), finalizing and restarting",
                        similarity * 100.0
                    );
                    self.finalize_current_sentence();
                }
            }

            // Pure append
            if !self.working_buffer.is_empty() {
                self.working_buffer.push(' ');
            }
            self.working_buffer.push_str(text);
            self.working_tokens
                .extend(new_tokens.iter().map(|s| s.to_string()));
            tail_changed = false;
        } else {
            // Anchor found - tail overwrite
            // The anchor tells us: existing[anchor_idx..anchor_idx+match_len] overlaps with new[0..match_len]
            // Strategy: Keep existing[..anchor_idx], then append the FULL new text
            // This replaces the overlapping part with the new version (which includes corrections)
            self.working_tokens.truncate(anchor_idx);

            // Rebuild working buffer from kept tokens + full new text
            self.working_buffer = self.working_tokens.join(" ");
            if !self.working_buffer.is_empty() {
                self.working_buffer.push(' ');
            }
            self.working_buffer.push_str(text);

            // Update working tokens with full new tokens
            self.working_tokens
                .extend(new_tokens.iter().map(|s| s.to_string()));

            tail_changed = true;
        }

        // Detect and clean up any repetition corruption in working tokens
        self.detect_repetition();

        // Max working buffer check: if too large, force-finalize oldest sentences
        if self.working_tokens.len() > 300 {
            eprintln!(
                "[GrowingTextMerger] Working buffer too large ({} tokens), force-finalizing oldest sentences",
                self.working_tokens.len()
            );
            // Keep only the last ~100 tokens
            let keep_start = self.working_tokens.len() - 100;
            let to_finalize: Vec<String> = self.working_tokens.drain(..keep_start).collect();
            let finalize_text = to_finalize.join(" ");
            if !finalize_text.is_empty() {
                self.finalized_sentences.push(FinalizedSentence {
                    text: finalize_text,
                    start_time: 0.0,
                    end_time: 0.0,
                });
            }
            self.working_buffer = self.working_tokens.join(" ");
        }

        // Check if this is a finalizing update (sentence complete)
        let is_sentence_complete = is_final && self.ends_with_sentence_terminator(&self.working_buffer);

        // Compute current_sentence BEFORE finalizing (so we can show the complete sentence)
        let current_sentence = self.extract_last_sentence(&self.working_buffer);

        // Now finalize if needed
        if is_sentence_complete {
            self.finalize_current_sentence();
        }

        // Compact working list if too many sentences
        self.compact_working_list();

        self.current_result_with_sentence(tail_changed, current_sentence)
    }

    /// Detect and truncate repetition corruption in working tokens.
    /// Scans for any word appearing 3+ times consecutively and truncates before
    /// the first occurrence of the repetition.
    fn detect_repetition(&mut self) {
        if self.working_tokens.len() < 4 {
            return;
        }

        // Check for 3+ consecutive identical words
        let mut consecutive_count = 1usize;
        for i in 1..self.working_tokens.len() {
            if self.working_tokens[i].to_lowercase() == self.working_tokens[i - 1].to_lowercase()
                && self.working_tokens[i].len() > 1
            {
                consecutive_count += 1;
                if consecutive_count >= 3 {
                    let truncate_at = i + 1 - consecutive_count;
                    if truncate_at > 0 {
                        eprintln!(
                            "[GrowingTextMerger] Repetition detected: '{}' x{}, truncating at word {}",
                            self.working_tokens[i], consecutive_count, truncate_at
                        );
                        self.working_tokens.truncate(truncate_at);
                        self.working_buffer = self.working_tokens.join(" ");
                        return;
                    }
                }
            } else {
                consecutive_count = 1;
            }
        }

        // Check for repeated phrases (2-3 word patterns)
        for pattern_len in 2..=3usize {
            if self.working_tokens.len() < pattern_len * 3 {
                continue;
            }
            for i in 0..=(self.working_tokens.len() - pattern_len * 3) {
                let mut pattern_count = 1;
                let mut j = i + pattern_len;
                while j + pattern_len <= self.working_tokens.len() {
                    let matches = (0..pattern_len).all(|k| {
                        self.working_tokens[i + k].to_lowercase()
                            == self.working_tokens[j + k].to_lowercase()
                    });
                    if matches {
                        pattern_count += 1;
                        if pattern_count >= 3 {
                            eprintln!(
                                "[GrowingTextMerger] Repeated phrase detected at {}: '{}'",
                                i,
                                self.working_tokens[i..i + pattern_len].join(" ")
                            );
                            if i > 0 {
                                self.working_tokens.truncate(i);
                                self.working_buffer = self.working_tokens.join(" ");
                            } else {
                                self.working_tokens.clear();
                                self.working_buffer.clear();
                            }
                            return;
                        }
                        j += pattern_len;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Find the best anchor point in the working buffer for the new tokens
    /// Returns (anchor_idx in working_tokens, match_length, score)
    fn find_best_anchor(&self, new_tokens: &[&str]) -> (usize, usize, f32) {
        if self.working_tokens.is_empty() || new_tokens.is_empty() {
            return (self.working_tokens.len(), 0, 0.0);
        }

        // Search in the last `search_back_tokens` of working buffer
        let search_start = self
            .working_tokens
            .len()
            .saturating_sub(self.config.search_back_tokens);

        let mut best_anchor = self.working_tokens.len();
        let mut best_match_len = 0;
        let mut best_score = 0.0f32;

        // Try each starting position in the search window
        for start_idx in search_start..self.working_tokens.len() {
            // Try match lengths from 1 to max_match_tokens
            let max_len = std::cmp::min(
                self.config.max_match_tokens,
                std::cmp::min(
                    self.working_tokens.len() - start_idx,
                    new_tokens.len(),
                ),
            );

            for match_len in 1..=max_len {
                let (avg_sim, all_above_threshold) = self.compute_match_score(
                    &self.working_tokens[start_idx..start_idx + match_len],
                    &new_tokens[..match_len],
                );

                // Accept match if meets thresholds
                if all_above_threshold && avg_sim >= self.config.min_avg_sim {
                    // Prefer longer matches, then higher scores
                    if match_len > best_match_len
                        || (match_len == best_match_len && avg_sim > best_score)
                    {
                        best_anchor = start_idx;
                        best_match_len = match_len;
                        best_score = avg_sim;
                    }
                }
            }
        }

        // If token-level anchor was found, return it
        if best_match_len > 0 {
            return (best_anchor, best_match_len, best_score);
        }

        // Fallback: sentence-level overlap check
        // If the new text's first sentence overlaps > 60% with any sentence in the
        // working buffer's tail (by word containment), treat the overlap point as the anchor.
        // This handles cases like "der M" vs "DM" where Levenshtein fails on short tokens.
        let new_first_sentence = Self::extract_first_sentence_words(new_tokens);
        if new_first_sentence.len() >= 3 {
            let new_normalized: Vec<String> = new_first_sentence
                .iter()
                .map(|w| Self::normalize_for_matching(w))
                .filter(|w| !w.is_empty())
                .collect();

            if !new_normalized.is_empty() {
                // Scan working buffer tail for sentence-level overlap
                // Look at the last few sentences' worth of tokens
                let scan_start = self.working_tokens.len().saturating_sub(self.config.search_back_tokens);
                for start_idx in scan_start..self.working_tokens.len() {
                    let remaining = &self.working_tokens[start_idx..];
                    let remaining_normalized: Vec<String> = remaining
                        .iter()
                        .map(|w| Self::normalize_for_matching(w))
                        .filter(|w| !w.is_empty())
                        .collect();

                    if remaining_normalized.is_empty() {
                        continue;
                    }

                    // Count how many of the new words appear in the remaining working tokens
                    let matching_words = new_normalized.iter()
                        .filter(|w| remaining_normalized.contains(w))
                        .count();
                    let overlap = matching_words as f32 / new_normalized.len() as f32;

                    if overlap >= 0.6 {
                        return (start_idx, 1, overlap);
                    }
                }
            }
        }

        (best_anchor, best_match_len, best_score)
    }

    /// Extract words from the first sentence of a token list (up to first sentence terminator)
    fn extract_first_sentence_words<'a>(tokens: &[&'a str]) -> Vec<&'a str> {
        let mut result = Vec::new();
        for &token in tokens {
            result.push(token);
            let trimmed = token.trim_end_matches(|c: char| c.is_ascii_punctuation());
            let _ = trimmed; // we just check the original
            if token.ends_with('.') || token.ends_with('!') || token.ends_with('?') {
                break;
            }
        }
        result
    }

    /// Normalize a token for matching, stripping articles and common prefixes
    /// that may differ between transcription passes (e.g., "der M" vs "DM")
    fn normalize_for_matching(token: &str) -> String {
        let normalized = Self::normalize_token(token);
        // Strip common German/Romance articles that cause matching failures
        let articles = ["der", "die", "das", "ein", "eine", "des", "dem", "den",
                        "le", "la", "les", "un", "une", "el", "los", "las",
                        "the", "a", "an"];
        if articles.contains(&normalized.as_str()) {
            return String::new(); // Skip articles in matching
        }
        normalized
    }

    /// Compute match score between two token sequences
    /// Returns (average_similarity, all_above_min_threshold)
    fn compute_match_score(&self, existing: &[String], new: &[&str]) -> (f32, bool) {
        if existing.len() != new.len() || existing.is_empty() {
            return (0.0, false);
        }

        let mut total_sim = 0.0;
        let mut all_above = true;

        for (e, n) in existing.iter().zip(new.iter()) {
            let sim = Self::token_similarity(e, n);
            total_sim += sim;
            if sim < self.config.min_each_sim {
                all_above = false;
            }
        }

        let avg_sim = total_sim / existing.len() as f32;
        (avg_sim, all_above)
    }

    /// Compute similarity between two tokens using character-level comparison
    /// Similar to Python's difflib.SequenceMatcher ratio
    fn token_similarity(a: &str, b: &str) -> f32 {
        let a_norm = Self::normalize_token(a);
        let b_norm = Self::normalize_token(b);

        if a_norm == b_norm {
            return 1.0;
        }

        if a_norm.is_empty() || b_norm.is_empty() {
            return 0.0;
        }

        // Use Levenshtein-based similarity ratio
        let distance = Self::levenshtein_distance(&a_norm, &b_norm);
        let max_len = std::cmp::max(a_norm.len(), b_norm.len());
        1.0 - (distance as f32 / max_len as f32)
    }

    /// Normalize token for matching (lowercase, strip punctuation)
    fn normalize_token(token: &str) -> String {
        token
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase()
    }

    /// Compute Levenshtein edit distance
    fn levenshtein_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();

        if m == 0 {
            return n;
        }
        if n == 0 {
            return m;
        }

        // Use two-row optimization for space efficiency
        let mut prev_row: Vec<usize> = (0..=n).collect();
        let mut curr_row: Vec<usize> = vec![0; n + 1];

        for i in 1..=m {
            curr_row[0] = i;
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] {
                    0
                } else {
                    1
                };
                curr_row[j] = std::cmp::min(
                    std::cmp::min(prev_row[j] + 1, curr_row[j - 1] + 1),
                    prev_row[j - 1] + cost,
                );
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        prev_row[n]
    }

    /// Check if a dot at the given byte position is inside a number (e.g., "25.000").
    /// A dot is inside a number if it's preceded by a digit AND followed by a digit.
    fn is_dot_in_number(text: &str, byte_pos: usize) -> bool {
        let bytes = text.as_bytes();
        byte_pos > 0
            && byte_pos + 1 < bytes.len()
            && bytes[byte_pos - 1].is_ascii_digit()
            && bytes[byte_pos + 1].is_ascii_digit()
    }

    /// Check if text ends with sentence terminator
    fn ends_with_sentence_terminator(&self, text: &str) -> bool {
        let trimmed = text.trim_end();
        trimmed.ends_with('.')
            || trimmed.ends_with('!')
            || trimmed.ends_with('?')
            || trimmed.ends_with('。') // Chinese period
            || trimmed.ends_with('！')
            || trimmed.ends_with('？')
    }

    /// Move current working buffer to finalized list
    fn finalize_current_sentence(&mut self) {
        if self.working_buffer.is_empty() {
            return;
        }

        let sentence = FinalizedSentence {
            text: std::mem::take(&mut self.working_buffer),
            start_time: 0.0, // Could be populated if we track timing
            end_time: 0.0,
        };

        self.finalized_sentences.push(sentence);
        self.working_tokens.clear();
    }

    /// Keep only the last N sentences in working list, move older to finalized
    fn compact_working_list(&mut self) {
        // Count sentences in working buffer by counting sentence terminators
        let sentence_count = self.count_sentences_in_working();

        if sentence_count > self.config.working_sentences {
            // Find position to split at
            let sentences_to_finalize = sentence_count - self.config.working_sentences;
            if let Some(split_pos) = self.find_sentence_split_position(sentences_to_finalize) {
                let to_finalize = self.working_buffer[..split_pos].trim().to_string();
                let to_keep = self.working_buffer[split_pos..].trim().to_string();

                if !to_finalize.is_empty() {
                    self.finalized_sentences.push(FinalizedSentence {
                        text: to_finalize,
                        start_time: 0.0,
                        end_time: 0.0,
                    });
                }

                self.working_buffer = to_keep;
                self.working_tokens = self
                    .working_buffer
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            }
        }
    }

    /// Count complete sentences in working buffer
    fn count_sentences_in_working(&self) -> usize {
        let mut count = 0;
        for (i, c) in self.working_buffer.char_indices() {
            match c {
                '!' | '?' | '。' => count += 1,
                '.' if !Self::is_dot_in_number(&self.working_buffer, i) => count += 1,
                _ => {}
            }
        }
        count
    }

    /// Find character position to split after N sentences
    fn find_sentence_split_position(&self, n: usize) -> Option<usize> {
        let mut count = 0;
        for (i, c) in self.working_buffer.char_indices() {
            let is_terminator = match c {
                '!' | '?' | '。' => true,
                '.' => !Self::is_dot_in_number(&self.working_buffer, i),
                _ => false,
            };
            if is_terminator {
                count += 1;
                if count == n {
                    return Some(i + c.len_utf8());
                }
            }
        }
        None
    }

    /// Get current result (extracts current_sentence from working buffer)
    fn current_result(&mut self, tail_changed: bool) -> GrowingTextResult {
        let current_sentence = self.extract_last_sentence(&self.working_buffer);
        self.current_result_with_sentence(tail_changed, current_sentence)
    }

    /// Get current result with a pre-computed current_sentence
    fn current_result_with_sentence(&mut self, tail_changed: bool, current_sentence: String) -> GrowingTextResult {
        // Build full buffer
        let mut buffer = String::new();
        for sentence in &self.finalized_sentences {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(&sentence.text);
        }
        if !self.working_buffer.is_empty() {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(&self.working_buffer);
        }

        // Calculate delta
        let delta = if buffer.len() > self.previous_buffer.len()
            && buffer.starts_with(&self.previous_buffer)
        {
            buffer[self.previous_buffer.len()..].trim_start().to_string()
        } else if tail_changed {
            // Tail was modified, delta is the whole working buffer
            self.working_buffer.clone()
        } else {
            String::new()
        };

        self.previous_buffer = buffer.clone();

        GrowingTextResult {
            buffer,
            current_sentence,
            delta,
            tail_changed,
            finalized_count: self.finalized_sentences.len(),
        }
    }

    /// Extract only the last sentence from text (after the last sentence terminator)
    fn extract_last_sentence(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        // Find the last sentence-ending punctuation (skipping dots inside numbers)
        let mut last_terminator_pos = None;
        for (i, c) in text.char_indices() {
            match c {
                '!' | '?' => last_terminator_pos = Some(i),
                '.' if !Self::is_dot_in_number(text, i) => last_terminator_pos = Some(i),
                _ => {}
            }
        }

        match last_terminator_pos {
            Some(pos) => {
                // Get everything after the last terminator
                let after = &text[pos + 1..];
                let trimmed = after.trim();
                if trimmed.is_empty() {
                    // If nothing after terminator, return the last complete sentence
                    // Find the second-to-last terminator (skipping dots in numbers)
                    let mut prev_pos = None;
                    for (i, c) in text[..pos].char_indices() {
                        match c {
                            '!' | '?' => prev_pos = Some(i),
                            '.' if !Self::is_dot_in_number(text, i) => prev_pos = Some(i),
                            _ => {}
                        }
                    }
                    if let Some(pp) = prev_pos {
                        text[pp + 1..].trim().to_string()
                    } else {
                        // No second terminator - return the whole text as the sentence
                        text.trim().to_string()
                    }
                } else {
                    trimmed.to_string()
                }
            }
            None => {
                // No terminator found, return the whole text
                text.trim().to_string()
            }
        }
    }

    /// Get the full transcript (finalized + working)
    pub fn get_full_transcript(&self) -> String {
        let mut buffer = String::new();
        for sentence in &self.finalized_sentences {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(&sentence.text);
        }
        if !self.working_buffer.is_empty() {
            if !buffer.is_empty() {
                buffer.push(' ');
            }
            buffer.push_str(&self.working_buffer);
        }
        buffer
    }

    /// Get finalized sentences only
    pub fn get_finalized_sentences(&self) -> &[FinalizedSentence] {
        &self.finalized_sentences
    }

    /// Reset the merger state
    pub fn reset(&mut self) {
        self.finalized_sentences.clear();
        self.working_buffer.clear();
        self.working_tokens.clear();
        self.previous_buffer.clear();
    }
}

impl Default for GrowingTextMerger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_append() {
        let mut merger = GrowingTextMerger::new();

        let result1 = merger.push("Hello", false);
        assert_eq!(result1.buffer, "Hello");
        assert!(!result1.tail_changed);

        let result2 = merger.push("Hello world", false);
        assert_eq!(result2.buffer, "Hello world");
        // Should find anchor and extend
    }

    #[test]
    fn test_tail_overwrite() {
        let mut merger = GrowingTextMerger::new();

        merger.push("wie jene bei Paleo K", false);
        let result = merger.push("wie jene bei Paleo Kastriza", false);

        // Should find anchor at "Paleo" and overwrite "K" with "Kastriza"
        assert!(result.buffer.contains("Kastriza"));
        assert!(!result.buffer.contains(" K "));
    }

    #[test]
    fn test_finalization() {
        let mut merger = GrowingTextMerger::new();

        merger.push("This is a test.", true);
        let result = merger.push("Another sentence", false);

        assert_eq!(merger.get_finalized_sentences().len(), 1);
        assert!(result.buffer.contains("This is a test."));
        assert!(result.buffer.contains("Another sentence"));
    }

    #[test]
    fn test_token_similarity() {
        assert_eq!(GrowingTextMerger::token_similarity("hello", "hello"), 1.0);
        assert_eq!(GrowingTextMerger::token_similarity("Hello", "hello"), 1.0);
        assert!(GrowingTextMerger::token_similarity("hello", "helo") > 0.7);
        assert!(GrowingTextMerger::token_similarity("hello", "world") < 0.5);
    }

    #[test]
    fn test_normalize_token() {
        assert_eq!(GrowingTextMerger::normalize_token("Hello!"), "hello");
        assert_eq!(GrowingTextMerger::normalize_token("don't"), "dont");
        assert_eq!(GrowingTextMerger::normalize_token("Test."), "test");
    }

    #[test]
    fn test_growing_sequence() {
        let mut merger = GrowingTextMerger::new();

        // Simulate growing transcription
        let texts = [
            "wie",
            "wie jene",
            "wie jene bei",
            "wie jene bei Paleo",
            "wie jene bei Paleo Kastriza",
            "wie jene bei Paleo Kastriza oder die Zwillingsbucht Porto Timoni.",
        ];

        for text in texts {
            let result = merger.push(text, text.ends_with('.'));
            println!("[{}] {}", if result.tail_changed { "CHANGED" } else { "APPEND" }, result.buffer);
        }

        let final_text = merger.get_full_transcript();
        assert!(final_text.contains("Paleo Kastriza"));
        assert!(final_text.contains("Porto Timoni"));
    }
}
