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

    // Stability tracking for trailing sentence finalization
    last_stable_buffer: String,
    stable_count: usize,

    // Anchor context: tokens from recently finalized text, used for overlap
    // detection when working_tokens is empty/small after finalization.
    // This prevents sliding-buffer models (like Canary) from re-introducing
    // already-finalized content.
    anchor_context: Vec<String>,
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
            last_stable_buffer: String::new(),
            stable_count: 0,
            anchor_context: Vec::new(),
        }
    }

    /// Process new transcription text and return what to emit
    ///
    /// The algorithm:
    /// 1. Tokenize new text
    /// 2. Strip any overlap with recently finalized content (anchor_context)
    /// 3. Find best anchor point in working buffer tail
    /// 4. Merge: keep buffer up to anchor, append new text from anchor
    /// 5. If is_final and ends with sentence terminator, finalize the sentence
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

        // Step 1: Strip any overlap with recently finalized content BEFORE anchor search.
        // This is critical for sliding-buffer models (Canary) where new transcriptions
        // contain: [already-finalized prefix] + [working buffer overlap] + [new content].
        // Without stripping first, the anchor search would find the working buffer overlap
        // but include the finalized prefix in the tail-overwrite.
        let context_skip = self.find_context_overlap(&new_tokens);
        let effective_tokens: Vec<&str> = if context_skip > 0 {
            eprintln!(
                "[GrowingTextMerger] Stripped {} overlapping tokens from finalized context",
                context_skip
            );
            new_tokens[context_skip..].to_vec()
        } else {
            new_tokens.clone()
        };
        let effective_text = if context_skip > 0 {
            effective_tokens.join(" ")
        } else {
            text.to_string()
        };

        if effective_tokens.is_empty() {
            // Entire input was finalized overlap — skip
            return self.current_result(false);
        }

        // Step 2: Find best anchor point using the (possibly stripped) effective tokens
        let (anchor_idx, match_len, _score) = self.find_best_anchor(&effective_tokens);

        let tail_changed;

        let mut divergence_fired = false;

        if anchor_idx == self.working_tokens.len() || match_len == 0 {
            // No anchor found in working tokens.

            // Check for divergence (complete restart detection)
            // If the new text's first 5 tokens share < 30% similarity with the
            // last 5 working tokens, this looks like a complete restart
            if self.working_tokens.len() >= 5 && effective_tokens.len() >= 5 {
                let tail_start = self.working_tokens.len() - 5;
                let tail_slice = &self.working_tokens[tail_start..];
                let new_head: Vec<String> = effective_tokens[..5].iter().map(|s| s.to_string()).collect();
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
                    divergence_fired = true;
                }
            }

            // Pure append
            if !self.working_buffer.is_empty() {
                self.working_buffer.push(' ');
            }
            self.working_buffer.push_str(&effective_text);
            self.working_tokens
                .extend(effective_tokens.iter().map(|s| s.to_string()));
            tail_changed = false;
        } else {
            // Anchor found - tail overwrite
            // The anchor tells us: existing[anchor_idx..anchor_idx+match_len] overlaps
            // with effective[0..match_len].
            // Strategy: Keep existing[..anchor_idx], then append the effective text
            // (which already has finalized prefix stripped).
            self.working_tokens.truncate(anchor_idx);

            // Rebuild working buffer from kept tokens + effective text
            self.working_buffer = self.working_tokens.join(" ");
            if !self.working_buffer.is_empty() {
                self.working_buffer.push(' ');
            }
            self.working_buffer.push_str(&effective_text);

            // Update working tokens with effective tokens (finalized prefix already stripped)
            self.working_tokens
                .extend(effective_tokens.iter().map(|s| s.to_string()));

            tail_changed = true;
        }

        // Expire anchor_context when working buffer is substantial enough for reliable
        // anchoring (prevents stale context from causing false positive stripping).
        // Using 60 tokens (was 30) to give sliding-buffer models more time to process
        // overlapping content before the context is discarded.
        if self.working_tokens.len() > 60 && !self.anchor_context.is_empty() {
            self.anchor_context.clear();
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

        // Proactively finalize complete sentences that have new text after them.
        // Skip when divergence just fired — the text just arrived fresh, let it
        // accumulate for at least one more push before splitting.
        if !divergence_fired {
            self.finalize_inner_sentences();
        }

        // Stability-based trailing sentence finalization:
        // If the working buffer ends with a sentence terminator and hasn't changed
        // for 3+ consecutive pushes, the model has stabilized — finalize it even
        // without is_final from the transcriber.
        // Skip when divergence just fired — buffer just changed, can't be stable.
        if !divergence_fired {
            if self.working_buffer == self.last_stable_buffer {
                self.stable_count += 1;
            } else {
                self.stable_count = 0;
                self.last_stable_buffer = self.working_buffer.clone();
            }
            if self.stable_count >= 3
                && !self.working_buffer.is_empty()
                && self.ends_with_sentence_terminator(&self.working_buffer)
            {
                eprintln!(
                    "[GrowingTextMerger] Stable trailing sentence detected ({} pushes), finalizing: \"{}\"",
                    self.stable_count,
                    &self.working_buffer.chars().take(80).collect::<String>()
                );
                self.finalize_current_sentence();
                self.stable_count = 0;
                self.last_stable_buffer.clear();
            }
        } else {
            // Reset stability tracking after divergence
            self.stable_count = 0;
            self.last_stable_buffer = self.working_buffer.clone();
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

        // Check for repeated phrases (2-3 word patterns, 3+ repetitions)
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

        // Check for longer phrases (3-5 words) repeated 2+ times.
        // These indicate model stuttering, e.g. "Der Kalender präsentiert. X. Der Kalender präsentiert."
        // Truncate at the start of the second occurrence.
        for pattern_len in 3..=5usize {
            if self.working_tokens.len() < pattern_len * 2 {
                continue;
            }
            for i in 0..=(self.working_tokens.len() - pattern_len * 2) {
                // Search for a second occurrence of the pattern after position i
                for j in (i + pattern_len)..=(self.working_tokens.len() - pattern_len) {
                    let matches = (0..pattern_len).all(|k| {
                        self.working_tokens[i + k].to_lowercase()
                            == self.working_tokens[j + k].to_lowercase()
                    });
                    if matches {
                        // Ensure at least one word in the pattern is a content word (not a trivial
                        // function word), to avoid false positives on "in der Stadt" etc.
                        let has_content = (0..pattern_len).any(|k| {
                            let w = self.working_tokens[i + k].to_lowercase();
                            !Self::is_dedup_stopword(&w) && w.len() > 2
                        });
                        if has_content {
                            eprintln!(
                                "[GrowingTextMerger] Long phrase repeated (2x) at {}: '{}'",
                                i,
                                self.working_tokens[i..i + pattern_len].join(" ")
                            );
                            // Truncate at the second occurrence
                            self.working_tokens.truncate(j);
                            self.working_buffer = self.working_tokens.join(" ");
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Find overlap between new tokens and anchor_context (recently finalized tokens).
    /// Returns the number of leading tokens from `new_tokens` that overlap with
    /// finalized content and should be stripped.
    ///
    /// Also checks for overlap in the MIDDLE/END of new_tokens (when the sliding
    /// buffer re-transcribes content that was already finalized).
    fn find_context_overlap(&self, new_tokens: &[&str]) -> usize {
        if self.anchor_context.is_empty() || new_tokens.len() < 3 {
            return 0;
        }

        // Strategy: find the longest run of new_tokens that matches a subsequence
        // of anchor_context. We use a more lenient threshold (0.6) than the normal
        // anchor search because sliding-buffer models produce varied re-transcriptions.
        // We also allow up to 1 non-matching token in a window (gap tolerance).
        //
        // We search ALL starting positions in new_tokens, not just the beginning,
        // because the overlap may be embedded in the middle of the new text.

        let ctx_len = self.anchor_context.len();
        let context_sim_threshold = 0.6; // More lenient than config.min_each_sim
        let mut best_new_start = 0usize;
        let mut best_match_len = 0usize;

        for new_start in 0..new_tokens.len() {
            let max_new_remaining = new_tokens.len() - new_start;
            if max_new_remaining < 3 {
                break;
            }

            for ctx_start in 0..ctx_len {
                let max_match = std::cmp::min(ctx_len - ctx_start, max_new_remaining);
                if max_match < 3 {
                    continue;
                }

                // Count matching tokens with gap tolerance (allow 1 mismatch)
                let mut match_len = 0;
                let mut mismatches = 0;
                for j in 0..max_match {
                    let sim = Self::token_similarity(
                        &self.anchor_context[ctx_start + j],
                        new_tokens[new_start + j],
                    );
                    if sim >= context_sim_threshold {
                        match_len = j + 1;
                    } else {
                        mismatches += 1;
                        if mismatches > 1 {
                            break;
                        }
                        // Allow 1 gap — still count position but don't reset
                        match_len = j + 1;
                    }
                }

                // Subtract trailing mismatches
                let effective_len = match_len.saturating_sub(mismatches);

                // Accept if:
                // 1. Effective match is at least 3 tokens long
                // 2. Match extends to within 3 tokens of the end of anchor_context
                //    (meaning these tokens were near the tail of finalized content)
                if effective_len >= 3 && (ctx_start + match_len + 3 >= ctx_len) {
                    let total_strip = new_start + match_len;
                    if total_strip > best_new_start + best_match_len {
                        best_new_start = new_start;
                        best_match_len = match_len;
                    }
                }
            }
        }

        if best_match_len >= 3 {
            // Strip everything from the beginning up to and including the overlap
            best_new_start + best_match_len
        } else {
            0
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

        // Dedup: skip if this text heavily overlaps with a recent finalized entry
        if self.is_duplicate_of_recent(&self.working_buffer) {
            // Don't finalize — just clear the working buffer to avoid re-processing
            // But accumulate tokens as anchor context so overlap detection still works
            self.anchor_context.extend(self.working_tokens.iter().cloned());
            let max_context = 100;
            if self.anchor_context.len() > max_context {
                let start = self.anchor_context.len() - max_context;
                self.anchor_context = self.anchor_context[start..].to_vec();
            }
            self.working_buffer.clear();
            self.working_tokens.clear();
            return;
        }

        // Accumulate anchor context for future overlap detection.
        // This is critical for sliding-buffer models (Canary) where the next
        // transcription will overlap with just-finalized content.
        // We EXTEND (not replace) to cover multiple consecutive finalizations.
        self.anchor_context.extend(self.working_tokens.iter().cloned());
        let max_context = 100;
        if self.anchor_context.len() > max_context {
            let start = self.anchor_context.len() - max_context;
            self.anchor_context = self.anchor_context[start..].to_vec();
        }

        let sentence = FinalizedSentence {
            text: std::mem::take(&mut self.working_buffer),
            start_time: 0.0, // Could be populated if we track timing
            end_time: 0.0,
        };

        self.finalized_sentences.push(sentence);
        self.working_tokens.clear();
    }

    /// Proactively finalize complete sentences in the working buffer when new text
    /// has started after them. If the buffer contains "First sentence. Second sentence.
    /// Third partial text", everything up to the last terminator with ≥ 2 words after it
    /// is finalized, and only "Third partial text" remains as the working buffer.
    fn finalize_inner_sentences(&mut self) {
        // Find the LAST sentence terminator that has ≥ 2 words after it
        let mut split_pos = None;
        for (i, c) in self.working_buffer.char_indices() {
            let is_terminator = match c {
                '!' | '?' | '。' | '！' | '？' => true,
                '.' => !Self::is_dot_in_number(&self.working_buffer, i),
                _ => false,
            };
            if is_terminator {
                let after_pos = i + c.len_utf8();
                let after = self.working_buffer[after_pos..].trim();
                if after.split_whitespace().count() >= 2 {
                    split_pos = Some(after_pos);
                }
            }
        }

        if let Some(pos) = split_pos {
            let to_finalize = self.working_buffer[..pos].trim().to_string();
            let to_keep = self.working_buffer[pos..].trim().to_string();

            if !to_finalize.is_empty()
                && to_finalize.len() > 2
                && to_finalize.chars().any(|c| c.is_alphanumeric())
                && !self.is_duplicate_of_recent(&to_finalize)
            {
                // Accumulate finalized tokens as anchor context for overlap detection
                let finalized_tokens: Vec<String> = to_finalize
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
                self.anchor_context.extend(finalized_tokens.iter().cloned());
                let max_context = 100;
                if self.anchor_context.len() > max_context {
                    let start = self.anchor_context.len() - max_context;
                    self.anchor_context = self.anchor_context[start..].to_vec();
                }

                self.finalized_sentences.push(FinalizedSentence {
                    text: to_finalize,
                    start_time: 0.0,
                    end_time: 0.0,
                });
                self.working_buffer = to_keep;
                self.working_tokens = self
                    .working_buffer
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();
            }
        }
    }

    /// Normalize a word for dedup comparison: lowercase + strip non-alphanumeric chars.
    fn normalize_word_for_dedup(word: &str) -> String {
        word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase()
    }

    /// Common function words excluded from content-word overlap calculation.
    /// These appear frequently across unrelated sentences and inflate bag-of-words scores.
    const DEDUP_STOPWORDS: &'static [&'static str] = &[
        // German articles
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einen", "einem", "eines",
        // German prepositions
        "in", "im", "am", "an", "auf", "aus", "bei", "mit", "nach", "von",
        "zu", "zum", "zur", "für", "über", "unter", "vor", "durch", "gegen", "um", "bis",
        // German conjunctions/particles
        "und", "oder", "aber", "doch", "sondern", "dass", "wenn", "weil", "ob",
        "ja", "nein", "na", "auch", "noch", "nur", "schon", "nicht", "denn", "mal",
        // German pronouns
        "ich", "du", "er", "sie", "es", "wir", "ihr", "mich", "mir", "dich", "dir",
        "sich", "uns", "euch", "ihm", "ihn", "ihnen", "man",
        // German auxiliary/modal verbs
        "ist", "sind", "war", "hat", "haben", "habe", "hatte",
        "wird", "werden", "wurde", "kann", "soll", "muss", "darf",
        // English articles/prepositions
        "the", "a", "an", "on", "at", "by", "for", "of", "to", "from", "with",
        "out", "up", "into", "about", "over",
        // English conjunctions/pronouns
        "and", "or", "but", "so", "as", "if", "that", "this", "than",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "them",
        // English auxiliary verbs
        "is", "are", "was", "were", "be", "been", "has", "have", "had",
        "do", "does", "did", "will", "would", "can", "could", "may", "not", "no",
    ];

    /// Check if a normalized word is a common function word (stopword).
    fn is_dedup_stopword(word: &str) -> bool {
        Self::DEDUP_STOPWORDS.contains(&word)
    }

    /// Check if text is a near-duplicate of any of the last N finalized entries.
    ///
    /// Uses three complementary checks:
    /// 1. All-word bag-of-words overlap (original, with tighter asymmetric threshold)
    /// 2. Content-word overlap (stopwords removed, fuzzy matching, lower thresholds)
    /// 3. Contiguous sequence match (4+ consecutive words matching with fuzzy similarity)
    fn is_duplicate_of_recent(&self, text: &str) -> bool {
        let candidate_words: Vec<String> = text.split_whitespace()
            .map(Self::normalize_word_for_dedup)
            .filter(|w| !w.is_empty())
            .collect();
        if candidate_words.len() < 3 {
            return false; // Too short to meaningfully dedup
        }

        // Pre-compute content words (non-stopwords) for the candidate
        let candidate_content: Vec<&String> = candidate_words.iter()
            .filter(|w| !Self::is_dedup_stopword(w))
            .collect();

        // Check against last 5 finalized entries
        let check_count = std::cmp::min(5, self.finalized_sentences.len());
        for i in (self.finalized_sentences.len() - check_count)..self.finalized_sentences.len() {
            let existing = &self.finalized_sentences[i].text;
            let existing_words: Vec<String> = existing.split_whitespace()
                .map(Self::normalize_word_for_dedup)
                .filter(|w| !w.is_empty())
                .collect();
            if existing_words.len() < 3 {
                continue;
            }

            // --- Check 1: All-word bag-of-words overlap ---
            let matching_all = candidate_words.iter()
                .filter(|w| existing_words.contains(w))
                .count();
            let o_cand_all = matching_all as f32 / candidate_words.len() as f32;
            let o_exist_all = matching_all as f32 / existing_words.len() as f32;

            // Symmetric: both directions have high overlap
            if o_cand_all >= 0.7 && o_exist_all >= 0.7 {
                return true;
            }
            // Asymmetric: one text is a near-complete subset of the other
            if (o_cand_all >= 0.80 && o_exist_all >= 0.40)
                || (o_exist_all >= 0.80 && o_cand_all >= 0.40)
            {
                return true;
            }

            // --- Check 2: Content-word overlap (stopwords removed, fuzzy matching) ---
            let existing_content: Vec<&String> = existing_words.iter()
                .filter(|w| !Self::is_dedup_stopword(w))
                .collect();

            if candidate_content.len() >= 2 && existing_content.len() >= 2 {
                let matching_content = candidate_content.iter()
                    .filter(|cw| {
                        existing_content.iter()
                            .any(|ew| Self::token_similarity(cw, ew) >= 0.8)
                    })
                    .count();
                let o_cand_content = matching_content as f32 / candidate_content.len() as f32;
                let o_exist_content = matching_content as f32 / existing_content.len() as f32;

                if o_cand_content >= 0.55 && o_exist_content >= 0.55 {
                    return true;
                }
                if (o_cand_content >= 0.70 && o_exist_content >= 0.35)
                    || (o_exist_content >= 0.70 && o_cand_content >= 0.35)
                {
                    return true;
                }
            }

            // --- Check 3: Contiguous sequence match (4+ consecutive words) ---
            // If 4+ consecutive words from the candidate appear in the same order
            // in the existing text (with fuzzy matching), it's a strong overlap signal.
            let min_contiguous = 4;
            if candidate_words.len() >= min_contiguous && existing_words.len() >= min_contiguous {
                let found_contiguous = Self::has_contiguous_match(
                    &candidate_words, &existing_words, min_contiguous,
                );
                if found_contiguous {
                    return true;
                }
            }
        }
        false
    }

    /// Check if two word sequences share a contiguous subsequence of `min_len` or more words,
    /// using fuzzy token matching (similarity >= 0.8).
    fn has_contiguous_match(a: &[String], b: &[String], min_len: usize) -> bool {
        if a.len() < min_len || b.len() < min_len {
            return false;
        }
        for a_start in 0..=(a.len() - min_len) {
            for b_start in 0..=(b.len() - min_len) {
                let max_match = std::cmp::min(a.len() - a_start, b.len() - b_start);
                let mut match_len = 0;
                for k in 0..max_match {
                    if Self::token_similarity(&a[a_start + k], &b[b_start + k]) >= 0.8 {
                        match_len += 1;
                        if match_len >= min_len {
                            return true;
                        }
                    } else {
                        break;
                    }
                }
            }
        }
        false
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

                if !to_finalize.is_empty()
                    && to_finalize.len() > 2
                    && to_finalize.chars().any(|c| c.is_alphanumeric())
                    && !self.is_duplicate_of_recent(&to_finalize)
                {
                    // Accumulate finalized tokens as anchor context for overlap detection
                    let finalized_tokens: Vec<String> = to_finalize
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                    self.anchor_context.extend(finalized_tokens.iter().cloned());
                    let max_context = 100;
                    if self.anchor_context.len() > max_context {
                        let start = self.anchor_context.len() - max_context;
                        self.anchor_context = self.anchor_context[start..].to_vec();
                    }

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

    /// Flush the working buffer: finalize whatever remains, even if incomplete.
    /// Call this when the session ends to avoid losing trailing text.
    /// Returns the flushed text (if any) so the caller can emit a FINAL.
    pub fn flush(&mut self) -> Option<String> {
        let trimmed = self.working_buffer.trim();
        if trimmed.is_empty() {
            return None;
        }
        let text = trimmed.to_string();
        // Skip degenerate fragments
        if text.len() <= 2 || !text.chars().any(|c| c.is_alphanumeric()) {
            self.working_buffer.clear();
            self.working_tokens.clear();
            return None;
        }
        // No dedup on flush — this is the last chance to emit trailing text.
        // The working buffer may partially overlap a recent FINAL (e.g. the
        // model revised text that was already finalized via divergence) but
        // it also contains new content that would otherwise be lost.
        self.finalized_sentences.push(FinalizedSentence {
            text: std::mem::take(&mut self.working_buffer),
            start_time: 0.0,
            end_time: 0.0,
        });
        self.working_tokens.clear();
        Some(text)
    }

    /// Reset the merger state
    pub fn reset(&mut self) {
        self.finalized_sentences.clear();
        self.working_buffer.clear();
        self.working_tokens.clear();
        self.previous_buffer.clear();
        self.last_stable_buffer.clear();
        self.stable_count = 0;
        self.anchor_context.clear();
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

    // ========================================================================
    // Basic operations
    // ========================================================================

    #[test]
    fn test_basic_append() {
        let mut merger = GrowingTextMerger::new();

        let result1 = merger.push("Hello", false);
        assert_eq!(result1.buffer, "Hello");
        assert!(!result1.tail_changed);

        let result2 = merger.push("Hello world", false);
        assert_eq!(result2.buffer, "Hello world");
    }

    #[test]
    fn test_empty_input() {
        let mut merger = GrowingTextMerger::new();

        let result = merger.push("", false);
        assert_eq!(result.buffer, "");
        assert!(!result.tail_changed);
        assert_eq!(result.finalized_count, 0);
    }

    #[test]
    fn test_whitespace_only_input() {
        let mut merger = GrowingTextMerger::new();

        let result = merger.push("   \t\n  ", false);
        assert_eq!(result.buffer, "");
    }

    #[test]
    fn test_single_word() {
        let mut merger = GrowingTextMerger::new();
        let result = merger.push("Hallo", false);
        assert_eq!(result.buffer, "Hallo");
        assert_eq!(result.current_sentence, "Hallo");
    }

    // ========================================================================
    // Tail overwrite / anchor matching
    // ========================================================================

    #[test]
    fn test_tail_overwrite() {
        let mut merger = GrowingTextMerger::new();

        merger.push("wie jene bei Paleo K", false);
        let result = merger.push("wie jene bei Paleo Kastriza", false);

        assert!(result.buffer.contains("Kastriza"));
        assert!(!result.buffer.contains(" K "));
    }

    #[test]
    fn test_tail_overwrite_correction() {
        let mut merger = GrowingTextMerger::new();

        // Simulating a growing transcription where the model corrects itself
        merger.push("Das ist ein Tets", false);
        let result = merger.push("Das ist ein Test der Qualität", false);

        // The corrected word "Test" should replace "Tets"
        assert!(result.buffer.contains("Test"));
        assert!(result.buffer.contains("Qualität"));
    }

    #[test]
    fn test_growing_sequence() {
        let mut merger = GrowingTextMerger::new();

        let texts = [
            "wie",
            "wie jene",
            "wie jene bei",
            "wie jene bei Paleo",
            "wie jene bei Paleo Kastriza",
            "wie jene bei Paleo Kastriza oder die Zwillingsbucht Porto Timoni.",
        ];

        for text in texts {
            merger.push(text, text.ends_with('.'));
        }

        let final_text = merger.get_full_transcript();
        assert!(final_text.contains("Paleo Kastriza"));
        assert!(final_text.contains("Porto Timoni"));
    }

    // ========================================================================
    // Finalization
    // ========================================================================

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
    fn test_finalization_with_is_final() {
        let mut merger = GrowingTextMerger::new();

        // Sentence ending + is_final should finalize
        merger.push("Guten Morgen.", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);
        assert_eq!(merger.get_finalized_sentences()[0].text, "Guten Morgen.");
    }

    #[test]
    fn test_no_finalization_without_terminator() {
        let mut merger = GrowingTextMerger::new();

        // is_final but no sentence terminator → should NOT finalize
        merger.push("Guten Morgen", true);
        assert_eq!(merger.get_finalized_sentences().len(), 0);
    }

    #[test]
    fn test_inner_sentence_finalization() {
        let mut merger = GrowingTextMerger::new();

        // Push text with a complete inner sentence
        merger.push("Erster Satz. Zweiter Satz. Dritter Teil", false);

        // The inner sentences should be finalized (since there are ≥2 words after the last terminator)
        let finalized = merger.get_finalized_sentences();
        assert!(finalized.len() >= 1, "Expected at least 1 finalized sentence, got {}", finalized.len());
    }

    #[test]
    fn test_question_mark_finalization() {
        let mut merger = GrowingTextMerger::new();
        merger.push("Wie geht es Ihnen?", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);
    }

    #[test]
    fn test_exclamation_finalization() {
        let mut merger = GrowingTextMerger::new();
        merger.push("Das ist toll!", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);
    }

    #[test]
    fn test_multiple_sentences_sequential() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Erster Satz.", true);
        merger.push("Zweiter Satz.", true);
        merger.push("Dritter Satz.", true);

        assert_eq!(merger.get_finalized_sentences().len(), 3);
    }

    // ========================================================================
    // Dot-in-number handling
    // ========================================================================

    #[test]
    fn test_dot_in_number() {
        assert!(GrowingTextMerger::is_dot_in_number("25.000", 2));
        assert!(GrowingTextMerger::is_dot_in_number("3.14", 1));
        assert!(!GrowingTextMerger::is_dot_in_number("end.", 3));
        assert!(!GrowingTextMerger::is_dot_in_number(".", 0));
    }

    #[test]
    fn test_number_in_text_not_finalized() {
        let mut merger = GrowingTextMerger::new();

        // "25.000" should NOT trigger inner finalization
        merger.push("Es waren 25.000 Menschen dort und es war super", false);
        let finalized = merger.get_finalized_sentences();
        // The "25.000" dot should not cause finalization
        assert_eq!(finalized.len(), 0, "Number dot should not trigger finalization");
    }

    // ========================================================================
    // Token similarity and normalization
    // ========================================================================

    #[test]
    fn test_token_similarity() {
        assert_eq!(GrowingTextMerger::token_similarity("hello", "hello"), 1.0);
        assert_eq!(GrowingTextMerger::token_similarity("Hello", "hello"), 1.0);
        assert!(GrowingTextMerger::token_similarity("hello", "helo") > 0.7);
        assert!(GrowingTextMerger::token_similarity("hello", "world") < 0.5);
    }

    #[test]
    fn test_token_similarity_empty() {
        // Both empty should be 1.0
        assert_eq!(GrowingTextMerger::token_similarity("", ""), 1.0);
        // One empty → 0.0
        assert_eq!(GrowingTextMerger::token_similarity("hello", ""), 0.0);
        assert_eq!(GrowingTextMerger::token_similarity("", "hello"), 0.0);
    }

    #[test]
    fn test_token_similarity_german_umlauts() {
        // Similar tokens with umlauts
        assert!(GrowingTextMerger::token_similarity("Qualität", "Qualitat") > 0.7);
        assert!(GrowingTextMerger::token_similarity("über", "uber") > 0.7);
    }

    #[test]
    fn test_normalize_token() {
        assert_eq!(GrowingTextMerger::normalize_token("Hello!"), "hello");
        assert_eq!(GrowingTextMerger::normalize_token("don't"), "dont");
        assert_eq!(GrowingTextMerger::normalize_token("Test."), "test");
    }

    #[test]
    fn test_normalize_token_unicode() {
        assert_eq!(GrowingTextMerger::normalize_token("Über"), "über");
        assert_eq!(GrowingTextMerger::normalize_token("café,"), "café");
    }

    // ========================================================================
    // Levenshtein distance
    // ========================================================================

    #[test]
    fn test_levenshtein_identical() {
        assert_eq!(GrowingTextMerger::levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        assert_eq!(GrowingTextMerger::levenshtein_distance("", ""), 0);
        assert_eq!(GrowingTextMerger::levenshtein_distance("abc", ""), 3);
        assert_eq!(GrowingTextMerger::levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn test_levenshtein_single_edit() {
        assert_eq!(GrowingTextMerger::levenshtein_distance("cat", "bat"), 1);
        assert_eq!(GrowingTextMerger::levenshtein_distance("cat", "cats"), 1);
        assert_eq!(GrowingTextMerger::levenshtein_distance("cat", "at"), 1);
    }

    #[test]
    fn test_levenshtein_known_values() {
        assert_eq!(GrowingTextMerger::levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(GrowingTextMerger::levenshtein_distance("sunday", "saturday"), 3);
    }

    // ========================================================================
    // Repetition detection
    // ========================================================================

    #[test]
    fn test_repetition_consecutive_words() {
        let mut merger = GrowingTextMerger::new();

        // Push text with 3 consecutive identical words
        merger.push("Das ist ist ist ein Problem", false);

        // After push, repetition should be detected and truncated
        let transcript = merger.get_full_transcript();
        // Should not contain "ist ist ist"
        let count = transcript.matches(" ist").count();
        assert!(count < 3, "Repetition should be truncated, got: {}", transcript);
    }

    #[test]
    fn test_repetition_phrase() {
        let mut merger = GrowingTextMerger::new();

        // Push text with repeated 2-word phrase
        merger.push("hello world hello world hello world other text", false);

        let transcript = merger.get_full_transcript();
        let count = transcript.matches("hello world").count();
        assert!(count < 3, "Phrase repetition should be truncated, got: {}", transcript);
    }

    #[test]
    fn test_no_false_repetition() {
        let mut merger = GrowingTextMerger::new();

        // Normal text with some repeated words (but not 3+)
        merger.push("Die Kinder und die Eltern sind da", false);
        let transcript = merger.get_full_transcript();
        assert!(transcript.contains("die Eltern"), "Normal text should not be truncated");
    }

    // ========================================================================
    // Dedup (is_duplicate_of_recent)
    // ========================================================================

    #[test]
    fn test_dedup_prevents_duplicate_finalization() {
        let mut merger = GrowingTextMerger::new();

        // Finalize the same sentence twice
        merger.push("Guten Morgen allerseits.", true);
        let count_after_first = merger.get_finalized_sentences().len();

        merger.push("Guten Morgen allerseits.", true);
        let count_after_second = merger.get_finalized_sentences().len();

        assert_eq!(count_after_first, 1);
        // Second should be deduped
        assert_eq!(count_after_second, count_after_first,
            "Duplicate sentence should not be finalized again");
    }

    #[test]
    fn test_dedup_allows_different_sentences() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Erster Satz hier.", true);
        merger.push("Zweiter Satz dort.", true);

        assert_eq!(merger.get_finalized_sentences().len(), 2);
    }

    // ========================================================================
    // Flush and reset
    // ========================================================================

    #[test]
    fn test_flush_returns_working_buffer() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Teilweise geschriebener Satz", false);
        let flushed = merger.flush();

        assert!(flushed.is_some());
        assert!(flushed.unwrap().contains("Teilweise"));
    }

    #[test]
    fn test_flush_empty() {
        let mut merger = GrowingTextMerger::new();
        assert!(merger.flush().is_none());
    }

    #[test]
    fn test_flush_after_finalization() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Complete sentence.", true);
        // Working buffer should be empty after finalization
        let flushed = merger.flush();
        assert!(flushed.is_none());
    }

    #[test]
    fn test_reset() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Some text.", true);
        merger.push("More text", false);

        merger.reset();

        assert_eq!(merger.get_finalized_sentences().len(), 0);
        assert_eq!(merger.get_full_transcript(), "");
    }

    // ========================================================================
    // Custom config
    // ========================================================================

    #[test]
    fn test_custom_config() {
        let config = GrowingTextConfig {
            search_back_tokens: 20,
            max_match_tokens: 10,
            min_each_sim: 0.9,
            min_avg_sim: 0.95,
            working_sentences: 3,
        };
        let mut merger = GrowingTextMerger::with_config(config);

        merger.push("Test mit Konfiguration.", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);
    }

    // ========================================================================
    // Delta calculation
    // ========================================================================

    #[test]
    fn test_delta_on_append() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Hello", false);
        let result = merger.push("Hello world", false);

        // Delta should be the newly added text
        assert!(!result.delta.is_empty());
    }

    #[test]
    fn test_delta_on_tail_change() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Das ist ein Tets", false);
        let result = merger.push("Das ist ein Test", false);

        // When tail changes, delta should reflect the working buffer
        if result.tail_changed {
            assert!(!result.delta.is_empty());
        }
    }

    // ========================================================================
    // Compact working list
    // ========================================================================

    #[test]
    fn test_compact_working_list_at_threshold() {
        let config = GrowingTextConfig {
            working_sentences: 3,
            ..Default::default()
        };
        let mut merger = GrowingTextMerger::with_config(config);

        // Push a long text with many sentences in a single push to trigger compaction
        let text = "Erster Satz hier. Zweiter Satz dort. Dritter Satz jetzt. \
                    Vierter Satz dann. Fünfter Satz noch. Sechster Satz am Ende.";
        merger.push(text, false);

        // With >5 sentences and working_sentences=3, compaction should occur
        let finalized = merger.get_finalized_sentences().len();
        // Either inner finalization or compact should have moved some to finalized
        assert!(finalized > 0, "Expected some sentences to be finalized, got 0. Full transcript: {}",
            merger.get_full_transcript());
    }

    // ========================================================================
    // Full transcript
    // ========================================================================

    #[test]
    fn test_full_transcript_includes_all() {
        let mut merger = GrowingTextMerger::new();

        merger.push("Erster Satz.", true);
        merger.push("Zweiter Satz.", true);
        merger.push("Dritter Teil", false);

        let full = merger.get_full_transcript();
        assert!(full.contains("Erster Satz."));
        assert!(full.contains("Zweiter Satz."));
        assert!(full.contains("Dritter Teil"));
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_very_long_text() {
        let mut merger = GrowingTextMerger::new();

        // Push 400 words (should trigger max working buffer check at 300 tokens)
        let long_text: String = (0..400).map(|i| format!("Wort{}", i)).collect::<Vec<_>>().join(" ");
        merger.push(&long_text, false);

        // Should not panic, and transcript should contain content
        let transcript = merger.get_full_transcript();
        assert!(!transcript.is_empty());
    }

    #[test]
    fn test_unicode_punctuation() {
        let mut merger = GrowingTextMerger::new();

        // Chinese period
        merger.push("这是一个测试。", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);
    }

    #[test]
    fn test_current_sentence_extraction() {
        let mut merger = GrowingTextMerger::new();

        merger.push("First sentence. Second partial", false);
        let finalized = merger.get_finalized_sentences();
        // Inner finalization should have finalized "First sentence."
        // and current_sentence should be something about the partial
        assert!(finalized.len() >= 1);
    }

    // ========================================================================
    // Anchor context (sliding-buffer overlap stripping)
    // ========================================================================

    #[test]
    fn test_anchor_context_strips_finalized_overlap() {
        let mut merger = GrowingTextMerger::new();

        // Simulate canary sliding buffer: first transcription
        merger.push("Ein Wort zum Thema Ortszentren lädt der ORF am 19.", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);

        // Next canary transcription overlaps: starts with old content
        let result = merger.push(
            "Wort zum Thema Ortszentren lädt der ORF am 19. November nach Bischofshofen.",
            false,
        );

        // The overlapping prefix should be stripped; only new content remains
        let transcript = merger.get_full_transcript();
        // Should NOT contain the old content duplicated
        let count = transcript.matches("Ortszentren").count();
        assert!(
            count <= 1,
            "Expected at most 1 occurrence of 'Ortszentren', got {}: {}",
            count,
            transcript
        );
        // New content should be present
        assert!(
            result.buffer.contains("November") || result.buffer.contains("Bischofshofen"),
            "New content should be present: {}",
            result.buffer
        );
    }

    #[test]
    fn test_anchor_context_strips_middle_overlap() {
        let mut merger = GrowingTextMerger::new();

        // Finalize a sentence
        merger.push("Der ORF lädt am 19 November nach Bischofshofen.", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);

        // New input has NEW content first, then repeats finalized content
        let result = merger.push(
            "Meine Anmeldung ist erforderlich. Der ORF lädt am 19 November nach Bischofshofen.",
            false,
        );

        // The working buffer should contain only the new content
        let transcript = merger.get_full_transcript();
        let count = transcript.matches("Bischofshofen").count();
        assert!(
            count <= 1,
            "Expected at most 1 occurrence of 'Bischofshofen', got {}: {}",
            count,
            transcript
        );
    }

    #[test]
    fn test_anchor_context_no_false_strip() {
        let mut merger = GrowingTextMerger::new();

        // Finalize a sentence
        merger.push("Heute ist ein schöner Tag.", true);
        assert_eq!(merger.get_finalized_sentences().len(), 1);

        // New input with completely different content (no overlap)
        let result = merger.push("Morgen wird es regnen und kalt sein.", false);

        // All new content should be preserved
        assert!(
            result.buffer.contains("Morgen wird es regnen"),
            "New content should not be stripped: {}",
            result.buffer
        );
    }

    #[test]
    fn test_anchor_context_cleared_after_use() {
        let mut merger = GrowingTextMerger::new();

        // Finalize
        merger.push("Erster Satz hier ist lang genug.", true);

        // Push with overlap — uses anchor_context
        merger.push("Satz hier ist lang genug. Zweiter Satz kommt jetzt.", false);

        // Push again with completely new content — anchor_context should be cleared
        let result = merger.push("Dritter Satz ist anders.", false);
        assert!(
            result.buffer.contains("Dritter Satz"),
            "Content should not be falsely stripped after context is cleared: {}",
            result.buffer
        );
    }
}
