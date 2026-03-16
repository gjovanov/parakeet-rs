//! Word-level confirmation for sliding-buffer ASR output.
//!
//! The canary model transcribes an 8-second sliding window every ~1 second.
//! Each pass produces 40-60 words with ~85-90% overlap with the previous pass.
//! This module tracks individual words across passes and confirms them when
//! they appear consistently in aligned positions across K consecutive passes.
//!
//! Architecture:
//!   Model output → WordConfirmer.push(text) → confirmed words + unconfirmed tail
//!   → GrowingTextMerger.push(confirmed_text) [sentence detection only]
//!   → FINAL emission when merger finalizes a sentence

// ============================================================================
// Text utilities (shared with growing_text.rs)
// ============================================================================

/// Normalize token for matching (lowercase, strip punctuation)
fn normalize_token(token: &str) -> String {
    token
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Compute Levenshtein edit distance between two strings
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 { return n; }
    if n == 0 { return m; }

    let mut prev_row: Vec<usize> = (0..=n).collect();
    let mut curr_row: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr_row[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr_row[j] = std::cmp::min(
                std::cmp::min(prev_row[j] + 1, curr_row[j - 1] + 1),
                prev_row[j - 1] + cost,
            );
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    prev_row[n]
}

/// Compute similarity between two tokens (0.0 to 1.0)
fn token_similarity(a: &str, b: &str) -> f32 {
    let a_norm = normalize_token(a);
    let b_norm = normalize_token(b);

    if a_norm == b_norm { return 1.0; }
    if a_norm.is_empty() || b_norm.is_empty() { return 0.0; }

    let distance = levenshtein_distance(&a_norm, &b_norm);
    let max_len = std::cmp::max(a_norm.len(), b_norm.len());
    1.0 - (distance as f32 / max_len as f32)
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the word confirmer
#[derive(Debug, Clone)]
pub struct WordConfirmerConfig {
    /// Number of consecutive aligned passes before a word is confirmed (default: 3)
    pub confirmation_threshold: u32,
    /// Minimum token_similarity for alignment (default: 0.8)
    pub similarity_threshold: f32,
    /// DP alignment band width (default: 8)
    pub alignment_band_width: usize,
    /// Force-confirm after this many passes without reaching threshold (default: 6)
    pub force_confirm_after_passes: u32,
    /// Maximum unconfirmed words before force-confirming oldest (default: 50)
    pub max_unconfirmed_words: usize,
    /// Maximum passes to keep in history (default: 8)
    pub max_pass_history: usize,
}

impl Default for WordConfirmerConfig {
    fn default() -> Self {
        Self {
            confirmation_threshold: 3,
            similarity_threshold: 0.8,
            alignment_band_width: 8,
            force_confirm_after_passes: 6,
            max_unconfirmed_words: 50,
            max_pass_history: 8,
        }
    }
}

// ============================================================================
// Core data structures
// ============================================================================

/// A word in the consensus sequence with confirmation tracking
#[derive(Debug, Clone)]
struct ConsensusWord {
    /// Best-voted form of this word
    text: String,
    /// Normalized form for matching
    normalized: String,
    /// Number of passes where this word appeared in aligned position
    appearances: u32,
    /// Pass index when first added to consensus
    first_seen_pass: u32,
    /// Most recent pass where it was aligned
    last_seen_pass: u32,
    /// Whether this word has been confirmed (appearances >= threshold)
    is_confirmed: bool,
    /// Variant forms seen and their counts (for voting on best spelling)
    variants: Vec<(String, u32)>,
}

impl ConsensusWord {
    fn new(text: &str, pass_idx: u32) -> Self {
        let normalized = normalize_token(text);
        Self {
            text: text.to_string(),
            normalized,
            appearances: 1,
            first_seen_pass: pass_idx,
            last_seen_pass: pass_idx,
            is_confirmed: false,
            variants: vec![(text.to_string(), 1)],
        }
    }

    /// Record another appearance, potentially updating the best text form
    fn record_appearance(&mut self, text: &str, pass_idx: u32) {
        self.appearances += 1;
        self.last_seen_pass = pass_idx;

        // Update variant counts
        if let Some(v) = self.variants.iter_mut().find(|(t, _)| t == text) {
            v.1 += 1;
        } else {
            self.variants.push((text.to_string(), 1));
        }

        // Use the most common variant as the display text
        if let Some((best, _)) = self.variants.iter().max_by_key(|(_, c)| *c) {
            self.text = best.clone();
        }
    }
}

/// Result of a push() call
#[derive(Debug, Clone)]
pub struct WordConfirmResult {
    /// Newly confirmed words since last push (may be empty)
    pub newly_confirmed: Vec<String>,
    /// All confirmed words so far (full confirmed transcript)
    pub confirmed_text: String,
    /// Unconfirmed tail words (for PARTIAL display)
    pub unconfirmed_tail: String,
    /// Total confirmed word count
    pub confirmed_count: usize,
    /// Total consensus length
    pub consensus_length: usize,
}

// ============================================================================
// WordConfirmer
// ============================================================================

/// Tracks words across inference passes and confirms them when stable.
pub struct WordConfirmer {
    config: WordConfirmerConfig,
    /// The consensus sequence — our best understanding of the transcript
    consensus: Vec<ConsensusWord>,
    /// Index: everything before this has been emitted
    emitted_cursor: usize,
    /// Pass counter
    pass_count: u32,
    /// Previous pass words (for compound word pre-normalization)
    prev_pass_words: Vec<String>,
}

impl WordConfirmer {
    pub fn new() -> Self {
        Self::with_config(WordConfirmerConfig::default())
    }

    pub fn with_config(config: WordConfirmerConfig) -> Self {
        Self {
            config,
            consensus: Vec::new(),
            emitted_cursor: 0,
            pass_count: 0,
            prev_pass_words: Vec::new(),
        }
    }

    /// Process a new inference pass and return confirmation results.
    ///
    /// Each call represents one full-buffer transcription from the canary model.
    /// Words are tracked across passes and confirmed when they appear in aligned
    /// positions across `confirmation_threshold` consecutive passes.
    pub fn push(&mut self, text: &str) -> WordConfirmResult {
        let text = text.trim();
        if text.is_empty() {
            return self.build_result(vec![]);
        }

        self.pass_count += 1;
        let pass_idx = self.pass_count;

        // Tokenize and apply compound word pre-normalization
        let raw_words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let words = self.normalize_compounds(&raw_words);

        // Align new words against the consensus
        let alignment = self.align_to_consensus(&words);

        // Update consensus based on alignment
        let newly_confirmed = self.update_consensus(&words, &alignment, pass_idx);

        // Age out unmatched consensus words (potential hallucinations)
        self.age_out_unmatched(pass_idx);

        // Force-confirm if too many unconfirmed words accumulate
        let mut force_confirmed = self.force_confirm_overflow(pass_idx);
        let mut all_newly_confirmed = newly_confirmed;
        all_newly_confirmed.append(&mut force_confirmed);

        // Store for next compound normalization
        self.prev_pass_words = words;

        self.build_result(all_newly_confirmed)
    }

    /// Get the full confirmed text
    pub fn confirmed_text(&self) -> String {
        self.consensus.iter()
            .filter(|w| w.is_confirmed)
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get confirmed text that hasn't been emitted yet (new since last advance)
    pub fn unemitted_confirmed_text(&self) -> String {
        self.consensus[self.emitted_cursor..].iter()
            .filter(|w| w.is_confirmed)
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Advance the emitted cursor past all currently confirmed words
    pub fn advance_cursor(&mut self, count: usize) {
        self.emitted_cursor = (self.emitted_cursor + count).min(self.consensus.len());
    }

    /// Flush: force-confirm all remaining words (call at session end)
    pub fn flush(&mut self) -> Vec<String> {
        let mut flushed = Vec::new();
        for word in &mut self.consensus[self.emitted_cursor..] {
            if !word.is_confirmed {
                word.is_confirmed = true;
                flushed.push(word.text.clone());
            }
        }
        flushed
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.consensus.clear();
        self.emitted_cursor = 0;
        self.pass_count = 0;
        self.prev_pass_words.clear();
    }

    // ========================================================================
    // Compound word pre-normalization
    // ========================================================================

    /// Try to rejoin compound words that the model split differently across passes.
    /// E.g., if previous pass had "Salzachblume" and current has "Salzach Blume",
    /// join them back into "Salzachblume" for alignment consistency.
    fn normalize_compounds(&self, words: &[String]) -> Vec<String> {
        if self.prev_pass_words.is_empty() || words.len() < 2 {
            return words.to_vec();
        }

        let mut result: Vec<String> = Vec::with_capacity(words.len());
        let mut i = 0;

        while i < words.len() {
            if i + 1 < words.len() {
                // Try joining consecutive words
                let joined = format!("{}{}", words[i], words[i + 1]);
                let joined_norm = normalize_token(&joined);

                // Check if any word in the previous pass matches the joined form
                let matches_prev = self.prev_pass_words.iter().any(|prev| {
                    token_similarity(&joined_norm, &normalize_token(prev)) > 0.9
                });

                if matches_prev && joined_norm.len() >= 6 {
                    result.push(joined);
                    i += 2;
                    continue;
                }
            }

            // Also check if current single word should be split to match prev pass
            // (less common, skip for now — joining is the dominant case)
            result.push(words[i].clone());
            i += 1;
        }

        result
    }

    // ========================================================================
    // Banded edit-distance alignment
    // ========================================================================

    /// Align new pass words to the consensus sequence using banded DP.
    /// Returns a mapping: for each new word index, the consensus index it aligns to
    /// (or None if it's an insertion).
    fn align_to_consensus(&self, new_words: &[String]) -> Vec<Option<usize>> {
        let n = new_words.len();
        let m = self.consensus.len();

        if m == 0 {
            // No consensus yet — all words are new
            return vec![None; n];
        }

        if n == 0 {
            return vec![];
        }

        let band = self.config.alignment_band_width;
        let sim_threshold = self.config.similarity_threshold;

        // Build similarity matrix (banded — only compute within band)
        // DP for edit-distance alignment with substitution cost based on token similarity
        // State: dp[i][j] = best alignment score for new_words[..i] and consensus[..j]
        // We want to maximize total similarity of aligned pairs

        // Use a simpler greedy anchored approach for efficiency:
        // 1. Find the best matching position in consensus for the first new word
        // 2. Align forward from there, allowing small gaps

        // Find the starting anchor: where does the first new word match in consensus?
        let first_norm = normalize_token(&new_words[0]);
        let mut best_start = 0usize;
        let mut best_sim = 0.0f32;

        // Search within a reasonable window from the emitted cursor
        let search_start = if self.emitted_cursor > band { self.emitted_cursor - band } else { 0 };
        let search_end = m.min(self.emitted_cursor + n + band);

        for j in search_start..search_end {
            let sim = token_similarity_normalized(&first_norm, &self.consensus[j].normalized);
            if sim > best_sim {
                best_sim = sim;
                best_start = j;
            }
        }

        // If no good anchor found, treat all words as new
        if best_sim < sim_threshold * 0.7 {
            return vec![None; n];
        }

        // Forward alignment from the anchor
        let mut alignment = vec![None; n];
        let mut ci = best_start; // consensus index
        let mut ni = 0usize;     // new word index

        while ni < n && ci < m {
            let sim = token_similarity_normalized(
                &normalize_token(&new_words[ni]),
                &self.consensus[ci].normalized,
            );

            if sim >= sim_threshold {
                // Good match — align
                alignment[ni] = Some(ci);
                ni += 1;
                ci += 1;
            } else {
                // Mismatch — try to recover by looking ahead in both sequences
                // Check: is the next consensus word a better match? (deletion in new)
                let skip_consensus = if ci + 1 < m {
                    token_similarity_normalized(
                        &normalize_token(&new_words[ni]),
                        &self.consensus[ci + 1].normalized,
                    )
                } else { 0.0 };

                // Check: is the next new word a better match? (insertion in new)
                let skip_new = if ni + 1 < n {
                    token_similarity_normalized(
                        &normalize_token(&new_words[ni + 1]),
                        &self.consensus[ci].normalized,
                    )
                } else { 0.0 };

                if skip_consensus >= sim_threshold && skip_consensus > skip_new {
                    // Skip one consensus word (it was deleted from new pass)
                    ci += 1;
                } else if skip_new >= sim_threshold {
                    // Current new word is an insertion — don't align it
                    alignment[ni] = None;
                    ni += 1;
                } else {
                    // Neither works — both are different. Mark as insertion and advance both.
                    alignment[ni] = None;
                    ni += 1;
                    ci += 1;
                }
            }
        }

        // Remaining new words have no consensus match
        // (alignment[ni..] stays None by default)

        alignment
    }

    // ========================================================================
    // Consensus update
    // ========================================================================

    /// Update the consensus based on alignment results.
    /// Returns the list of words that became newly confirmed in this pass.
    fn update_consensus(
        &mut self,
        new_words: &[String],
        alignment: &[Option<usize>],
        pass_idx: u32,
    ) -> Vec<String> {
        let mut newly_confirmed = Vec::new();
        let threshold = self.config.confirmation_threshold;

        // Track which consensus words were matched this pass
        let mut matched_consensus: Vec<bool> = vec![false; self.consensus.len()];

        // Process aligned words
        let mut insert_after: Vec<(usize, String)> = Vec::new();

        for (ni, aligned_ci) in alignment.iter().enumerate() {
            match aligned_ci {
                Some(ci) => {
                    // Word aligned to existing consensus entry — update it
                    let cw = &mut self.consensus[*ci];
                    cw.record_appearance(&new_words[ni], pass_idx);
                    matched_consensus[*ci] = true;

                    // Check if this word just became confirmed
                    if !cw.is_confirmed && cw.appearances >= threshold {
                        cw.is_confirmed = true;
                        newly_confirmed.push(cw.text.clone());
                    }
                }
                None => {
                    // New word not in consensus — find insertion position
                    // Insert after the last aligned consensus word, or at end
                    let insert_pos = alignment[..ni].iter().rev()
                        .find_map(|a| *a)
                        .map(|ci| ci + 1)
                        .unwrap_or(if self.consensus.is_empty() { 0 } else { self.consensus.len() });

                    insert_after.push((insert_pos, new_words[ni].clone()));
                }
            }
        }

        // Insert new words into consensus (reverse order to preserve indices)
        // Group insertions by position and insert in order
        insert_after.sort_by_key(|(pos, _)| *pos);
        let mut offset = 0usize;
        for (pos, word) in insert_after {
            let actual_pos = (pos + offset).min(self.consensus.len());
            self.consensus.insert(actual_pos, ConsensusWord::new(&word, pass_idx));
            offset += 1;
            // Adjust emitted_cursor if insertion is before it (keeps cursor pointing to same word)
            if actual_pos < self.emitted_cursor {
                self.emitted_cursor += 1;
            }
        }

        newly_confirmed
    }

    /// Remove consensus words that haven't been seen in recent passes
    /// (likely hallucinations that appeared in only 1-2 passes)
    fn age_out_unmatched(&mut self, current_pass: u32) {
        let staleness_limit = self.config.force_confirm_after_passes;

        // Don't remove confirmed words or words before emitted cursor
        let mut i = self.emitted_cursor;
        while i < self.consensus.len() {
            let w = &self.consensus[i];
            if !w.is_confirmed
                && current_pass - w.last_seen_pass >= staleness_limit
                && w.appearances < 2
            {
                // Hallucination: appeared once and wasn't seen again
                self.consensus.remove(i);
                // Don't increment i — next element shifts into this position
            } else {
                i += 1;
            }
        }
    }

    /// Force-confirm oldest unconfirmed words if we exceed the buffer limit
    fn force_confirm_overflow(&mut self, current_pass: u32) -> Vec<String> {
        let mut force_confirmed = Vec::new();
        let unconfirmed_count = self.consensus[self.emitted_cursor..].iter()
            .filter(|w| !w.is_confirmed)
            .count();

        if unconfirmed_count <= self.config.max_unconfirmed_words {
            return force_confirmed;
        }

        // Force-confirm the oldest unconfirmed words
        let excess = unconfirmed_count - self.config.max_unconfirmed_words;
        let mut confirmed = 0;

        for word in &mut self.consensus[self.emitted_cursor..] {
            if confirmed >= excess { break; }
            if !word.is_confirmed
                && (current_pass - word.first_seen_pass >= self.config.force_confirm_after_passes
                    || word.appearances >= 2)
            {
                word.is_confirmed = true;
                force_confirmed.push(word.text.clone());
                confirmed += 1;
            }
        }

        if !force_confirmed.is_empty() {
            eprintln!(
                "[WordConfirmer] Force-confirmed {} words (unconfirmed buffer overflow: {})",
                force_confirmed.len(), unconfirmed_count
            );
        }

        force_confirmed
    }

    // ========================================================================
    // Result building
    // ========================================================================

    fn build_result(&self, newly_confirmed: Vec<String>) -> WordConfirmResult {
        // Find the contiguous confirmed prefix from emitted_cursor
        // and then everything after the last confirmed word is the tail
        let slice = &self.consensus[self.emitted_cursor..];

        // Find the last confirmed word position in the slice
        let last_confirmed_idx = slice.iter().rposition(|w| w.is_confirmed);

        let (confirmed_words, tail_words) = match last_confirmed_idx {
            Some(last_idx) => {
                // Confirmed = all words up to and including last confirmed (including gaps)
                let confirmed: Vec<&str> = slice[..=last_idx].iter()
                    .filter(|w| w.is_confirmed)
                    .map(|w| w.text.as_str())
                    .collect();
                // Tail = everything after last confirmed word
                let tail: Vec<&str> = slice[last_idx + 1..].iter()
                    .map(|w| w.text.as_str())
                    .collect();
                (confirmed, tail)
            }
            None => {
                // No confirmed words — everything is tail
                let tail: Vec<&str> = slice.iter().map(|w| w.text.as_str()).collect();
                (vec![], tail)
            }
        };

        let total_confirmed = self.consensus.iter().filter(|w| w.is_confirmed).count();

        WordConfirmResult {
            newly_confirmed,
            confirmed_text: confirmed_words.join(" "),
            unconfirmed_tail: tail_words.join(" "),
            confirmed_count: total_confirmed,
            consensus_length: self.consensus.len(),
        }
    }
}

/// Token similarity using pre-normalized strings (avoids double normalization)
fn token_similarity_normalized(a_norm: &str, b_norm: &str) -> f32 {
    if a_norm == b_norm { return 1.0; }
    if a_norm.is_empty() || b_norm.is_empty() { return 0.0; }
    let distance = levenshtein_distance(a_norm, b_norm);
    let max_len = std::cmp::max(a_norm.len(), b_norm.len());
    1.0 - (distance as f32 / max_len as f32)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_confirmation() {
        let mut wc = WordConfirmer::new();

        // Pass 1: initial text
        let r1 = wc.push("Der Kalender präsentiert von der Salzachblume");
        assert_eq!(r1.confirmed_count, 0); // Nothing confirmed yet

        // Pass 2: same text (shifted by one word, some overlap)
        let r2 = wc.push("Kalender präsentiert von der Salzachblume in Nussdorf");
        assert!(r2.confirmed_count < 6); // Some words may have 2 appearances

        // Pass 3: overlapping again
        let r3 = wc.push("präsentiert von der Salzachblume in Nussdorf Mara");
        // After 3 passes, core overlapping words should be confirmed
        assert!(r3.confirmed_count > 0, "Should have confirmed words after 3 passes");
        assert!(r3.confirmed_text.contains("Salzachblume"),
            "Salzachblume should be confirmed: {}", r3.confirmed_text);
    }

    #[test]
    fn test_identical_passes_confirm_fast() {
        let mut wc = WordConfirmer::new();

        let text = "Guten Abend meine Damen und Herren";
        wc.push(text);
        wc.push(text);
        let r = wc.push(text);

        // After 3 identical passes, all words should be confirmed
        assert_eq!(r.confirmed_count, 6);
        assert_eq!(r.confirmed_text, text);
        assert!(r.unconfirmed_tail.is_empty());
    }

    #[test]
    fn test_shifting_window() {
        let mut wc = WordConfirmer::new();

        wc.push("eins zwei drei vier fünf");
        wc.push("zwei drei vier fünf sechs");
        let r = wc.push("drei vier fünf sechs sieben");

        // "drei vier fünf" appeared in all 3 passes
        assert!(r.confirmed_text.contains("drei"));
        assert!(r.confirmed_text.contains("vier"));
        assert!(r.confirmed_text.contains("fünf"));
    }

    #[test]
    fn test_compound_word_normalization() {
        let mut wc = WordConfirmer::new();

        // Pass 1: compound word
        wc.push("Die Salzachblume ist schön");
        // Pass 2: compound split
        wc.push("Die Salzach Blume ist schön");
        // Pass 3: compound word again
        let r = wc.push("Die Salzachblume ist schön");

        // "Die" and "ist" and "schön" should be confirmed
        assert!(r.confirmed_count >= 3, "Should confirm stable words: confirmed={}", r.confirmed_count);
    }

    #[test]
    fn test_unconfirmed_tail() {
        let mut wc = WordConfirmer::new();

        wc.push("alpha beta gamma delta");
        wc.push("alpha beta gamma delta epsilon");
        let r = wc.push("alpha beta gamma delta epsilon zeta");

        // "epsilon" has 2 appearances (not yet 3)
        // "zeta" has 1 appearance
        assert!(!r.unconfirmed_tail.is_empty(),
            "Should have unconfirmed tail words");
    }

    #[test]
    fn test_force_confirm_on_overflow() {
        let config = WordConfirmerConfig {
            confirmation_threshold: 3,
            max_unconfirmed_words: 5,
            force_confirm_after_passes: 2,
            ..Default::default()
        };
        let mut wc = WordConfirmer::with_config(config);

        // Push enough unique words across enough passes to trigger overflow
        wc.push("eins zwei drei vier fünf sechs sieben acht");
        wc.push("eins zwei drei vier fünf sechs sieben acht neun zehn");
        // Pass 3: words from pass 1 are now 3 passes old, and >5 unconfirmed
        wc.push("neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn");
        let r = wc.push("dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn zwanzig");

        // Should have force-confirmed some words (overflow > 5 unconfirmed)
        assert!(r.confirmed_count > 0,
            "Force confirmation should trigger: unconfirmed in consensus={}, confirmed={}",
            r.consensus_length - r.confirmed_count, r.confirmed_count);
    }

    #[test]
    fn test_hallucination_removal() {
        let config = WordConfirmerConfig {
            confirmation_threshold: 3,
            force_confirm_after_passes: 3,
            ..Default::default()
        };
        let mut wc = WordConfirmer::with_config(config);

        // Pass 1: includes a hallucinated word
        wc.push("Der Kalender HALLUZINATION präsentiert");
        // Passes 2-4: without the hallucination
        wc.push("Der Kalender präsentiert von");
        wc.push("Der Kalender präsentiert von der");
        let r = wc.push("Der Kalender präsentiert von der Salzachblume");

        // "HALLUZINATION" should have been aged out
        let all_text = format!("{} {}", r.confirmed_text, r.unconfirmed_tail);
        assert!(!all_text.contains("HALLUZINATION"),
            "Hallucinated word should be removed: {}", all_text);
    }

    #[test]
    fn test_empty_input() {
        let mut wc = WordConfirmer::new();
        let r = wc.push("");
        assert_eq!(r.confirmed_count, 0);
        assert!(r.confirmed_text.is_empty());
    }

    #[test]
    fn test_flush() {
        let mut wc = WordConfirmer::new();
        wc.push("alpha beta gamma");
        wc.push("alpha beta gamma delta");

        let flushed = wc.flush();
        assert!(!flushed.is_empty(), "Flush should force-confirm remaining words");
    }

    #[test]
    fn test_fuzzy_matching() {
        let mut wc = WordConfirmer::new();

        // Slight spelling variations across passes
        wc.push("Wirtschaftskammer Österreich");
        wc.push("Wirtschaftskamer Österreich"); // typo: missing 'm'
        let r = wc.push("Wirtschaftskammer Österreich");

        // Should still confirm despite the typo in pass 2
        assert!(r.confirmed_count >= 1,
            "Fuzzy matching should handle minor typos: confirmed={}", r.confirmed_count);
    }

    #[test]
    fn test_token_similarity_basic() {
        assert_eq!(token_similarity("hello", "hello"), 1.0);
        assert_eq!(token_similarity("hello", ""), 0.0);
        assert!(token_similarity("Wirtschaftskammer", "Wirtschaftskamer") > 0.9);
        assert!(token_similarity("Salzachblume", "Salzach") < 0.7);
    }

    #[test]
    fn test_normalize_token() {
        assert_eq!(normalize_token("Hello!"), "hello");
        assert_eq!(normalize_token("Dr."), "dr");
        assert_eq!(normalize_token("self-test"), "selftest");
    }
}
