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
    // Banded DP alignment
    // ========================================================================

    /// Find the best-matching contiguous region in the consensus for the new pass.
    /// Returns (start_idx, score) — the consensus index where the new pass text
    /// best overlaps. Uses a sliding window of n-gram matches.
    fn find_best_region(&self, new_norms: &[String]) -> (usize, f32) {
        let m = self.consensus.len();
        let n = new_norms.len();
        if m == 0 || n == 0 { return (0, 0.0); }

        let sample_size = n.min(8);
        let min_overlap = n.min(m).min(3); // Need at least 3 matching positions
        let mut best_offset = 0usize;
        let mut best_score = 0.0f32;
        let mut best_matched = 0usize;

        // Slide new words across consensus to find best overlap position
        // Only consider offsets near the emitted cursor (new text can't be far from it)
        let search_start = if self.emitted_cursor > n + 5 { self.emitted_cursor - n - 5 } else { 0 };
        let search_end = m.min(self.emitted_cursor + n * 2 + 10);

        for offset in search_start..search_end {
            // How many words can overlap at this offset?
            let overlap = n.min(m.saturating_sub(offset));
            if overlap < min_overlap { continue; }

            let check_count = overlap.min(sample_size);
            let mut score = 0.0f32;
            let mut matched = 0usize;

            // Check consecutive words in the overlap region
            for i in 0..check_count {
                let ci = offset + i;
                if ci < m && i < n {
                    let sim = token_similarity_normalized(&new_norms[i], &self.consensus[ci].normalized);
                    if sim >= 0.5 { matched += 1; }
                    score += sim;
                }
            }

            if check_count > 0 {
                let avg = score / check_count as f32;
                // Prefer higher scores, break ties by preferring more overlap
                if avg > best_score || (avg == best_score && matched > best_matched) {
                    best_score = avg;
                    best_offset = offset;
                    best_matched = matched;
                }
            }
        }

        (best_offset, best_score)
    }

    /// Align new pass words to the consensus using banded DP.
    ///
    /// The DP finds the optimal alignment between `new_words` and a region of the
    /// consensus starting at `region_start`. Each cell `dp[i][j]` represents the
    /// best score for aligning `new_words[0..i]` with `consensus[region_start..region_start+j]`.
    ///
    /// Operations: match (score = similarity), gap in new (skip consensus word),
    /// gap in consensus (skip new word = insertion).
    fn align_to_consensus(&self, new_words: &[String]) -> Vec<Option<usize>> {
        let n = new_words.len();
        let m = self.consensus.len();

        if m == 0 {
            return vec![None; n];
        }
        if n == 0 {
            return vec![];
        }

        let band = self.config.alignment_band_width;
        let sim_threshold = self.config.similarity_threshold;

        // Pre-compute normalized forms for new words
        let new_norms: Vec<String> = new_words.iter().map(|w| normalize_token(w)).collect();

        // Step 1: Find the best-matching region in the consensus
        let (region_start, region_score) = self.find_best_region(&new_norms);

        // If no good region found, all words are new
        if region_score < sim_threshold * 0.5 {
            return vec![None; n];
        }

        // Step 2: Banded DP alignment within the region
        // Align new_words[0..n] against consensus[region_start..region_end]
        let region_end = m.min(region_start + n + band);
        let region_len = region_end - region_start;

        // DP matrix: dp[i][j] = best alignment score for new[0..i] vs region[0..j]
        // We want to maximize total similarity.
        // Operations:
        //   match/substitute: dp[i-1][j-1] + similarity(new[i], consensus[region_start+j])
        //   gap in consensus (insert new word): dp[i-1][j] + gap_penalty
        //   gap in new (skip consensus word): dp[i][j-1] + gap_penalty
        let gap_penalty: f32 = -0.1;

        // Only allocate within band: for row i, columns from max(0, i-band) to min(region_len, i+band)
        // Use full DP for simplicity (n * region_len is typically 60 * 80 = 4800, fast enough)
        let rows = n + 1;
        let cols = region_len + 1;
        let mut dp = vec![vec![f32::NEG_INFINITY; cols]; rows];
        let mut trace = vec![vec![0u8; cols]; rows]; // 0=none, 1=match, 2=gap_consensus, 3=gap_new

        // Base cases
        dp[0][0] = 0.0;
        for j in 1..cols {
            if j <= band {
                dp[0][j] = dp[0][j - 1] + gap_penalty;
                trace[0][j] = 3; // skip consensus word
            }
        }
        for i in 1..rows {
            if i <= band {
                dp[i][0] = dp[i - 1][0] + gap_penalty;
                trace[i][0] = 2; // skip new word
            }
        }

        // Fill DP within band
        for i in 1..rows {
            let j_min = if i > band { i - band } else { 1 };
            let j_max = (i + band + 1).min(cols);

            for j in j_min..j_max {
                let ci = region_start + j - 1; // consensus index
                let sim = token_similarity_normalized(&new_norms[i - 1], &self.consensus[ci].normalized);

                // Match/substitute
                if dp[i - 1][j - 1] > f32::NEG_INFINITY {
                    let score = dp[i - 1][j - 1] + if sim >= sim_threshold { sim } else { sim - 0.5 };
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        trace[i][j] = 1;
                    }
                }

                // Gap in consensus (skip new word = insertion)
                if dp[i - 1][j] > f32::NEG_INFINITY {
                    let score = dp[i - 1][j] + gap_penalty;
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        trace[i][j] = 2;
                    }
                }

                // Gap in new (skip consensus word)
                if dp[i][j - 1] > f32::NEG_INFINITY {
                    let score = dp[i][j - 1] + gap_penalty;
                    if score > dp[i][j] {
                        dp[i][j] = score;
                        trace[i][j] = 3;
                    }
                }
            }
        }

        // Step 3: Traceback from best endpoint
        // Find the best score in the last row (all new words consumed)
        let mut best_j = 0;
        let mut best_score = f32::NEG_INFINITY;
        for j in 0..cols {
            if dp[n][j] > best_score {
                best_score = dp[n][j];
                best_j = j;
            }
        }

        // If DP produced no good alignment, all words are new
        if best_score < 0.0 {
            return vec![None; n];
        }

        // Traceback
        let mut alignment = vec![None; n];
        let mut i = n;
        let mut j = best_j;

        while i > 0 && j > 0 {
            match trace[i][j] {
                1 => {
                    // Match: new[i-1] aligns to consensus[region_start + j - 1]
                    let ci = region_start + j - 1;
                    let sim = token_similarity_normalized(&new_norms[i - 1], &self.consensus[ci].normalized);
                    if sim >= sim_threshold {
                        alignment[i - 1] = Some(ci);
                    }
                    // else: substitution (low similarity), treat as unaligned
                    i -= 1;
                    j -= 1;
                }
                2 => {
                    // Gap in consensus: new[i-1] has no match (insertion)
                    i -= 1;
                }
                3 => {
                    // Gap in new: consensus word skipped
                    j -= 1;
                }
                _ => {
                    // Shouldn't happen, but break to avoid infinite loop
                    i -= 1;
                }
            }
        }
        // Handle remaining new words (if traceback ended at j=0)
        // They stay as None (insertions)

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

        // Phase 1: Update existing consensus words that have alignments
        for (ni, aligned_ci) in alignment.iter().enumerate() {
            if let Some(ci) = aligned_ci {
                let cw = &mut self.consensus[*ci];
                cw.record_appearance(&new_words[ni], pass_idx);

                if !cw.is_confirmed && cw.appearances >= threshold {
                    cw.is_confirmed = true;
                    newly_confirmed.push(cw.text.clone());
                }
            }
        }

        // Phase 2: Insert unaligned words at correct positions
        // Find the last aligned consensus index — new words after this extend the consensus
        let _last_aligned_ci: Option<usize> = alignment.iter().rev().find_map(|a| *a);
        let _first_aligned_ci = alignment.iter().find_map(|a| *a);

        // Collect insertions: (position_in_consensus, word)
        let mut insertions: Vec<(usize, String)> = Vec::new();

        for (ni, aligned_ci) in alignment.iter().enumerate() {
            if aligned_ci.is_some() { continue; }

            // Find the nearest aligned neighbors to determine insertion position
            let prev_aligned = alignment[..ni].iter().rev().find_map(|a| *a);
            let next_aligned = alignment[ni + 1..].iter().find_map(|a| *a);

            let insert_pos = match (prev_aligned, next_aligned) {
                (Some(prev), _) => prev + 1,           // After previous aligned word
                (None, Some(next)) => next,             // Before next aligned word
                (None, None) => {
                    // No aligned neighbors at all — append after consensus
                    self.consensus.len()
                }
            };

            insertions.push((insert_pos, new_words[ni].clone()));
        }

        // Sort insertions by position (ascending) and apply with offset tracking.
        // Each insertion shifts subsequent positions by 1.
        insertions.sort_by_key(|(pos, _)| *pos);
        let mut offset = 0usize;
        for (pos, word) in insertions {
            let actual_pos = (pos + offset).min(self.consensus.len());
            self.consensus.insert(actual_pos, ConsensusWord::new(&word, pass_idx));
            offset += 1;
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
