//! Sentence-aware buffering for transcription segments
//!
//! This module provides a buffer that merges short transcription segments
//! into complete sentences before emitting them, improving readability
//! while maintaining low latency.

use std::time::Instant;
use crate::streaming_transcriber::TranscriptionSegment;

/// Sentence completion mode - determines how aggressively to buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SentenceBufferMode {
    /// No buffering - emit segments immediately (current behavior)
    Off,
    /// Buffer until punctuation detected, max ~500ms extra latency
    #[default]
    Minimal,
    /// Buffer with 1-2s tolerance for better sentences
    Balanced,
    /// Wait for clear sentence boundaries, 3-5s tolerance
    Complete,
}

impl SentenceBufferMode {
    /// Parse from string (for API/config)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "off" | "none" | "disabled" => Self::Off,
            "minimal" | "min" => Self::Minimal,
            "balanced" | "medium" => Self::Balanced,
            "complete" | "full" | "max" => Self::Complete,
            _ => Self::Minimal, // Default
        }
    }

    /// Convert to string for API responses
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Balanced => "balanced",
            Self::Complete => "complete",
        }
    }
}

/// Configuration for sentence buffering
#[derive(Debug, Clone)]
pub struct SentenceBufferConfig {
    /// Buffering mode
    pub mode: SentenceBufferMode,
    /// Maximum time to buffer before forcing emission (seconds)
    pub max_buffer_secs: f32,
    /// Pause duration that triggers emission (milliseconds)
    pub long_pause_ms: u64,
    /// Minimum segments to accumulate before checking for sentence end
    pub min_segments: usize,
}

impl Default for SentenceBufferConfig {
    fn default() -> Self {
        Self::minimal()
    }
}

impl SentenceBufferConfig {
    /// No buffering - pass through immediately
    pub fn off() -> Self {
        Self {
            mode: SentenceBufferMode::Off,
            max_buffer_secs: 0.0,
            long_pause_ms: 0,
            min_segments: 0,
        }
    }

    /// Minimal buffering - quick sentence detection
    /// Emit on punctuation or after 2s, pause after 2s gap
    pub fn minimal() -> Self {
        Self {
            mode: SentenceBufferMode::Minimal,
            max_buffer_secs: 2.0,
            long_pause_ms: 2000,  // 2s gap = real pause
            min_segments: 1,
        }
    }

    /// Balanced buffering - better sentences with moderate latency
    /// Emit on punctuation or after 4s, pause after 3s gap
    pub fn balanced() -> Self {
        Self {
            mode: SentenceBufferMode::Balanced,
            max_buffer_secs: 4.0,
            long_pause_ms: 3000,  // 3s gap = real pause
            min_segments: 1,      // Don't require multiple segments
        }
    }

    /// Complete buffering - wait for full sentences
    /// Emit on punctuation or after 6s, pause after 4s gap
    pub fn complete() -> Self {
        Self {
            mode: SentenceBufferMode::Complete,
            max_buffer_secs: 6.0,
            long_pause_ms: 4000,  // 4s gap = real pause
            min_segments: 1,      // Don't require multiple segments
        }
    }

    /// Create config from mode
    pub fn from_mode(mode: SentenceBufferMode) -> Self {
        match mode {
            SentenceBufferMode::Off => Self::off(),
            SentenceBufferMode::Minimal => Self::minimal(),
            SentenceBufferMode::Balanced => Self::balanced(),
            SentenceBufferMode::Complete => Self::complete(),
        }
    }
}

/// Buffer that accumulates segments and emits complete sentences
pub struct SentenceBuffer {
    /// Pending segments waiting to be merged
    pending_segments: Vec<TranscriptionSegment>,
    /// Combined text from pending segments
    combined_text: String,
    /// Start time of first pending segment
    start_time: Option<f32>,
    /// End time of last pending segment
    end_time: f32,
    /// Speaker from first segment (for consistency)
    speaker: Option<usize>,
    /// Time when last segment was added
    last_segment_time: Option<Instant>,
    /// Configuration
    config: SentenceBufferConfig,
    /// Total inference time accumulated
    total_inference_ms: Option<u32>,
}

impl SentenceBuffer {
    /// Create a new sentence buffer with the given configuration
    pub fn new(config: SentenceBufferConfig) -> Self {
        Self {
            pending_segments: Vec::new(),
            combined_text: String::new(),
            start_time: None,
            end_time: 0.0,
            speaker: None,
            last_segment_time: None,
            config,
            total_inference_ms: None,
        }
    }

    /// Create with default (minimal) configuration
    pub fn with_mode(mode: SentenceBufferMode) -> Self {
        Self::new(SentenceBufferConfig::from_mode(mode))
    }

    /// Check if buffering is disabled
    pub fn is_disabled(&self) -> bool {
        self.config.mode == SentenceBufferMode::Off
    }

    /// Push a segment into the buffer
    /// Returns a merged segment if ready to emit, None otherwise
    pub fn push(&mut self, segment: TranscriptionSegment) -> Option<TranscriptionSegment> {
        // If disabled, pass through immediately
        if self.is_disabled() {
            return Some(segment);
        }

        let now = Instant::now();

        // Check for long pause since last segment (indicates sentence boundary)
        let long_pause = if let Some(last_time) = self.last_segment_time {
            let elapsed_ms = last_time.elapsed().as_millis() as u64;
            elapsed_ms >= self.config.long_pause_ms
        } else {
            false
        };

        // If there was a long pause and we have pending content, emit it first
        if long_pause && !self.pending_segments.is_empty() {
            let merged = self.emit_merged();
            // Then add the new segment to start fresh buffer
            self.add_segment(segment, now);
            return merged;
        }

        // Add segment to buffer
        self.add_segment(segment, now);

        // Check if we should emit
        if self.should_emit() {
            return self.emit_merged();
        }

        None
    }

    /// Add a segment to the pending buffer
    fn add_segment(&mut self, segment: TranscriptionSegment, now: Instant) {
        // Set start time from first segment
        if self.start_time.is_none() {
            self.start_time = Some(segment.start_time);
            self.speaker = segment.speaker;
        }

        // Update end time
        self.end_time = segment.end_time;

        // Accumulate inference time
        if let Some(inf_ms) = segment.inference_time_ms {
            self.total_inference_ms = Some(
                self.total_inference_ms.unwrap_or(0) + inf_ms
            );
        }

        // Append text with space separator
        if !self.combined_text.is_empty() && !segment.text.is_empty() {
            self.combined_text.push(' ');
        }
        self.combined_text.push_str(&segment.text);

        // Store segment and update time
        self.pending_segments.push(segment);
        self.last_segment_time = Some(now);
    }

    /// Check if we should emit the buffered content
    fn should_emit(&self) -> bool {
        if self.pending_segments.is_empty() {
            return false;
        }

        // Check minimum segments requirement
        if self.pending_segments.len() < self.config.min_segments {
            return false;
        }

        // Check for sentence-ending punctuation
        let text = self.combined_text.trim();
        if Self::ends_with_sentence_punctuation(text) {
            return true;
        }

        // Check max buffer duration
        if let Some(start) = self.start_time {
            let duration = self.end_time - start;
            if duration >= self.config.max_buffer_secs {
                return true;
            }
        }

        false
    }

    /// Check if text ends with sentence-ending punctuation
    fn ends_with_sentence_punctuation(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return false;
        }

        // Check for common sentence-ending punctuation
        // Including quotes and parentheses that might follow punctuation
        let endings = ['.', '!', '?', '。', '！', '？'];
        let last_char = trimmed.chars().last().unwrap();

        // Direct punctuation
        if endings.contains(&last_char) {
            return true;
        }

        // Punctuation followed by closing quote/paren
        // Using Unicode escapes for curly quotes: U+201D (right double), U+2019 (right single)
        let closing = ['"', '\'', '\u{201D}', '\u{2019}', ')', ']', '»'];
        if closing.contains(&last_char) {
            // Check second-to-last char
            let chars: Vec<char> = trimmed.chars().collect();
            if chars.len() >= 2 {
                let second_last = chars[chars.len() - 2];
                if endings.contains(&second_last) {
                    return true;
                }
            }
        }

        false
    }

    /// Emit merged segment and clear buffer
    fn emit_merged(&mut self) -> Option<TranscriptionSegment> {
        if self.pending_segments.is_empty() {
            return None;
        }

        let merged = TranscriptionSegment {
            text: self.combined_text.trim().to_string(),
            start_time: self.start_time.unwrap_or(0.0),
            end_time: self.end_time,
            speaker: self.speaker,
            confidence: None, // Could average confidences if needed
            is_final: true,
            inference_time_ms: self.total_inference_ms,
        };

        // Clear buffer
        self.pending_segments.clear();
        self.combined_text.clear();
        self.start_time = None;
        self.end_time = 0.0;
        self.speaker = None;
        self.last_segment_time = None;
        self.total_inference_ms = None;

        Some(merged)
    }

    /// Flush any remaining buffered content
    /// Call this when the stream ends or on disconnect
    pub fn flush(&mut self) -> Option<TranscriptionSegment> {
        self.emit_merged()
    }

    /// Check if buffer has pending content
    pub fn has_pending(&self) -> bool {
        !self.pending_segments.is_empty()
    }

    /// Get number of pending segments
    pub fn pending_count(&self) -> usize {
        self.pending_segments.len()
    }

    /// Get the current buffered text (for debugging/display)
    pub fn buffered_text(&self) -> &str {
        &self.combined_text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(text: &str, start: f32, end: f32) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_string(),
            start_time: start,
            end_time: end,
            speaker: Some(0),
            confidence: None,
            is_final: true,
            inference_time_ms: Some(50),
        }
    }

    // ========================================================================
    // Mode configuration
    // ========================================================================

    #[test]
    fn test_off_mode_passthrough() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::off());
        let seg = make_segment("Hello world", 0.0, 1.0);
        let result = buffer.push(seg.clone());
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Hello world");
    }

    #[test]
    fn test_off_mode_is_disabled() {
        let buffer = SentenceBuffer::new(SentenceBufferConfig::off());
        assert!(buffer.is_disabled());
    }

    #[test]
    fn test_minimal_mode_not_disabled() {
        let buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());
        assert!(!buffer.is_disabled());
    }

    #[test]
    fn test_mode_from_str() {
        assert_eq!(SentenceBufferMode::from_str("off"), SentenceBufferMode::Off);
        assert_eq!(SentenceBufferMode::from_str("minimal"), SentenceBufferMode::Minimal);
        assert_eq!(SentenceBufferMode::from_str("balanced"), SentenceBufferMode::Balanced);
        assert_eq!(SentenceBufferMode::from_str("complete"), SentenceBufferMode::Complete);
        // Unknown defaults to minimal
        assert_eq!(SentenceBufferMode::from_str("unknown"), SentenceBufferMode::Minimal);
    }

    #[test]
    fn test_mode_as_str() {
        assert_eq!(SentenceBufferMode::Off.as_str(), "off");
        assert_eq!(SentenceBufferMode::Minimal.as_str(), "minimal");
        assert_eq!(SentenceBufferMode::Balanced.as_str(), "balanced");
        assert_eq!(SentenceBufferMode::Complete.as_str(), "complete");
    }

    #[test]
    fn test_with_mode_factory() {
        let buffer = SentenceBuffer::with_mode(SentenceBufferMode::Balanced);
        assert!(!buffer.is_disabled());
    }

    // ========================================================================
    // Sentence punctuation detection
    // ========================================================================

    #[test]
    fn test_sentence_detection() {
        assert!(SentenceBuffer::ends_with_sentence_punctuation("Hello."));
        assert!(SentenceBuffer::ends_with_sentence_punctuation("What?"));
        assert!(SentenceBuffer::ends_with_sentence_punctuation("Wow!"));
        assert!(SentenceBuffer::ends_with_sentence_punctuation("He said \"Hello.\""));
        assert!(!SentenceBuffer::ends_with_sentence_punctuation("Hello"));
        assert!(!SentenceBuffer::ends_with_sentence_punctuation("Hello,"));
    }

    #[test]
    fn test_sentence_detection_unicode() {
        // Chinese punctuation
        assert!(SentenceBuffer::ends_with_sentence_punctuation("测试。"));
        assert!(SentenceBuffer::ends_with_sentence_punctuation("测试！"));
        assert!(SentenceBuffer::ends_with_sentence_punctuation("测试？"));
    }

    #[test]
    fn test_sentence_detection_curly_quotes() {
        // English curly quotes: U+201C opening, U+201D closing (supported)
        let english = format!("He said \u{201C}Hello.\u{201D}");
        assert!(SentenceBuffer::ends_with_sentence_punctuation(&english));
        // Closing single quote U+2019
        let single = format!("He said \u{2018}Hello.\u{2019}");
        assert!(SentenceBuffer::ends_with_sentence_punctuation(&single));
        // Guillemets
        assert!(SentenceBuffer::ends_with_sentence_punctuation("Il a dit .»"));
    }

    #[test]
    fn test_sentence_detection_empty() {
        assert!(!SentenceBuffer::ends_with_sentence_punctuation(""));
    }

    // ========================================================================
    // Buffering behavior
    // ========================================================================

    #[test]
    fn test_minimal_buffering() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        let seg1 = make_segment("Hello", 0.0, 0.5);
        assert!(buffer.push(seg1).is_none());

        let seg2 = make_segment("world.", 0.5, 1.0);
        let result = buffer.push(seg2);
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Hello world.");
    }

    #[test]
    fn test_max_duration_emit() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig {
            mode: SentenceBufferMode::Minimal,
            max_buffer_secs: 1.0,
            long_pause_ms: 500,
            min_segments: 1,
        });

        let seg = make_segment("This is a long segment without punctuation", 0.0, 1.5);
        let result = buffer.push(seg);
        assert!(result.is_some());
    }

    #[test]
    fn test_multiple_segments_accumulate() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        let seg1 = make_segment("Erster", 0.0, 0.3);
        assert!(buffer.push(seg1).is_none());
        assert_eq!(buffer.pending_count(), 1);

        let seg2 = make_segment("Teil", 0.3, 0.6);
        assert!(buffer.push(seg2).is_none());
        assert_eq!(buffer.pending_count(), 2);

        assert!(buffer.has_pending());

        let seg3 = make_segment("Ende.", 0.6, 1.0);
        let result = buffer.push(seg3);
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Erster Teil Ende.");
    }

    #[test]
    fn test_buffered_text() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        let seg = make_segment("Pending text", 0.0, 0.5);
        buffer.push(seg);

        assert_eq!(buffer.buffered_text(), "Pending text");
    }

    #[test]
    fn test_balanced_mode_buffering() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::balanced());

        let seg = make_segment("Short.", 0.0, 0.3);
        let result = buffer.push(seg);
        // Balanced mode has longer buffer - short sentence may still be emitted
        // since it ends with punctuation
        assert!(result.is_some());
    }

    #[test]
    fn test_inference_time_accumulation() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        let seg1 = TranscriptionSegment {
            text: "Hello".to_string(),
            start_time: 0.0,
            end_time: 0.5,
            speaker: Some(0),
            confidence: None,
            is_final: true,
            inference_time_ms: Some(100),
        };
        buffer.push(seg1);

        let seg2 = TranscriptionSegment {
            text: "world.".to_string(),
            start_time: 0.5,
            end_time: 1.0,
            speaker: Some(0),
            confidence: None,
            is_final: true,
            inference_time_ms: Some(200),
        };
        let result = buffer.push(seg2).unwrap();

        // Should take max inference time from accumulated segments
        assert!(result.inference_time_ms.is_some());
    }

    // ========================================================================
    // Flush
    // ========================================================================

    #[test]
    fn test_flush() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        let seg = make_segment("Incomplete", 0.0, 0.5);
        assert!(buffer.push(seg).is_none());

        let result = buffer.flush();
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Incomplete");

        assert!(!buffer.has_pending());
    }

    #[test]
    fn test_flush_empty() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());
        assert!(buffer.flush().is_none());
    }

    #[test]
    fn test_flush_preserves_timing() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        // Use short timings that stay under max_buffer_secs (2.0)
        let seg1 = make_segment("Part one", 0.0, 0.5);
        buffer.push(seg1);
        let seg2 = make_segment("part two", 0.5, 1.0);
        buffer.push(seg2);

        let flushed = buffer.flush().unwrap();
        assert_eq!(flushed.start_time, 0.0);
        assert_eq!(flushed.end_time, 1.0);
    }

    // ========================================================================
    // Speaker handling
    // ========================================================================

    #[test]
    fn test_speaker_preserved() {
        let mut buffer = SentenceBuffer::new(SentenceBufferConfig::minimal());

        let seg = TranscriptionSegment {
            text: "Hello world.".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            speaker: Some(2),
            confidence: None,
            is_final: true,
            inference_time_ms: None,
        };

        let result = buffer.push(seg).unwrap();
        assert_eq!(result.speaker, Some(2));
    }
}
