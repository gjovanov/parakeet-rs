//! Quasi-realtime transcription using ParakeetTDT with ring buffer.
//!
//! Inspired by the EOU streaming approach:
//! - Maintains a ring buffer for audio context (default 15 seconds)
//! - Processes buffer periodically (every ~2 seconds of new audio)
//! - Extracts full context from buffer for accurate transcription
//! - Only emits "confirmed" tokens that won't change
//!
//! ## Key differences from EOU streaming:
//! - TDT has no encoder cache, so we reprocess the buffer each time
//! - Higher latency (~5-10s) but much better transcription quality
//! - Word-level timestamps for accurate speaker attribution

use crate::decoder::TimedToken;
use crate::error::Result;
use crate::execution::ModelConfig as ExecutionConfig;
use crate::parakeet_tdt::ParakeetTDT;
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use std::collections::VecDeque;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Calculate RMS (root mean square) energy of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Configuration for quasi-realtime TDT processing
#[derive(Debug, Clone)]
pub struct RealtimeTDTConfig {
    /// Ring buffer size in seconds (default: 15.0)
    /// Larger buffer = more context = better quality but more processing
    pub buffer_size_secs: f32,

    /// How often to process (in seconds of new audio, default: 2.0)
    /// Smaller = lower latency but more CPU usage
    pub process_interval_secs: f32,

    /// Confirmed zone: tokens older than this won't change (default: 3.0)
    /// This is the "safe zone" before the buffer end
    pub confirm_threshold_secs: f32,

    /// Enable pause-based confirmation (default: false)
    /// When true, tokens are confirmed when a pause is detected, enabling
    /// lower latency with better quality
    pub pause_based_confirm: bool,

    /// Minimum pause duration to trigger confirmation (default: 0.4s)
    /// Only used when pause_based_confirm is true
    pub pause_threshold_secs: f32,

    /// Energy threshold for silence detection (default: 0.01)
    /// Audio RMS below this is considered silence
    pub silence_energy_threshold: f32,

    /// Enable lookahead mode for better transcription quality (default: false)
    /// When true, waits for multiple pause segments before emitting,
    /// giving ASR more context for better accuracy
    pub lookahead_mode: bool,

    /// Number of pause segments to look ahead (default: 2)
    /// After initial ramp-up, always keeps this many future segments
    /// for context when transcribing the segment to emit
    pub lookahead_segments: usize,
}

impl Default for RealtimeTDTConfig {
    fn default() -> Self {
        Self {
            buffer_size_secs: 15.0,
            process_interval_secs: 2.0,
            confirm_threshold_secs: 3.0,
            pause_based_confirm: false,
            pause_threshold_secs: 0.4,
            silence_energy_threshold: 0.01,
            lookahead_mode: false,
            lookahead_segments: 2,
        }
    }
}

impl RealtimeTDTConfig {
    /// Low latency mode: faster updates, smaller buffer
    pub fn low_latency() -> Self {
        Self {
            buffer_size_secs: 10.0,
            process_interval_secs: 1.5,
            confirm_threshold_secs: 2.0,
            pause_based_confirm: false,
            pause_threshold_secs: 0.4,
            silence_energy_threshold: 0.01,
            lookahead_mode: false,
            lookahead_segments: 2,
        }
    }

    /// High quality mode: larger buffer, more context
    pub fn high_quality() -> Self {
        Self {
            buffer_size_secs: 20.0,
            process_interval_secs: 3.0,
            confirm_threshold_secs: 4.0,
            pause_based_confirm: false,
            pause_threshold_secs: 0.4,
            silence_energy_threshold: 0.01,
            lookahead_mode: false,
            lookahead_segments: 2,
        }
    }

    /// Pause-based low latency mode: uses silence detection for confirmation
    /// This gives both low latency AND good quality by confirming at natural pauses
    pub fn pause_based() -> Self {
        Self {
            buffer_size_secs: 10.0,      // Keep good context
            process_interval_secs: 0.3,  // Process frequently
            confirm_threshold_secs: 1.5, // Fallback threshold - increased for better quality
            pause_based_confirm: true,   // Enable pause detection
            pause_threshold_secs: 0.5,   // 500ms pause triggers confirmation (increased for accuracy)
            silence_energy_threshold: 0.005, // More sensitive silence detection
            lookahead_mode: false,
            lookahead_segments: 2,
        }
    }

    /// Speedy pause-based mode: optimized for lower latency while maintaining quality
    /// Uses pause detection with balanced threshold for complete sentences
    pub fn speedy() -> Self {
        Self {
            buffer_size_secs: 8.0,       // Slightly smaller buffer
            process_interval_secs: 0.2,  // Process very frequently
            confirm_threshold_secs: 0.5, // Fallback threshold
            pause_based_confirm: true,   // Enable pause detection
            pause_threshold_secs: 0.6,   // 600ms pause for better sentence boundaries
            silence_energy_threshold: 0.008, // Balanced sensitivity
            lookahead_mode: false,
            lookahead_segments: 2,
        }
    }

    /// Lookahead mode: uses sliding window of pause segments for best quality
    /// Transcribes multiple segments together but only emits the oldest one,
    /// giving ASR future context for better accuracy at segment boundaries
    pub fn lookahead() -> Self {
        Self {
            buffer_size_secs: 20.0,      // Larger buffer for multiple segments
            process_interval_secs: 0.3,  // Process frequently
            confirm_threshold_secs: 1.5, // Fallback threshold
            pause_based_confirm: true,   // Must use pause detection
            pause_threshold_secs: 0.5,   // 500ms pause threshold
            silence_energy_threshold: 0.005, // Sensitive silence detection
            lookahead_mode: true,        // Enable lookahead
            lookahead_segments: 2,       // Keep 2 future segments for context
        }
    }

    // Keep old API for compatibility
    pub fn chunk_size_secs(&self) -> f32 {
        self.buffer_size_secs
    }

    pub fn overlap_secs(&self) -> f32 {
        self.confirm_threshold_secs
    }
}

/// A confirmed segment with timestamps
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub tokens: Vec<TimedToken>,
    pub is_final: bool,
}

/// A pause-delimited segment for lookahead transcription
/// Stores audio between two pauses along with timing information
#[derive(Debug, Clone)]
struct PauseSegmentInfo {
    /// Global start time of this segment (when speech started after previous pause)
    start_time: f32,
    /// Global end time of this segment (when pause was detected)
    end_time: f32,
    /// Audio samples for this segment
    audio: Vec<f32>,
}

/// Result from processing
#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub segments: Vec<Segment>,
    pub full_text: String,
    pub buffer_time: f32,
    pub needs_more_audio: bool,
}

/// Quasi-realtime TDT transcriber with ring buffer
pub struct RealtimeTDT {
    model: ParakeetTDT,
    config: RealtimeTDTConfig,

    /// Ring buffer for audio (like EOU's 4s buffer, but larger for TDT)
    audio_buffer: VecDeque<f32>,
    buffer_size_samples: usize,

    /// Tracking state
    total_samples_received: usize,
    samples_since_last_process: usize,
    process_interval_samples: usize,

    /// Time tracking for confirmed zone
    confirmed_until: f32,

    /// Accumulated results
    finalized_segments: Vec<Segment>,
    pending_tokens: Vec<TimedToken>,

    /// Pause detection state
    silence_start_time: Option<f32>,  // When current silence started (global time)
    last_pause_end_time: f32,         // When last detected pause ended (for confirmation)
    pause_boundary_time: Option<f32>, // Time boundary where a pause was detected (confirm tokens before this)

    /// Track emitted token end times to prevent duplicates
    /// This is more reliable than text-based deduplication
    last_emitted_token_end: f32,

    // === Lookahead mode state ===
    /// Queue of pause-delimited segments for lookahead transcription
    /// When lookahead_mode is enabled, we accumulate segments here
    lookahead_segments: VecDeque<PauseSegmentInfo>,

    /// Audio accumulated since last pause (current segment being built)
    current_segment_audio: Vec<f32>,

    /// Start time of current segment being built
    current_segment_start: f32,

    /// Whether we're currently in speech (vs silence)
    in_speech: bool,
}

impl RealtimeTDT {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeTDTConfig>,
    ) -> Result<Self> {
        let model = ParakeetTDT::from_pretrained(model_path, exec_config)?;
        let config = config.unwrap_or_default();

        let buffer_size_samples = (config.buffer_size_secs * SAMPLE_RATE as f32) as usize;
        let process_interval_samples = (config.process_interval_secs * SAMPLE_RATE as f32) as usize;

        Ok(Self {
            model,
            config,
            audio_buffer: VecDeque::with_capacity(buffer_size_samples),
            buffer_size_samples,
            total_samples_received: 0,
            samples_since_last_process: 0,
            process_interval_samples,
            confirmed_until: 0.0,
            finalized_segments: Vec::new(),
            pending_tokens: Vec::new(),
            silence_start_time: None,
            last_pause_end_time: 0.0,
            pause_boundary_time: None,
            last_emitted_token_end: 0.0,
            // Lookahead mode state
            lookahead_segments: VecDeque::new(),
            current_segment_audio: Vec::new(),
            current_segment_start: 0.0,
            in_speech: false,
        })
    }

    /// Push audio samples (like EOU's transcribe method)
    ///
    /// Ring buffer approach:
    /// 1. Add samples to ring buffer
    /// 2. Trim buffer to max size (keeping most recent)
    /// 3. Detect pauses for smarter confirmation (if enabled)
    /// 4. Process when enough new audio accumulated
    /// 5. Only emit tokens in "confirmed zone"
    ///
    /// Lookahead mode:
    /// - Accumulates audio into pause-delimited segments
    /// - When enough segments are buffered, transcribes all together
    /// - Only emits the oldest segment (with future context)
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<ChunkResult> {
        // Calculate current time before adding samples
        let current_time = self.total_samples_received as f32 / SAMPLE_RATE as f32;

        // Lookahead mode uses a different processing path
        if self.config.lookahead_mode {
            return self.push_audio_lookahead(samples, current_time);
        }

        // Pause detection (if enabled)
        if self.config.pause_based_confirm {
            self.detect_pause(samples, current_time);
        }

        // Add to ring buffer
        self.audio_buffer.extend(samples.iter().copied());
        self.total_samples_received += samples.len();
        self.samples_since_last_process += samples.len();

        // Trim buffer to keep only most recent samples (ring buffer behavior)
        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Check if we should process
        // Need: minimum buffer AND enough new audio since last process
        let min_buffer_samples = (self.config.confirm_threshold_secs * 2.0 * SAMPLE_RATE as f32) as usize;

        // Debug: log buffer state occasionally
        if self.total_samples_received % (SAMPLE_RATE * 5) < samples.len() {
            eprintln!("[RealtimeTDT] buffer: {} samples ({:.1}s), min: {}, since_process: {}, interval: {}",
                self.audio_buffer.len(), buffer_secs, min_buffer_samples,
                self.samples_since_last_process, self.process_interval_samples);
        }

        if self.audio_buffer.len() < min_buffer_samples {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: buffer_secs,
                needs_more_audio: true,
            });
        }

        if self.samples_since_last_process < self.process_interval_samples {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: buffer_secs,
                needs_more_audio: true,
            });
        }

        // Process the buffer
        self.samples_since_last_process = 0;
        self.process_buffer()
    }

    /// Push audio in lookahead mode
    /// Accumulates segments delimited by pauses, then transcribes with future context
    fn push_audio_lookahead(&mut self, samples: &[f32], current_time: f32) -> Result<ChunkResult> {
        let rms = calculate_rms(samples);
        let is_silence = rms < self.config.silence_energy_threshold;
        let chunk_duration = samples.len() as f32 / SAMPLE_RATE as f32;
        let chunk_end_time = current_time + chunk_duration;

        // Update total samples tracking
        self.total_samples_received += samples.len();

        // State machine for segment accumulation
        // IMPORTANT: Always accumulate audio when in_speech, even during brief silences
        // This prevents losing words that are spoken softly or during transition
        if is_silence {
            // In silence
            if self.in_speech {
                // Still accumulate audio during silence gaps within speech
                // This ensures we don't lose words during brief pauses
                self.current_segment_audio.extend(samples.iter().copied());

                // Transition: speech -> silence (potential pause)
                // Check if this is a significant pause
                if self.silence_start_time.is_none() {
                    self.silence_start_time = Some(current_time);
                }

                let silence_duration = chunk_end_time - self.silence_start_time.unwrap();

                if silence_duration >= self.config.pause_threshold_secs {
                    // Pause detected! Finalize current segment
                    if !self.current_segment_audio.is_empty() {
                        let segment = PauseSegmentInfo {
                            start_time: self.current_segment_start,
                            end_time: chunk_end_time, // Include silence up to now
                            audio: std::mem::take(&mut self.current_segment_audio),
                        };
                        self.lookahead_segments.push_back(segment);
                    }
                    self.in_speech = false;
                    self.silence_start_time = None; // Reset for next pause detection
                }
            }
            // When not in_speech and silence, just wait for speech to start
        } else {
            // In speech
            if !self.in_speech {
                // Transition: silence -> speech
                self.in_speech = true;
                self.current_segment_start = current_time;
            }
            self.silence_start_time = None; // Reset silence tracking on speech
            // Accumulate audio
            self.current_segment_audio.extend(samples.iter().copied());
        }

        // Also maintain the ring buffer for fallback processing
        self.audio_buffer.extend(samples.iter().copied());
        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Time-based fallback: force segment break to keep latency low
        // Use 3s segments for better quality while maintaining ~5-7s latency
        let max_segment_duration = 3.0; // seconds - balance quality and latency
        let current_segment_duration = chunk_end_time - self.current_segment_start;
        if self.in_speech && current_segment_duration > max_segment_duration && !self.current_segment_audio.is_empty() {
            // Force segment break
            let segment = PauseSegmentInfo {
                start_time: self.current_segment_start,
                end_time: chunk_end_time,
                audio: std::mem::take(&mut self.current_segment_audio),
            };
            self.lookahead_segments.push_back(segment);
            self.current_segment_start = chunk_end_time;
        }

        // Check if we have enough segments to process with lookahead
        // We need: lookahead_segments + 1 segments to emit one
        // Example: with lookahead_segments=2, we need 3 segments to emit segment 0
        let required_segments = self.config.lookahead_segments + 1;

        if self.lookahead_segments.len() >= required_segments {
            return self.process_lookahead();
        }

        // Ramp-up: emit immediately once we have at least 1 complete segment
        // This minimizes initial latency - first output after ~2-4s
        if !self.lookahead_segments.is_empty() {
            return self.process_lookahead();
        }

        Ok(ChunkResult {
            segments: Vec::new(),
            full_text: self.get_full_text(),
            buffer_time: buffer_secs,
            needs_more_audio: true,
        })
    }

    /// Process accumulated segments in lookahead mode
    /// Transcribes all segments together but only emits the oldest one
    /// Includes past context for better ASR quality
    fn process_lookahead(&mut self) -> Result<ChunkResult> {
        if self.lookahead_segments.is_empty() {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: true,
            });
        }

        // Add PAST context from ring buffer for better ASR quality
        // Use up to 5 seconds of past audio (doesn't affect latency since it's already emitted)
        let past_context_secs = 5.0;
        let past_context_samples = (past_context_secs * SAMPLE_RATE as f32) as usize;

        let first_segment_start = self.lookahead_segments.front().unwrap().start_time;
        let first_segment_end = self.lookahead_segments.front().unwrap().end_time;

        // Calculate how much past audio we need from the ring buffer
        // The ring buffer contains audio up to the current time
        // We want audio from (first_segment_start - past_context_secs) to first_segment_start
        let total_audio_samples: usize = self.lookahead_segments.iter()
            .map(|s| s.audio.len())
            .sum();
        let total_audio_samples = total_audio_samples + self.current_segment_audio.len();

        // Get past context from ring buffer (already processed audio)
        // The ring buffer might have more context than the segments
        let buffer_len = self.audio_buffer.len();
        let past_samples = past_context_samples.min(buffer_len.saturating_sub(total_audio_samples));

        let mut combined_audio: Vec<f32> = Vec::new();
        let mut time_offset = 0.0;

        // Add past context from ring buffer if available
        if past_samples > 0 {
            // Get the oldest 'past_samples' from buffer that precede our segments
            let start_idx = buffer_len.saturating_sub(total_audio_samples + past_samples);
            let end_idx = buffer_len.saturating_sub(total_audio_samples);
            if end_idx > start_idx {
                combined_audio.extend(self.audio_buffer.range(start_idx..end_idx));
                time_offset = (end_idx - start_idx) as f32 / SAMPLE_RATE as f32;
            }
        }

        // Add all lookahead segment audio
        for segment in &self.lookahead_segments {
            combined_audio.extend(&segment.audio);
        }

        // Add current segment audio if any (for additional context)
        if !self.current_segment_audio.is_empty() {
            combined_audio.extend(&self.current_segment_audio);
        }

        if combined_audio.is_empty() {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: true,
            });
        }

        // Transcribe all segments together (past + current + future context)
        let result = self.model.transcribe_samples(
            combined_audio,
            SAMPLE_RATE as u32,
            1,
            Some(TimestampMode::Words),
        )?;

        // Adjust timestamps to global time
        // Account for the past context we prepended
        let adjusted_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += first_segment_start - time_offset;
                t.end += first_segment_start - time_offset;
                t
            })
            .collect();

        // Extract only tokens from the FIRST segment (the one we're emitting)
        // Use stricter deduplication to prevent repeated sentences at segment boundaries.
        // The key insight: tokens should START at or after our last emitted position.
        // Using a small negative tolerance (0.2s) allows for minor ASR timestamp variance
        // while still preventing the same word from being emitted twice.
        let emit_tokens: Vec<TimedToken> = adjusted_tokens
            .into_iter()
            .filter(|t| {
                // Token must START at or after our last emitted token end
                // Small tolerance for ASR timestamp variance between transcriptions
                let is_new_content = t.start >= self.last_emitted_token_end - 0.2;

                // Token must belong to the first segment being processed
                let in_first_segment = t.start >= first_segment_start - 0.2 &&
                                       t.start <= first_segment_end + 0.3;

                is_new_content && in_first_segment
            })
            .collect();

        let mut new_segments = Vec::new();

        if !emit_tokens.is_empty() {
            // Filter out punctuation-only tokens at start
            // Also apply text-based deduplication: skip tokens that match the last emitted word
            // Find the last actual WORD (not punctuation) from previous segments
            let last_emitted_word = self.finalized_segments.last()
                .and_then(|seg| {
                    seg.tokens.iter().rev()
                        .find(|t| t.text.trim().chars().any(|c| c.is_alphanumeric()))
                        .map(|t| {
                            // Extract the word without punctuation
                            t.text.trim()
                                .chars()
                                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                                .collect::<String>()
                                .trim()
                                .to_lowercase()
                        })
                });

            let tokens_to_use: Vec<_> = emit_tokens
                .into_iter()
                .skip_while(|t| {
                    let text = t.text.trim();
                    // Skip punctuation-only tokens
                    if text.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace()) {
                        return true;
                    }
                    // Skip if this token matches the last emitted word (text-based dedup)
                    if let Some(ref last_word) = last_emitted_word {
                        if !last_word.is_empty() {
                            // Extract just the alphanumeric part for comparison
                            let current_word: String = text.chars()
                                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                                .collect::<String>()
                                .trim()
                                .to_lowercase();
                            // Check for exact match or if one contains the other (handles partial words)
                            if !current_word.is_empty() &&
                               (current_word == *last_word ||
                                last_word.contains(&current_word) ||
                                current_word.contains(last_word.as_str())) {
                                return true;
                            }
                        }
                    }
                    false
                })
                .collect();

            let has_real_content = tokens_to_use.iter().any(|t|
                t.text.chars().any(|c| c.is_alphanumeric())
            );

            if !tokens_to_use.is_empty() && has_real_content {
                // Join words with spaces, handling punctuation properly
                let segment_text: String = {
                    let mut output = String::new();
                    for (i, token) in tokens_to_use.iter().enumerate() {
                        let is_standalone_punct = token.text.len() == 1
                            && token.text.chars().all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                        if i > 0 && !is_standalone_punct {
                            output.push(' ');
                        }
                        output.push_str(token.text.trim());
                    }
                    output.trim().to_string()
                };

                let segment = Segment {
                    text: segment_text,
                    start_time: tokens_to_use.first().map(|t| t.start).unwrap_or(0.0),
                    end_time: tokens_to_use.last().map(|t| t.end).unwrap_or(0.0),
                    tokens: tokens_to_use.clone(),
                    is_final: true,
                };

                // Update tracking - use SEGMENT boundary not token time for more stable dedup
                // This ensures the next process will correctly skip tokens from this segment
                if let Some(last) = tokens_to_use.last() {
                    self.confirmed_until = last.end;
                    // Use the segment end time as the boundary, clamped to at least the last token
                    self.last_emitted_token_end = last.end.max(first_segment_end);
                }

                self.finalized_segments.push(segment.clone());
                new_segments.push(segment);
            }
        }

        // Remove the first segment from the queue (we just processed it)
        // Even if no tokens were emitted, update tracking to prevent re-processing
        if new_segments.is_empty() {
            // No tokens emitted but we're done with this segment
            self.last_emitted_token_end = self.last_emitted_token_end.max(first_segment_end);
        }
        self.lookahead_segments.pop_front();

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: buffer_secs,
            needs_more_audio: true,
        })
    }

    /// Detect pause/silence in audio for smarter token confirmation
    ///
    /// Tracks transitions between speech and silence. When silence exceeds
    /// the threshold duration, marks a pause boundary for confirmation.
    ///
    /// Key insight: We add a small buffer (200ms) before the silence start
    /// to ensure ASR has enough context to produce stable timestamps for
    /// the last word before the pause.
    fn detect_pause(&mut self, samples: &[f32], current_time: f32) {
        let rms = calculate_rms(samples);
        let is_silence = rms < self.config.silence_energy_threshold;

        // Calculate the end time of this chunk
        let chunk_duration = samples.len() as f32 / SAMPLE_RATE as f32;
        let chunk_end_time = current_time + chunk_duration;

        if is_silence {
            // Currently in silence
            if self.silence_start_time.is_none() {
                // Just entered silence - record start time
                self.silence_start_time = Some(current_time);
            } else {
                // Continuing silence - check if it exceeds threshold
                let silence_start = self.silence_start_time.unwrap();
                let silence_duration = chunk_end_time - silence_start;

                if silence_duration >= self.config.pause_threshold_secs {
                    // Detected a pause! Mark the boundary AT the silence start
                    // The confirm zone should include all words before the pause
                    // Use a small buffer to account for ASR timestamp variance
                    let pause_buffer = 0.1; // 100ms buffer - reduced to avoid cutting words
                    let boundary = (silence_start - pause_buffer).max(self.last_emitted_token_end);

                    if self.pause_boundary_time.is_none()
                        || boundary > self.pause_boundary_time.unwrap()
                    {
                        self.pause_boundary_time = Some(boundary);
                        self.last_pause_end_time = chunk_end_time;
                    }
                }
            }
        } else {
            // Speech detected - reset silence tracking
            if self.silence_start_time.is_some() {
                // Coming out of silence
                self.silence_start_time = None;
            }
        }
    }

    /// Process the ring buffer and emit confirmed tokens
    fn process_buffer(&mut self) -> Result<ChunkResult> {
        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Convert ring buffer to vec for processing
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();

        // Calculate global time offset
        // The buffer starts at: total_received - buffer_length
        let buffer_start_time = (self.total_samples_received - self.audio_buffer.len()) as f32 / SAMPLE_RATE as f32;

        // Process through TDT
        let result = self.model.transcribe_samples(
            audio,
            SAMPLE_RATE as u32,
            1,
            Some(TimestampMode::Words),
        )?;

        // Adjust timestamps to global time
        let raw_token_count = result.tokens.len();
        let adjusted_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += buffer_start_time;
                t.end += buffer_start_time;
                t
            })
            .collect();

        // Debug: log model output every 5 seconds
        if self.total_samples_received % (SAMPLE_RATE * 5) < (self.config.process_interval_secs * SAMPLE_RATE as f32) as usize {
            eprintln!("[RealtimeTDT] process_buffer: raw_tokens={}, buffer_start={:.2}s, last_emitted_end={:.2}s",
                raw_token_count, buffer_start_time, self.last_emitted_token_end);
        }

        // Determine confirmed zone
        // In pause-based mode: use detected pause boundary if available
        // Otherwise: fall back to fixed threshold from buffer end
        let buffer_end_time = buffer_start_time + buffer_secs;
        let confirm_until = if self.config.pause_based_confirm {
            if let Some(pause_time) = self.pause_boundary_time {
                // Use pause boundary - confirm tokens ending before the pause
                // Only if it's ahead of what we've already confirmed AND still
                // within the current buffer. If the pause boundary is before
                // the buffer start, it's stale (inference was too slow and the
                // buffer scrolled past the pause) - fall back to time-based.
                if pause_time > self.confirmed_until && pause_time >= buffer_start_time {
                    pause_time
                } else {
                    // Pause is stale or before confirmed point, clear it and use fallback
                    self.pause_boundary_time = None;
                    buffer_end_time - self.config.confirm_threshold_secs
                }
            } else {
                // No pause detected yet, use fallback threshold
                buffer_end_time - self.config.confirm_threshold_secs
            }
        } else {
            // Standard time-based confirmation
            buffer_end_time - self.config.confirm_threshold_secs
        };

        // Split tokens into confirmed and pending
        // Use timestamp-based filtering to prevent duplicates
        // A token is considered "already emitted" if its END time is at or before our last emitted position
        // Use a small margin to catch ASR timestamp variance without skipping distinct words
        let dedup_margin = 0.15; // 150ms margin - balance between dedup and not skipping words
        let mut confirmed_tokens: Vec<TimedToken> = Vec::new();
        let mut new_pending: Vec<TimedToken> = Vec::new();

        for token in adjusted_tokens {
            // Skip tokens that have already been emitted
            // Use END time comparison - if token ends before (or at) our last emitted position,
            // it's definitely a duplicate
            if token.end <= self.last_emitted_token_end + dedup_margin {
                // Token ends before our last emitted token end
                // This is a duplicate from re-processing
                continue;
            }

            if token.end < confirm_until {
                // In confirmed zone
                confirmed_tokens.push(token);
            } else {
                // In pending zone (might change with more context)
                new_pending.push(token);
            }
        }

        // Debug: log token filtering results
        if self.total_samples_received % (SAMPLE_RATE * 5) < (self.config.process_interval_secs * SAMPLE_RATE as f32) as usize {
            let dedup_filter = self.last_emitted_token_end + dedup_margin;
            eprintln!("[RealtimeTDT] filtering: confirm_until={:.2}s, dedup_filter={:.2}s, confirmed={}, pending={}",
                confirm_until, dedup_filter, confirmed_tokens.len(), new_pending.len());
        }

        let mut new_segments = Vec::new();

        // Find sentence boundary for clean output
        if !confirmed_tokens.is_empty() {
            let sentence_end_idx = find_last_sentence_boundary(&confirmed_tokens, confirm_until);

            let (final_tokens, extra_pending) = if let Some(idx) = sentence_end_idx {
                let (final_part, extra) = confirmed_tokens.split_at(idx + 1);
                (final_part.to_vec(), extra.to_vec())
            } else {
                // No sentence boundary - emit all to avoid infinite buffering
                (confirmed_tokens, Vec::new())
            };

            if !final_tokens.is_empty() {
                // The deduplication is now done at token filtering stage (above)
                // Filter out tokens that are pure punctuation at the start (artifacts from boundary)
                let tokens_to_use: Vec<_> = final_tokens
                    .into_iter()
                    .skip_while(|t| t.text.chars().all(|c| c.is_ascii_punctuation() || c.is_whitespace()))
                    .collect();

                // Skip segments that are too short or contain only punctuation
                let has_real_content = tokens_to_use.iter().any(|t|
                    t.text.chars().any(|c| c.is_alphanumeric())
                );

                if !tokens_to_use.is_empty() && has_real_content {
                    // Join words with spaces, handling punctuation properly
                    let segment_text: String = {
                        let mut output = String::new();
                        for (i, token) in tokens_to_use.iter().enumerate() {
                            let is_standalone_punct = token.text.len() == 1
                                && token.text.chars().all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                            if i > 0 && !is_standalone_punct {
                                output.push(' ');
                            }
                            output.push_str(token.text.trim());
                        }
                        output.trim().to_string()
                    };

                    let segment = Segment {
                        text: segment_text,
                        start_time: tokens_to_use.first().map(|t| t.start).unwrap_or(0.0),
                        end_time: tokens_to_use.last().map(|t| t.end).unwrap_or(0.0),
                        tokens: tokens_to_use.clone(),
                        is_final: true,
                    };

                    // Update both confirmed_until and last_emitted_token_end
                    if let Some(last) = tokens_to_use.last() {
                        self.confirmed_until = last.end;
                        self.last_emitted_token_end = last.end;

                        // Clear pause boundary if we've confirmed past it
                        if self.config.pause_based_confirm {
                            if let Some(pause_time) = self.pause_boundary_time {
                                if self.confirmed_until >= pause_time {
                                    self.pause_boundary_time = None;
                                }
                            }
                        }
                    }

                    eprintln!("[RealtimeTDT] EMIT segment: \"{}\" [{:.2}s-{:.2}s]",
                        segment.text.chars().take(60).collect::<String>(),
                        segment.start_time, segment.end_time);
                    self.finalized_segments.push(segment.clone());
                    new_segments.push(segment);
                } else {
                    eprintln!("[RealtimeTDT] SKIP segment: tokens_to_use={}, has_real_content={}",
                        tokens_to_use.len(), has_real_content);
                }
            }

            // Add extra to pending
            new_pending = [extra_pending, new_pending].concat();
        }

        self.pending_tokens = new_pending;

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: buffer_secs,
            needs_more_audio: true,
        })
    }

    /// Finalize and get remaining text
    pub fn finalize(&mut self) -> Result<ChunkResult> {
        if self.audio_buffer.is_empty() && self.pending_tokens.is_empty() {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: false,
            });
        }

        // Process remaining buffer with no confirm threshold
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();
        if audio.is_empty() {
            // Just emit pending tokens
            let mut new_segments = Vec::new();
            if !self.pending_tokens.is_empty() {
                // Join words with spaces, handling punctuation properly
                let segment_text: String = {
                    let mut output = String::new();
                    for (i, token) in self.pending_tokens.iter().enumerate() {
                        let is_standalone_punct = token.text.len() == 1
                            && token.text.chars().all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                        if i > 0 && !is_standalone_punct {
                            output.push(' ');
                        }
                        output.push_str(token.text.trim());
                    }
                    output.trim().to_string()
                };

                let segment = Segment {
                    text: segment_text,
                    start_time: self.pending_tokens.first().map(|t| t.start).unwrap_or(0.0),
                    end_time: self.pending_tokens.last().map(|t| t.end).unwrap_or(0.0),
                    tokens: std::mem::take(&mut self.pending_tokens),
                    is_final: true,
                };

                self.finalized_segments.push(segment.clone());
                new_segments.push(segment);
            }

            return Ok(ChunkResult {
                segments: new_segments,
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: false,
            });
        }

        let buffer_start_time = (self.total_samples_received - self.audio_buffer.len()) as f32 / SAMPLE_RATE as f32;

        let result = self.model.transcribe_samples(
            audio,
            SAMPLE_RATE as u32,
            1,
            Some(TimestampMode::Words),
        )?;

        // Get all remaining tokens
        let final_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += buffer_start_time;
                t.end += buffer_start_time;
                t
            })
            .filter(|t| t.end > self.confirmed_until)
            .collect();

        let mut new_segments = Vec::new();

        if !final_tokens.is_empty() {
            // Join words with spaces, handling punctuation properly
            let segment_text: String = {
                let mut output = String::new();
                for (i, token) in final_tokens.iter().enumerate() {
                    let is_standalone_punct = token.text.len() == 1
                        && token.text.chars().all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                    if i > 0 && !is_standalone_punct {
                        output.push(' ');
                    }
                    output.push_str(token.text.trim());
                }
                output.trim().to_string()
            };

            let segment = Segment {
                text: segment_text,
                start_time: final_tokens.first().map(|t| t.start).unwrap_or(0.0),
                end_time: final_tokens.last().map(|t| t.end).unwrap_or(0.0),
                tokens: final_tokens,
                is_final: true,
            };

            self.finalized_segments.push(segment.clone());
            new_segments.push(segment);
        }

        self.audio_buffer.clear();
        self.pending_tokens.clear();

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: 0.0,
            needs_more_audio: false,
        })
    }

    pub fn segments(&self) -> &[Segment] {
        &self.finalized_segments
    }

    pub fn get_full_text(&self) -> String {
        let mut text: String = self.finalized_segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if !self.pending_tokens.is_empty() {
            // Join words with spaces, handling punctuation properly
            let pending: String = {
                let mut output = String::new();
                for (i, token) in self.pending_tokens.iter().enumerate() {
                    let is_standalone_punct = token.text.len() == 1
                        && token.text.chars().all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
                    if i > 0 && !is_standalone_punct {
                        output.push(' ');
                    }
                    output.push_str(token.text.trim());
                }
                output
            };

            if !text.is_empty() && !pending.trim().is_empty() {
                text.push(' ');
            }
            text.push_str(pending.trim());
        }

        text
    }

    pub fn current_time(&self) -> f32 {
        self.total_samples_received as f32 / SAMPLE_RATE as f32
    }

    /// Get the current buffer size in samples
    pub fn buffer_len(&self) -> usize {
        self.audio_buffer.len()
    }

    /// Get the current buffer duration in seconds
    pub fn buffer_duration(&self) -> f32 {
        self.audio_buffer.len() as f32 / SAMPLE_RATE as f32
    }

    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.total_samples_received = 0;
        self.samples_since_last_process = 0;
        self.confirmed_until = 0.0;
        self.finalized_segments.clear();
        self.pending_tokens.clear();
        // Reset pause detection state
        self.silence_start_time = None;
        self.last_pause_end_time = 0.0;
        self.pause_boundary_time = None;
        // Reset deduplication tracking
        self.last_emitted_token_end = 0.0;
        // Reset lookahead state
        self.lookahead_segments.clear();
        self.current_segment_audio.clear();
        self.current_segment_start = 0.0;
        self.in_speech = false;
    }
}

/// Find the last sentence boundary (., !, ?) before a given time
fn find_last_sentence_boundary(tokens: &[TimedToken], before_time: f32) -> Option<usize> {
    let mut last_boundary_idx: Option<usize> = None;

    for (i, token) in tokens.iter().enumerate() {
        if token.end >= before_time {
            break;
        }
        let text = token.text.trim();
        if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
            last_boundary_idx = Some(i);
        }
    }

    // Fallback: if no sentence boundary, return last token before time
    if last_boundary_idx.is_none() {
        for (i, token) in tokens.iter().enumerate().rev() {
            if token.end < before_time {
                return Some(i);
            }
        }
    }

    last_boundary_idx
}

// ============================================================================
// Diarized version
// ============================================================================

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

#[cfg(feature = "sortformer")]
pub struct RealtimeTDTDiarized {
    tdt: RealtimeTDT,
    diarization: SortformerStream,
}

#[cfg(feature = "sortformer")]
impl RealtimeTDTDiarized {
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        tdt_model_path: P1,
        diar_model_path: P2,
        exec_config: Option<ExecutionConfig>,
        tdt_config: Option<RealtimeTDTConfig>,
    ) -> Result<Self> {
        let tdt = RealtimeTDT::new(tdt_model_path, exec_config.clone(), tdt_config)?;
        let diarization = SortformerStream::with_config(
            diar_model_path,
            exec_config,
            crate::sortformer::DiarizationConfig::callhome(),
        )?;

        Ok(Self { tdt, diarization })
    }

    pub fn push_audio(&mut self, samples: &[f32]) -> Result<DiarizedChunkResult> {
        let tdt_result = self.tdt.push_audio(samples)?;
        let _ = self.diarization.push_audio(samples)?;

        let diarized_segments: Vec<DiarizedSegment> = tdt_result.segments
            .into_iter()
            .map(|seg| {
                let mid_time = (seg.start_time + seg.end_time) / 2.0;
                let speaker = self.diarization.get_speaker_at(mid_time);

                DiarizedSegment {
                    text: seg.text,
                    start_time: seg.start_time,
                    end_time: seg.end_time,
                    speaker,
                    tokens: seg.tokens,
                    is_final: seg.is_final,
                }
            })
            .collect();

        Ok(DiarizedChunkResult {
            segments: diarized_segments,
            full_text: tdt_result.full_text,
            buffer_time: tdt_result.buffer_time,
            needs_more_audio: tdt_result.needs_more_audio,
        })
    }

    pub fn finalize(&mut self) -> Result<DiarizedChunkResult> {
        let tdt_result = self.tdt.finalize()?;

        let diarized_segments: Vec<DiarizedSegment> = tdt_result.segments
            .into_iter()
            .map(|seg| {
                let mid_time = (seg.start_time + seg.end_time) / 2.0;
                let speaker = self.diarization.get_speaker_at(mid_time);

                DiarizedSegment {
                    text: seg.text,
                    start_time: seg.start_time,
                    end_time: seg.end_time,
                    speaker,
                    tokens: seg.tokens,
                    is_final: seg.is_final,
                }
            })
            .collect();

        Ok(DiarizedChunkResult {
            segments: diarized_segments,
            full_text: tdt_result.full_text,
            buffer_time: 0.0,
            needs_more_audio: false,
        })
    }

    pub fn reset(&mut self) {
        self.tdt.reset();
        self.diarization.reset();
    }

    /// Get the current total audio duration processed
    pub fn current_time(&self) -> f32 {
        self.tdt.current_time()
    }

    /// Get the current buffer duration
    pub fn buffer_duration(&self) -> f32 {
        self.tdt.buffer_duration()
    }
}

// ============================================================================
// StreamingTranscriber implementation for RealtimeTDTDiarized
// ============================================================================

#[cfg(feature = "sortformer")]
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};

#[cfg(feature = "sortformer")]
impl StreamingTranscriber for RealtimeTDTDiarized {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "parakeet-tdt".to_string(),
            display_name: "Parakeet TDT 0.6B".to_string(),
            description: "NVIDIA's Parakeet TDT model for high-quality speech recognition with word-level timestamps".to_string(),
            supports_diarization: true,
            languages: vec!["en".to_string()],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        // Call the inherent method using fully-qualified syntax
        let result = RealtimeTDTDiarized::push_audio(self, samples)?;

        let segments = result.segments
            .into_iter()
            .map(|seg| TranscriptionSegment {
                text: seg.text,
                start_time: seg.start_time,
                end_time: seg.end_time,
                speaker: seg.speaker,
                confidence: None,
                is_final: seg.is_final,
                inference_time_ms: None,  // Set by transcription handler
            })
            .collect();

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: result.buffer_time,
            total_duration: self.current_time(),
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        // Call the inherent method using fully-qualified syntax
        let result = RealtimeTDTDiarized::finalize(self)?;

        let segments = result.segments
            .into_iter()
            .map(|seg| TranscriptionSegment {
                text: seg.text,
                start_time: seg.start_time,
                end_time: seg.end_time,
                speaker: seg.speaker,
                confidence: None,
                is_final: seg.is_final,
                inference_time_ms: None,
            })
            .collect();

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.current_time(),
        })
    }

    fn reset(&mut self) {
        // Call the inherent method using fully-qualified syntax
        RealtimeTDTDiarized::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        // Call the inherent method using fully-qualified syntax
        RealtimeTDTDiarized::buffer_duration(self)
    }

    fn total_duration(&self) -> f32 {
        self.current_time()
    }
}

#[cfg(feature = "sortformer")]
#[derive(Debug, Clone)]
pub struct DiarizedSegment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub speaker: Option<usize>,
    pub tokens: Vec<TimedToken>,
    pub is_final: bool,
}

#[cfg(feature = "sortformer")]
impl DiarizedSegment {
    pub fn speaker_display(&self) -> String {
        match self.speaker {
            Some(id) => format!("Speaker {}", id),
            None => "Speaker ?".to_string(),
        }
    }
}

#[cfg(feature = "sortformer")]
#[derive(Debug, Clone)]
pub struct DiarizedChunkResult {
    pub segments: Vec<DiarizedSegment>,
    pub full_text: String,
    pub buffer_time: f32,
    pub needs_more_audio: bool,
}
