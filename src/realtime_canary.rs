//! Streaming wrapper for Canary 1B v2 model
//!
//! This module provides a streaming interface for the Canary encoder-decoder model,
//! similar to RealtimeTDT but adapted for the encoder-decoder architecture.
//!
//! ## Text Confirmation Strategy
//!
//! Since Canary re-transcribes a sliding window of audio, we need to handle
//! overlapping/repeated text carefully:
//!
//! 1. Each transcription covers the current buffer window
//! 2. We compare with previous transcription to find the common prefix (confirmed text)
//! 3. Only NEW text (beyond the common prefix) is emitted as partial
//! 4. After several consecutive matches, the prefix is "confirmed" and emitted as final
//!
//! This prevents the same words from being emitted repeatedly.

use crate::canary::{CanaryConfig, CanaryModel};
use crate::error::Result;
use crate::execution::ModelConfig as ExecutionConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use std::collections::VecDeque;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Minimum number of consecutive matches before confirming text as final
const MIN_STABLE_COUNT: u32 = 2;

/// Configuration for streaming Canary processing
#[derive(Debug, Clone)]
pub struct RealtimeCanaryConfig {
    /// Buffer size in seconds before processing
    pub buffer_size_secs: f32,
    /// Minimum audio to accumulate before first transcription
    pub min_audio_secs: f32,
    /// How often to process (in seconds of new audio)
    pub process_interval_secs: f32,
    /// Target language code
    pub language: String,
}

impl Default for RealtimeCanaryConfig {
    fn default() -> Self {
        Self {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language: "en".to_string(),
        }
    }
}

/// Streaming Canary transcriber
///
/// Accumulates audio in a buffer and processes periodically,
/// using the encoder-decoder model for transcription.
pub struct RealtimeCanary {
    model: CanaryModel,
    config: RealtimeCanaryConfig,

    /// Audio buffer
    audio_buffer: VecDeque<f32>,
    buffer_size_samples: usize,

    /// Processing state
    total_samples_received: usize,
    samples_since_last_process: usize,
    process_interval_samples: usize,
    min_audio_samples: usize,

    /// Last transcription result (for incremental updates)
    last_transcription: String,

    /// Confirmed (final) text that has been stable across multiple transcriptions
    confirmed_text: String,
    /// Word count in confirmed text
    confirmed_word_count: usize,
    /// End time of confirmed text
    confirmed_end_time: f32,

    /// Pending text being evaluated for confirmation
    pending_text: String,
    /// How many consecutive times the pending text has matched
    pending_stable_count: u32,

    /// Segments pending to be emitted as final
    pending_final_segments: Vec<(String, f32, f32)>, // (text, start_time, end_time)
}

impl RealtimeCanary {
    /// Create a new streaming Canary transcriber
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeCanaryConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        let canary_config = CanaryConfig {
            language: config.language.clone(),
            ..Default::default()
        };

        let model = CanaryModel::from_pretrained(model_path, exec_config, Some(canary_config))?;

        let buffer_size_samples = (config.buffer_size_secs * SAMPLE_RATE as f32) as usize;
        let process_interval_samples = (config.process_interval_secs * SAMPLE_RATE as f32) as usize;
        let min_audio_samples = (config.min_audio_secs * SAMPLE_RATE as f32) as usize;

        Ok(Self {
            model,
            config,
            audio_buffer: VecDeque::with_capacity(buffer_size_samples),
            buffer_size_samples,
            total_samples_received: 0,
            samples_since_last_process: 0,
            process_interval_samples,
            min_audio_samples,
            last_transcription: String::new(),
            confirmed_text: String::new(),
            confirmed_word_count: 0,
            confirmed_end_time: 0.0,
            pending_text: String::new(),
            pending_stable_count: 0,
            pending_final_segments: Vec::new(),
        })
    }

    /// Push audio samples and get transcription results
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<CanaryChunkResult> {
        // Add to buffer
        self.audio_buffer.extend(samples.iter().copied());
        self.total_samples_received += samples.len();
        self.samples_since_last_process += samples.len();

        // Trim buffer to max size
        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Check if we should process
        if self.audio_buffer.len() < self.min_audio_samples {
            return Ok(CanaryChunkResult {
                text: String::new(),
                is_partial: true,
                buffer_time: buffer_secs,
            });
        }

        if self.samples_since_last_process < self.process_interval_samples {
            return Ok(CanaryChunkResult {
                text: String::new(),
                is_partial: true,
                buffer_time: buffer_secs,
            });
        }

        // Process the buffer
        self.samples_since_last_process = 0;
        self.process_buffer()
    }

    fn process_buffer(&mut self) -> Result<CanaryChunkResult> {
        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Convert buffer to vec
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();

        // Transcribe
        let text = self.model.transcribe(&audio)?;
        let text = text.trim().to_string();

        // Calculate timing
        let current_time = self.total_samples_received as f32 / SAMPLE_RATE as f32;

        // Split into words for comparison
        let current_words: Vec<&str> = text.split_whitespace().collect();
        let last_words: Vec<&str> = self.last_transcription.split_whitespace().collect();

        // Find longest common prefix between current and last transcription
        let common_prefix_len = current_words.iter()
            .zip(last_words.iter())
            .take_while(|(a, b)| a == b)
            .count();

        // Words that are stable (appeared in both transcriptions in same position)
        // These are candidates for confirmation
        let stable_words: Vec<&str> = current_words[..common_prefix_len].to_vec();

        // New/unstable words at the end
        let unstable_words: Vec<&str> = current_words[common_prefix_len..].to_vec();

        // Check if we should confirm more words
        // Confirm words that have been stable and are old enough (buffer is moving past them)
        let words_to_confirm = if common_prefix_len > self.confirmed_word_count {
            // We have more stable words than confirmed - confirm the difference
            let new_confirmed: Vec<&str> = stable_words[self.confirmed_word_count..].to_vec();
            if !new_confirmed.is_empty() {
                let new_text = new_confirmed.join(" ");

                // Only confirm if the text has been seen before (stability check)
                if self.pending_text == new_text {
                    self.pending_stable_count += 1;
                    if self.pending_stable_count >= MIN_STABLE_COUNT {
                        Some(new_text)
                    } else {
                        None
                    }
                } else {
                    self.pending_text = new_text;
                    self.pending_stable_count = 1;
                    None
                }
            } else {
                None
            }
        } else {
            // Stable prefix shrunk or same - reset pending
            if common_prefix_len < self.confirmed_word_count {
                // This shouldn't happen often - the model changed its mind about confirmed text
                // For now, just continue with partial output
            }
            None
        };

        // Process any confirmed words
        if let Some(new_confirmed_text) = words_to_confirm {
            let start_time = self.confirmed_end_time;
            self.confirmed_end_time = current_time - (unstable_words.len() as f32 * 0.15); // Estimate

            // Update confirmed state
            if !self.confirmed_text.is_empty() {
                self.confirmed_text.push(' ');
            }
            self.confirmed_text.push_str(&new_confirmed_text);
            self.confirmed_word_count = self.confirmed_text.split_whitespace().count();

            // Queue as final segment
            self.pending_final_segments.push((
                new_confirmed_text.clone(),
                start_time,
                self.confirmed_end_time,
            ));

            eprintln!("[RealtimeCanary] Confirmed: \"{}\" (total: {} words)",
                new_confirmed_text, self.confirmed_word_count);

            // Reset pending
            self.pending_text.clear();
            self.pending_stable_count = 0;
        }

        self.last_transcription = text.clone();

        // Return the unstable portion as partial (what's still being refined)
        let partial_text = unstable_words.join(" ");
        Ok(CanaryChunkResult {
            text: partial_text,
            is_partial: true,
            buffer_time: buffer_secs,
        })
    }

    /// Finalize transcription
    pub fn finalize(&mut self) -> Result<CanaryChunkResult> {
        if self.audio_buffer.is_empty() {
            return Ok(CanaryChunkResult {
                text: self.last_transcription.clone(),
                is_partial: false,
                buffer_time: 0.0,
            });
        }

        // Final transcription of remaining audio
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();
        let text = self.model.transcribe(&audio)?;

        self.audio_buffer.clear();

        Ok(CanaryChunkResult {
            text,
            is_partial: false,
            buffer_time: 0.0,
        })
    }

    /// Reset the transcriber state
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.total_samples_received = 0;
        self.samples_since_last_process = 0;
        self.last_transcription.clear();
        self.confirmed_text.clear();
        self.confirmed_word_count = 0;
        self.confirmed_end_time = 0.0;
        self.pending_text.clear();
        self.pending_stable_count = 0;
        self.pending_final_segments.clear();
    }

    /// Take any pending final segments (drains them)
    pub fn take_final_segments(&mut self) -> Vec<(String, f32, f32)> {
        std::mem::take(&mut self.pending_final_segments)
    }

    /// Get current buffer duration in seconds
    pub fn buffer_duration(&self) -> f32 {
        self.audio_buffer.len() as f32 / SAMPLE_RATE as f32
    }

    /// Get total audio duration processed
    pub fn total_duration(&self) -> f32 {
        self.total_samples_received as f32 / SAMPLE_RATE as f32
    }

    /// Set the target language
    pub fn set_language(&mut self, lang: &str) {
        self.model.set_language(lang);
        self.config.language = lang.to_string();
    }
}

/// Result from Canary chunk processing
#[derive(Debug, Clone)]
pub struct CanaryChunkResult {
    /// Transcribed text
    pub text: String,
    /// Whether this is a partial (in-progress) transcription
    pub is_partial: bool,
    /// Current buffer duration in seconds
    pub buffer_time: f32,
}

// ============================================================================
// StreamingTranscriber implementation
// ============================================================================

impl StreamingTranscriber for RealtimeCanary {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "canary-1b".to_string(),
            display_name: "Canary 1B v2".to_string(),
            description: "NVIDIA's Canary 1B encoder-decoder model for multilingual ASR".to_string(),
            supports_diarization: false,
            languages: vec![
                "en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string(),
                "it".to_string(), "pt".to_string(), "nl".to_string(), "pl".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = RealtimeCanary::push_audio(self, samples)?;

        let mut segments = Vec::new();

        // First, emit any confirmed (final) segments
        let final_segs = self.take_final_segments();
        for (text, start, end) in final_segs {
            segments.push(TranscriptionSegment {
                text,
                start_time: start,
                end_time: end,
                speaker: None, // Canary doesn't support diarization
                confidence: None,
                is_final: true,
            });
        }

        // Then, emit the partial segment if there's new text
        if !result.text.is_empty() {
            segments.push(TranscriptionSegment {
                text: result.text,
                start_time: self.confirmed_end_time,
                end_time: self.total_duration(),
                speaker: None,
                confidence: None,
                is_final: false,
            });
        }

        if !segments.is_empty() {
            eprintln!("[StreamingTranscriber] Returning {} segment(s) ({} final, {} partial)",
                segments.len(),
                segments.iter().filter(|s| s.is_final).count(),
                segments.iter().filter(|s| !s.is_final).count());
        }

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: result.buffer_time,
            total_duration: self.total_duration(),
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let result = RealtimeCanary::finalize(self)?;

        let segments = if !result.text.is_empty() {
            vec![TranscriptionSegment {
                text: result.text,
                start_time: 0.0,
                end_time: self.total_duration(),
                speaker: None,
                confidence: None,
                is_final: true,
            }]
        } else {
            vec![]
        };

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.total_duration(),
        })
    }

    fn reset(&mut self) {
        RealtimeCanary::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        RealtimeCanary::buffer_duration(self)
    }

    fn total_duration(&self) -> f32 {
        RealtimeCanary::total_duration(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RealtimeCanaryConfig::default();
        assert_eq!(config.buffer_size_secs, 10.0);
        assert_eq!(config.language, "en");
    }
}
