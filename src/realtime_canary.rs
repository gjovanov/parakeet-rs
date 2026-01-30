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

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

const SAMPLE_RATE: usize = 16000;

/// Minimum number of consecutive matches before confirming text as final
const MIN_STABLE_COUNT: u32 = 2;

/// Maximum expected words per second of audio (for hallucination length guard)
const MAX_WORDS_PER_SEC: f32 = 5.0;

/// Calculate RMS (root mean square) energy of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Multiplier for inference time anomaly detection
const INFERENCE_TIME_ANOMALY_MULTIPLIER: f32 = 5.0;

/// Truncate text at first detected hallucination (3+ consecutive repeated words
/// or 3+ repeated 2-3 word phrases)
fn truncate_hallucination(text: &str) -> String {
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
                let truncate_at = i - consecutive_count + 1;
                if truncate_at > 0 {
                    eprintln!(
                        "[RealtimeCanary] Truncating hallucination at word {}: '{}'",
                        truncate_at, words[i]
                    );
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
                        eprintln!(
                            "[RealtimeCanary] Truncating repeated phrase at {}: '{}'",
                            i, pattern.join(" ")
                        );
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
    /// Enable pause-based confirmation (default: true for speedy mode)
    /// When true, detected pauses force-confirm text, bypassing MIN_STABLE_COUNT
    pub pause_based_confirm: bool,
    /// Minimum pause duration in seconds to trigger confirmation (default: 0.6)
    pub pause_threshold_secs: f32,
    /// Audio RMS below this is considered silence (default: 0.008)
    pub silence_energy_threshold: f32,
}

impl Default for RealtimeCanaryConfig {
    fn default() -> Self {
        Self {
            buffer_size_secs: 10.0,
            min_audio_secs: 2.0,
            process_interval_secs: 2.0,
            language: "en".to_string(),
            pause_based_confirm: false,
            pause_threshold_secs: 0.6,
            silence_energy_threshold: 0.008,
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

    /// Rolling inference time tracker for anomaly detection
    inference_times_ms: Vec<u32>,
    /// Median inference time (updated after each inference)
    median_inference_time_ms: u32,

    /// Pause detection state: when current silence started (global time)
    silence_start_time: Option<f32>,
    /// Time boundary where a pause was detected (force-confirm text before this)
    pause_boundary_time: Option<f32>,
    /// When the last detected pause ended
    last_pause_end_time: f32,

    /// Optional speaker diarizer (requires sortformer feature)
    #[cfg(feature = "sortformer")]
    diarizer: Option<SortformerStream>,
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
            inference_times_ms: Vec::new(),
            median_inference_time_ms: 0,
            silence_start_time: None,
            pause_boundary_time: None,
            last_pause_end_time: 0.0,
            #[cfg(feature = "sortformer")]
            diarizer: None,
        })
    }

    /// Create a new streaming Canary transcriber with optional diarization
    #[cfg(feature = "sortformer")]
    pub fn new_with_diarization<P1: AsRef<Path>, P2: AsRef<Path>>(
        model_path: P1,
        diar_path: Option<P2>,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeCanaryConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        let canary_config = CanaryConfig {
            language: config.language.clone(),
            ..Default::default()
        };

        let model = CanaryModel::from_pretrained(&model_path, exec_config.clone(), Some(canary_config))?;

        let buffer_size_samples = (config.buffer_size_secs * SAMPLE_RATE as f32) as usize;
        let process_interval_samples = (config.process_interval_secs * SAMPLE_RATE as f32) as usize;
        let min_audio_samples = (config.min_audio_secs * SAMPLE_RATE as f32) as usize;

        // Create diarizer if path provided
        let diarizer = if let Some(diar_path) = diar_path {
            eprintln!("[RealtimeCanary] Creating diarizer from {:?}", diar_path.as_ref());
            Some(SortformerStream::with_config(
                diar_path,
                exec_config,
                Default::default(),
            )?)
        } else {
            None
        };

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
            inference_times_ms: Vec::new(),
            median_inference_time_ms: 0,
            silence_start_time: None,
            pause_boundary_time: None,
            last_pause_end_time: 0.0,
            diarizer,
        })
    }

    /// Check if diarization is available
    #[cfg(feature = "sortformer")]
    pub fn has_diarization(&self) -> bool {
        self.diarizer.is_some()
    }

    /// Check if diarization is available (always false without sortformer feature)
    #[cfg(not(feature = "sortformer"))]
    pub fn has_diarization(&self) -> bool {
        false
    }

    /// Get speaker at a given time
    #[cfg(feature = "sortformer")]
    fn get_speaker_at(&self, time: f32) -> Option<usize> {
        if let Some(diarizer) = &self.diarizer {
            diarizer.get_speaker_at(time)
        } else {
            None
        }
    }

    /// Get speaker at a given time (always None without sortformer feature)
    #[cfg(not(feature = "sortformer"))]
    fn get_speaker_at(&self, _time: f32) -> Option<usize> {
        None
    }

    /// Detect pause/silence in audio for smarter confirmation.
    /// Tracks transitions between speech and silence. When silence exceeds
    /// the threshold duration, marks a pause boundary for confirmation.
    fn detect_pause(&mut self, samples: &[f32], current_time: f32) {
        let rms = calculate_rms(samples);
        let is_silence = rms < self.config.silence_energy_threshold;

        let chunk_duration = samples.len() as f32 / SAMPLE_RATE as f32;
        let chunk_end_time = current_time + chunk_duration;

        if is_silence {
            if self.silence_start_time.is_none() {
                self.silence_start_time = Some(current_time);
            } else {
                let silence_start = self.silence_start_time.unwrap();
                let silence_duration = chunk_end_time - silence_start;

                if silence_duration >= self.config.pause_threshold_secs {
                    let pause_buffer = 0.1;
                    let boundary = (silence_start - pause_buffer).max(self.last_pause_end_time);

                    if self.pause_boundary_time.is_none()
                        || boundary > self.pause_boundary_time.unwrap()
                    {
                        self.pause_boundary_time = Some(boundary);
                        self.last_pause_end_time = chunk_end_time;
                        eprintln!(
                            "[RealtimeCanary] Pause detected at {:.2}s (boundary: {:.2}s)",
                            chunk_end_time, boundary
                        );
                    }
                }
            }
        } else {
            if self.silence_start_time.is_some() {
                self.silence_start_time = None;
            }
        }
    }

    /// Push audio samples and get transcription results
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<CanaryChunkResult> {
        // Push audio to diarizer for speaker tracking (if available)
        #[cfg(feature = "sortformer")]
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        // Pause detection (before buffering to get accurate timing)
        let current_time = self.total_samples_received as f32 / SAMPLE_RATE as f32;
        if self.config.pause_based_confirm {
            self.detect_pause(samples, current_time);
        }

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

    /// Update rolling inference time tracker and return the median
    fn update_inference_stats(&mut self, time_ms: u32) -> u32 {
        self.inference_times_ms.push(time_ms);
        // Keep last 20 measurements
        if self.inference_times_ms.len() > 20 {
            self.inference_times_ms.remove(0);
        }
        let mut sorted = self.inference_times_ms.clone();
        sorted.sort();
        let median = sorted[sorted.len() / 2];
        self.median_inference_time_ms = median;
        median
    }

    fn process_buffer(&mut self) -> Result<CanaryChunkResult> {
        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Convert buffer to vec
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();

        // Transcribe with timing
        let inference_start = std::time::Instant::now();
        let text = self.model.transcribe(&audio)?;
        let inference_ms = inference_start.elapsed().as_millis() as u32;
        let text = text.trim().to_string();

        // Update inference stats
        let median = self.update_inference_stats(inference_ms);

        // Anomaly detection: if inference time > 5x the rolling median, skip result
        if median > 0 && inference_ms > (median as f32 * INFERENCE_TIME_ANOMALY_MULTIPLIER) as u32 {
            eprintln!(
                "[RealtimeCanary] WARNING: Inference anomaly detected ({}ms vs {}ms median), skipping result",
                inference_ms, median
            );
            return Ok(CanaryChunkResult {
                text: String::new(),
                is_partial: true,
                buffer_time: buffer_secs,
            });
        }

        // Apply hallucination truncation
        let text = truncate_hallucination(&text);

        // Length guard: if output is too long for the buffer duration, truncate
        let max_expected_words = (buffer_secs * MAX_WORDS_PER_SEC * 3.0) as usize;
        let word_count = text.split_whitespace().count();
        let text = if word_count > max_expected_words && max_expected_words > 0 {
            eprintln!(
                "[RealtimeCanary] WARNING: Output too long ({} words for {:.1}s buffer), truncating to {}",
                word_count, buffer_secs, max_expected_words
            );
            text.split_whitespace()
                .take(max_expected_words)
                .collect::<Vec<&str>>()
                .join(" ")
        } else {
            text
        };

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

        // Calculate buffer start time for accurate partial segment timing
        let buffer_start_time = (self.total_samples_received - self.audio_buffer.len()) as f32 / SAMPLE_RATE as f32;

        // Always advance confirmed_end_time when the common prefix grows,
        // even before stability count is met. This prevents start_time from freezing.
        if common_prefix_len > self.confirmed_word_count {
            // Estimate end time proportionally based on word position in the buffer
            let total_words = current_words.len().max(1) as f32;
            let stable_fraction = common_prefix_len as f32 / total_words;
            let estimated_end = buffer_start_time + (buffer_secs * stable_fraction);
            // Always keep confirmed_end_time moving forward
            if estimated_end > self.confirmed_end_time {
                self.confirmed_end_time = estimated_end;
            }
        }

        // Check for pause-based force confirmation
        // When a pause boundary is detected, force-confirm all stable text, bypassing MIN_STABLE_COUNT
        let pause_force_confirm = if self.config.pause_based_confirm {
            if let Some(_pause_time) = self.pause_boundary_time {
                // Force confirm all stable words up to this point
                if common_prefix_len > self.confirmed_word_count {
                    let new_confirmed: Vec<&str> = stable_words[self.confirmed_word_count..].to_vec();
                    if !new_confirmed.is_empty() {
                        eprintln!(
                            "[RealtimeCanary] Pause-based force confirm: {} words",
                            new_confirmed.len()
                        );
                        self.pause_boundary_time = None; // Consume the pause
                        Some(new_confirmed.join(" "))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Check if we should confirm more words (emit as final segment)
        // Pause-based confirmation takes priority
        let words_to_confirm = if pause_force_confirm.is_some() {
            pause_force_confirm
        } else if common_prefix_len > self.confirmed_word_count {
            let new_confirmed: Vec<&str> = stable_words[self.confirmed_word_count..].to_vec();
            if !new_confirmed.is_empty() {
                let new_text = new_confirmed.join(" ");

                // Only emit as final if the text has been seen before (stability check)
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
            // Stable prefix shrunk or same
            None
        };

        // Process any confirmed words
        if let Some(new_confirmed_text) = words_to_confirm {
            let start_time = self.confirmed_end_time
                - (new_confirmed_text.split_whitespace().count() as f32 * 0.15);
            let start_time = start_time.max(
                self.pending_final_segments
                    .last()
                    .map(|(_, _, end)| *end)
                    .unwrap_or(0.0),
            );

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
        // Use buffer_start_time for partial segment start rather than confirmed_end_time
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
        self.inference_times_ms.clear();
        self.median_inference_time_ms = 0;
        self.silence_start_time = None;
        self.pause_boundary_time = None;
        self.last_pause_end_time = 0.0;
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
            supports_diarization: self.has_diarization(),
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
            // Get speaker from diarizer at the midpoint of this segment
            let mid_time = (start + end) / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            segments.push(TranscriptionSegment {
                text,
                start_time: start,
                end_time: end,
                speaker,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            });
        }

        // Then, emit the partial segment if there's new text
        if !result.text.is_empty() {
            let start_time = self.confirmed_end_time;
            let end_time = self.total_duration();
            let mid_time = (start_time + end_time) / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            segments.push(TranscriptionSegment {
                text: result.text,
                start_time,
                end_time,
                speaker,
                confidence: None,
                is_final: false,
                inference_time_ms: None,
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
            let end_time = self.total_duration();
            let mid_time = end_time / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            vec![TranscriptionSegment {
                text: result.text,
                start_time: 0.0,
                end_time,
                speaker,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
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
