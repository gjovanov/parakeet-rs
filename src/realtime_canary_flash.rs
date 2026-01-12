//! Streaming wrapper for Canary 180M Flash model
//!
//! This module provides a streaming interface for the Canary Flash encoder-decoder model,
//! similar to RealtimeCanary but using the faster 180M Flash variant with KV cache support.
//!
//! Key differences from RealtimeCanary:
//! - Uses smaller 180M parameter model
//! - O(n) decoding with KV cache (vs O(n²))
//! - Faster processing intervals due to faster inference
//! - Same 4 language support (en, de, fr, es)

use crate::canary_flash::{CanaryFlashConfig, CanaryFlashModel};
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

/// Configuration for streaming Canary Flash processing
#[derive(Debug, Clone)]
pub struct RealtimeCanaryFlashConfig {
    /// Buffer size in seconds before processing
    pub buffer_size_secs: f32,
    /// Minimum audio to accumulate before first transcription
    pub min_audio_secs: f32,
    /// How often to process (in seconds of new audio)
    pub process_interval_secs: f32,
    /// Target language code
    pub language: String,
}

impl Default for RealtimeCanaryFlashConfig {
    fn default() -> Self {
        Self {
            // Smaller buffer due to faster inference
            buffer_size_secs: 8.0,
            min_audio_secs: 1.0,
            // More frequent processing due to faster model
            process_interval_secs: 0.5,
            language: "en".to_string(),
        }
    }
}

/// Streaming Canary Flash transcriber
pub struct RealtimeCanaryFlash {
    model: CanaryFlashModel,
    config: RealtimeCanaryFlashConfig,

    /// Audio buffer
    audio_buffer: VecDeque<f32>,
    buffer_size_samples: usize,

    /// Processing state
    total_samples_received: usize,
    samples_since_last_process: usize,
    process_interval_samples: usize,
    min_audio_samples: usize,

    /// Last transcription result
    last_transcription: String,

    /// Confirmed (final) text
    confirmed_text: String,
    confirmed_word_count: usize,
    confirmed_end_time: f32,

    /// Pending text being evaluated
    pending_text: String,
    pending_stable_count: u32,

    /// Segments pending to be emitted as final
    pending_final_segments: Vec<(String, f32, f32)>,

    /// Optional speaker diarizer
    #[cfg(feature = "sortformer")]
    diarizer: Option<SortformerStream>,
}

impl RealtimeCanaryFlash {
    /// Create a new streaming Canary Flash transcriber
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeCanaryFlashConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        let flash_config = CanaryFlashConfig {
            language: config.language.clone(),
            ..Default::default()
        };

        let model = CanaryFlashModel::from_pretrained(model_path, exec_config, Some(flash_config))?;

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
            #[cfg(feature = "sortformer")]
            diarizer: None,
        })
    }

    /// Create with optional diarization
    #[cfg(feature = "sortformer")]
    pub fn new_with_diarization<P1: AsRef<Path>, P2: AsRef<Path>>(
        model_path: P1,
        diar_path: Option<P2>,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeCanaryFlashConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        let flash_config = CanaryFlashConfig {
            language: config.language.clone(),
            ..Default::default()
        };

        let model = CanaryFlashModel::from_pretrained(&model_path, exec_config.clone(), Some(flash_config))?;

        let buffer_size_samples = (config.buffer_size_secs * SAMPLE_RATE as f32) as usize;
        let process_interval_samples = (config.process_interval_secs * SAMPLE_RATE as f32) as usize;
        let min_audio_samples = (config.min_audio_secs * SAMPLE_RATE as f32) as usize;

        let diarizer = if let Some(diar_path) = diar_path {
            eprintln!("[RealtimeCanaryFlash] Creating diarizer from {:?}", diar_path.as_ref());
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
            diarizer,
        })
    }

    #[cfg(feature = "sortformer")]
    pub fn has_diarization(&self) -> bool {
        self.diarizer.is_some()
    }

    #[cfg(not(feature = "sortformer"))]
    pub fn has_diarization(&self) -> bool {
        false
    }

    #[cfg(feature = "sortformer")]
    fn get_speaker_at(&self, time: f32) -> Option<usize> {
        if let Some(diarizer) = &self.diarizer {
            diarizer.get_speaker_at(time)
        } else {
            None
        }
    }

    #[cfg(not(feature = "sortformer"))]
    fn get_speaker_at(&self, _time: f32) -> Option<usize> {
        None
    }

    /// Push audio samples and get transcription results
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<CanaryFlashChunkResult> {
        #[cfg(feature = "sortformer")]
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        self.audio_buffer.extend(samples.iter().copied());
        self.total_samples_received += samples.len();
        self.samples_since_last_process += samples.len();

        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        if self.audio_buffer.len() < self.min_audio_samples {
            return Ok(CanaryFlashChunkResult {
                text: String::new(),
                is_partial: true,
                buffer_time: buffer_secs,
            });
        }

        if self.samples_since_last_process < self.process_interval_samples {
            return Ok(CanaryFlashChunkResult {
                text: String::new(),
                is_partial: true,
                buffer_time: buffer_secs,
            });
        }

        self.samples_since_last_process = 0;
        self.process_buffer()
    }

    fn process_buffer(&mut self) -> Result<CanaryFlashChunkResult> {
        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();

        let text = self.model.transcribe(&audio)?;
        let text = text.trim().to_string();

        let current_time = self.total_samples_received as f32 / SAMPLE_RATE as f32;

        // Split into owned words to avoid borrow conflicts
        let current_words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let last_words: Vec<&str> = self.last_transcription.split_whitespace().collect();

        let common_prefix_len = current_words.iter()
            .zip(last_words.iter())
            .take_while(|(a, b)| a.as_str() == **b)
            .count();

        let stable_words: Vec<&str> = current_words[..common_prefix_len].iter().map(|s| s.as_str()).collect();
        let unstable_words: Vec<&str> = current_words[common_prefix_len..].iter().map(|s| s.as_str()).collect();

        let words_to_confirm = if common_prefix_len > self.confirmed_word_count {
            let new_confirmed: Vec<&str> = stable_words[self.confirmed_word_count..].to_vec();
            if !new_confirmed.is_empty() {
                let new_text = new_confirmed.join(" ");

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
            None
        };

        if let Some(new_confirmed_text) = words_to_confirm {
            let start_time = self.confirmed_end_time;
            self.confirmed_end_time = current_time - (unstable_words.len() as f32 * 0.15);

            if !self.confirmed_text.is_empty() {
                self.confirmed_text.push(' ');
            }
            self.confirmed_text.push_str(&new_confirmed_text);
            self.confirmed_word_count = self.confirmed_text.split_whitespace().count();

            self.pending_final_segments.push((
                new_confirmed_text.clone(),
                start_time,
                self.confirmed_end_time,
            ));

            eprintln!("[RealtimeCanaryFlash] Confirmed: \"{}\" (total: {} words)",
                new_confirmed_text, self.confirmed_word_count);

            self.pending_text.clear();
            self.pending_stable_count = 0;
        }

        let partial_text = unstable_words.join(" ");
        self.last_transcription = text;

        Ok(CanaryFlashChunkResult {
            text: partial_text,
            is_partial: true,
            buffer_time: buffer_secs,
        })
    }

    /// Finalize transcription
    pub fn finalize(&mut self) -> Result<CanaryFlashChunkResult> {
        if self.audio_buffer.is_empty() {
            return Ok(CanaryFlashChunkResult {
                text: self.last_transcription.clone(),
                is_partial: false,
                buffer_time: 0.0,
            });
        }

        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();
        let text = self.model.transcribe(&audio)?;

        self.audio_buffer.clear();

        Ok(CanaryFlashChunkResult {
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

    /// Take any pending final segments
    pub fn take_final_segments(&mut self) -> Vec<(String, f32, f32)> {
        std::mem::take(&mut self.pending_final_segments)
    }

    /// Get current buffer duration
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

    /// Check if model has KV cache support
    pub fn has_kv_cache(&self) -> bool {
        self.model.has_kv_cache()
    }
}

/// Result from Canary Flash chunk processing
#[derive(Debug, Clone)]
pub struct CanaryFlashChunkResult {
    /// Transcribed text
    pub text: String,
    /// Whether this is a partial transcription
    pub is_partial: bool,
    /// Current buffer duration in seconds
    pub buffer_time: f32,
}

// ============================================================================
// StreamingTranscriber implementation
// ============================================================================

impl StreamingTranscriber for RealtimeCanaryFlash {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "canary-180m-flash".to_string(),
            display_name: "Canary 180M Flash".to_string(),
            description: "NVIDIA's fast Canary 180M Flash model optimized for real-time ASR".to_string(),
            supports_diarization: self.has_diarization(),
            languages: vec![
                "en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryFlash::push_audio(self, samples)?;

        let mut segments = Vec::new();

        let final_segs = self.take_final_segments();
        for (text, start, end) in final_segs {
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
        let result = RealtimeCanaryFlash::finalize(self)?;

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
        RealtimeCanaryFlash::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        RealtimeCanaryFlash::buffer_duration(self)
    }

    fn total_duration(&self) -> f32 {
        RealtimeCanaryFlash::total_duration(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RealtimeCanaryFlashConfig::default();
        assert_eq!(config.buffer_size_secs, 8.0);
        assert_eq!(config.process_interval_secs, 0.5);
        assert_eq!(config.language, "en");
    }
}
