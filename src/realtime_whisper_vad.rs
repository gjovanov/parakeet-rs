//! VAD-triggered Whisper transcription with optional segment buffering
//!
//! Uses Silero VAD to detect speech segments, then transcribes complete utterances
//! with the Whisper model. Supports two modes:
//!
//! ## Immediate Mode (default)
//! - Transcribes each VAD segment immediately after pause detection
//! - Lower latency, but may produce fragmented short transcripts
//!
//! ## Buffered Mode
//! - Buffers VAD segments until minimum duration is reached
//! - Better quality by giving ASR more context
//! - Transcribes when: buffer >= min_duration OR long pause OR max_duration reached
//!
//! ## Benefits
//! - Lower CPU usage (transcription only on pauses)
//! - Better accuracy (complete utterances, no mid-word cuts)
//! - Natural sentence boundaries
//! - Supports 99 languages with auto-detection

use crate::error::Result;
use crate::execution::ModelConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::vad::{VadConfig, VadSegmenter, VadSegment, VAD_SAMPLE_RATE};
use crate::whisper::{WhisperConfig, WhisperModel, WhisperVariant};
use std::path::Path;

/// Configuration for VAD-triggered Whisper
#[derive(Debug, Clone)]
pub struct RealtimeWhisperVadConfig {
    /// VAD configuration
    pub vad: VadConfig,
    /// Target language code (e.g., "en", "de", "auto")
    pub language: String,
    /// Task: "transcribe" or "translate"
    pub task: String,
    /// Minimum buffer duration (seconds) before transcribing (0 = immediate mode)
    pub min_buffer_duration: f32,
    /// Maximum buffer duration (seconds) before forcing transcription (max 30s for Whisper)
    pub max_buffer_duration: f32,
    /// Long pause threshold (seconds) - forces transcription even if min_buffer not reached
    pub long_pause_threshold: f32,
}

impl Default for RealtimeWhisperVadConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            language: "en".to_string(),
            task: "transcribe".to_string(),
            min_buffer_duration: 0.0, // Immediate mode by default
            max_buffer_duration: 25.0, // Leave headroom for Whisper's 30s limit
            long_pause_threshold: 1.5,
        }
    }
}

impl RealtimeWhisperVadConfig {
    /// Create config from mode string
    pub fn from_mode(mode: &str, language: String) -> Self {
        Self {
            vad: VadConfig::from_mode(mode),
            language,
            ..Default::default()
        }
    }

    /// Create buffered mode config for sentence-level transcription
    pub fn buffered(language: String) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            task: "transcribe".to_string(),
            min_buffer_duration: 1.5,  // Minimum 1.5s of speech for quality
            max_buffer_duration: 25.0, // Whisper max is 30s
            long_pause_threshold: 1.0, // 1s pause = sentence boundary
        }
    }

    /// Create buffered mode with custom durations
    pub fn buffered_custom(
        language: String,
        min_secs: f32,
        max_secs: f32,
        long_pause_secs: f32,
    ) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            task: "transcribe".to_string(),
            min_buffer_duration: min_secs,
            max_buffer_duration: max_secs.min(25.0), // Cap at 25s for safety
            long_pause_threshold: long_pause_secs,
        }
    }

    /// Enable translation mode (to English)
    pub fn with_translation(mut self) -> Self {
        self.task = "translate".to_string();
        self
    }
}

/// Buffered VAD segment with audio samples
#[derive(Debug, Clone)]
struct BufferedSegment {
    samples: Vec<f32>,
    start_time: f32,
    end_time: f32,
}

/// VAD-triggered Whisper transcriber
///
/// Uses voice activity detection to segment audio into utterances,
/// then transcribes each complete utterance with the Whisper model.
/// Supports buffered mode for better quality transcription.
pub struct RealtimeWhisperVad {
    model: WhisperModel,
    segmenter: VadSegmenter,
    config: RealtimeWhisperVadConfig,

    /// Total samples received
    total_samples: usize,

    /// Pending transcription segments (output)
    pending_segments: Vec<TranscriptionSegment>,

    /// Buffered VAD segments waiting to be transcribed
    segment_buffer: Vec<BufferedSegment>,

    /// Total duration of buffered audio (seconds)
    buffer_duration: f32,

    /// Time since last speech ended (for detecting long pauses)
    silence_duration: f32,

    /// Whether we're in buffered mode
    buffered_mode: bool,
}

impl RealtimeWhisperVad {
    /// Create a new VAD-triggered Whisper transcriber
    ///
    /// # Arguments
    /// * `whisper_model_path` - Path to Whisper ONNX model directory
    /// * `vad_model_path` - Path to silero_vad.onnx
    /// * `exec_config` - Optional execution config (CPU/GPU)
    /// * `config` - Optional transcriber config
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        whisper_model_path: P1,
        vad_model_path: P2,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeWhisperVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffered_mode = config.min_buffer_duration > 0.0;

        // Create Whisper model
        let whisper_config = WhisperConfig {
            language: config.language.clone(),
            task: config.task.clone(),
            timestamps: false,
            ..Default::default()
        };
        let model = WhisperModel::from_pretrained(
            whisper_model_path,
            exec_config.clone(),
            Some(whisper_config),
        )?;

        // Create VAD segmenter
        let segmenter = VadSegmenter::new(
            vad_model_path,
            config.vad.clone(),
            exec_config,
        )?;

        if buffered_mode {
            eprintln!(
                "[WhisperVAD] Buffered mode: min={:.1}s, max={:.1}s, long_pause={:.1}s",
                config.min_buffer_duration,
                config.max_buffer_duration,
                config.long_pause_threshold
            );
        }

        eprintln!(
            "[WhisperVAD] Model: {}, Language: {}, Task: {}",
            model.variant().as_str(),
            config.language,
            config.task
        );

        Ok(Self {
            model,
            segmenter,
            config,
            total_samples: 0,
            pending_segments: Vec::new(),
            segment_buffer: Vec::new(),
            buffer_duration: 0.0,
            silence_duration: 0.0,
            buffered_mode,
        })
    }

    /// Push audio samples and get transcription results
    ///
    /// Audio is accumulated until a speech pause is detected,
    /// then the complete utterance is transcribed.
    ///
    /// In buffered mode, segments are accumulated until:
    /// - Buffer duration >= min_buffer_duration, OR
    /// - A long pause (>= long_pause_threshold) is detected, OR
    /// - Buffer duration >= max_buffer_duration
    pub fn push_audio_internal(&mut self, samples: &[f32]) -> Result<WhisperVadResult> {
        self.total_samples += samples.len();
        let chunk_duration = samples.len() as f32 / VAD_SAMPLE_RATE as f32;

        // Get any completed speech segments from VAD
        let vad_segments = self.segmenter.push_audio(samples)?;

        if self.buffered_mode {
            // Buffered mode: accumulate segments
            let is_speaking = self.segmenter.is_speaking();

            // Track silence duration when not speaking
            if !is_speaking && self.segment_buffer.is_empty() {
                // No buffered speech, just track silence
                self.silence_duration += chunk_duration;
            } else if !is_speaking && !self.segment_buffer.is_empty() {
                // We have buffered speech and now silence
                self.silence_duration += chunk_duration;
            } else {
                // Speaking - reset silence counter
                self.silence_duration = 0.0;
            }

            // Add new segments to buffer
            for segment in vad_segments {
                let seg_duration = segment.end_time - segment.start_time;
                self.buffer_duration += seg_duration;
                self.segment_buffer.push(BufferedSegment {
                    samples: segment.samples,
                    start_time: segment.start_time,
                    end_time: segment.end_time,
                });
                // Reset silence counter when we get a new segment
                self.silence_duration = 0.0;
            }

            // Decide whether to transcribe the buffer
            const MIN_SPEECH_FOR_TRANSCRIPTION: f32 = 1.0;

            let should_transcribe = if self.segment_buffer.is_empty() {
                false
            } else if self.buffer_duration >= self.config.max_buffer_duration {
                eprintln!("[WhisperVAD] Max buffer reached ({:.1}s), transcribing", self.buffer_duration);
                true
            } else if self.silence_duration >= self.config.long_pause_threshold
                      && self.buffer_duration >= MIN_SPEECH_FOR_TRANSCRIPTION {
                eprintln!(
                    "[WhisperVAD] Long pause ({:.1}s >= {:.1}s), transcribing {:.1}s buffer",
                    self.silence_duration,
                    self.config.long_pause_threshold,
                    self.buffer_duration
                );
                true
            } else if self.buffer_duration >= self.config.min_buffer_duration && !is_speaking {
                eprintln!(
                    "[WhisperVAD] Min buffer reached ({:.1}s), transcribing",
                    self.buffer_duration
                );
                true
            } else {
                false
            };

            if should_transcribe {
                self.transcribe_buffer()?;
            }
        } else {
            // Immediate mode: transcribe each segment as it arrives
            for segment in vad_segments {
                self.transcribe_segment_immediate(segment)?;
            }
        }

        // Return and clear pending segments
        let segments = std::mem::take(&mut self.pending_segments);

        Ok(WhisperVadResult {
            segments,
            is_speaking: self.segmenter.is_speaking(),
            total_duration: self.total_duration(),
        })
    }

    /// Transcribe a single VAD segment immediately (non-buffered mode)
    fn transcribe_segment_immediate(&mut self, segment: VadSegment) -> Result<()> {
        // Transcribe the speech segment
        let text = self.model.transcribe(&segment.samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            eprintln!(
                "[WhisperVAD] Transcribed segment [{:.2}s - {:.2}s]: \"{}\"",
                segment.start_time, segment.end_time, text
            );

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time: segment.start_time,
                end_time: segment.end_time,
                speaker: None,
                confidence: None,
                is_final: true,
            });
        }

        Ok(())
    }

    /// Transcribe all buffered segments together (buffered mode)
    fn transcribe_buffer(&mut self) -> Result<()> {
        if self.segment_buffer.is_empty() {
            return Ok(());
        }

        // Combine all buffered segments into one audio stream
        let start_time = self.segment_buffer.first().unwrap().start_time;
        let end_time = self.segment_buffer.last().unwrap().end_time;

        // Calculate total samples and combine
        let total_samples: usize = self.segment_buffer.iter().map(|s| s.samples.len()).sum();
        let mut combined_samples = Vec::with_capacity(total_samples);

        for segment in &self.segment_buffer {
            combined_samples.extend_from_slice(&segment.samples);
        }

        eprintln!(
            "[WhisperVAD] Transcribing buffer: {} segments, {:.2}s - {:.2}s ({} samples)",
            self.segment_buffer.len(),
            start_time,
            end_time,
            combined_samples.len()
        );

        // Transcribe the combined audio
        let text = self.model.transcribe(&combined_samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            eprintln!(
                "[WhisperVAD] Buffered transcription [{:.2}s - {:.2}s]: \"{}\"",
                start_time, end_time, text
            );

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time,
                end_time,
                speaker: None,
                confidence: None,
                is_final: true,
            });
        }

        // Clear buffer
        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;

        Ok(())
    }

    /// Finalize any pending speech
    pub fn finalize_internal(&mut self) -> Result<WhisperVadResult> {
        // Get any remaining speech from VAD
        if let Some(segment) = self.segmenter.finalize()? {
            if self.buffered_mode {
                // Add to buffer
                let seg_duration = segment.end_time - segment.start_time;
                self.buffer_duration += seg_duration;
                self.segment_buffer.push(BufferedSegment {
                    samples: segment.samples,
                    start_time: segment.start_time,
                    end_time: segment.end_time,
                });
            } else {
                self.transcribe_segment_immediate(segment)?;
            }
        }

        // In buffered mode, transcribe any remaining buffer
        if self.buffered_mode && !self.segment_buffer.is_empty() {
            eprintln!(
                "[WhisperVAD] Finalizing: transcribing remaining {:.1}s buffer",
                self.buffer_duration
            );
            self.transcribe_buffer()?;
        }

        let segments = std::mem::take(&mut self.pending_segments);

        Ok(WhisperVadResult {
            segments,
            is_speaking: false,
            total_duration: self.total_duration(),
        })
    }

    /// Reset the transcriber state
    pub fn reset_internal(&mut self) {
        self.segmenter.reset();
        self.total_samples = 0;
        self.pending_segments.clear();
        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;
    }

    /// Get total audio duration in seconds
    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / VAD_SAMPLE_RATE as f32
    }

    /// Check if currently detecting speech
    pub fn is_speaking(&self) -> bool {
        self.segmenter.is_speaking()
    }

    /// Set the target language
    pub fn set_language(&mut self, lang: &str) {
        self.model.set_language(lang);
        self.config.language = lang.to_string();
    }

    /// Get the model variant
    pub fn variant(&self) -> WhisperVariant {
        self.model.variant()
    }

    /// Get VAD state name for debugging
    pub fn vad_state(&self) -> &'static str {
        self.segmenter.state_name()
    }
}

/// Result from VAD-triggered Whisper processing
#[derive(Debug, Clone)]
pub struct WhisperVadResult {
    /// Completed transcription segments
    pub segments: Vec<TranscriptionSegment>,
    /// Whether speech is currently being detected
    pub is_speaking: bool,
    /// Total audio duration processed
    pub total_duration: f32,
}

// ============================================================================
// StreamingTranscriber implementation
// ============================================================================

impl StreamingTranscriber for RealtimeWhisperVad {
    fn model_info(&self) -> ModelInfo {
        let variant = self.model.variant();
        ModelInfo {
            id: format!("{}-vad", variant.as_str()),
            display_name: match variant {
                WhisperVariant::LargeV2 => "Whisper Large v2 (VAD)".to_string(),
                WhisperVariant::LargeV3 => "Whisper Large v3 (VAD)".to_string(),
            },
            description: format!("OpenAI {} with Silero VAD for utterance detection", variant.as_str()),
            supports_diarization: false,
            languages: vec![
                "en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string(),
                "it".to_string(), "pt".to_string(), "nl".to_string(), "pl".to_string(),
                "ru".to_string(), "zh".to_string(), "ja".to_string(), "ko".to_string(),
                "ar".to_string(), "hi".to_string(), "tr".to_string(), "vi".to_string(),
                "auto".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = self.push_audio_internal(samples)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: self.buffer_duration,
            total_duration: result.total_duration,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let result = self.finalize_internal()?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0,
            total_duration: result.total_duration,
        })
    }

    fn reset(&mut self) {
        self.reset_internal();
    }

    fn buffer_duration(&self) -> f32 {
        self.buffer_duration
    }

    fn total_duration(&self) -> f32 {
        self.total_duration()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_mode() {
        let config = RealtimeWhisperVadConfig::from_mode("speedy", "de".to_string());
        assert_eq!(config.language, "de");
        assert_eq!(config.vad.silence_trigger_ms, 300);
    }

    #[test]
    fn test_config_buffered() {
        let config = RealtimeWhisperVadConfig::buffered("en".to_string());
        assert_eq!(config.min_buffer_duration, 1.5);
        assert!(config.max_buffer_duration <= 25.0);
    }

    #[test]
    fn test_config_translation() {
        let config = RealtimeWhisperVadConfig::default().with_translation();
        assert_eq!(config.task, "translate");
    }
}
