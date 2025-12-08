//! VAD-triggered Canary transcription with optional segment buffering
//!
//! Uses Silero VAD to detect speech segments, then transcribes complete utterances
//! with the Canary model. Supports two modes:
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

use crate::canary::{CanaryConfig, CanaryModel};
use crate::error::Result;
use crate::execution::ModelConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::vad::{VadConfig, VadSegmenter, VadSegment, VAD_SAMPLE_RATE};
use std::path::Path;

/// Configuration for VAD-triggered Canary
#[derive(Debug, Clone)]
pub struct RealtimeCanaryVadConfig {
    /// VAD configuration
    pub vad: VadConfig,
    /// Target language code
    pub language: String,
    /// Minimum buffer duration (seconds) before transcribing (0 = immediate mode)
    pub min_buffer_duration: f32,
    /// Maximum buffer duration (seconds) before forcing transcription
    pub max_buffer_duration: f32,
    /// Long pause threshold (seconds) - forces transcription even if min_buffer not reached
    pub long_pause_threshold: f32,
}

impl Default for RealtimeCanaryVadConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            language: "en".to_string(),
            min_buffer_duration: 0.0, // Immediate mode by default
            max_buffer_duration: 15.0,
            long_pause_threshold: 1.5,
        }
    }
}

impl RealtimeCanaryVadConfig {
    /// Create config from mode string
    pub fn from_mode(mode: &str, language: String) -> Self {
        Self {
            vad: VadConfig::from_mode(mode),
            language,
            ..Default::default()
        }
    }

    /// Create buffered mode config for sentence-level transcription
    /// Balanced for German: allows complete sentences with subordinate clauses
    pub fn buffered(language: String) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            min_buffer_duration: 1.5,  // Minimum 1.5s of speech for quality
            max_buffer_duration: 6.0,  // Allow up to 6s for complete German sentences
            long_pause_threshold: 1.0, // 1s pause = sentence boundary
        }
    }

    /// Create buffered mode with custom durations
    pub fn buffered_custom(language: String, min_secs: f32, max_secs: f32, long_pause_secs: f32) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            min_buffer_duration: min_secs,
            max_buffer_duration: max_secs,
            long_pause_threshold: long_pause_secs,
        }
    }
}

/// Buffered VAD segment with audio samples
#[derive(Debug, Clone)]
struct BufferedSegment {
    samples: Vec<f32>,
    start_time: f32,
    end_time: f32,
}

/// VAD-triggered Canary transcriber
///
/// Uses voice activity detection to segment audio into utterances,
/// then transcribes each complete utterance with the Canary model.
/// Supports buffered mode for better quality transcription.
pub struct RealtimeCanaryVad {
    model: CanaryModel,
    segmenter: VadSegmenter,
    config: RealtimeCanaryVadConfig,

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

impl RealtimeCanaryVad {
    /// Create a new VAD-triggered Canary transcriber
    ///
    /// # Arguments
    /// * `canary_model_path` - Path to Canary ONNX model directory
    /// * `vad_model_path` - Path to silero_vad.onnx
    /// * `exec_config` - Optional execution config (CPU/GPU)
    /// * `config` - Optional transcriber config
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        canary_model_path: P1,
        vad_model_path: P2,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeCanaryVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffered_mode = config.min_buffer_duration > 0.0;

        // Create Canary model
        let canary_config = CanaryConfig {
            language: config.language.clone(),
            ..Default::default()
        };
        let model = CanaryModel::from_pretrained(
            canary_model_path,
            exec_config.clone(),
            Some(canary_config),
        )?;

        // Create VAD segmenter
        let segmenter = VadSegmenter::new(
            vad_model_path,
            config.vad.clone(),
            exec_config,
        )?;

        if buffered_mode {
            eprintln!(
                "[CanaryVAD] Buffered mode: min={:.1}s, max={:.1}s, long_pause={:.1}s",
                config.min_buffer_duration,
                config.max_buffer_duration,
                config.long_pause_threshold
            );
        }

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
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<CanaryVadResult> {
        let prev_samples = self.total_samples;
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
            // Minimum speech required before we'll consider transcribing (avoid single words)
            const MIN_SPEECH_FOR_TRANSCRIPTION: f32 = 1.0; // At least 1 second of speech

            let should_transcribe = if self.segment_buffer.is_empty() {
                false
            } else if self.buffer_duration >= self.config.max_buffer_duration {
                // Max buffer reached - force transcription
                eprintln!("[CanaryVAD] Max buffer reached ({:.1}s), transcribing", self.buffer_duration);
                true
            } else if self.silence_duration >= self.config.long_pause_threshold
                      && self.buffer_duration >= MIN_SPEECH_FOR_TRANSCRIPTION {
                // Long pause detected AND we have enough speech to transcribe
                eprintln!(
                    "[CanaryVAD] Long pause ({:.1}s >= {:.1}s), transcribing {:.1}s buffer",
                    self.silence_duration,
                    self.config.long_pause_threshold,
                    self.buffer_duration
                );
                true
            } else if self.buffer_duration >= self.config.min_buffer_duration && !is_speaking {
                // Min buffer reached and not currently speaking
                eprintln!(
                    "[CanaryVAD] Min buffer reached ({:.1}s), transcribing",
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

        Ok(CanaryVadResult {
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
                "[CanaryVAD] Transcribed segment [{:.2}s - {:.2}s]: \"{}\"",
                segment.start_time, segment.end_time, text
            );

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time: segment.start_time,
                end_time: segment.end_time,
                speaker: None, // Canary doesn't support diarization
                confidence: None,
                is_final: true, // VAD segments are always final (complete utterances)
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
            "[CanaryVAD] Transcribing buffer: {} segments, {:.2}s - {:.2}s ({} samples)",
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
                "[CanaryVAD] Buffered transcription [{:.2}s - {:.2}s]: \"{}\"",
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
    pub fn finalize(&mut self) -> Result<CanaryVadResult> {
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
                "[CanaryVAD] Finalizing: transcribing remaining {:.1}s buffer",
                self.buffer_duration
            );
            self.transcribe_buffer()?;
        }

        let segments = std::mem::take(&mut self.pending_segments);

        Ok(CanaryVadResult {
            segments,
            is_speaking: false,
            total_duration: self.total_duration(),
        })
    }

    /// Reset the transcriber state
    pub fn reset(&mut self) {
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

    /// Get VAD state name for debugging
    pub fn vad_state(&self) -> &'static str {
        self.segmenter.state_name()
    }
}

/// Result from VAD-triggered Canary processing
#[derive(Debug, Clone)]
pub struct CanaryVadResult {
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

impl StreamingTranscriber for RealtimeCanaryVad {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "canary-1b-vad".to_string(),
            display_name: "Canary 1B v2 (VAD)".to_string(),
            description: "NVIDIA Canary with Silero VAD for utterance detection".to_string(),
            supports_diarization: false,
            languages: vec![
                "en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string(),
                "it".to_string(), "pt".to_string(), "nl".to_string(), "pl".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryVad::push_audio(self, samples)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0, // VAD doesn't have a fixed buffer
            total_duration: result.total_duration,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryVad::finalize(self)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0,
            total_duration: result.total_duration,
        })
    }

    fn reset(&mut self) {
        RealtimeCanaryVad::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        0.0 // VAD doesn't use a fixed buffer
    }

    fn total_duration(&self) -> f32 {
        RealtimeCanaryVad::total_duration(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_mode() {
        let config = RealtimeCanaryVadConfig::from_mode("speedy", "de".to_string());
        assert_eq!(config.language, "de");
        assert_eq!(config.vad.silence_trigger_ms, 300);
    }
}
