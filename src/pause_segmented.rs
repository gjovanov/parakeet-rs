//! Pause-segmented transcription mode.
//!
//! Segments audio by acoustic pauses, transcribes each speech chunk exactly once.
//! No sliding buffer, no GrowingTextMerger, no echo dedup.
//! Each FINAL = one speech segment with precise [start_time, end_time].

use crate::canary::{CanaryConfig, CanaryModel};
use crate::error::Result;
use crate::ExecutionConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Calculate RMS energy of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Truncate text at first detected hallucination (repeated words/phrases)
fn truncate_hallucination(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 { return text.to_string(); }
    let mut consecutive_count = 0;
    for i in 1..words.len() {
        if words[i].to_lowercase() == words[i - 1].to_lowercase() && words[i].len() > 1 {
            consecutive_count += 1;
            if consecutive_count >= 3 {
                let truncate_at = i - consecutive_count + 1;
                if truncate_at > 0 {
                    return words[..truncate_at].join(" ");
                }
                return String::new();
            }
        } else {
            consecutive_count = 0;
        }
    }
    text.to_string()
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct PauseSegmentedConfig {
    /// Silence duration to trigger transcription (default: 0.3s = 300ms)
    pub pause_threshold_secs: f32,
    /// RMS below this is silence (default: 0.008)
    pub silence_energy_threshold: f32,
    /// Don't transcribe segments shorter than this (default: 0.5s)
    pub min_segment_secs: f32,
    /// Force-transcribe after this duration even without pause (default: 15.0s)
    pub max_segment_secs: f32,
    /// How often to emit PARTIALs during speech (default: 1.5s)
    pub partial_interval_secs: f32,
    /// Target language
    pub language: String,
}

impl Default for PauseSegmentedConfig {
    fn default() -> Self {
        Self {
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            min_segment_secs: 0.5,
            max_segment_secs: 15.0,
            partial_interval_secs: 1.5,
            language: "de".to_string(),
        }
    }
}

// ============================================================================
// Core struct
// ============================================================================

pub struct PauseSegmentedCanary {
    model: CanaryModel,
    config: PauseSegmentedConfig,

    /// Accumulated speech samples for the current chunk
    speech_buffer: Vec<f32>,
    /// Total samples received since session start
    total_samples: usize,
    /// Global sample index where current speech started
    speech_start_sample: Option<usize>,
    /// When silence started (in seconds)
    silence_start_time: Option<f32>,
    /// Whether currently in speech
    is_speaking: bool,
    /// Samples accumulated since last PARTIAL emission
    samples_since_partial: usize,
    /// Pending segments to emit
    pending_segments: Vec<TranscriptionSegment>,
    /// Last emitted FINAL end time (for monotonic timestamps)
    last_final_end_time: f32,

    /// Optional diarizer
    #[cfg(feature = "sortformer")]
    diarizer: Option<crate::sortformer_stream::SortformerStream>,
}

impl PauseSegmentedCanary {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<PauseSegmentedConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let canary_config = CanaryConfig {
            language: config.language.clone(),
            ..Default::default()
        };
        let model = CanaryModel::from_pretrained(model_path, exec_config, Some(canary_config))?;

        Ok(Self {
            model,
            config,
            speech_buffer: Vec::with_capacity(SAMPLE_RATE * 15), // pre-alloc 15s
            total_samples: 0,
            speech_start_sample: None,
            silence_start_time: None,
            is_speaking: false,
            samples_since_partial: 0,
            pending_segments: Vec::new(),
            last_final_end_time: 0.0,
            #[cfg(feature = "sortformer")]
            diarizer: None,
        })
    }

    #[cfg(feature = "sortformer")]
    pub fn new_with_diarization<P: AsRef<Path>>(
        model_path: P,
        diar_path: Option<P>,
        exec_config: Option<ExecutionConfig>,
        config: Option<PauseSegmentedConfig>,
    ) -> Result<Self> {
        let mut instance = Self::new(model_path, exec_config, config)?;
        if let Some(diar) = diar_path {
            instance.diarizer = Some(crate::sortformer_stream::SortformerStream::new(diar)?);
        }
        Ok(instance)
    }

    /// Get current time in seconds
    fn current_time(&self) -> f32 {
        self.total_samples as f32 / SAMPLE_RATE as f32
    }

    /// Get speech buffer duration in seconds
    fn speech_duration(&self) -> f32 {
        self.speech_buffer.len() as f32 / SAMPLE_RATE as f32
    }

    /// Get speech start time in seconds
    fn speech_start_time(&self) -> f32 {
        self.speech_start_sample
            .map(|s| s as f32 / SAMPLE_RATE as f32)
            .unwrap_or(self.current_time())
    }

    /// Get speaker at a given time
    #[cfg(feature = "sortformer")]
    fn get_speaker_at(&mut self, time: f32) -> Option<usize> {
        self.diarizer.as_mut().and_then(|d| d.get_speaker_at(time))
    }

    #[cfg(not(feature = "sortformer"))]
    fn get_speaker_at(&self, _time: f32) -> Option<usize> {
        None
    }

    /// Transcribe the current speech buffer and emit as FINAL
    fn transcribe_and_emit_final(&mut self) -> Result<()> {
        if self.speech_buffer.is_empty() {
            return Ok(());
        }

        let duration = self.speech_duration();
        if duration < self.config.min_segment_secs {
            // Too short — likely noise. Discard.
            self.speech_buffer.clear();
            self.speech_start_sample = None;
            self.samples_since_partial = 0;
            return Ok(());
        }

        let start_time = self.speech_start_time();
        let end_time = start_time + duration;

        // Transcribe
        let text = self.model.transcribe(&self.speech_buffer)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            let text = truncate_hallucination(&text);
            let mid_time = (start_time + end_time) / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            eprintln!(
                "[PauseSegmented] FINAL [{:.2}s-{:.2}s] ({:.1}s) \"{}\"",
                start_time, end_time, duration,
                &text.chars().take(80).collect::<String>()
            );

            self.pending_segments.push(TranscriptionSegment {
                text,
                raw_text: None,
                start_time,
                end_time,
                speaker,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            });

            self.last_final_end_time = end_time;
        }

        // Reset for next chunk
        self.speech_buffer.clear();
        self.speech_start_sample = None;
        self.samples_since_partial = 0;

        Ok(())
    }

    /// Transcribe current buffer and emit as PARTIAL (for live display)
    fn emit_partial(&mut self) -> Result<()> {
        if self.speech_buffer.is_empty() {
            return Ok(());
        }

        let start_time = self.speech_start_time();
        let duration = self.speech_duration();
        let end_time = start_time + duration;

        let text = self.model.transcribe(&self.speech_buffer)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            let text = truncate_hallucination(&text);
            let mid_time = (start_time + end_time) / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            self.pending_segments.push(TranscriptionSegment {
                text,
                raw_text: None,
                start_time,
                end_time,
                speaker,
                confidence: None,
                is_final: false,
                inference_time_ms: None,
            });
        }

        self.samples_since_partial = 0;
        Ok(())
    }

    /// Process incoming audio samples
    pub fn push_audio_samples(&mut self, samples: &[f32]) -> Result<()> {
        // Feed diarizer
        #[cfg(feature = "sortformer")]
        if let Some(ref mut diarizer) = self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        let rms = calculate_rms(samples);
        let is_speech = rms >= self.config.silence_energy_threshold;
        let current_time = self.current_time();

        self.total_samples += samples.len();

        if is_speech {
            // Speech detected
            self.silence_start_time = None;

            if !self.is_speaking {
                // Transition: silence → speech
                self.is_speaking = true;
                if self.speech_start_sample.is_none() {
                    self.speech_start_sample = Some(self.total_samples - samples.len());
                }
            }

            // Accumulate samples
            self.speech_buffer.extend_from_slice(samples);
            self.samples_since_partial += samples.len();

            // Check max segment duration — force transcribe
            if self.speech_duration() >= self.config.max_segment_secs {
                eprintln!(
                    "[PauseSegmented] Max segment ({:.1}s) reached, force-transcribing",
                    self.config.max_segment_secs
                );
                self.transcribe_and_emit_final()?;
            }
            // Check PARTIAL emission interval
            else if self.samples_since_partial as f32 / SAMPLE_RATE as f32 >= self.config.partial_interval_secs {
                self.emit_partial()?;
            }
        } else {
            // Silence detected
            if self.is_speaking {
                // Still accumulate the silence samples (they're part of the transition)
                self.speech_buffer.extend_from_slice(samples);
            }

            if self.silence_start_time.is_none() {
                self.silence_start_time = Some(current_time);
            }

            let silence_duration = self.current_time() - self.silence_start_time.unwrap_or(current_time);

            if silence_duration >= self.config.pause_threshold_secs && self.is_speaking {
                // Pause detected — transcribe accumulated speech
                self.is_speaking = false;
                self.transcribe_and_emit_final()?;
            }
        }

        Ok(())
    }

    /// Take pending segments (drains them)
    fn take_segments(&mut self) -> Vec<TranscriptionSegment> {
        std::mem::take(&mut self.pending_segments)
    }
}

// ============================================================================
// StreamingTranscriber implementation
// ============================================================================

impl StreamingTranscriber for PauseSegmentedCanary {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "canary-1b".to_string(),
            display_name: "Canary 1B (Pause-Segmented)".to_string(),
            description: "Canary 1B with pause-based audio segmentation".to_string(),
            supports_diarization: {
                #[cfg(feature = "sortformer")]
                { self.diarizer.is_some() }
                #[cfg(not(feature = "sortformer"))]
                { false }
            },
            languages: vec![
                "en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        self.push_audio_samples(samples)?;

        let segments = self.take_segments();
        let _inference_time: Option<u32> = None;

        // Set inference time on segments that don't have it
        for seg in &segments {
            if seg.is_final {
                // Already logged in transcribe_and_emit_final
            }
        }

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.speech_duration(),
            total_duration: self.current_time(),
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        // Transcribe any remaining speech
        if !self.speech_buffer.is_empty() {
            self.transcribe_and_emit_final()?;
        }

        let segments = self.take_segments();
        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.current_time(),
        })
    }

    fn reset(&mut self) {
        self.speech_buffer.clear();
        self.total_samples = 0;
        self.speech_start_sample = None;
        self.silence_start_time = None;
        self.is_speaking = false;
        self.samples_since_partial = 0;
        self.pending_segments.clear();
        self.last_final_end_time = 0.0;
    }

    fn buffer_duration(&self) -> f32 {
        self.speech_duration()
    }

    fn total_duration(&self) -> f32 {
        self.current_time()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = PauseSegmentedConfig::default();
        assert_eq!(config.pause_threshold_secs, 0.3);
        assert_eq!(config.min_segment_secs, 0.5);
        assert_eq!(config.max_segment_secs, 15.0);
        assert_eq!(config.partial_interval_secs, 1.5);
        assert_eq!(config.language, "de");
    }

    #[test]
    fn test_calculate_rms() {
        assert_eq!(calculate_rms(&[]), 0.0);
        let silence: Vec<f32> = vec![0.0; 1600];
        assert_eq!(calculate_rms(&silence), 0.0);
        let signal: Vec<f32> = vec![0.1; 1600];
        assert!((calculate_rms(&signal) - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_truncate_hallucination() {
        assert_eq!(truncate_hallucination("hello world"), "hello world");
        // Single-char words skipped by len>1 check
        assert_eq!(truncate_hallucination("a a a a"), "a a a a");
        assert_eq!(truncate_hallucination("ok ok ok ok ok"), "ok");
    }
}
