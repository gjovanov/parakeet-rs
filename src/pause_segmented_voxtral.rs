//! Pause-segmented transcription for Voxtral 4B model.
//!
//! Segments audio by acoustic pauses, transcribes each speech chunk exactly once.
//! No sliding buffer, no GrowingTextMerger, no echo dedup.
//! Each FINAL = one speech segment with precise [start_time, end_time].

use crate::voxtral::{VoxtralConfig, VoxtralModel};
use crate::pause_segmented::{calculate_rms, truncate_hallucination, strip_context_prefix, PauseSegmentedConfig};
use crate::error::Result;
use crate::ExecutionConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

// ============================================================================
// Configuration
// ============================================================================

// ============================================================================
// Core struct
// ============================================================================

pub struct PauseSegmentedVoxtral {
    model: VoxtralModel,
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

    /// Ring buffer of previous segments' audio (for context)
    /// Each entry: (audio_samples, emitted_text)
    context_ring: std::collections::VecDeque<(Vec<f32>, String)>,

    /// Optional diarizer
    #[cfg(feature = "sortformer")]
    diarizer: Option<crate::sortformer_stream::SortformerStream>,
}

impl PauseSegmentedVoxtral {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<PauseSegmentedConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let canary_config = VoxtralConfig {
            language: config.language.clone(),
            ..Default::default()
        };
        let model = VoxtralModel::from_pretrained(model_path, exec_config, Some(canary_config))?;

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
            context_ring: std::collections::VecDeque::new(),
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

    /// Transcribe the current speech buffer and emit as FINAL.
    /// When context_segments > 1, prepends previous segments' audio for better
    /// model context, then strips the known prefix text.
    fn transcribe_and_emit_final(&mut self) -> Result<()> {
        if self.speech_buffer.is_empty() {
            return Ok(());
        }

        let duration = self.speech_duration();
        if duration < self.config.min_segment_secs {
            self.speech_buffer.clear();
            self.speech_start_sample = None;
            self.samples_since_partial = 0;
            return Ok(());
        }

        let start_time = self.speech_start_time();
        let end_time = start_time + duration;
        let ctx = self.config.context_segments;

        // Build audio: context segments + current segment
        let (audio_to_transcribe, context_text) = if ctx > 1 && !self.context_ring.is_empty() {
            let mut combined_audio: Vec<f32> = Vec::new();
            let mut combined_text = String::new();

            // Take the last (ctx-1) segments as context
            let context_count = (ctx - 1).min(self.context_ring.len());
            let start_idx = self.context_ring.len() - context_count;

            for (audio, text) in self.context_ring.iter().skip(start_idx) {
                combined_audio.extend_from_slice(audio);
                if !combined_text.is_empty() { combined_text.push(' '); }
                combined_text.push_str(text);
            }

            // Append current segment
            combined_audio.extend_from_slice(&self.speech_buffer);

            let ctx_secs = combined_audio.len() as f32 / SAMPLE_RATE as f32;
            eprintln!(
                "[PauseSegmentedVoxtral] Transcribing with {} context seg(s) ({:.1}s total audio, {:.1}s current)",
                context_count, ctx_secs, duration
            );

            (combined_audio, combined_text)
        } else {
            (self.speech_buffer.clone(), String::new())
        };

        // Transcribe combined audio
        let full_text = self.model.transcribe(&audio_to_transcribe)?;
        let full_text = full_text.trim().to_string();

        // Strip the context prefix to get only the new segment's text
        let text = if !context_text.is_empty() && !full_text.is_empty() {
            strip_context_prefix(&full_text, &context_text)
        } else {
            full_text.clone()
        };

        let text = text.trim().to_string();

        if !text.is_empty() {
            let text = truncate_hallucination(&text);
            let mid_time = (start_time + end_time) / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            eprintln!(
                "[PauseSegmentedVoxtral] FINAL [{:.2}s-{:.2}s] ({:.1}s) \"{}\"",
                start_time, end_time, duration,
                &text.chars().take(80).collect::<String>()
            );

            // Store in context ring for future segments
            let max_ctx = self.config.context_segments.max(1);
            self.context_ring.push_back((self.speech_buffer.clone(), text.clone()));
            while self.context_ring.len() > max_ctx {
                self.context_ring.pop_front();
            }

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
        } else {
            // Even empty text, store audio in context ring
            let max_ctx = self.config.context_segments.max(1);
            self.context_ring.push_back((self.speech_buffer.clone(), String::new()));
            while self.context_ring.len() > max_ctx {
                self.context_ring.pop_front();
            }
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
                    "[PauseSegmentedVoxtral] Max segment ({:.1}s) reached, force-transcribing",
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

impl StreamingTranscriber for PauseSegmentedVoxtral {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "voxtral-4b".to_string(),
            display_name: "Voxtral 4B (Pause-Segmented)".to_string(),
            description: "Voxtral 4B with pause-based audio segmentation".to_string(),
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
        self.context_ring.clear();
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
