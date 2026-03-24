//! Pause-segmented transcription mode for Whisper models
//!
//! Segments audio by acoustic pauses, transcribes each speech chunk exactly once.
//! No sliding buffer, no GrowingTextMerger, no echo dedup.
//! Each FINAL = one speech segment with precise [start_time, end_time].
//!
//! Follows the same pattern as PauseSegmentedCanary.

use crate::error::Result;
use crate::pause_segmented::{calculate_rms, strip_context_prefix, truncate_hallucination, PauseSegmentedConfig, FRAME_SAMPLES};
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::whisper::{WhisperModel, WhisperModelConfig};
use std::collections::VecDeque;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Pause-segmented Whisper transcriber
pub struct PauseSegmentedWhisper {
    model: WhisperModel,
    config: PauseSegmentedConfig,
    /// Model identifier for API responses
    model_id: String,

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
    /// Consecutive silent frames (20ms each) for frame-level pause detection
    consecutive_silence_frames: usize,
    /// Pending segments to emit
    pending_segments: Vec<TranscriptionSegment>,
    /// Last emitted FINAL end time
    last_final_end_time: f32,

    /// Ring buffer of previous segments' audio (for context)
    context_ring: VecDeque<(Vec<f32>, String)>,

    /// Beam search size (stored for potential future reconfiguration)
    #[allow(dead_code)]
    beam_size: i32,
    /// Thread count
    #[allow(dead_code)]
    n_threads: i32,
    /// Use GPU
    #[allow(dead_code)]
    use_gpu: bool,
}

impl PauseSegmentedWhisper {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: Option<PauseSegmentedConfig>,
        model_id: Option<String>,
        beam_size: Option<i32>,
        n_threads: Option<i32>,
        use_gpu: bool,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let beam_size = beam_size.unwrap_or(5);
        let n_threads = n_threads.unwrap_or(4);
        let model_id = model_id.unwrap_or_else(|| "whisper".to_string());

        let whisper_config = WhisperModelConfig {
            language: config.language.clone(),
            beam_size,
            n_threads,
            use_gpu,
            ..Default::default()
        };

        let model = WhisperModel::from_file(model_path, Some(whisper_config))?;

        Ok(Self {
            model,
            config,
            model_id,
            speech_buffer: Vec::with_capacity(SAMPLE_RATE * 15),
            total_samples: 0,
            speech_start_sample: None,
            silence_start_time: None,
            is_speaking: false,
            samples_since_partial: 0,
            consecutive_silence_frames: 0,
            pending_segments: Vec::new(),
            last_final_end_time: 0.0,
            context_ring: VecDeque::new(),
            beam_size,
            n_threads,
            use_gpu,
        })
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

    /// Transcribe the current speech buffer and emit as FINAL
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

            let context_count = (ctx - 1).min(self.context_ring.len());
            let start_idx = self.context_ring.len() - context_count;

            for (audio, text) in self.context_ring.iter().skip(start_idx) {
                combined_audio.extend_from_slice(audio);
                if !combined_text.is_empty() {
                    combined_text.push(' ');
                }
                combined_text.push_str(text);
            }

            combined_audio.extend_from_slice(&self.speech_buffer);

            eprintln!(
                "[PauseSegmentedWhisper] Transcribing with {} context seg(s) ({:.1}s total, {:.1}s current)",
                context_count,
                combined_audio.len() as f32 / SAMPLE_RATE as f32,
                duration
            );

            (combined_audio, combined_text)
        } else {
            (self.speech_buffer.clone(), String::new())
        };

        // Transcribe
        let inference_start = std::time::Instant::now();
        let full_text = self.model.transcribe_text(&audio_to_transcribe)?;
        let inference_ms = inference_start.elapsed().as_millis() as u32;
        let full_text = full_text.trim().to_string();

        // Strip context prefix
        let text = if !context_text.is_empty() && !full_text.is_empty() {
            strip_context_prefix(&full_text, &context_text)
        } else {
            full_text
        };

        let text = text.trim().to_string();

        if !text.is_empty() {
            let text = truncate_hallucination(&text);

            eprintln!(
                "[PauseSegmentedWhisper] FINAL [{:.2}s-{:.2}s] ({:.1}s, {}ms) \"{}\"",
                start_time,
                end_time,
                duration,
                inference_ms,
                &text.chars().take(80).collect::<String>()
            );

            // Store in context ring
            let max_ctx = self.config.context_segments.max(1);
            self.context_ring
                .push_back((self.speech_buffer.clone(), text.clone()));
            while self.context_ring.len() > max_ctx {
                self.context_ring.pop_front();
            }

            self.pending_segments.push(TranscriptionSegment {
                text,
                raw_text: None,
                start_time,
                end_time,
                speaker: None,
                confidence: None,
                is_final: true,
                inference_time_ms: Some(inference_ms),
            });

            self.last_final_end_time = end_time;
        } else {
            let max_ctx = self.config.context_segments.max(1);
            self.context_ring
                .push_back((self.speech_buffer.clone(), String::new()));
            while self.context_ring.len() > max_ctx {
                self.context_ring.pop_front();
            }
        }

        self.speech_buffer.clear();
        self.speech_start_sample = None;
        self.samples_since_partial = 0;

        Ok(())
    }

    /// Transcribe current buffer and emit as PARTIAL
    fn emit_partial(&mut self) -> Result<()> {
        if self.speech_buffer.is_empty() {
            return Ok(());
        }

        let start_time = self.speech_start_time();
        let duration = self.speech_duration();
        let end_time = start_time + duration;

        let inference_start = std::time::Instant::now();
        let text = self.model.transcribe_text(&self.speech_buffer)?;
        let inference_ms = inference_start.elapsed().as_millis() as u32;
        let text = text.trim().to_string();

        if !text.is_empty() {
            let text = truncate_hallucination(&text);

            self.pending_segments.push(TranscriptionSegment {
                text,
                raw_text: None,
                start_time,
                end_time,
                speaker: None,
                confidence: None,
                is_final: false,
                inference_time_ms: Some(inference_ms),
            });
        }

        self.samples_since_partial = 0;
        Ok(())
    }

    /// Process incoming audio samples using frame-level (20ms) pause detection.
    pub fn push_audio_samples(&mut self, samples: &[f32]) -> Result<()> {
        let threshold = self.config.silence_energy_threshold;
        let pause_frames_needed = (self.config.pause_threshold_secs * SAMPLE_RATE as f32 / FRAME_SAMPLES as f32).ceil() as usize;

        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + FRAME_SAMPLES).min(samples.len());
            let frame = &samples[offset..end];
            offset = end;

            if frame.len() < FRAME_SAMPLES / 2 {
                self.total_samples += frame.len();
                break;
            }

            let rms = calculate_rms(frame);
            let is_speech = rms >= threshold;

            self.total_samples += frame.len();

            if is_speech {
                self.consecutive_silence_frames = 0;

                if !self.is_speaking {
                    self.is_speaking = true;
                    if self.speech_start_sample.is_none() {
                        self.speech_start_sample = Some(self.total_samples - frame.len());
                    }
                }

                self.speech_buffer.extend_from_slice(frame);
                self.samples_since_partial += frame.len();

                if self.speech_duration() >= self.config.max_segment_secs {
                    self.transcribe_and_emit_final()?;
                } else if self.samples_since_partial as f32 / SAMPLE_RATE as f32
                    >= self.config.partial_interval_secs
                {
                    self.emit_partial()?;
                }
            } else {
                self.consecutive_silence_frames += 1;

                if self.is_speaking {
                    self.speech_buffer.extend_from_slice(frame);
                }

                if self.consecutive_silence_frames >= pause_frames_needed && self.is_speaking {
                    self.is_speaking = false;
                    self.transcribe_and_emit_final()?;
                }
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

impl StreamingTranscriber for PauseSegmentedWhisper {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.model_id.clone(),
            display_name: format!("Whisper ({}, Pause-Segmented)", self.model_id),
            description: "Whisper with pause-based audio segmentation".to_string(),
            supports_diarization: false,
            languages: vec![
                "en".to_string(),
                "de".to_string(),
                "fr".to_string(),
                "es".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        self.push_audio_samples(samples)?;

        let segments = self.take_segments();

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.speech_duration(),
            total_duration: self.current_time(),
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pause_segmented_config_defaults() {
        let config = PauseSegmentedConfig::default();
        assert_eq!(config.pause_threshold_secs, 0.3);
        assert_eq!(config.min_segment_secs, 0.5);
        assert_eq!(config.max_segment_secs, 15.0);
    }
}
