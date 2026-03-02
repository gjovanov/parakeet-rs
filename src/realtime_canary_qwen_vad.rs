//! VAD-triggered Canary-Qwen transcription with optional segment buffering and diarization
//!
//! Uses Silero VAD to detect speech segments, then transcribes complete utterances
//! with the Canary-Qwen 2.5B SALM model. Supports buffered and sliding_window modes.
//!
//! Mirrors realtime_canary_vad.rs but uses CanaryQwenModel instead of CanaryModel.

use crate::canary_qwen::{CanaryQwenConfig, CanaryQwenModel};
use crate::error::Result;
use crate::execution::ModelConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::vad::{VadConfig, VadSegmenter, VadSegment, VAD_SAMPLE_RATE};
use std::path::Path;

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

/// Buffer mode for VAD transcription
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadBufferMode {
    Immediate,
    Buffered,
    SlidingWindow,
}

impl Default for VadBufferMode {
    fn default() -> Self {
        VadBufferMode::Immediate
    }
}

/// Configuration for VAD-triggered Canary-Qwen
#[derive(Debug, Clone)]
pub struct RealtimeCanaryQwenVadConfig {
    pub vad: VadConfig,
    pub language: String,
    pub buffer_mode: VadBufferMode,
    pub min_buffer_duration: f32,
    pub max_buffer_duration: f32,
    pub long_pause_threshold: f32,
    pub enable_diarization: bool,
    pub max_window_segments: usize,
    pub overlap_segments: usize,
}

impl Default for RealtimeCanaryQwenVadConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            language: "en".to_string(),
            buffer_mode: VadBufferMode::Immediate,
            min_buffer_duration: 0.0,
            max_buffer_duration: 15.0,
            long_pause_threshold: 1.5,
            enable_diarization: true,
            max_window_segments: 10,
            overlap_segments: 2,
        }
    }
}

impl RealtimeCanaryQwenVadConfig {
    pub fn from_mode(mode: &str, language: String) -> Self {
        Self {
            vad: VadConfig::from_mode(mode),
            language,
            ..Default::default()
        }
    }

    pub fn buffered(language: String) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            buffer_mode: VadBufferMode::Buffered,
            min_buffer_duration: 1.5,
            max_buffer_duration: 6.0,
            long_pause_threshold: 1.0,
            enable_diarization: true,
            max_window_segments: 10,
            overlap_segments: 2,
        }
    }

    pub fn sliding_window(language: String) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            buffer_mode: VadBufferMode::Buffered,
            min_buffer_duration: 1.0,
            max_buffer_duration: 5.0,
            long_pause_threshold: 0.35,
            enable_diarization: true,
            max_window_segments: 2,
            overlap_segments: 0,
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

/// VAD-triggered Canary-Qwen transcriber with optional diarization
#[cfg(feature = "sortformer")]
pub struct RealtimeCanaryQwenVad {
    model: CanaryQwenModel,
    diarizer: Option<SortformerStream>,
    segmenter: VadSegmenter,
    config: RealtimeCanaryQwenVadConfig,
    total_samples: usize,
    pending_segments: Vec<TranscriptionSegment>,
    segment_buffer: Vec<BufferedSegment>,
    buffer_duration: f32,
    silence_duration: f32,
    buffer_mode: VadBufferMode,
    last_window_text: String,
    committed_segment_count: usize,
    last_emitted_end_time: f32,
}

#[cfg(not(feature = "sortformer"))]
pub struct RealtimeCanaryQwenVad {
    model: CanaryQwenModel,
    segmenter: VadSegmenter,
    config: RealtimeCanaryQwenVadConfig,
    total_samples: usize,
    pending_segments: Vec<TranscriptionSegment>,
    segment_buffer: Vec<BufferedSegment>,
    buffer_duration: f32,
    silence_duration: f32,
    buffer_mode: VadBufferMode,
    last_window_text: String,
    committed_segment_count: usize,
    last_emitted_end_time: f32,
}

#[cfg(feature = "sortformer")]
impl RealtimeCanaryQwenVad {
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        canary_qwen_model_path: P1,
        diar_model_path: Option<P2>,
        vad_model_path: P3,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeCanaryQwenVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffer_mode = config.buffer_mode;

        let qwen_config = CanaryQwenConfig {
            language: config.language.clone(),
            ..CanaryQwenConfig::from_env()
        };
        let model = CanaryQwenModel::from_pretrained(
            canary_qwen_model_path,
            exec_config.clone(),
            Some(qwen_config),
        )?;

        let diarizer = if config.enable_diarization {
            if let Some(diar_path) = diar_model_path {
                eprintln!("[CanaryQwenVAD] Creating diarizer from {:?}", diar_path.as_ref());
                Some(SortformerStream::with_config(
                    diar_path,
                    exec_config.clone(),
                    Default::default(),
                )?)
            } else {
                None
            }
        } else {
            None
        };

        let segmenter = VadSegmenter::new(vad_model_path, config.vad.clone(), exec_config)?;

        eprintln!("[CanaryQwenVAD] Initialized ({:?} mode, diarization={})",
            buffer_mode, diarizer.is_some());

        Ok(Self {
            model,
            diarizer,
            segmenter,
            config,
            total_samples: 0,
            pending_segments: Vec::new(),
            segment_buffer: Vec::new(),
            buffer_duration: 0.0,
            silence_duration: 0.0,
            buffer_mode,
            last_window_text: String::new(),
            committed_segment_count: 0,
            last_emitted_end_time: 0.0,
        })
    }

    pub fn has_diarization(&self) -> bool {
        self.diarizer.is_some()
    }
}

#[cfg(not(feature = "sortformer"))]
impl RealtimeCanaryQwenVad {
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        canary_qwen_model_path: P1,
        vad_model_path: P2,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeCanaryQwenVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffer_mode = config.buffer_mode;

        let qwen_config = CanaryQwenConfig {
            language: config.language.clone(),
            ..CanaryQwenConfig::from_env()
        };
        let model = CanaryQwenModel::from_pretrained(
            canary_qwen_model_path,
            exec_config.clone(),
            Some(qwen_config),
        )?;

        let segmenter = VadSegmenter::new(vad_model_path, config.vad.clone(), exec_config)?;

        eprintln!("[CanaryQwenVAD] Initialized ({:?} mode)", buffer_mode);

        Ok(Self {
            model,
            segmenter,
            config,
            total_samples: 0,
            pending_segments: Vec::new(),
            segment_buffer: Vec::new(),
            buffer_duration: 0.0,
            silence_duration: 0.0,
            buffer_mode,
            last_window_text: String::new(),
            committed_segment_count: 0,
            last_emitted_end_time: 0.0,
        })
    }

    pub fn has_diarization(&self) -> bool {
        false
    }
}

impl RealtimeCanaryQwenVad {
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<CanaryQwenVadResult> {
        self.total_samples += samples.len();
        let chunk_duration = samples.len() as f32 / VAD_SAMPLE_RATE as f32;

        #[cfg(feature = "sortformer")]
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        let vad_segments = self.segmenter.push_audio(samples)?;

        match self.buffer_mode {
            VadBufferMode::Buffered => {
                self.process_buffered_mode(vad_segments, chunk_duration)?;
            }
            VadBufferMode::SlidingWindow => {
                self.process_sliding_window_mode(vad_segments, chunk_duration)?;
            }
            VadBufferMode::Immediate => {
                for segment in vad_segments {
                    self.transcribe_segment_immediate(segment)?;
                }
            }
        }

        let segments = std::mem::take(&mut self.pending_segments);

        Ok(CanaryQwenVadResult {
            segments,
            is_speaking: self.segmenter.is_speaking(),
            total_duration: self.total_duration(),
        })
    }

    fn process_buffered_mode(&mut self, vad_segments: Vec<VadSegment>, chunk_duration: f32) -> Result<()> {
        let is_speaking = self.segmenter.is_speaking();

        if !is_speaking {
            self.silence_duration += chunk_duration;
        } else {
            self.silence_duration = 0.0;
        }

        for segment in vad_segments {
            let seg_duration = segment.end_time - segment.start_time;
            self.buffer_duration += seg_duration;
            self.segment_buffer.push(BufferedSegment {
                samples: segment.samples,
                start_time: segment.start_time,
                end_time: segment.end_time,
            });
            self.silence_duration = 0.0;
        }

        const MIN_SPEECH_FOR_TRANSCRIPTION: f32 = 1.0;

        let should_transcribe = if self.segment_buffer.is_empty() {
            false
        } else if self.buffer_duration >= self.config.max_buffer_duration {
            true
        } else if self.silence_duration >= self.config.long_pause_threshold
                  && self.buffer_duration >= MIN_SPEECH_FOR_TRANSCRIPTION {
            true
        } else if self.buffer_duration >= self.config.min_buffer_duration && !is_speaking {
            true
        } else {
            false
        };

        if should_transcribe {
            self.transcribe_buffer()?;
        }

        Ok(())
    }

    fn process_sliding_window_mode(&mut self, vad_segments: Vec<VadSegment>, chunk_duration: f32) -> Result<()> {
        let is_speaking = self.segmenter.is_speaking();

        if !is_speaking {
            self.silence_duration += chunk_duration;
        } else {
            self.silence_duration = 0.0;
        }

        for segment in vad_segments {
            let seg_duration = segment.end_time - segment.start_time;
            self.buffer_duration += seg_duration;
            self.segment_buffer.push(BufferedSegment {
                samples: segment.samples,
                start_time: segment.start_time,
                end_time: segment.end_time,
            });
            self.silence_duration = 0.0;
        }

        let window_full_by_segments = self.segment_buffer.len() >= self.config.max_window_segments;
        let window_full_by_duration = self.buffer_duration >= self.config.max_buffer_duration;
        let pause_trigger = self.silence_duration >= self.config.long_pause_threshold;

        let should_transcribe = !self.segment_buffer.is_empty()
            && !is_speaking
            && (window_full_by_segments || window_full_by_duration || pause_trigger);

        if should_transcribe {
            self.transcribe_sliding_window()?;
        }

        Ok(())
    }

    #[allow(unused_variables)]
    fn transcribe_sliding_window(&mut self) -> Result<()> {
        if self.segment_buffer.is_empty() {
            return Ok(());
        }

        let start_time = self.segment_buffer.first().unwrap().start_time;
        let end_time = self.segment_buffer.last().unwrap().end_time;

        let total_samples: usize = self.segment_buffer.iter().map(|s| s.samples.len()).sum();
        let mut combined_samples = Vec::with_capacity(total_samples);
        for segment in &self.segment_buffer {
            combined_samples.extend_from_slice(&segment.samples);
        }

        let text = self.model.transcribe(&combined_samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            #[cfg(feature = "sortformer")]
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (start_time + end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };
            #[cfg(not(feature = "sortformer"))]
            let speaker: Option<usize> = None;

            let new_text = self.extract_new_text(&text);

            if !new_text.is_empty() {
                let newest_segment = self.segment_buffer.last().unwrap();
                let new_start = self.last_emitted_end_time.max(newest_segment.start_time);
                let new_end = newest_segment.end_time;

                self.pending_segments.push(TranscriptionSegment {
                    text: new_text,
                    start_time: new_start,
                    end_time: new_end,
                    speaker,
                    confidence: None,
                    is_final: true,
                    inference_time_ms: None,
                });

                self.last_emitted_end_time = new_end;
            }

            self.last_window_text = text;
        }

        self.slide_window();
        Ok(())
    }

    fn extract_new_text(&self, current_text: &str) -> String {
        if self.last_window_text.is_empty() {
            return current_text.to_string();
        }

        let last_words: Vec<&str> = self.last_window_text.split_whitespace().collect();
        let current_words: Vec<&str> = current_text.split_whitespace().collect();

        let mut best_overlap = 0;
        for overlap_len in 1..=last_words.len().min(current_words.len()) {
            let last_suffix = &last_words[last_words.len() - overlap_len..];
            let current_prefix = &current_words[..overlap_len];
            if last_suffix == current_prefix {
                best_overlap = overlap_len;
            }
        }

        if best_overlap > 0 && best_overlap < current_words.len() {
            current_words[best_overlap..].join(" ")
        } else if best_overlap == 0 {
            current_text.to_string()
        } else {
            String::new()
        }
    }

    fn slide_window(&mut self) {
        let overlap = self.config.overlap_segments;
        let to_remove = self.segment_buffer.len().saturating_sub(overlap);

        if to_remove > 0 {
            let removed: Vec<_> = self.segment_buffer.drain(..to_remove).collect();
            let removed_duration: f32 = removed.iter()
                .map(|s| s.end_time - s.start_time)
                .sum();
            self.buffer_duration = (self.buffer_duration - removed_duration).max(0.0);
            self.committed_segment_count += to_remove;
        }
    }

    fn transcribe_segment_immediate(&mut self, segment: VadSegment) -> Result<()> {
        let text = self.model.transcribe(&segment.samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            #[cfg(feature = "sortformer")]
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (segment.start_time + segment.end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };
            #[cfg(not(feature = "sortformer"))]
            let speaker: Option<usize> = None;

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time: segment.start_time,
                end_time: segment.end_time,
                speaker,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            });
        }

        Ok(())
    }

    fn transcribe_buffer(&mut self) -> Result<()> {
        if self.segment_buffer.is_empty() {
            return Ok(());
        }

        let start_time = self.segment_buffer.first().unwrap().start_time;
        let end_time = self.segment_buffer.last().unwrap().end_time;

        let total_samples: usize = self.segment_buffer.iter().map(|s| s.samples.len()).sum();
        let mut combined_samples = Vec::with_capacity(total_samples);
        for segment in &self.segment_buffer {
            combined_samples.extend_from_slice(&segment.samples);
        }

        let text = self.model.transcribe(&combined_samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            #[cfg(feature = "sortformer")]
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (start_time + end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };
            #[cfg(not(feature = "sortformer"))]
            let speaker: Option<usize> = None;

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time,
                end_time,
                speaker,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            });
        }

        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;

        Ok(())
    }

    pub fn finalize(&mut self) -> Result<CanaryQwenVadResult> {
        if let Some(segment) = self.segmenter.finalize()? {
            match self.buffer_mode {
                VadBufferMode::Buffered | VadBufferMode::SlidingWindow => {
                    let seg_duration = segment.end_time - segment.start_time;
                    self.buffer_duration += seg_duration;
                    self.segment_buffer.push(BufferedSegment {
                        samples: segment.samples,
                        start_time: segment.start_time,
                        end_time: segment.end_time,
                    });
                }
                VadBufferMode::Immediate => {
                    self.transcribe_segment_immediate(segment)?;
                }
            }
        }

        if !self.segment_buffer.is_empty() {
            match self.buffer_mode {
                VadBufferMode::SlidingWindow => {
                    self.transcribe_sliding_window()?;
                }
                VadBufferMode::Buffered => {
                    self.transcribe_buffer()?;
                }
                VadBufferMode::Immediate => {}
            }
        }

        let segments = std::mem::take(&mut self.pending_segments);

        Ok(CanaryQwenVadResult {
            segments,
            is_speaking: false,
            total_duration: self.total_duration(),
        })
    }

    pub fn reset(&mut self) {
        self.segmenter.reset();
        self.total_samples = 0;
        self.pending_segments.clear();
        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;
        self.last_window_text.clear();
        self.committed_segment_count = 0;
        self.last_emitted_end_time = 0.0;
    }

    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / VAD_SAMPLE_RATE as f32
    }

    pub fn is_speaking(&self) -> bool {
        self.segmenter.is_speaking()
    }
}

/// Result from VAD-triggered Canary-Qwen processing
#[derive(Debug, Clone)]
pub struct CanaryQwenVadResult {
    pub segments: Vec<TranscriptionSegment>,
    pub is_speaking: bool,
    pub total_duration: f32,
}

// ============================================================================
// StreamingTranscriber implementation
// ============================================================================

impl StreamingTranscriber for RealtimeCanaryQwenVad {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "canary-qwen-2b-vad".to_string(),
            display_name: "Canary-Qwen 2.5B (VAD)".to_string(),
            description: "NVIDIA Canary-Qwen SALM with Silero VAD for utterance detection".to_string(),
            supports_diarization: self.has_diarization(),
            languages: vec!["en".to_string()],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryQwenVad::push_audio(self, samples)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0,
            total_duration: result.total_duration,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryQwenVad::finalize(self)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0,
            total_duration: result.total_duration,
        })
    }

    fn reset(&mut self) {
        RealtimeCanaryQwenVad::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        0.0
    }

    fn total_duration(&self) -> f32 {
        RealtimeCanaryQwenVad::total_duration(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_mode() {
        let config = RealtimeCanaryQwenVadConfig::from_mode("speedy", "en".to_string());
        assert_eq!(config.language, "en");
    }

    #[test]
    fn test_config_default() {
        let config = RealtimeCanaryQwenVadConfig::default();
        assert_eq!(config.language, "en");
        assert_eq!(config.max_buffer_duration, 15.0);
    }

    #[test]
    fn test_buffered_config() {
        let config = RealtimeCanaryQwenVadConfig::buffered("en".to_string());
        assert_eq!(config.buffer_mode, VadBufferMode::Buffered);
        assert_eq!(config.min_buffer_duration, 1.5);
    }

    #[test]
    fn test_sliding_window_config() {
        let config = RealtimeCanaryQwenVadConfig::sliding_window("en".to_string());
        assert_eq!(config.max_window_segments, 2);
        assert_eq!(config.overlap_segments, 0);
    }
}
