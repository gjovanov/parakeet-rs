//! VAD-triggered TDT transcription with diarization
//!
//! Uses Silero VAD to detect speech segments, then transcribes complete utterances
//! with the Parakeet TDT model and optional speaker diarization.
//!
//! ## Modes
//! - **Immediate**: Transcribes each VAD segment as it arrives
//! - **Buffered**: Accumulates segments until min/max duration reached
//! - **SlidingWindow**: Buffers N segments, transcribes window, slides forward with overlap
//!
//! ## Benefits
//! - Lower CPU usage (transcription only on pauses)
//! - Better accuracy (complete utterances, no mid-word cuts)
//! - Natural sentence boundaries
//! - Optional speaker diarization per segment

use crate::error::Result;
use crate::execution::ModelConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::vad::{VadConfig, VadSegmenter, VadSegment, VAD_SAMPLE_RATE};
use crate::parakeet_tdt::ParakeetTDT;
use crate::transcriber::Transcriber;
use std::path::Path;

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

/// Buffer mode for VAD transcription
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadBufferMode {
    /// Immediate: transcribe each VAD segment as it arrives
    Immediate,
    /// Buffered: accumulate segments until min/max duration reached
    Buffered,
    /// Sliding window: keep N segments, transcribe window, slide forward
    SlidingWindow,
}

impl Default for VadBufferMode {
    fn default() -> Self {
        VadBufferMode::Immediate
    }
}

/// Buffered VAD segment with audio samples
#[derive(Debug, Clone)]
struct BufferedSegment {
    samples: Vec<f32>,
    start_time: f32,
    end_time: f32,
}

/// Configuration for VAD-triggered TDT
#[derive(Debug, Clone)]
pub struct RealtimeTdtVadConfig {
    /// VAD configuration
    pub vad: VadConfig,
    /// Whether to enable diarization
    pub enable_diarization: bool,
    /// Buffer mode
    pub buffer_mode: VadBufferMode,
    /// Maximum buffer duration (seconds) before forcing transcription
    pub max_buffer_duration: f32,
    /// Maximum segments in sliding window
    pub max_window_segments: usize,
    /// Overlap segments to keep after sliding
    pub overlap_segments: usize,
}

impl Default for RealtimeTdtVadConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            enable_diarization: true,
            buffer_mode: VadBufferMode::Immediate,
            max_buffer_duration: 6.0,
            max_window_segments: 5,
            overlap_segments: 1,
        }
    }
}

impl RealtimeTdtVadConfig {
    /// Create config from mode string
    pub fn from_mode(mode: &str) -> Self {
        Self {
            vad: VadConfig::from_mode(mode),
            enable_diarization: true,
            ..Default::default()
        }
    }

    /// Create sliding window mode config
    /// Buffers up to max_segments OR max_duration, transcribes, then slides forward
    pub fn sliding_window() -> Self {
        Self {
            vad: VadConfig::pause_based(),
            enable_diarization: true,
            buffer_mode: VadBufferMode::SlidingWindow,
            max_buffer_duration: 6.0,  // Max 6s per window for faster output
            max_window_segments: 3,    // Max 3 segments per window
            overlap_segments: 1,       // Keep 1 segment for context
        }
    }
}

/// VAD-triggered TDT transcriber with diarization
///
/// Uses voice activity detection to segment audio into utterances,
/// then transcribes each complete utterance with the TDT model.
/// Supports immediate, buffered, and sliding window modes.
#[cfg(feature = "sortformer")]
pub struct RealtimeTdtVad {
    model: ParakeetTDT,
    diarizer: Option<SortformerStream>,
    segmenter: VadSegmenter,
    config: RealtimeTdtVadConfig,

    /// Total samples received
    total_samples: usize,

    /// Pending transcription segments (output)
    pending_segments: Vec<TranscriptionSegment>,

    /// Buffered VAD segments waiting to be transcribed
    segment_buffer: Vec<BufferedSegment>,

    /// Total duration of buffered audio (seconds)
    buffer_duration: f32,

    /// Time since last speech ended (for detecting pauses)
    silence_duration: f32,

    /// Buffer mode from config
    buffer_mode: VadBufferMode,

    /// Last transcription text for sliding window deduplication
    last_window_text: String,
}

#[cfg(feature = "sortformer")]
impl RealtimeTdtVad {
    /// Create a new VAD-triggered TDT transcriber
    ///
    /// # Arguments
    /// * `tdt_model_path` - Path to Parakeet TDT ONNX model directory
    /// * `diar_model_path` - Optional path to diarization model
    /// * `vad_model_path` - Path to silero_vad.onnx
    /// * `exec_config` - Optional execution config (CPU/GPU)
    /// * `config` - Optional transcriber config
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        tdt_model_path: P1,
        diar_model_path: Option<P2>,
        vad_model_path: P3,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeTdtVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffer_mode = config.buffer_mode;

        // Create TDT model
        let model = ParakeetTDT::from_pretrained(
            tdt_model_path,
            exec_config.clone(),
        )?;

        // Create diarizer if path provided and enabled
        let diarizer = if config.enable_diarization {
            if let Some(diar_path) = diar_model_path {
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

        // Create VAD segmenter
        let segmenter = VadSegmenter::new(
            vad_model_path,
            config.vad.clone(),
            exec_config,
        )?;

        match buffer_mode {
            VadBufferMode::SlidingWindow => {
                eprintln!(
                    "[TDT-VAD] Sliding window mode: max_segments={}, max_duration={:.1}s, overlap={}, diarization={}",
                    config.max_window_segments,
                    config.max_buffer_duration,
                    config.overlap_segments,
                    diarizer.is_some()
                );
            }
            VadBufferMode::Buffered => {
                eprintln!(
                    "[TDT-VAD] Buffered mode: max_duration={:.1}s, diarization={}",
                    config.max_buffer_duration,
                    diarizer.is_some()
                );
            }
            VadBufferMode::Immediate => {
                eprintln!("[TDT-VAD] Immediate mode: diarization={}", diarizer.is_some());
            }
        }

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
        })
    }

    /// Push audio samples and get transcription results
    ///
    /// Audio is accumulated until a speech pause is detected,
    /// then the complete utterance is transcribed.
    /// In sliding window mode, buffers segments and transcribes periodically.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<TdtVadResult> {
        self.total_samples += samples.len();
        let chunk_duration = samples.len() as f32 / VAD_SAMPLE_RATE as f32;

        // Also push audio to diarizer for speaker tracking
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        // Get any completed speech segments from VAD
        let vad_segments = self.segmenter.push_audio(samples)?;

        match self.buffer_mode {
            VadBufferMode::SlidingWindow => {
                self.process_sliding_window_mode(vad_segments, chunk_duration)?;
            }
            VadBufferMode::Buffered => {
                self.process_buffered_mode(vad_segments, chunk_duration)?;
            }
            VadBufferMode::Immediate => {
                // Immediate mode: transcribe each segment as it arrives
                for segment in vad_segments {
                    self.transcribe_segment_immediate(segment)?;
                }
            }
        }

        // Return and clear pending segments
        let segments = std::mem::take(&mut self.pending_segments);

        Ok(TdtVadResult {
            segments,
            is_speaking: self.segmenter.is_speaking(),
            total_duration: self.total_duration(),
        })
    }

    /// Process audio in sliding window mode
    fn process_sliding_window_mode(&mut self, vad_segments: Vec<VadSegment>, chunk_duration: f32) -> Result<()> {
        let is_speaking = self.segmenter.is_speaking();

        // Track silence duration
        if !is_speaking {
            self.silence_duration += chunk_duration;
        } else {
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
            self.silence_duration = 0.0;
        }

        // Check if window is full (by segment count OR duration)
        let window_full_by_segments = self.segment_buffer.len() >= self.config.max_window_segments;
        let window_full_by_duration = self.buffer_duration >= self.config.max_buffer_duration;

        // Transcribe when window is full and we're in a pause
        let should_transcribe = !self.segment_buffer.is_empty()
            && !is_speaking
            && (window_full_by_segments || window_full_by_duration);

        if should_transcribe {
            eprintln!(
                "[TDT-VAD] Sliding window full: {} segments, {:.1}s duration, transcribing",
                self.segment_buffer.len(),
                self.buffer_duration
            );
            self.transcribe_sliding_window()?;
        }

        Ok(())
    }

    /// Process audio in buffered mode
    fn process_buffered_mode(&mut self, vad_segments: Vec<VadSegment>, chunk_duration: f32) -> Result<()> {
        let is_speaking = self.segmenter.is_speaking();

        // Track silence duration
        if !is_speaking && !self.segment_buffer.is_empty() {
            self.silence_duration += chunk_duration;
        } else if is_speaking {
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
            self.silence_duration = 0.0;
        }

        // Transcribe when max duration reached or pause detected
        let should_transcribe = !self.segment_buffer.is_empty()
            && (self.buffer_duration >= self.config.max_buffer_duration
                || (self.silence_duration >= 0.5 && !is_speaking));

        if should_transcribe {
            self.transcribe_buffer()?;
        }

        Ok(())
    }

    /// Transcribe the sliding window and emit only new content
    fn transcribe_sliding_window(&mut self) -> Result<()> {
        if self.segment_buffer.is_empty() {
            return Ok(());
        }

        // Combine all buffered segments into one audio stream
        let start_time = self.segment_buffer.first().unwrap().start_time;
        let end_time = self.segment_buffer.last().unwrap().end_time;

        let total_samples: usize = self.segment_buffer.iter().map(|s| s.samples.len()).sum();
        let mut combined_samples = Vec::with_capacity(total_samples);

        for segment in &self.segment_buffer {
            combined_samples.extend_from_slice(&segment.samples);
        }

        eprintln!(
            "[TDT-VAD] Transcribing sliding window: {} segments, {:.2}s - {:.2}s ({} samples)",
            self.segment_buffer.len(),
            start_time,
            end_time,
            combined_samples.len()
        );

        // Transcribe the combined audio
        let result = self.model.transcribe_samples(
            combined_samples,
            VAD_SAMPLE_RATE as u32,
            1,
            None,
        )?;
        let text = result.text.trim().to_string();

        if !text.is_empty() {
            // Get speaker from diarizer if available
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (start_time + end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };

            // Find new text by comparing with previous window transcription
            let new_text = self.extract_new_text(&text);

            if !new_text.is_empty() {
                // Calculate approximate timing for new content
                let text_ratio = new_text.len() as f32 / text.len().max(1) as f32;
                let duration = end_time - start_time;
                let new_start = end_time - (duration * text_ratio);

                eprintln!(
                    "[TDT-VAD] Sliding window NEW text [{:.2}s - {:.2}s] Speaker {:?}: \"{}\"",
                    new_start, end_time, speaker, new_text
                );

                self.pending_segments.push(TranscriptionSegment {
                    text: new_text,
                    start_time: new_start,
                    end_time,
                    speaker,
                    confidence: None,
                    is_final: true,
                    inference_time_ms: None,
                });
            }

            // Update last window text for next comparison
            self.last_window_text = text;
        }

        // Slide the window: keep overlap_segments for context
        self.slide_window();

        Ok(())
    }

    /// Transcribe the buffer and clear it
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

        let result = self.model.transcribe_samples(
            combined_samples,
            VAD_SAMPLE_RATE as u32,
            1,
            None,
        )?;
        let text = result.text.trim().to_string();

        if !text.is_empty() {
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (start_time + end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };

            eprintln!(
                "[TDT-VAD] Buffered transcription [{:.2}s - {:.2}s] Speaker {:?}: \"{}\"",
                start_time, end_time, speaker, text
            );

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

        // Clear buffer
        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;

        Ok(())
    }

    /// Extract new text that wasn't in the previous transcription
    fn extract_new_text(&self, current_text: &str) -> String {
        if self.last_window_text.is_empty() {
            return current_text.to_string();
        }

        let last_words: Vec<&str> = self.last_window_text.split_whitespace().collect();
        let current_words: Vec<&str> = current_text.split_whitespace().collect();

        // Find overlap
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

    /// Slide the window forward, keeping overlap_segments for context
    fn slide_window(&mut self) {
        let overlap = self.config.overlap_segments;
        let to_remove = self.segment_buffer.len().saturating_sub(overlap);

        if to_remove > 0 {
            let removed: Vec<_> = self.segment_buffer.drain(..to_remove).collect();
            let removed_duration: f32 = removed.iter()
                .map(|s| s.end_time - s.start_time)
                .sum();
            self.buffer_duration = (self.buffer_duration - removed_duration).max(0.0);

            eprintln!(
                "[TDT-VAD] Slid window: removed {} segments, keeping {} for context, buffer now {:.1}s",
                to_remove,
                self.segment_buffer.len(),
                self.buffer_duration
            );
        }
    }

    /// Transcribe a VAD segment immediately with optional diarization
    fn transcribe_segment_immediate(&mut self, segment: VadSegment) -> Result<()> {
        // Transcribe with TDT using the Transcriber trait
        let result = self.model.transcribe_samples(
            segment.samples.clone(),
            VAD_SAMPLE_RATE as u32,
            1, // mono
            None, // no timestamp mode needed
        )?;

        let text = result.text.trim().to_string();

        if text.is_empty() {
            return Ok(());
        }

        // Get speaker if diarization is enabled
        let speaker = if let Some(diarizer) = &self.diarizer {
            // Get speaker at the midpoint of this segment
            let mid_time = (segment.start_time + segment.end_time) / 2.0;
            diarizer.get_speaker_at(mid_time)
        } else {
            None
        };

        eprintln!(
            "[TDT-VAD] Transcribed segment [{:.2}s - {:.2}s] Speaker {:?}: \"{}\"",
            segment.start_time, segment.end_time, speaker, text
        );

        self.pending_segments.push(TranscriptionSegment {
            text,
            start_time: segment.start_time,
            end_time: segment.end_time,
            speaker,
            confidence: None,
            is_final: true, // VAD segments are always final
            inference_time_ms: None,
        });

        Ok(())
    }

    /// Finalize any pending speech
    pub fn finalize(&mut self) -> Result<TdtVadResult> {
        // Get any remaining speech from VAD
        if let Some(segment) = self.segmenter.finalize()? {
            match self.buffer_mode {
                VadBufferMode::SlidingWindow | VadBufferMode::Buffered => {
                    // Add to buffer
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

        // Transcribe any remaining buffer
        if !self.segment_buffer.is_empty() {
            eprintln!(
                "[TDT-VAD] Finalizing: transcribing remaining {:.1}s buffer ({} segments)",
                self.buffer_duration,
                self.segment_buffer.len()
            );
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

        Ok(TdtVadResult {
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
        self.last_window_text.clear();
    }

    /// Get total audio duration in seconds
    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / VAD_SAMPLE_RATE as f32
    }

    /// Check if currently detecting speech
    pub fn is_speaking(&self) -> bool {
        self.segmenter.is_speaking()
    }

    /// Get VAD state name for debugging
    pub fn vad_state(&self) -> &'static str {
        self.segmenter.state_name()
    }
}

/// Result from VAD-triggered TDT processing
#[derive(Debug, Clone)]
pub struct TdtVadResult {
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

#[cfg(feature = "sortformer")]
impl StreamingTranscriber for RealtimeTdtVad {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "parakeet-tdt-vad".to_string(),
            display_name: "Parakeet TDT 0.6B (VAD)".to_string(),
            description: "NVIDIA Parakeet TDT with Silero VAD for utterance detection".to_string(),
            supports_diarization: self.diarizer.is_some(),
            languages: vec!["en".to_string()],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = RealtimeTdtVad::push_audio(self, samples)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0, // VAD doesn't have a fixed buffer
            total_duration: result.total_duration,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let result = RealtimeTdtVad::finalize(self)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0,
            total_duration: result.total_duration,
        })
    }

    fn reset(&mut self) {
        RealtimeTdtVad::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        0.0 // VAD doesn't use a fixed buffer
    }

    fn total_duration(&self) -> f32 {
        RealtimeTdtVad::total_duration(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_mode() {
        let config = RealtimeTdtVadConfig::from_mode("speedy");
        assert_eq!(config.vad.silence_trigger_ms, 300);
        assert!(config.enable_diarization);
    }
}
