//! VAD-triggered TDT transcription with diarization
//!
//! Uses Silero VAD to detect speech segments, then transcribes complete utterances
//! with the Parakeet TDT model and optional speaker diarization.
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

/// Configuration for VAD-triggered TDT
#[derive(Debug, Clone)]
pub struct RealtimeTdtVadConfig {
    /// VAD configuration
    pub vad: VadConfig,
    /// Whether to enable diarization
    pub enable_diarization: bool,
}

impl Default for RealtimeTdtVadConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            enable_diarization: true,
        }
    }
}

impl RealtimeTdtVadConfig {
    /// Create config from mode string
    pub fn from_mode(mode: &str) -> Self {
        Self {
            vad: VadConfig::from_mode(mode),
            enable_diarization: true,
        }
    }
}

/// VAD-triggered TDT transcriber with diarization
///
/// Uses voice activity detection to segment audio into utterances,
/// then transcribes each complete utterance with the TDT model.
#[cfg(feature = "sortformer")]
pub struct RealtimeTdtVad {
    model: ParakeetTDT,
    diarizer: Option<SortformerStream>,
    segmenter: VadSegmenter,
    config: RealtimeTdtVadConfig,

    /// Total samples received
    total_samples: usize,

    /// Pending transcription segments
    pending_segments: Vec<TranscriptionSegment>,
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

        Ok(Self {
            model,
            diarizer,
            segmenter,
            config,
            total_samples: 0,
            pending_segments: Vec::new(),
        })
    }

    /// Push audio samples and get transcription results
    ///
    /// Audio is accumulated until a speech pause is detected,
    /// then the complete utterance is transcribed.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<TdtVadResult> {
        self.total_samples += samples.len();

        // Also push audio to diarizer for speaker tracking
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        // Get any completed speech segments from VAD
        let vad_segments = self.segmenter.push_audio(samples)?;

        // Transcribe each completed segment
        for segment in vad_segments {
            self.transcribe_segment(segment)?;
        }

        // Return and clear pending segments
        let segments = std::mem::take(&mut self.pending_segments);

        Ok(TdtVadResult {
            segments,
            is_speaking: self.segmenter.is_speaking(),
            total_duration: self.total_duration(),
        })
    }

    /// Transcribe a VAD segment with optional diarization
    fn transcribe_segment(&mut self, segment: VadSegment) -> Result<()> {
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
        });

        Ok(())
    }

    /// Finalize any pending speech
    pub fn finalize(&mut self) -> Result<TdtVadResult> {
        // Get any remaining speech from VAD
        if let Some(segment) = self.segmenter.finalize()? {
            self.transcribe_segment(segment)?;
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
