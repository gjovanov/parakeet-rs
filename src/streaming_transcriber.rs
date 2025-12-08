//! Streaming transcriber trait and types
//!
//! This module defines the common interface for all streaming transcription models,
//! enabling a unified API for different model architectures (TDT, Canary, Whisper, etc.)

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// A segment of transcribed text with timing and optional speaker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// The transcribed text
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Speaker ID (if diarization is enabled)
    pub speaker: Option<usize>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: Option<f32>,
    /// Whether this segment is finalized (won't change)
    pub is_final: bool,
}

/// Result of processing an audio chunk
#[derive(Debug, Clone, Default)]
pub struct StreamingChunkResult {
    /// Transcription segments from this chunk
    pub segments: Vec<TranscriptionSegment>,
    /// Current buffer duration in seconds
    pub buffer_duration: f32,
    /// Total audio processed in seconds
    pub total_duration: f32,
}

impl StreamingChunkResult {
    /// Create an empty result
    pub fn empty() -> Self {
        Self::default()
    }

    /// Check if there are any segments
    pub fn has_segments(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Get only the final (confirmed) segments
    pub fn final_segments(&self) -> impl Iterator<Item = &TranscriptionSegment> {
        self.segments.iter().filter(|s| s.is_final)
    }

    /// Get only the partial (unconfirmed) segments
    pub fn partial_segments(&self) -> impl Iterator<Item = &TranscriptionSegment> {
        self.segments.iter().filter(|s| !s.is_final)
    }
}

/// Model information for display and selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique model identifier
    pub id: String,
    /// Human-readable display name
    pub display_name: String,
    /// Model description
    pub description: String,
    /// Whether the model supports speaker diarization
    pub supports_diarization: bool,
    /// Supported languages (empty = all languages)
    pub languages: Vec<String>,
    /// Whether the model is currently loaded and ready
    pub is_loaded: bool,
}

/// Trait for streaming transcription models
///
/// This trait provides a unified interface for different transcription model architectures.
/// Implementations should handle their own buffering and state management.
///
/// # Example
///
/// ```ignore
/// let mut transcriber = create_transcriber("parakeet-tdt")?;
///
/// // Process audio chunks
/// loop {
///     let samples = read_audio_chunk();
///     let result = transcriber.push_audio(&samples)?;
///
///     for segment in result.final_segments() {
///         println!("{}: {}", segment.start_time, segment.text);
///     }
/// }
///
/// // Finalize to get remaining transcription
/// let final_result = transcriber.finalize()?;
/// ```
pub trait StreamingTranscriber: Send {
    /// Get model information
    fn model_info(&self) -> ModelInfo;

    /// Push audio samples and get transcription results
    ///
    /// # Arguments
    /// * `samples` - Audio samples as f32 values normalized to [-1.0, 1.0]
    ///               Expected sample rate: 16kHz, mono channel
    ///
    /// # Returns
    /// Transcription result containing any new segments
    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult>;

    /// Finalize transcription and get any remaining text
    ///
    /// This should be called when the audio stream ends to flush
    /// any buffered audio and get the final transcription.
    fn finalize(&mut self) -> Result<StreamingChunkResult>;

    /// Reset the transcriber state for a new stream
    ///
    /// This clears all internal buffers and state, preparing
    /// the transcriber for a new audio stream.
    fn reset(&mut self);

    /// Get the current buffer duration in seconds
    fn buffer_duration(&self) -> f32;

    /// Get the total audio duration processed
    fn total_duration(&self) -> f32;
}

/// Factory function type for creating transcribers
pub type TranscriberFactory = Box<dyn Fn() -> Result<Box<dyn StreamingTranscriber>> + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_chunk_result() {
        let mut result = StreamingChunkResult::empty();
        assert!(!result.has_segments());

        result.segments.push(TranscriptionSegment {
            text: "Hello".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            speaker: Some(0),
            confidence: Some(0.95),
            is_final: true,
        });

        result.segments.push(TranscriptionSegment {
            text: "world".to_string(),
            start_time: 1.0,
            end_time: 2.0,
            speaker: Some(0),
            confidence: Some(0.90),
            is_final: false,
        });

        assert!(result.has_segments());
        assert_eq!(result.final_segments().count(), 1);
        assert_eq!(result.partial_segments().count(), 1);
    }
}
