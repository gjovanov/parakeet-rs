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
    /// Inference time in milliseconds (how long the model took to generate this segment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_time_ms: Option<u32>,
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

    fn make_segment(text: &str, is_final: bool) -> TranscriptionSegment {
        TranscriptionSegment {
            text: text.to_string(),
            start_time: 0.0,
            end_time: 1.0,
            speaker: None,
            confidence: None,
            is_final,
            inference_time_ms: None,
        }
    }

    #[test]
    fn test_streaming_chunk_result_empty() {
        let result = StreamingChunkResult::empty();
        assert!(!result.has_segments());
        assert_eq!(result.final_segments().count(), 0);
        assert_eq!(result.partial_segments().count(), 0);
        assert_eq!(result.buffer_duration, 0.0);
        assert_eq!(result.total_duration, 0.0);
    }

    #[test]
    fn test_streaming_chunk_result_mixed() {
        let mut result = StreamingChunkResult::empty();

        result.segments.push(TranscriptionSegment {
            text: "Hello".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            speaker: Some(0),
            confidence: Some(0.95),
            is_final: true,
            inference_time_ms: None,
        });

        result.segments.push(TranscriptionSegment {
            text: "world".to_string(),
            start_time: 1.0,
            end_time: 2.0,
            speaker: Some(0),
            confidence: Some(0.90),
            is_final: false,
            inference_time_ms: None,
        });

        assert!(result.has_segments());
        assert_eq!(result.final_segments().count(), 1);
        assert_eq!(result.partial_segments().count(), 1);
    }

    #[test]
    fn test_all_final_segments() {
        let mut result = StreamingChunkResult::empty();
        result.segments.push(make_segment("One.", true));
        result.segments.push(make_segment("Two.", true));

        assert_eq!(result.final_segments().count(), 2);
        assert_eq!(result.partial_segments().count(), 0);
    }

    #[test]
    fn test_all_partial_segments() {
        let mut result = StreamingChunkResult::empty();
        result.segments.push(make_segment("partial", false));

        assert_eq!(result.final_segments().count(), 0);
        assert_eq!(result.partial_segments().count(), 1);
    }

    #[test]
    fn test_segment_default_values() {
        let seg = make_segment("test", false);
        assert_eq!(seg.speaker, None);
        assert_eq!(seg.confidence, None);
        assert_eq!(seg.inference_time_ms, None);
        assert!(!seg.is_final);
    }

    #[test]
    fn test_segment_with_all_fields() {
        let seg = TranscriptionSegment {
            text: "Full segment".to_string(),
            start_time: 1.5,
            end_time: 3.2,
            speaker: Some(2),
            confidence: Some(0.87),
            is_final: true,
            inference_time_ms: Some(150),
        };

        assert_eq!(seg.text, "Full segment");
        assert_eq!(seg.start_time, 1.5);
        assert_eq!(seg.end_time, 3.2);
        assert_eq!(seg.speaker, Some(2));
        assert_eq!(seg.confidence, Some(0.87));
        assert!(seg.is_final);
        assert_eq!(seg.inference_time_ms, Some(150));
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo {
            id: "canary-1b".to_string(),
            display_name: "Canary 1B".to_string(),
            description: "Multilingual ASR model".to_string(),
            supports_diarization: true,
            languages: vec!["en".to_string(), "de".to_string()],
            is_loaded: true,
        };

        assert_eq!(info.id, "canary-1b");
        assert!(info.supports_diarization);
        assert_eq!(info.languages.len(), 2);
    }

    #[test]
    fn test_segment_serialization() {
        let seg = make_segment("test", true);
        let json = serde_json::to_string(&seg).unwrap();
        assert!(json.contains("\"text\":\"test\""));
        assert!(json.contains("\"is_final\":true"));
        // inference_time_ms should be skipped when None
        assert!(!json.contains("inference_time_ms"));
    }

    #[test]
    fn test_segment_serialization_with_inference_time() {
        let seg = TranscriptionSegment {
            text: "test".to_string(),
            start_time: 0.0,
            end_time: 1.0,
            speaker: None,
            confidence: None,
            is_final: true,
            inference_time_ms: Some(200),
        };
        let json = serde_json::to_string(&seg).unwrap();
        assert!(json.contains("\"inference_time_ms\":200"));
    }

    #[test]
    fn test_segment_deserialization() {
        let json = r#"{"text":"hello","start_time":0.5,"end_time":1.5,"speaker":1,"confidence":0.9,"is_final":true}"#;
        let seg: TranscriptionSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.text, "hello");
        assert_eq!(seg.start_time, 0.5);
        assert_eq!(seg.speaker, Some(1));
        assert!(seg.is_final);
    }
}
