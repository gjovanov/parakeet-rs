//! Whisper model wrapper using whisper-rs (whisper.cpp bindings)
//!
//! Provides a unified interface for Whisper GGML models, supporting both
//! GPU (CUDA via cuBLAS) and CPU inference. Wraps whisper-rs WhisperContext
//! and WhisperState for single-shot transcription of audio buffers.

use crate::error::{Error, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

/// Configuration for Whisper model loading and inference
#[derive(Debug, Clone)]
pub struct WhisperModelConfig {
    /// Target language code (e.g., "de", "en", "auto")
    pub language: String,
    /// Beam search size (0 = greedy)
    pub beam_size: i32,
    /// Number of threads for decoding
    pub n_threads: i32,
    /// Initial decoding temperature (0.0 = greedy)
    pub temperature: f32,
    /// Suppress blank tokens at start
    pub suppress_blank: bool,
    /// No-speech probability threshold
    pub no_speech_thold: f32,
    /// Enable token-level timestamps
    pub token_timestamps: bool,
    /// Use GPU if available (whisper feature must include cuda)
    pub use_gpu: bool,
}

impl Default for WhisperModelConfig {
    fn default() -> Self {
        Self {
            language: "de".to_string(),
            beam_size: 5,
            n_threads: 4,
            temperature: 0.0,
            suppress_blank: true,
            no_speech_thold: 0.6,
            token_timestamps: false,
            use_gpu: false,
        }
    }
}

/// A transcribed segment from Whisper
#[derive(Debug, Clone)]
pub struct WhisperSegment {
    /// Transcribed text
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// No-speech probability (higher = more likely silence/noise)
    pub no_speech_prob: f32,
}

/// Whisper model wrapper
///
/// Holds a WhisperContext (shared model weights) and creates WhisperState
/// per-transcription call. Thread-safe: WhisperContext is Send + Sync.
pub struct WhisperModel {
    ctx: WhisperContext,
    config: WhisperModelConfig,
}

impl WhisperModel {
    /// Load a Whisper model from a GGML file
    ///
    /// # Arguments
    /// * `model_path` - Path to the .bin GGML model file
    /// * `config` - Model configuration (language, beam_size, etc.)
    pub fn from_file<P: AsRef<Path>>(
        model_path: P,
        config: Option<WhisperModelConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let path = model_path.as_ref();

        if !path.exists() {
            return Err(Error::Whisper(format!(
                "Whisper model file not found: {}",
                path.display()
            )));
        }

        let mut ctx_params = WhisperContextParameters::default();
        ctx_params.use_gpu(config.use_gpu);

        let path_str = path.to_str().ok_or_else(|| {
            Error::Whisper(format!("Invalid model path: {}", path.display()))
        })?;

        eprintln!(
            "[Whisper] Loading model from {} (gpu: {}, lang: {}, beam: {})",
            path.display(),
            config.use_gpu,
            config.language,
            config.beam_size
        );

        let ctx = WhisperContext::new_with_params(path_str, ctx_params).map_err(|e| {
            Error::Whisper(format!("Failed to load Whisper model: {}", e))
        })?;

        eprintln!("[Whisper] Model loaded successfully");

        Ok(Self { ctx, config })
    }

    /// Transcribe audio samples and return segments
    ///
    /// # Arguments
    /// * `samples` - f32 PCM audio at 16kHz mono, normalized to [-1.0, 1.0]
    ///
    /// # Returns
    /// Vector of transcribed segments with text and timestamps
    pub fn transcribe(&self, samples: &[f32]) -> Result<Vec<WhisperSegment>> {
        if samples.is_empty() {
            return Ok(vec![]);
        }

        let mut state = self.ctx.create_state().map_err(|e| {
            Error::Whisper(format!("Failed to create Whisper state: {}", e))
        })?;

        let params = self.build_params();

        state.full(params, samples).map_err(|e| {
            Error::Whisper(format!("Whisper inference failed: {}", e))
        })?;

        self.extract_segments(&state)
    }

    /// Transcribe and return concatenated text (convenience method)
    pub fn transcribe_text(&self, samples: &[f32]) -> Result<String> {
        let segments = self.transcribe(samples)?;
        let text: String = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        Ok(text.trim().to_string())
    }

    /// Set the target language
    pub fn set_language(&mut self, lang: &str) {
        self.config.language = lang.to_string();
    }

    /// Get current language
    pub fn language(&self) -> &str {
        &self.config.language
    }

    /// Build FullParams from current config
    fn build_params(&self) -> FullParams<'_, '_> {
        let strategy = if self.config.beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size: self.config.beam_size,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };

        let mut params = FullParams::new(strategy);

        // Language
        let lang = if self.config.language == "auto" {
            None
        } else {
            Some(self.config.language.as_str())
        };
        params.set_language(lang);

        // Threading
        params.set_n_threads(self.config.n_threads);

        // Decoding
        params.set_temperature(self.config.temperature);
        params.set_suppress_blank(self.config.suppress_blank);
        params.set_no_speech_thold(self.config.no_speech_thold);

        // Timestamps
        params.set_token_timestamps(self.config.token_timestamps);

        // Disable console output from whisper.cpp
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Don't translate, just transcribe
        params.set_translate(false);

        params
    }

    /// Extract segments from WhisperState after inference
    fn extract_segments(&self, state: &WhisperState) -> Result<Vec<WhisperSegment>> {
        let mut segments = Vec::new();

        // Use the iterator API (whisper-rs 0.16+)
        for segment in state.as_iter() {
            let text = segment
                .to_str_lossy()
                .map(|cow| cow.to_string())
                .unwrap_or_default();

            // Timestamps are in centiseconds (10ms units)
            let start_time = segment.start_timestamp() as f32 / 100.0;
            let end_time = segment.end_timestamp() as f32 / 100.0;

            let no_speech_prob = segment.no_speech_probability();

            segments.push(WhisperSegment {
                text,
                start_time,
                end_time,
                no_speech_prob,
            });
        }

        Ok(segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = WhisperModelConfig::default();
        assert_eq!(config.language, "de");
        assert_eq!(config.beam_size, 5);
        assert_eq!(config.n_threads, 4);
        assert_eq!(config.temperature, 0.0);
        assert!(!config.use_gpu);
    }

    #[test]
    fn test_config_custom() {
        let config = WhisperModelConfig {
            language: "en".to_string(),
            beam_size: 1,
            use_gpu: true,
            ..Default::default()
        };
        assert_eq!(config.language, "en");
        assert_eq!(config.beam_size, 1);
        assert!(config.use_gpu);
    }
}
