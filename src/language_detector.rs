//! Language detection and routing for automatic model selection
//!
//! When `language: "auto"` is specified, the system detects the spoken language
//! from initial audio and routes to the most appropriate ASR model:
//! - English → Parakeet TDT (fastest, English-only)
//! - German/French/Spanish → Canary 1B (multilingual)
//! - Other → Canary Qwen 2B (broadest language support)

use std::collections::HashMap;
use std::path::Path;

/// Detected language result
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    /// ISO 639-1 language code (e.g., "en", "de", "fr")
    pub language: String,
    /// Confidence score 0.0-1.0
    pub confidence: f32,
    /// Top-N alternatives with confidence scores
    pub alternatives: Vec<(String, f32)>,
}

/// Trait for language detection backends
pub trait LanguageDetector: Send + Sync {
    /// Detect language from audio samples (16kHz mono f32)
    /// Returns None if not enough audio has been accumulated
    fn detect(&mut self, samples: &[f32]) -> Option<LanguageDetection>;

    /// Minimum audio duration needed for reliable detection (seconds)
    fn min_audio_secs(&self) -> f32;

    /// Name of this detector for logging
    fn name(&self) -> &'static str;
}

/// Routes detected language to the best model ID
#[derive(Debug, Clone)]
pub struct LanguageRouter {
    /// Mapping from language code to model ID
    routes: HashMap<String, String>,
    /// Default model for unrecognized languages
    fallback_model: String,
}

impl LanguageRouter {
    /// Create a router with default routes:
    /// en → parakeet-tdt, de/fr/es → canary-1b, other → canary-qwen-2b
    pub fn default_routes() -> Self {
        let mut routes = HashMap::new();
        routes.insert("en".to_string(), "parakeet-tdt".to_string());
        routes.insert("de".to_string(), "canary-1b".to_string());
        routes.insert("fr".to_string(), "canary-1b".to_string());
        routes.insert("es".to_string(), "canary-1b".to_string());

        Self {
            routes,
            fallback_model: "canary-qwen-2b".to_string(),
        }
    }

    /// Get the model ID for a detected language
    pub fn route(&self, language: &str) -> &str {
        self.routes
            .get(language)
            .map(|s| s.as_str())
            .unwrap_or(&self.fallback_model)
    }

    /// Check if a language has an explicit route
    pub fn has_route(&self, language: &str) -> bool {
        self.routes.contains_key(language)
    }
}

/// Placeholder language detector using simple energy-based heuristics.
///
/// TODO: Replace with Silero LangID ONNX model for production use.
/// The Silero model provides ~95% accuracy on 3-5s of audio for 90+ languages.
/// Model: silero-langid.onnx (~5MB, INT8 quantized)
/// Input: 16kHz mono audio, minimum 2 seconds
/// Output: language probabilities for 90+ languages
pub struct SimpleLanguageDetector {
    /// Accumulated audio samples
    buffer: Vec<f32>,
    /// Whether detection has been performed
    detected: bool,
    /// Cached detection result
    result: Option<LanguageDetection>,
    /// Minimum samples needed (2 seconds at 16kHz)
    min_samples: usize,
}

impl SimpleLanguageDetector {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            detected: false,
            result: None,
            min_samples: 32000, // 2 seconds at 16kHz
        }
    }
}

impl Default for SimpleLanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageDetector for SimpleLanguageDetector {
    fn detect(&mut self, samples: &[f32]) -> Option<LanguageDetection> {
        if self.detected {
            return self.result.clone();
        }

        self.buffer.extend_from_slice(samples);

        if self.buffer.len() < self.min_samples {
            return None;
        }

        // Placeholder: always returns "de" (German) as default
        // TODO: Replace with actual Silero LangID inference
        // The inference would:
        // 1. Compute mel spectrogram from self.buffer
        // 2. Run ONNX model inference
        // 3. Softmax over language logits
        // 4. Return top-1 with confidence
        self.detected = true;
        let detection = LanguageDetection {
            language: "de".to_string(),
            confidence: 0.0, // 0.0 signals "not actually detected"
            alternatives: vec![],
        };
        self.result = Some(detection.clone());
        Some(detection)
    }

    fn min_audio_secs(&self) -> f32 {
        2.0
    }

    fn name(&self) -> &'static str {
        "SimpleDetector"
    }
}

/// Silero LangID detector (placeholder for future implementation)
///
/// TODO: Implement actual ONNX inference:
/// ```ignore
/// let session = ort::Session::builder()?
///     .with_execution_providers([CUDAExecutionProvider::default().build()])?
///     .commit_from_file("silero-langid.onnx")?;
///
/// let mel = compute_mel_spectrogram(&audio_samples);
/// let outputs = session.run(ort::inputs!["input" => mel]?)?;
/// let probs = outputs["output"].extract_tensor::<f32>()?;
/// ```
pub struct SileroLangDetector {
    _model_path: std::path::PathBuf,
    inner: SimpleLanguageDetector,
}

impl SileroLangDetector {
    /// Create a new Silero language detector
    /// Returns None if the model file doesn't exist
    pub fn new(model_path: impl AsRef<Path>) -> Option<Self> {
        let path = model_path.as_ref();
        if !path.exists() {
            eprintln!("[SileroLangDetector] Model not found at {:?}", path);
            return None;
        }
        eprintln!("[SileroLangDetector] Model found at {:?} (using placeholder until ONNX integration)", path);
        Some(Self {
            _model_path: path.to_path_buf(),
            inner: SimpleLanguageDetector::new(),
        })
    }
}

impl LanguageDetector for SileroLangDetector {
    fn detect(&mut self, samples: &[f32]) -> Option<LanguageDetection> {
        // TODO: Replace with actual Silero ONNX inference
        self.inner.detect(samples)
    }

    fn min_audio_secs(&self) -> f32 {
        3.0 // Silero needs ~3s for reliable detection
    }

    fn name(&self) -> &'static str {
        "SileroLangID"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_router_default_routes() {
        let router = LanguageRouter::default_routes();
        assert_eq!(router.route("en"), "parakeet-tdt");
        assert_eq!(router.route("de"), "canary-1b");
        assert_eq!(router.route("fr"), "canary-1b");
        assert_eq!(router.route("es"), "canary-1b");
        assert_eq!(router.route("ja"), "canary-qwen-2b"); // fallback
        assert_eq!(router.route("zh"), "canary-qwen-2b"); // fallback
    }

    #[test]
    fn test_language_router_has_route() {
        let router = LanguageRouter::default_routes();
        assert!(router.has_route("en"));
        assert!(router.has_route("de"));
        assert!(!router.has_route("ja"));
    }

    #[test]
    fn test_simple_detector_needs_min_samples() {
        let mut detector = SimpleLanguageDetector::new();
        // Too few samples
        let result = detector.detect(&[0.0; 16000]); // 1 second
        assert!(result.is_none());
    }

    #[test]
    fn test_simple_detector_returns_result() {
        let mut detector = SimpleLanguageDetector::new();
        // Enough samples (2 seconds)
        let result = detector.detect(&[0.0; 32000]);
        assert!(result.is_some());
        let detection = result.unwrap();
        assert_eq!(detection.language, "de"); // placeholder default
    }

    #[test]
    fn test_simple_detector_caches_result() {
        let mut detector = SimpleLanguageDetector::new();
        detector.detect(&[0.0; 32000]);
        // Second call returns cached result without needing more audio
        let result = detector.detect(&[]);
        assert!(result.is_some());
    }

    #[test]
    fn test_silero_detector_missing_model() {
        let detector = SileroLangDetector::new("/nonexistent/path/model.onnx");
        assert!(detector.is_none());
    }
}
