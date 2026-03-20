//! Model Registry for managing available transcription models
//!
//! The ModelRegistry provides a centralized way to:
//! - Register available models with their configurations
//! - List available models for the frontend
//! - Create transcriber instances on demand

use crate::execution::ModelConfig as ExecutionConfig;
use crate::streaming_transcriber::ModelInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Model type identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelType {
    /// Parakeet TDT 0.6B model
    ParakeetTdt,
    /// Canary 1B model
    Canary1B,
}

impl ModelType {
    /// Get the string identifier for this model type
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::ParakeetTdt => "parakeet-tdt",
            ModelType::Canary1B => "canary-1b",
        }
    }

    /// Get the display name for this model type
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelType::ParakeetTdt => "Parakeet TDT 0.6B",
            ModelType::Canary1B => "Canary 1B",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "parakeet-tdt" => Some(ModelType::ParakeetTdt),
            "canary-1b" => Some(ModelType::Canary1B),
            _ => None,
        }
    }

    /// Get supported languages for this model type
    pub fn languages(&self) -> Vec<&'static str> {
        match self {
            ModelType::ParakeetTdt => vec!["en"],
            ModelType::Canary1B => vec!["en", "de", "fr", "es"],
        }
    }
}

/// Configuration for a registered model
#[derive(Debug, Clone)]
pub struct RegisteredModel {
    /// Model type
    pub model_type: ModelType,
    /// Path to the model directory or file
    pub model_path: PathBuf,
    /// Path to diarization model (optional, for models that support it)
    pub diarization_path: Option<PathBuf>,
    /// Execution configuration (GPU/CPU, threads)
    pub exec_config: ExecutionConfig,
    /// Whether the model files exist and are valid
    pub is_available: bool,
    /// Description of the model
    pub description: String,
    /// Supported languages
    pub languages: Vec<String>,
}

impl RegisteredModel {
    /// Convert to ModelInfo for API responses
    pub fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.model_type.as_str().to_string(),
            display_name: self.model_type.display_name().to_string(),
            description: self.description.clone(),
            supports_diarization: self.diarization_path.is_some(),
            languages: self.languages.clone(),
            is_loaded: self.is_available,
        }
    }
}

/// Registry for managing available transcription models
pub struct ModelRegistry {
    /// Registered models by ID
    models: HashMap<String, RegisteredModel>,
    /// Default execution config
    default_exec_config: ExecutionConfig,
}

impl ModelRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            default_exec_config: ExecutionConfig::from_env(),
        }
    }

    /// Create registry from environment variables
    ///
    /// Reads model paths from environment:
    /// - TDT_MODEL_PATH: Path to Parakeet TDT model
    /// - CANARY_MODEL_PATH: Path to Canary 1B model
    /// - DIAR_MODEL_PATH: Path to diarization model
    pub fn from_env() -> Self {
        let mut registry = Self::new();

        // Get paths from environment
        let tdt_path = std::env::var("TDT_MODEL_PATH")
            .unwrap_or_else(|_| "./tdt".to_string());
        let diar_path = std::env::var("DIAR_MODEL_PATH")
            .ok()
            .map(PathBuf::from);
        let canary_path = std::env::var("CANARY_MODEL_PATH")
            .ok()
            .map(PathBuf::from);

        // Register Parakeet TDT if available
        let tdt_available = std::path::Path::new(&tdt_path).exists();
        registry.register(RegisteredModel {
            model_type: ModelType::ParakeetTdt,
            model_path: PathBuf::from(&tdt_path),
            diarization_path: diar_path.clone(),
            exec_config: registry.default_exec_config.clone(),
            is_available: tdt_available,
            description: "NVIDIA's Parakeet TDT model for high-quality speech recognition with word-level timestamps".to_string(),
            languages: vec!["en".to_string()],
        });

        // Register Canary 1B if path is configured
        if let Some(path) = canary_path {
            let canary_available = path.exists();
            registry.register(RegisteredModel {
                model_type: ModelType::Canary1B,
                model_path: path,
                diarization_path: diar_path.clone(), // Use same diarization model as TDT
                exec_config: registry.default_exec_config.clone(),
                is_available: canary_available,
                description: "NVIDIA's Canary 1B encoder-decoder model for multilingual ASR and translation".to_string(),
                languages: vec!["en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string()],
            });
        }

        eprintln!("[ModelRegistry] Registered {} models", registry.models.len());
        for (id, model) in &registry.models {
            eprintln!(
                "[ModelRegistry]   {} ({}): {}",
                id,
                if model.is_available { "available" } else { "not found" },
                model.model_path.display()
            );
        }

        registry
    }

    /// Register a model
    pub fn register(&mut self, model: RegisteredModel) {
        let id = model.model_type.as_str().to_string();
        self.models.insert(id, model);
    }

    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.models
            .values()
            .map(|m| m.to_model_info())
            .collect()
    }


    /// Get model configuration by ID
    pub fn get_model(&self, model_id: &str) -> Option<&RegisteredModel> {
        self.models.get(model_id)
    }

    /// Check if a model is available
    pub fn is_available(&self, model_id: &str) -> bool {
        self.models
            .get(model_id)
            .map(|m| m.is_available)
            .unwrap_or(false)
    }

}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Thread-safe reference to ModelRegistry
pub type SharedModelRegistry = Arc<ModelRegistry>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_type_roundtrip() {
        assert_eq!(ModelType::from_str("parakeet-tdt"), Some(ModelType::ParakeetTdt));
        assert_eq!(ModelType::from_str("canary-1b"), Some(ModelType::Canary1B));
        assert_eq!(ModelType::from_str("unknown"), None);

        assert_eq!(ModelType::ParakeetTdt.as_str(), "parakeet-tdt");
        assert_eq!(ModelType::Canary1B.as_str(), "canary-1b");
    }

    #[test]
    fn test_registry_creation() {
        let registry = ModelRegistry::new();
        assert!(registry.models.is_empty());
    }
}
