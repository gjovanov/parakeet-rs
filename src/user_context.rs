//! User context for personalized transcription formatting
//!
//! Carries per-session vocabulary, conversation history, and tone hints
//! that the text formatter uses to improve output quality.

use crate::text_formatter::FormattingTone;
use serde::{Deserialize, Serialize};

/// User context for personalized formatting
#[derive(Debug, Clone, Default)]
pub struct UserContext {
    /// Custom vocabulary terms with preferred casing (e.g., "iPhone", "NVIDIA")
    pub vocabulary: Vec<String>,
    /// Recent conversation history for LLM context
    pub history: Vec<String>,
    /// Application context hint (e.g., "medical dictation", "legal transcript")
    pub app_hint: Option<String>,
    /// Preferred formatting tone
    pub tone: FormattingTone,
}

impl UserContext {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a correction to learn vocabulary
    pub fn learn_correction(&mut self, _original: &str, corrected: &str) {
        // Extract words that look like proper nouns or brand names:
        // - Words starting with uppercase (e.g., "Berlin", "NVIDIA")
        // - Words with mixed case (e.g., "iPhone", "macOS")
        for word in corrected.split_whitespace() {
            if word.len() <= 1 || self.vocabulary.contains(&word.to_string()) {
                continue;
            }
            let has_upper = word.chars().any(|c| c.is_uppercase());
            let has_mixed_case = has_upper && word.chars().any(|c| c.is_lowercase());
            let starts_upper = word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
            // Mixed case like "iPhone" or starts with uppercase like "Berlin"
            if (has_mixed_case && !starts_upper) || starts_upper {
                self.vocabulary.push(word.to_string());
            }
        }
    }

    /// Push text to conversation history (keeps last 10)
    pub fn push_history(&mut self, text: &str) {
        self.history.push(text.to_string());
        if self.history.len() > 10 {
            self.history.remove(0);
        }
    }
}

/// Request struct for user context in session creation API
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserContextRequest {
    /// Custom vocabulary terms with preferred casing
    #[serde(default)]
    pub vocabulary: Option<Vec<String>>,
    /// Application context hint
    #[serde(default)]
    pub app_hint: Option<String>,
}

/// WebSocket correction message from client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionMessage {
    /// Original text the ASR produced
    pub original: String,
    /// User's corrected version
    pub corrected: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learn_correction_proper_nouns() {
        let mut ctx = UserContext::new();
        ctx.learn_correction("the iphone is great", "The iPhone is great");
        assert!(ctx.vocabulary.contains(&"The".to_string()));
        assert!(ctx.vocabulary.contains(&"iPhone".to_string()));
    }

    #[test]
    fn test_push_history_limit() {
        let mut ctx = UserContext::new();
        for i in 0..15 {
            ctx.push_history(&format!("sentence {}", i));
        }
        assert_eq!(ctx.history.len(), 10);
        assert_eq!(ctx.history[0], "sentence 5");
    }

    #[test]
    fn test_user_context_request_deserialization() {
        let json = r#"{"vocabulary": ["iPhone", "NVIDIA"], "app_hint": "tech blog"}"#;
        let req: UserContextRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.vocabulary.unwrap(), vec!["iPhone", "NVIDIA"]);
        assert_eq!(req.app_hint.unwrap(), "tech blog");
    }

    #[test]
    fn test_user_context_request_empty() {
        let json = r#"{}"#;
        let req: UserContextRequest = serde_json::from_str(json).unwrap();
        assert!(req.vocabulary.is_none());
        assert!(req.app_hint.is_none());
    }
}
