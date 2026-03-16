//! Text formatting backends for post-processing ASR output
//!
//! Provides a trait-based abstraction for cleaning up raw transcription text:
//! removing filler words, collapsing self-corrections, applying tone-specific rules.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Formatting tone hint — controls how aggressively to clean text
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FormattingTone {
    #[default]
    Casual,
    Email,
    Technical,
    Formal,
    Subtitle,
}

impl FormattingTone {
    pub fn from_str(s: &str) -> Self {
        match s {
            "email" => Self::Email,
            "technical" => Self::Technical,
            "formal" => Self::Formal,
            "subtitle" => Self::Subtitle,
            _ => Self::Casual,
        }
    }
}

/// Context passed to the formatter for each segment
pub struct FormattingContext {
    pub tone: FormattingTone,
    pub language: String,
    pub vocabulary: Vec<String>,
    pub recent_text: Vec<String>,
}

impl Default for FormattingContext {
    fn default() -> Self {
        Self {
            tone: FormattingTone::default(),
            language: "de".to_string(),
            vocabulary: vec![],
            recent_text: vec![],
        }
    }
}

/// Trait for text formatting backends
pub trait TextFormatter: Send + Sync {
    /// Format a finalized text segment. Returns formatted text.
    fn format(&self, text: &str, context: &FormattingContext) -> String;

    /// Name of this formatter for logging
    fn name(&self) -> &'static str;
}

/// Rule-based formatter using regex patterns — zero latency cost
pub struct RuleBasedFormatter;

impl RuleBasedFormatter {
    pub fn new() -> Self {
        Self
    }
}

// German filler words (standalone interjections only)
static FILLER_DE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(ähm?|ähh?|hmm?|mhm|sozusagen|quasi|halt|irgendwie|na ja|naja)\b[,]?\s*").unwrap()
});

// English filler words (standalone interjections only)
static FILLER_EN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(umm?|uhh?|hmm?|like,?\s|you know,?\s|I mean,?\s|basically,?\s|sort of,?\s|kind of,?\s)\b").unwrap()
});

// Self-correction: "X, I mean Y" / "X, also ich meine Y" → "Y"
static SELF_CORRECTION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)[^.!?]+[,]?\s*(?:I mean|ich meine|also ich meine|no wait|wait no|actually|eigentlich)[,]?\s+(.+)").unwrap()
});

// Restart correction: "comp-- the company" → "the company" (word-prefix restart)
static RESTART_CORRECTION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(\w{2,})--\s+(?:the\s+|die\s+|der\s+|das\s+|ein\s+|eine\s+)?(\w+)").unwrap()
});

/// Remove adjacent duplicate words: "the the" → "the" (no backreference needed)
fn remove_adjacent_duplicates(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 2 {
        return text.to_string();
    }
    let mut result = Vec::with_capacity(words.len());
    result.push(words[0]);
    for i in 1..words.len() {
        if words[i].len() >= 2 && words[i].eq_ignore_ascii_case(words[i - 1]) {
            continue;
        }
        result.push(words[i]);
    }
    result.join(" ")
}

// Multiple spaces
static MULTI_SPACE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"  +").unwrap()
});

impl TextFormatter for RuleBasedFormatter {
    fn format(&self, text: &str, context: &FormattingContext) -> String {
        if text.trim().is_empty() {
            return String::new();
        }

        let mut result = text.to_string();

        // 1. Self-correction collapse (must run before filler removal to detect "I mean" etc.)
        result = SELF_CORRECTION.replace_all(&result, "$1").to_string();

        // 2. Restart correction ("comp-- the company" → "the company")
        result = RESTART_CORRECTION.replace_all(&result, "$2").to_string();

        // 3. Remove filler words (language-aware)
        match context.language.as_str() {
            "de" => {
                result = FILLER_DE.replace_all(&result, "").to_string();
                result = FILLER_EN.replace_all(&result, "").to_string();
            }
            "en" => {
                result = FILLER_EN.replace_all(&result, "").to_string();
            }
            _ => {
                // Apply both for unknown languages
                result = FILLER_DE.replace_all(&result, "").to_string();
                result = FILLER_EN.replace_all(&result, "").to_string();
            }
        }

        // 4. Adjacent duplicate removal ("the the" → "the")
        result = remove_adjacent_duplicates(&result);

        // 5. Vocabulary preservation (case correction)
        for term in &context.vocabulary {
            let lower = term.to_lowercase();
            // Simple case-insensitive replacement preserving the user's casing
            let re = regex::RegexBuilder::new(&regex::escape(&lower))
                .case_insensitive(true)
                .build();
            if let Ok(re) = re {
                result = re.replace_all(&result, term.as_str()).to_string();
            }
        }

        // 6. Tone-specific rules
        match context.tone {
            FormattingTone::Subtitle => {
                // Limit line length for subtitles (char count, not bytes)
                if result.chars().count() > 84 {
                    // Split roughly in half at a word boundary
                    // Find a char-safe byte offset near char position 42
                    let byte_offset_42: usize = result.char_indices()
                        .nth(42)
                        .map(|(i, _)| i)
                        .unwrap_or(result.len());
                    if let Some(mid) = result[..byte_offset_42].rfind(' ') {
                        result.insert(mid + 1, '\n');
                    }
                }
            }
            FormattingTone::Formal => {
                // Expand common contractions
                result = result.replace("don't", "do not");
                result = result.replace("can't", "cannot");
                result = result.replace("won't", "will not");
                result = result.replace("isn't", "is not");
                result = result.replace("aren't", "are not");
                result = result.replace("doesn't", "does not");
                result = result.replace("didn't", "did not");
                result = result.replace("wasn't", "was not");
                result = result.replace("weren't", "were not");
                result = result.replace("haven't", "have not");
                result = result.replace("hasn't", "has not");
                result = result.replace("wouldn't", "would not");
                result = result.replace("couldn't", "could not");
                result = result.replace("shouldn't", "should not");
            }
            _ => {}
        }

        // 7. Clean up whitespace
        result = MULTI_SPACE.replace_all(&result, " ").to_string();
        result = result.trim().to_string();

        // 8. Capitalize first letter
        if !result.is_empty() {
            let mut chars = result.chars();
            if let Some(first) = chars.next() {
                result = first.to_uppercase().to_string() + chars.as_str();
            }
        }

        result
    }

    fn name(&self) -> &'static str {
        "RuleBased"
    }
}

// ---------------------------------------------------------------------------
// LLM-based formatter (Qwen2.5-0.5B-Instruct ONNX)
// ---------------------------------------------------------------------------

use ndarray::{Array2, Array4};
use ort::session::Session;

/// Number of decoder layers in Qwen2.5-0.5B
const LLM_NUM_LAYERS: usize = 24;
/// Number of KV heads in Qwen2.5-0.5B (GQA: 2 KV heads)
const LLM_KV_HEADS: usize = 2;
/// Head dimension in Qwen2.5-0.5B
const LLM_HEAD_DIM: usize = 64;
/// Maximum tokens to generate for formatted output
const LLM_MAX_NEW_TOKENS: usize = 128;
/// EOS token ID for Qwen2.5 (<|im_end|>)
const LLM_EOS_TOKEN_ID: u32 = 151645;
/// Default timeout in ms for LLM generation (overridable via FORMATTER_TIMEOUT_MS)
const LLM_DEFAULT_TIMEOUT_MS: u64 = 5000;

/// LLM-based text formatter using Qwen2.5-0.5B-Instruct ONNX with KV cache.
///
/// Runs autoregressive generation with a formatting prompt.
/// Falls back to `RuleBasedFormatter` on any error or timeout.
pub struct LlmFormatter {
    model_path: std::path::PathBuf,
    session: std::sync::Mutex<Option<Session>>,
    tokenizer: once_cell::sync::OnceCell<tokenizers::Tokenizer>,
    fallback: RuleBasedFormatter,
}

impl LlmFormatter {
    /// Create a new LlmFormatter pointing at the given model directory.
    /// Model and tokenizer are loaded lazily on first `format()` call.
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
        let model_path = model_path.into();
        eprintln!("[LlmFormatter] Created with model path: {}", model_path.display());
        Self {
            model_path,
            session: std::sync::Mutex::new(None),
            tokenizer: once_cell::sync::OnceCell::new(),
            fallback: RuleBasedFormatter::new(),
        }
    }

    /// Check whether the model path exists on disk.
    pub fn is_available(&self) -> bool {
        self.model_path.exists()
    }

    /// Return the model path for this formatter.
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    /// Get the timeout in ms for LLM generation.
    /// Reads FORMATTER_TIMEOUT_MS env var, defaults to LLM_DEFAULT_TIMEOUT_MS.
    fn timeout_ms() -> u128 {
        std::env::var("FORMATTER_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(LLM_DEFAULT_TIMEOUT_MS as u128)
    }

    /// Lazily load and return a locked ONNX session
    fn load_session_if_needed(&self) -> Option<std::sync::MutexGuard<'_, Option<Session>>> {
        let mut guard = self.session.lock().ok()?;
        if guard.is_none() {
            // Try Q4F16 first, then quantized INT8, then Q4
            let candidates = ["onnx/model_q4f16.onnx", "onnx/model_quantized.onnx", "onnx/model_q4.onnx"];
            let model_file = candidates.iter()
                .map(|f| self.model_path.join(f))
                .find(|p| p.exists());

            let model_file = match model_file {
                Some(f) => f,
                None => {
                    eprintln!("[LlmFormatter] No ONNX model found in {:?}", self.model_path);
                    return None;
                }
            };

            eprintln!("[LlmFormatter] Loading ONNX model from {:?}", model_file);
            let builder = match Session::builder() {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[LlmFormatter] Failed to create session builder: {}", e);
                    return None;
                }
            };

            // Apply execution config with ModelRole::Formatter
            let exec_config = crate::execution::ModelConfig::from_env()
                .with_role(crate::execution::ModelRole::Formatter);
            let mut builder = match exec_config.apply_to_session_builder(builder) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[LlmFormatter] Failed to apply exec config: {}", e);
                    return None;
                }
            };

            let session = match builder.commit_from_file(&model_file) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("[LlmFormatter] Failed to load model: {}", e);
                    return None;
                }
            };

            eprintln!("[LlmFormatter] Model loaded successfully ({} inputs, {} outputs)",
                session.inputs().len(), session.outputs().len());
            *guard = Some(session);
        }
        Some(guard)
    }

    /// Get or lazily load the tokenizer
    fn get_tokenizer(&self) -> Option<&tokenizers::Tokenizer> {
        self.tokenizer.get_or_try_init(|| {
            let path = self.model_path.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&path).map_err(|e| {
                eprintln!("[LlmFormatter] Failed to load tokenizer: {}", e);
            })?;
            eprintln!("[LlmFormatter] Tokenizer loaded ({} tokens)", tokenizer.get_vocab_size(true));
            Ok::<_, ()>(tokenizer)
        }).ok()
    }

    /// Build the chat-template prompt for formatting
    fn build_prompt(&self, text: &str, context: &FormattingContext) -> String {
        let tone_str = match context.tone {
            FormattingTone::Casual => "casual conversational",
            FormattingTone::Email => "professional email",
            FormattingTone::Technical => "technical documentation",
            FormattingTone::Formal => "formal written",
            FormattingTone::Subtitle => "subtitle/caption",
        };

        let vocab_hint = if context.vocabulary.is_empty() {
            String::new()
        } else {
            format!(" Preserve these terms exactly: {}.", context.vocabulary.join(", "))
        };

        format!(
            "<|im_start|>system\nYou are a text formatter. Clean up speech-to-text output: fix punctuation, capitalization, remove filler words (um, uh, ähm), collapse self-corrections. Style: {}. Language: {}.{} Output ONLY the cleaned text, nothing else.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            tone_str, context.language, vocab_hint, text
        )
    }

    /// Run autoregressive generation with KV cache
    fn generate(&self, session: &mut Session, tokenizer: &tokenizers::Tokenizer, prompt_ids: &[u32]) -> Option<String> {
        let seq_len = prompt_ids.len();

        // Prefill: run full prompt through model
        let input_ids = Array2::from_shape_vec(
            (1, seq_len),
            prompt_ids.iter().map(|&id| id as i64).collect(),
        ).ok()?;

        let attention_mask = Array2::from_elem((1, seq_len), 1i64);

        let position_ids = Array2::from_shape_vec(
            (1, seq_len),
            (0..seq_len as i64).collect(),
        ).ok()?;

        // Empty KV cache for prefill
        let empty_kv = Array4::<f32>::zeros((1, LLM_KV_HEADS, 0, LLM_HEAD_DIM));

        let mut inputs = ort::inputs!(
            "input_ids" => ort::value::Value::from_array(input_ids).ok()?,
            "attention_mask" => ort::value::Value::from_array(attention_mask).ok()?,
            "position_ids" => ort::value::Value::from_array(position_ids).ok()?
        );

        for layer in 0..LLM_NUM_LAYERS {
            inputs.push((
                format!("past_key_values.{}.key", layer).into(),
                ort::value::Value::from_array(empty_kv.clone()).ok()?.into(),
            ));
            inputs.push((
                format!("past_key_values.{}.value", layer).into(),
                ort::value::Value::from_array(empty_kv.clone()).ok()?.into(),
            ));
        }

        // Prefill: extract first token and KV cache, then drop outputs
        let (first_token, mut past_keys, mut past_values) = {
            let outputs = session.run(inputs).ok()?;

            let (logits_shape, logits_data) = outputs["logits"]
                .try_extract_tensor::<f32>().ok()?;
            let vocab_size = logits_shape.as_ref()[2] as usize;
            let last_pos_offset = (seq_len - 1) * vocab_size;
            let last_logits = &logits_data[last_pos_offset..last_pos_offset + vocab_size];
            let token = argmax(last_logits);

            let mut keys: Vec<ndarray::Array4<f32>> = Vec::new();
            let mut values: Vec<ndarray::Array4<f32>> = Vec::new();
            for layer in 0..LLM_NUM_LAYERS {
                keys.push(extract_4d(&outputs, &format!("present.{}.key", layer))?);
                values.push(extract_4d(&outputs, &format!("present.{}.value", layer))?);
            }
            (token, keys, values)
        }; // outputs dropped here

        if first_token == LLM_EOS_TOKEN_ID {
            return None;
        }

        let mut generated = vec![first_token];

        // Autoregressive generation with KV cache
        for _step in 1..LLM_MAX_NEW_TOKENS {
            let last_token = *generated.last().unwrap();
            let total_seq = seq_len + generated.len();

            let input_ids = Array2::from_elem((1, 1), last_token as i64);
            let attention_mask = Array2::from_elem((1, total_seq), 1i64);
            let position_ids = Array2::from_elem((1, 1), (total_seq - 1) as i64);

            let mut inputs = ort::inputs!(
                "input_ids" => ort::value::Value::from_array(input_ids).ok()?,
                "attention_mask" => ort::value::Value::from_array(attention_mask).ok()?,
                "position_ids" => ort::value::Value::from_array(position_ids).ok()?
            );

            for layer in 0..LLM_NUM_LAYERS {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    ort::value::Value::from_array(past_keys[layer].clone()).ok()?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    ort::value::Value::from_array(past_values[layer].clone()).ok()?.into(),
                ));
            }

            // Extract token and KV cache, drop outputs before next iteration
            let (next_token, new_keys, new_values) = {
                let outputs = session.run(inputs).ok()?;

                let (_shape, logits_data) = outputs["logits"]
                    .try_extract_tensor::<f32>().ok()?;
                let token = argmax(&logits_data);

                let mut keys: Vec<ndarray::Array4<f32>> = Vec::new();
                let mut values: Vec<ndarray::Array4<f32>> = Vec::new();
                for layer in 0..LLM_NUM_LAYERS {
                    keys.push(extract_4d(&outputs, &format!("present.{}.key", layer))?);
                    values.push(extract_4d(&outputs, &format!("present.{}.value", layer))?);
                }
                (token, keys, values)
            }; // outputs dropped here

            past_keys = new_keys;
            past_values = new_values;

            if next_token == LLM_EOS_TOKEN_ID {
                break;
            }
            generated.push(next_token);
        }

        // Decode generated tokens (skip special tokens)
        let filtered: Vec<u32> = generated.iter()
            .filter(|&&id| id != LLM_EOS_TOKEN_ID && id != 151643) // filter EOS and PAD
            .copied()
            .collect();
        tokenizer.decode(&filtered, true).ok()
    }
}

/// Argmax over a float slice
fn argmax(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

/// Extract a 4D tensor from session outputs
fn extract_4d(outputs: &ort::session::SessionOutputs, name: &str) -> Option<ndarray::Array4<f32>> {
    let (shape, data) = outputs[name].try_extract_tensor::<f32>().ok()?;
    let dims = shape.as_ref();
    if dims.len() != 4 { return None; }
    ndarray::Array4::from_shape_vec(
        (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize),
        data.to_vec(),
    ).ok()
}

impl TextFormatter for LlmFormatter {
    fn format(&self, text: &str, context: &FormattingContext) -> String {
        if text.trim().is_empty() {
            return String::new();
        }

        // Try LLM formatting
        let mut session_guard = match self.load_session_if_needed() {
            Some(g) => g,
            None => return self.fallback.format(text, context),
        };
        let session = match session_guard.as_mut() {
            Some(s) => s,
            None => return self.fallback.format(text, context),
        };
        let tokenizer = match self.get_tokenizer() {
            Some(t) => t,
            None => return self.fallback.format(text, context),
        };

        let start = std::time::Instant::now();

        let prompt = self.build_prompt(text, context);
        let prompt_ids = match tokenizer.encode(prompt.as_str(), false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                eprintln!("[LlmFormatter] Tokenization failed: {}, using fallback", e);
                return self.fallback.format(text, context);
            }
        };

        let result = self.generate(session, tokenizer, &prompt_ids);
        let elapsed = start.elapsed();

        match result {
            Some(formatted) if !formatted.trim().is_empty()
                && formatted.len() <= text.len() * 3
                && elapsed.as_millis() < Self::timeout_ms() =>
            {
                eprintln!("[LlmFormatter] Generated in {:?}: {:?} → {:?}",
                    elapsed, text.chars().take(40).collect::<String>(), formatted.chars().take(40).collect::<String>());
                formatted.trim().to_string()
            }
            Some(formatted) => {
                if elapsed.as_millis() >= Self::timeout_ms() {
                    eprintln!("[LlmFormatter] Timeout ({:?} >= {}ms), using fallback", elapsed, Self::timeout_ms());
                } else {
                    eprintln!("[LlmFormatter] Output rejected (empty or too long: {} chars), using fallback", formatted.len());
                }
                self.fallback.format(text, context)
            }
            None => {
                eprintln!("[LlmFormatter] Generation failed, using fallback");
                self.fallback.format(text, context)
            }
        }
    }

    fn name(&self) -> &'static str {
        "LlmFormatter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(text: &str) -> String {
        let formatter = RuleBasedFormatter::new();
        let ctx = FormattingContext::default();
        formatter.format(text, &ctx)
    }

    fn fmt_en(text: &str) -> String {
        let formatter = RuleBasedFormatter::new();
        let ctx = FormattingContext {
            language: "en".to_string(),
            ..Default::default()
        };
        formatter.format(text, &ctx)
    }

    fn fmt_with_vocab(text: &str, vocab: Vec<String>) -> String {
        let formatter = RuleBasedFormatter::new();
        let ctx = FormattingContext {
            vocabulary: vocab,
            ..Default::default()
        };
        formatter.format(text, &ctx)
    }

    #[test]
    fn test_empty_passthrough() {
        assert_eq!(fmt(""), "");
        assert_eq!(fmt("  "), "");
    }

    #[test]
    fn test_no_changes() {
        assert_eq!(fmt("Dies ist ein normaler Satz."), "Dies ist ein normaler Satz.");
    }

    #[test]
    fn test_german_fillers() {
        assert_eq!(fmt("Ähm das ist gut"), "Das ist gut");
        assert_eq!(fmt("Das ist sozusagen eine Lösung"), "Das ist eine Lösung");
        assert_eq!(fmt("Na ja das ist halt so"), "Das ist so");
    }

    #[test]
    fn test_english_fillers() {
        assert_eq!(fmt_en("Um the thing is"), "The thing is");
        assert_eq!(fmt_en("It is like really good"), "It is really good");
    }

    #[test]
    fn test_self_correction() {
        assert_eq!(fmt("Die alte Lösung, I mean die neue Lösung"), "Die neue Lösung");
    }

    #[test]
    fn test_adjacent_duplicates() {
        assert_eq!(fmt("Die die Firma ist gut"), "Die Firma ist gut");
    }

    #[test]
    fn test_vocabulary_preservation() {
        let result = fmt_with_vocab("das iphone ist gut", vec!["iPhone".to_string()]);
        assert_eq!(result, "Das iPhone ist gut");
    }

    #[test]
    fn test_formal_contractions() {
        let formatter = RuleBasedFormatter::new();
        let ctx = FormattingContext {
            tone: FormattingTone::Formal,
            language: "en".to_string(),
            ..Default::default()
        };
        assert_eq!(formatter.format("I don't think so", &ctx), "I do not think so");
    }

    #[test]
    fn test_capitalization() {
        assert_eq!(fmt("das ist gut"), "Das ist gut");
    }

    #[test]
    fn test_whitespace_cleanup() {
        assert_eq!(fmt("Too  many   spaces"), "Too many spaces");
    }

    #[test]
    fn test_formatting_tone_from_str() {
        assert_eq!(FormattingTone::from_str("casual"), FormattingTone::Casual);
        assert_eq!(FormattingTone::from_str("email"), FormattingTone::Email);
        assert_eq!(FormattingTone::from_str("technical"), FormattingTone::Technical);
        assert_eq!(FormattingTone::from_str("formal"), FormattingTone::Formal);
        assert_eq!(FormattingTone::from_str("subtitle"), FormattingTone::Subtitle);
        assert_eq!(FormattingTone::from_str("unknown"), FormattingTone::Casual);
    }

    // -----------------------------------------------------------------------
    // LlmFormatter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_llm_formatter_name() {
        let formatter = LlmFormatter::new("/tmp/nonexistent-model");
        assert_eq!(formatter.name(), "LlmFormatter");
    }

    #[test]
    fn test_llm_formatter_is_available_false_for_missing_path() {
        let formatter = LlmFormatter::new("/tmp/definitely-does-not-exist-parakeet-test");
        assert!(!formatter.is_available());
    }

    #[test]
    fn test_llm_formatter_delegates_to_rule_based() {
        // Since LlmFormatter is a placeholder, it should produce identical output to RuleBasedFormatter
        let llm = LlmFormatter::new("/tmp/nonexistent-model");
        let rule = RuleBasedFormatter::new();
        let ctx = FormattingContext::default();

        let input = "Ähm das ist gut";
        assert_eq!(llm.format(input, &ctx), rule.format(input, &ctx));
    }

    #[test]
    fn test_llm_formatter_model_path() {
        let formatter = LlmFormatter::new("/some/model/path");
        assert_eq!(formatter.model_path(), std::path::Path::new("/some/model/path"));
    }
}
