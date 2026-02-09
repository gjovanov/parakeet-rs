//! Canary 180M Flash ONNX model implementation with KV cache support
//!
//! This is NVIDIA's smaller, faster variant of Canary optimized for real-time inference.
//! Key differences from Canary 1B:
//! - 182M parameters (vs 1B)
//! - 17 encoder layers, 4 decoder layers
//! - Supports KV caching for O(n) decoding
//! - 4 languages (en, de, fr, es)
//!
//! Model source: https://huggingface.co/nvidia/canary-180m-flash

use crate::canary::CanaryTokenizer;
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::{Session, SessionOutputs};
use std::path::{Path, PathBuf};

// ============================================================================
// Constants
// ============================================================================

const SAMPLE_RATE: usize = 16000;
const N_MELS: usize = 128;
const N_FFT: usize = 512;
const HOP_LENGTH: usize = 160;
const WIN_LENGTH: usize = 400;
const PREEMPHASIS: f32 = 0.97;

// Special token IDs (may differ from Canary 1B - verify with vocab.txt)
const ENDOFTEXT_ID: i64 = 3;
const STARTOFTRANSCRIPT_ID: i64 = 4;
const PNC_ID: i64 = 5;
const STARTOFCONTEXT_ID: i64 = 7;
const NOITN_ID: i64 = 9;
const NOTIMESTAMP_ID: i64 = 11;
const NODIARIZE_ID: i64 = 13;
const EMO_UNDEFINED_ID: i64 = 16;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Canary Flash model
#[derive(Debug, Clone)]
pub struct CanaryFlashConfig {
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Maximum sequence length for decoder
    pub max_sequence_length: usize,
    /// Target language code (e.g., "en" for English)
    pub language: String,
    /// Number of decoder layers (for KV cache dimensions)
    pub num_decoder_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden dimension per head
    pub head_dim: usize,
}

impl Default for CanaryFlashConfig {
    fn default() -> Self {
        Self {
            n_mels: N_MELS,
            sample_rate: SAMPLE_RATE,
            max_sequence_length: 512,
            language: "en".to_string(),
            num_decoder_layers: 3,  // Canary 180M Flash has 3 decoder layers
            num_heads: 8,
            head_dim: 128,          // 8 heads * 128 = 1024 hidden dim
        }
    }
}

// ============================================================================
// KV Cache
// ============================================================================

/// KV Cache for incremental decoder inference
///
/// This enables O(n) decoding instead of O(n²) by caching key-value pairs
/// from previous decoder steps.
pub struct DecoderKVCache {
    /// Self-attention key cache: [num_layers, batch, heads, seq_len, head_dim]
    pub cache: Option<Array4<f32>>,
    /// Current sequence position (number of tokens processed)
    pub position: usize,
    /// Configuration
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl DecoderKVCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            cache: None,
            position: 0,
            num_layers,
            num_heads,
            head_dim,
        }
    }

    /// Reset cache for new sequence
    pub fn reset(&mut self) {
        self.cache = None;
        self.position = 0;
    }

    /// Get cache tensor for decoder input (empty on first call)
    pub fn get_past_key_values(&self, batch_size: usize) -> Array4<f32> {
        self.cache.clone().unwrap_or_else(|| {
            // Empty cache: [num_layers * 2, batch, heads, 0, head_dim]
            // Factor of 2 for key and value
            Array4::zeros((self.num_layers * 2, batch_size, self.num_heads, 0))
        })
    }

    /// Update cache with present key-values from decoder output
    pub fn update(&mut self, present_key_values: Array4<f32>) {
        self.cache = Some(present_key_values);
        self.position += 1;
    }
}

// ============================================================================
// Canary Flash Model
// ============================================================================

/// Canary 180M Flash encoder-decoder model with KV cache support
pub struct CanaryFlashModel {
    encoder: Session,
    decoder: Session,
    tokenizer: CanaryTokenizer,
    config: CanaryFlashConfig,
    /// Whether the decoder has KV cache outputs
    has_kv_cache: bool,
}

impl CanaryFlashModel {
    /// Load Canary Flash model from directory
    ///
    /// Expected files:
    /// - encoder-model.onnx (or encoder.onnx)
    /// - decoder-model.onnx (or decoder.onnx)
    /// - vocab.txt
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<CanaryFlashConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec_config = exec_config.unwrap_or_else(ExecutionConfig::from_env);
        let config = config.unwrap_or_default();

        // Find model files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_path = Self::find_decoder(model_dir)?;
        let vocab_path = model_dir.join("vocab.txt");

        eprintln!("[CanaryFlash] Loading encoder from {:?}", encoder_path);
        eprintln!("[CanaryFlash] Loading decoder from {:?}", decoder_path);

        // Load tokenizer (reuse from Canary 1B)
        let tokenizer = CanaryTokenizer::from_file(&vocab_path)?;

        // Load encoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder = builder.commit_from_file(&decoder_path)?;

        // Check if decoder has KV cache support
        let has_kv_cache = decoder.outputs.iter().any(|o| {
            o.name.contains("present") || o.name.contains("cache") || o.name.contains("past")
        });

        eprintln!("[CanaryFlash] Models loaded successfully");
        eprintln!("[CanaryFlash] Encoder inputs: {:?}",
            encoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        eprintln!("[CanaryFlash] Decoder inputs: {:?}",
            decoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        eprintln!("[CanaryFlash] Decoder outputs: {:?}",
            decoder.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());
        eprintln!("[CanaryFlash] KV cache support: {}", has_kv_cache);

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            config,
            has_kv_cache,
        })
    }

    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "encoder-model.int8.onnx",
            "encoder-model.onnx",
            "encoder.onnx",
            "model_encoder.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No encoder model found in {}",
            dir.display()
        )))
    }

    fn find_decoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "decoder-model.int8.onnx",
            "decoder-model.onnx",
            "decoder.onnx",
            "model_decoder.onnx",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No decoder model found in {}",
            dir.display()
        )))
    }

    /// Extract mel spectrogram features from audio samples
    fn extract_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        use crate::audio::{apply_preemphasis, stft};

        let audio = apply_preemphasis(audio, PREEMPHASIS);
        let spectrogram = stft(&audio, N_FFT, HOP_LENGTH, WIN_LENGTH);
        let mel_filterbank = self.create_mel_filterbank();
        let mel_spectrogram = mel_filterbank.dot(&spectrogram);
        let mel_spectrogram = mel_spectrogram.mapv(|x| (x.max(1e-10)).ln());
        let mel_spectrogram = mel_spectrogram.t().to_owned();
        let mel_spectrogram = self.normalize_features(mel_spectrogram)?;

        let time_steps = mel_spectrogram.shape()[0];
        let n_mels = mel_spectrogram.shape()[1];
        let mel_3d = mel_spectrogram
            .into_shape((1, time_steps, n_mels))
            .map_err(|e| Error::Audio(format!("Failed to reshape features: {}", e)))?;

        Ok(mel_3d)
    }

    fn create_mel_filterbank(&self) -> Array2<f32> {
        let freq_bins = N_FFT / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((self.config.n_mels, freq_bins));

        let min_mel = Self::hz_to_mel(0.0);
        let max_mel = Self::hz_to_mel(self.config.sample_rate as f32 / 2.0);

        let mel_points: Vec<f32> = (0..=self.config.n_mels + 1)
            .map(|i| {
                Self::mel_to_hz(
                    min_mel + (max_mel - min_mel) * i as f32 / (self.config.n_mels + 1) as f32,
                )
            })
            .collect();

        let freq_bin_width = self.config.sample_rate as f32 / N_FFT as f32;

        for mel_idx in 0..self.config.n_mels {
            let left = mel_points[mel_idx];
            let center = mel_points[mel_idx + 1];
            let right = mel_points[mel_idx + 2];

            for freq_idx in 0..freq_bins {
                let freq = freq_idx as f32 * freq_bin_width;

                if freq >= left && freq <= center {
                    filterbank[[mel_idx, freq_idx]] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    filterbank[[mel_idx, freq_idx]] = (right - freq) / (right - center);
                }
            }
        }

        filterbank
    }

    fn hz_to_mel(freq: f32) -> f32 {
        2595.0 * (1.0 + freq / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn normalize_features(&self, mut features: Array2<f32>) -> Result<Array2<f32>> {
        let num_frames = features.shape()[0];
        let num_features = features.shape()[1];

        for feat_idx in 0..num_features {
            let mut column = features.column_mut(feat_idx);
            let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
            let variance: f32 =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_frames as f32;
            let std = variance.sqrt().max(1e-10);

            for val in column.iter_mut() {
                *val = (*val - mean) / std;
            }
        }

        Ok(features)
    }

    /// Run encoder on audio features
    fn run_encoder(&mut self, features: &Array3<f32>) -> Result<(Array3<f32>, Array2<i64>)> {
        let time_steps = features.shape()[1];
        let batch_size = 1;
        let n_mels = features.shape()[2];

        let features_transposed = features
            .clone()
            .into_shape((batch_size, time_steps, n_mels))
            .map_err(|e| Error::Model(format!("Failed to reshape: {}", e)))?;

        let mut transposed = Array3::<f32>::zeros((batch_size, n_mels, time_steps));
        for t in 0..time_steps {
            for m in 0..n_mels {
                transposed[[0, m, t]] = features_transposed[[0, t, m]];
            }
        }

        let length = Array1::from_vec(vec![time_steps as i64]);

        let input_value = ort::value::Value::from_array(transposed)?;
        let length_value = ort::value::Value::from_array(length)?;

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => input_value,
            "length" => length_value
        ))?;

        let embeddings = &outputs["encoder_embeddings"];
        let mask = &outputs["encoder_mask"];

        let (emb_shape, emb_data) = embeddings
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder embeddings: {}", e)))?;

        let (mask_shape, mask_data) = mask
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder mask: {}", e)))?;

        let emb_dims = emb_shape.as_ref();
        let mask_dims = mask_shape.as_ref();

        let encoder_out = Array3::from_shape_vec(
            (emb_dims[0] as usize, emb_dims[1] as usize, emb_dims[2] as usize),
            emb_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create encoder array: {}", e)))?;

        let encoder_mask = Array2::from_shape_vec(
            (mask_dims[0] as usize, mask_dims[1] as usize),
            mask_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create mask array: {}", e)))?;

        Ok((encoder_out, encoder_mask))
    }

    /// Run greedy decoding - automatically selects cached or non-cached path
    fn greedy_decode(
        &mut self,
        encoder_embeddings: &Array3<f32>,
        encoder_mask: &Array2<i64>,
    ) -> Result<Vec<i64>> {
        if self.has_kv_cache {
            self.greedy_decode_cached(encoder_embeddings, encoder_mask)
        } else {
            self.greedy_decode_full(encoder_embeddings, encoder_mask)
        }
    }

    /// O(n) decoding with KV cache
    fn greedy_decode_cached(
        &mut self,
        encoder_embeddings: &Array3<f32>,
        encoder_mask: &Array2<i64>,
    ) -> Result<Vec<i64>> {
        let mut cache = DecoderKVCache::new(
            self.config.num_decoder_layers,
            self.config.num_heads,
            self.config.head_dim,
        );

        let lang_id = self.tokenizer.get_language_id(&self.config.language);
        let initial_prompt: Vec<i64> = vec![
            STARTOFCONTEXT_ID,
            STARTOFTRANSCRIPT_ID,
            EMO_UNDEFINED_ID,
            lang_id,
            lang_id,
            PNC_ID,
            NOITN_ID,
            NOTIMESTAMP_ID,
            NODIARIZE_ID,
        ];

        let mut tokens = initial_prompt.clone();

        // First step: process all prompt tokens
        let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.clone())
            .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?;

        let past_kv = cache.get_past_key_values(1);

        // Try to find the correct input names for this model
        let decoder_inputs = self.decoder.inputs.iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>();

        let has_past_key_values = decoder_inputs.iter().any(|n| n.contains("past"));

        // First step: process all prompt tokens
        let (first_token, present_kv_name) = {
            let outputs = if has_past_key_values {
                self.decoder.run(ort::inputs!(
                    "input_ids" => ort::value::Value::from_array(input_ids)?,
                    "encoder_embeddings" => ort::value::Value::from_array(encoder_embeddings.clone())?,
                    "encoder_mask" => ort::value::Value::from_array(encoder_mask.clone())?,
                    "past_key_values" => ort::value::Value::from_array(past_kv)?
                ))?
            } else {
                // Fallback to decoder_mems format (like Canary 1B)
                let decoder_mems = Array4::<f32>::zeros((
                    self.config.num_decoder_layers * 2,
                    1,
                    0,
                    self.config.num_heads * self.config.head_dim
                ));
                self.decoder.run(ort::inputs!(
                    "input_ids" => ort::value::Value::from_array(input_ids)?,
                    "encoder_embeddings" => ort::value::Value::from_array(encoder_embeddings.clone())?,
                    "encoder_mask" => ort::value::Value::from_array(encoder_mask.clone())?,
                    "decoder_mems" => ort::value::Value::from_array(decoder_mems)?
                ))?
            };

            // Try to get present_key_values from output
            let present_kv_name: Option<String> = outputs.keys()
                .find(|k| k.contains("present") || k.contains("cache"))
                .map(|s| s.to_string());

            // Update cache if we found present_key_values
            if let Some(ref name) = present_kv_name {
                if let Ok((shape, data)) = outputs[name.as_str()].try_extract_tensor::<f32>() {
                    let dims: &[i64] = shape.as_ref();
                    if dims.len() == 4 {
                        if let Ok(present) = Array4::from_shape_vec(
                            (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize),
                            data.to_vec(),
                        ) {
                            cache.update(present);
                        }
                    }
                }
            }

            // Get first generated token
            let first_token = extract_next_token(&outputs)?;
            (first_token, present_kv_name)
        }; // outputs dropped here

        if first_token == ENDOFTEXT_ID {
            return Ok(tokens);
        }
        tokens.push(first_token);

        // Subsequent steps: only pass last token when we have cache
        for _step in 1..self.config.max_sequence_length {
            let use_cached = present_kv_name.is_some() && cache.position > 0;

            let input_ids = if use_cached {
                Array2::from_shape_vec((1, 1), vec![*tokens.last().unwrap()])
                    .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?
            } else {
                Array2::from_shape_vec((1, tokens.len()), tokens.clone())
                    .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?
            };

            let outputs = if has_past_key_values && use_cached {
                let past_kv = cache.get_past_key_values(1);
                self.decoder.run(ort::inputs!(
                    "input_ids" => ort::value::Value::from_array(input_ids)?,
                    "encoder_embeddings" => ort::value::Value::from_array(encoder_embeddings.clone())?,
                    "encoder_mask" => ort::value::Value::from_array(encoder_mask.clone())?,
                    "past_key_values" => ort::value::Value::from_array(past_kv)?
                ))?
            } else {
                let decoder_mems = Array4::<f32>::zeros((
                    self.config.num_decoder_layers * 2,
                    1,
                    0,
                    self.config.num_heads * self.config.head_dim
                ));
                self.decoder.run(ort::inputs!(
                    "input_ids" => ort::value::Value::from_array(input_ids)?,
                    "encoder_embeddings" => ort::value::Value::from_array(encoder_embeddings.clone())?,
                    "encoder_mask" => ort::value::Value::from_array(encoder_mask.clone())?,
                    "decoder_mems" => ort::value::Value::from_array(decoder_mems)?
                ))?
            };

            // Update cache if available
            if let Some(ref name) = present_kv_name {
                if let Ok((shape, data)) = outputs[name.as_str()].try_extract_tensor::<f32>() {
                    let dims: &[i64] = shape.as_ref();
                    if dims.len() == 4 {
                        if let Ok(present) = Array4::from_shape_vec(
                            (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize),
                            data.to_vec(),
                        ) {
                            cache.update(present);
                        }
                    }
                }
            }

            let next_token = extract_next_token(&outputs)?;
            if next_token == ENDOFTEXT_ID {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// O(n²) decoding without KV cache (fallback)
    fn greedy_decode_full(
        &mut self,
        encoder_embeddings: &Array3<f32>,
        encoder_mask: &Array2<i64>,
    ) -> Result<Vec<i64>> {
        let lang_id = self.tokenizer.get_language_id(&self.config.language);

        // Initialize with 9-token Canary prompt format:
        // <|startofcontext|><|startoftranscript|><|emo:undefined|><|lang|><|lang|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
        let initial_prompt: Vec<i64> = vec![
            STARTOFCONTEXT_ID,
            STARTOFTRANSCRIPT_ID,
            EMO_UNDEFINED_ID,
            lang_id,
            lang_id,
            PNC_ID,
            NOITN_ID,
            NOTIMESTAMP_ID,
            NODIARIZE_ID,
        ];

        let mut tokens = initial_prompt.clone();
        let decoder_mems = Array4::<f32>::zeros((
            self.config.num_decoder_layers * 2,
            1,
            0,
            self.config.num_heads * self.config.head_dim
        ));

        for _step in 0..self.config.max_sequence_length {
            let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.clone())
                .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?;

            let outputs = self.decoder.run(ort::inputs!(
                "input_ids" => ort::value::Value::from_array(input_ids)?,
                "encoder_embeddings" => ort::value::Value::from_array(encoder_embeddings.clone())?,
                "encoder_mask" => ort::value::Value::from_array(encoder_mask.clone())?,
                "decoder_mems" => ort::value::Value::from_array(decoder_mems.clone())?
            ))?;

            let next_token = extract_next_token(&outputs)?;
            if next_token == ENDOFTEXT_ID {
                break;
            }
            tokens.push(next_token);
        }

        Ok(tokens)
    }

    /// Transcribe audio samples
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let features = self.extract_features(samples)?;
        let (encoder_embeddings, encoder_mask) = self.run_encoder(&features)?;
        let token_ids = self.greedy_decode(&encoder_embeddings, &encoder_mask)?;
        let text = self.tokenizer.decode(&token_ids);

        Ok(text)
    }

    /// Set the target language
    pub fn set_language(&mut self, lang: &str) {
        self.config.language = lang.to_string();
    }

    /// Get the tokenizer reference
    pub fn tokenizer(&self) -> &CanaryTokenizer {
        &self.tokenizer
    }

    /// Check if model has KV cache support
    pub fn has_kv_cache(&self) -> bool {
        self.has_kv_cache
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract next token from decoder output logits
fn extract_next_token(outputs: &SessionOutputs<'_>) -> Result<i64> {
    let (logits_shape, logits_data) = outputs["logits"]
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract logits: {}", e)))?;

    let logits_dims: &[i64] = logits_shape.as_ref();
    let vocab_size = logits_dims[2] as usize;
    let seq_len = logits_dims[1] as usize;

    // Get logits from the last sequence position
    let last_pos_start = (seq_len - 1) * vocab_size;
    let last_logits = &logits_data[last_pos_start..last_pos_start + vocab_size];

    // Greedy decoding: select token with highest logit
    let next_token: i64 = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i64)
        .unwrap_or(ENDOFTEXT_ID);

    Ok(next_token)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CanaryFlashConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.language, "en");
        assert_eq!(config.num_decoder_layers, 3);
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = DecoderKVCache::new(4, 8, 64);
        assert_eq!(cache.position, 0);
        assert!(cache.cache.is_none());
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = DecoderKVCache::new(4, 8, 64);
        cache.position = 10;
        cache.reset();
        assert_eq!(cache.position, 0);
        assert!(cache.cache.is_none());
    }

    #[test]
    fn test_kv_cache_get_empty() {
        let cache = DecoderKVCache::new(4, 8, 64);
        let past = cache.get_past_key_values(1);
        assert_eq!(past.shape(), &[8, 1, 8, 0]); // num_layers*2, batch, heads, seq_len=0
    }
}
