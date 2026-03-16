//! Canary-Qwen 2.5B SALM (Speech-Augmented Language Model) implementation
//!
//! Combines a FastConformer encoder with a Qwen3-1.7B LLM decoder via a linear
//! projection layer. Achieves state-of-the-art English ASR (5.63% mean WER).
//!
//! ONNX pipeline:
//!   audio → encoder.onnx → projection.onnx → [prompt_embeds, audio_embeds] → decoder → tokens
//!
//! Model source: nvidia/canary-qwen-2.5b (ONNX exports)

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
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

/// Maximum audio duration in seconds (model training limit)
const MAX_AUDIO_SECS: f32 = 40.0;

/// Audio placeholder token ID used in SALM prompt format.
/// This token gets replaced with projected audio embeddings.
const AUDIO_PLACEHOLDER_TOKEN_ID: i64 = 151669;

/// Chat-template prompt token IDs for "Transcribe the following: <|audioplaceholder|>"
/// Format: <|im_start|>user\nTranscribe the following: <|audioplaceholder|><|im_end|>\n<|im_start|>assistant\n
const PROMPT_TOKEN_IDS: &[i64] = &[
    151644, // <|im_start|>
    872,    // user
    198,    // \n
    3167,   // Trans
    3114,   // cribe
    279,    // the
    2701,   // following
    25,     // :
    220,    // (space)
    151669, // <|audioplaceholder|>  -- will be replaced with audio embeddings
    151645, // <|im_end|>
    198,    // \n
    151644, // <|im_start|>
    77091,  // assistant
    198,    // \n
];

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for Canary-Qwen SALM model
#[derive(Debug, Clone)]
pub struct CanaryQwenConfig {
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Maximum audio duration in seconds
    pub max_audio_secs: f32,
    /// Maximum sequence length for Qwen decoder
    pub max_sequence_length: usize,
    /// Target language (English only for this model)
    pub language: String,
    /// Repetition penalty for greedy decoding
    pub repetition_penalty: f32,
    /// Number of decoder layers (Qwen3-1.7B)
    pub num_decoder_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA)
    pub num_kv_heads: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Hidden size of the LLM
    pub hidden_size: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Quantization preference: None=auto (Q4 on CPU, FP16 on GPU), Some(true)=force Q4, Some(false)=force FP16
    pub prefer_quantized: Option<bool>,
}

impl Default for CanaryQwenConfig {
    fn default() -> Self {
        Self {
            n_mels: N_MELS,
            sample_rate: SAMPLE_RATE,
            max_audio_secs: MAX_AUDIO_SECS,
            max_sequence_length: 2048,
            language: "en".to_string(),
            repetition_penalty: 1.2,
            // Qwen3-1.7B defaults
            num_decoder_layers: 28,
            num_attention_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 2048,
            vocab_size: 151936,
            prefer_quantized: None,
        }
    }
}

impl CanaryQwenConfig {
    /// Create config from environment variable for quantization preference
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(quant) = std::env::var("CANARY_QWEN_QUANTIZATION") {
            config.prefer_quantized = match quant.to_lowercase().as_str() {
                "q4" => Some(true),
                "fp16" => Some(false),
                _ => None, // "auto" or anything else
            };
        }
        config
    }
}

// ============================================================================
// Tokenizer (BPE via tokenizers crate)
// ============================================================================

/// BPE tokenizer wrapper for Qwen3 decoder
pub struct QwenTokenizer {
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: u32,
    pad_token_id: u32,
}

impl QwenTokenizer {
    /// Load tokenizer from tokenizer.json
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;

        // EOS = <|im_end|> (151645 per config.json)
        // PAD = <|endoftext|> (151643 per tokenizer_config.json)
        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(151645); // Qwen3 default eos

        let pad_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<pad>"))
            .unwrap_or(151643);

        eprintln!("[QwenTokenizer] Loaded tokenizer with {} tokens", tokenizer.get_vocab_size(true));
        eprintln!("[QwenTokenizer] EOS token ID: {}, PAD token ID: {}", eos_token_id, pad_token_id);

        Ok(Self {
            tokenizer,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("Failed to encode: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        // Filter out special tokens
        let filtered: Vec<u32> = ids.iter()
            .filter(|&&id| id != self.eos_token_id && id != self.pad_token_id)
            .copied()
            .collect();
        self.tokenizer
            .decode(&filtered, true)
            .map_err(|e| Error::Tokenizer(format!("Failed to decode: {}", e)))
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

// ============================================================================
// KV Cache for Qwen3 decoder
// ============================================================================

/// Per-layer KV cache for Qwen3 decoder
///
/// Stores past key-value tensors for each decoder layer.
/// Shape per layer: [batch, num_kv_heads, seq_len, head_dim]
pub struct QwenKVCache {
    /// Past key tensors per layer
    pub past_keys: Vec<ndarray::Array4<f32>>,
    /// Past value tensors per layer
    pub past_values: Vec<ndarray::Array4<f32>>,
    /// Current sequence position
    pub position: usize,
    /// Config
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl QwenKVCache {
    pub fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let past_keys = (0..num_layers)
            .map(|_| ndarray::Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)))
            .collect();
        let past_values = (0..num_layers)
            .map(|_| ndarray::Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)))
            .collect();

        Self {
            past_keys,
            past_values,
            position: 0,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Reset cache for a new sequence
    pub fn reset(&mut self) {
        for i in 0..self.num_layers {
            self.past_keys[i] = ndarray::Array4::<f32>::zeros((1, self.num_kv_heads, 0, self.head_dim));
            self.past_values[i] = ndarray::Array4::<f32>::zeros((1, self.num_kv_heads, 0, self.head_dim));
        }
        self.position = 0;
    }

    /// Update cache with present key-values from decoder output
    pub fn update(&mut self, present_keys: Vec<ndarray::Array4<f32>>, present_values: Vec<ndarray::Array4<f32>>) {
        self.past_keys = present_keys;
        self.past_values = present_values;
        self.position += 1;
    }
}

// ============================================================================
// Canary-Qwen SALM Model
// ============================================================================

/// Canary-Qwen 2.5B SALM model
///
/// Pipeline:
///   1. audio → mel spectrogram → encoder → encoder_embeddings
///   2. encoder_embeddings → projection → projected_embeddings (in LLM dim)
///   3. prompt_tokens → embed_tokens → prompt_embeddings
///   4. concat [prompt_embeddings, projected_embeddings] → inputs_embeds
///   5. decoder(inputs_embeds) → autoregressive greedy decoding → text
pub struct CanaryQwenModel {
    encoder: Session,
    projection: Session,
    decoder: Session,
    embed_tokens: Session,
    tokenizer: QwenTokenizer,
    config: CanaryQwenConfig,
}

impl CanaryQwenModel {
    /// Load Canary-Qwen model from directory
    ///
    /// Expected files:
    /// - encoder.onnx
    /// - projection.onnx
    /// - decoder_model_merged_q4.onnx / decoder_model_merged_fp16.onnx
    /// - embed_tokens.onnx
    /// - tokenizer.json
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<CanaryQwenConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec_config = exec_config.unwrap_or_else(ExecutionConfig::from_env);
        let config = config.unwrap_or_else(CanaryQwenConfig::from_env);

        // Find model files
        let encoder_path = Self::find_file(model_dir, &["encoder.onnx", "encoder-model.onnx"], "encoder")?;
        let projection_path = Self::find_file(model_dir, &["projection.onnx"], "projection")?;
        let decoder_path = Self::find_decoder(model_dir, &exec_config, &config)?;
        let embed_tokens_path = Self::find_file(model_dir, &["embed_tokens.onnx"], "embed_tokens")?;
        let tokenizer_path = model_dir.join("tokenizer.json");

        eprintln!("[CanaryQwen] Loading encoder from {:?}", encoder_path);
        eprintln!("[CanaryQwen] Loading projection from {:?}", projection_path);
        eprintln!("[CanaryQwen] Loading decoder from {:?}", decoder_path);
        eprintln!("[CanaryQwen] Loading embed_tokens from {:?}", embed_tokens_path);

        // Load tokenizer
        let tokenizer = QwenTokenizer::from_file(&tokenizer_path)?;

        // Load ONNX sessions
        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let projection = builder.commit_from_file(&projection_path)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let decoder = builder.commit_from_file(&decoder_path)?;

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let embed_tokens = builder.commit_from_file(&embed_tokens_path)?;

        eprintln!("[CanaryQwen] Models loaded successfully");
        eprintln!("[CanaryQwen] Encoder inputs: {:?}", encoder.inputs().iter().map(|i| i.name()).collect::<Vec<_>>());
        eprintln!("[CanaryQwen] Encoder outputs: {:?}", encoder.outputs().iter().map(|o| o.name()).collect::<Vec<_>>());
        eprintln!("[CanaryQwen] Projection inputs: {:?}", projection.inputs().iter().map(|i| i.name()).collect::<Vec<_>>());
        eprintln!("[CanaryQwen] Decoder inputs: {:?}", decoder.inputs().iter().map(|i| i.name()).collect::<Vec<_>>());
        eprintln!("[CanaryQwen] Decoder outputs: {:?}", decoder.outputs().iter().map(|o| o.name()).collect::<Vec<_>>());
        eprintln!("[CanaryQwen] EmbedTokens inputs: {:?}", embed_tokens.inputs().iter().map(|i| i.name()).collect::<Vec<_>>());

        Ok(Self {
            encoder,
            projection,
            decoder,
            embed_tokens,
            tokenizer,
            config,
        })
    }

    fn find_file(dir: &Path, candidates: &[&str], name: &str) -> Result<PathBuf> {
        for candidate in candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No {} model found in {}",
            name,
            dir.display()
        )))
    }

    /// Find the appropriate decoder model based on quantization preference
    fn find_decoder(dir: &Path, exec_config: &ExecutionConfig, config: &CanaryQwenConfig) -> Result<PathBuf> {
        let use_quantized = match config.prefer_quantized {
            Some(pref) => pref,
            None => {
                // Auto-select: Q4 for CPU, FP16 for GPU
                !exec_config.execution_provider.is_gpu()
            }
        };

        let (primary, fallback) = if use_quantized {
            (
                vec!["decoder_model_merged_q4.onnx", "decoder_model_merged_q4f16.onnx"],
                vec!["decoder_model_merged_fp16.onnx", "decoder_model_merged.onnx"],
            )
        } else {
            (
                vec!["decoder_model_merged_fp16.onnx", "decoder_model_merged.onnx"],
                vec!["decoder_model_merged_q4.onnx", "decoder_model_merged_q4f16.onnx"],
            )
        };

        for candidate in &primary {
            let path = dir.join(candidate);
            if path.exists() {
                eprintln!("[CanaryQwen] Selected decoder: {} (preferred: {})",
                    candidate, if use_quantized { "Q4" } else { "FP16" });
                return Ok(path);
            }
        }

        for candidate in &fallback {
            let path = dir.join(candidate);
            if path.exists() {
                eprintln!("[CanaryQwen] Selected decoder: {} (fallback)", candidate);
                return Ok(path);
            }
        }

        Err(Error::Config(format!(
            "No decoder model found in {}",
            dir.display()
        )))
    }

    /// Extract mel spectrogram features from audio samples (same pipeline as Canary)
    fn extract_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        use crate::audio::{apply_preemphasis, stft};

        let audio = apply_preemphasis(audio, PREEMPHASIS);
        let spectrogram = stft(&audio, N_FFT, HOP_LENGTH, WIN_LENGTH);
        let mel_filterbank = self.create_mel_filterbank();
        let mel_spectrogram = mel_filterbank.dot(&spectrogram);
        let mel_spectrogram = mel_spectrogram.mapv(|x| (x.max(1e-10)).ln());
        let mel_spectrogram = mel_spectrogram.t().as_standard_layout().to_owned();
        let mel_spectrogram = self.normalize_features(mel_spectrogram)?;

        let time_steps = mel_spectrogram.shape()[0];
        let n_mels = mel_spectrogram.shape()[1];
        let mel_3d = mel_spectrogram
            .into_shape_with_order((1, time_steps, n_mels))
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

    /// Run FastConformer encoder on audio features
    fn run_encoder(&mut self, features: &Array3<f32>) -> Result<(Array3<f32>, Array1<i64>)> {
        let time_steps = features.shape()[1];
        let batch_size = 1;
        let n_mels = features.shape()[2];

        // Transpose to (batch, mels, time) for NeMo
        let mut transposed = Array3::<f32>::zeros((batch_size, n_mels, time_steps));
        for t in 0..time_steps {
            for m in 0..n_mels {
                transposed[[0, m, t]] = features[[0, t, m]];
            }
        }

        let length = Array1::from_vec(vec![time_steps as i64]);

        let input_value = ort::value::Value::from_array(transposed)?;
        let length_value = ort::value::Value::from_array(length)?;

        let outputs = self.encoder.run(ort::inputs!(
            "audio_signal" => input_value,
            "length" => length_value
        ))?;

        let embeddings = &outputs["encoder_output"];
        let enc_length = &outputs["encoder_length"];

        let (emb_shape, emb_data) = embeddings
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {}", e)))?;

        let (len_shape, len_data) = enc_length
            .try_extract_tensor::<i64>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder length: {}", e)))?;

        let emb_dims = emb_shape.as_ref();
        let len_dims = len_shape.as_ref();

        let encoder_out = Array3::from_shape_vec(
            (emb_dims[0] as usize, emb_dims[1] as usize, emb_dims[2] as usize),
            emb_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create encoder array: {}", e)))?;

        let encoder_length = Array1::from_shape_vec(
            len_dims[0] as usize,
            len_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create length array: {}", e)))?;

        Ok((encoder_out, encoder_length))
    }

    /// Run linear projection: encoder_dim → LLM hidden_dim
    fn run_projection(&mut self, encoder_embeddings: &Array3<f32>) -> Result<Array3<f32>> {
        let input_value = ort::value::Value::from_array(encoder_embeddings.clone())?;

        // Read output name before run() to avoid overlapping borrows on self.projection
        let output_name = self.projection.outputs().first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "projected_output".to_string());

        let outputs = self.projection.run(ort::inputs!(
            "encoder_output" => input_value
        ))?;

        let (shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract projected embeddings: {}", e)))?;

        let dims: &[i64] = &**shape;
        Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape projected embeddings: {}", e)))
    }

    /// Embed token IDs via embed_tokens.onnx → embeddings
    fn embed_prompt(&mut self, token_ids: &[i64]) -> Result<Array3<f32>> {
        let input = Array2::from_shape_vec((1, token_ids.len()), token_ids.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?;

        let input_value = ort::value::Value::from_array(input)?;

        // Read output name before run() to avoid overlapping borrows on self.embed_tokens
        let output_name = self.embed_tokens.outputs().first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "inputs_embeds".to_string());

        let outputs = self.embed_tokens.run(ort::inputs!(
            "input_ids" => input_value
        ))?;

        let (shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract embeddings: {}", e)))?;

        let dims: &[i64] = &**shape;
        Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape embeddings: {}", e)))
    }

    /// Build the decoder input by replacing the audio placeholder in the prompt
    /// embeddings with projected audio embeddings.
    ///
    /// The SALM prompt format contains a single `<|audioplaceholder|>` token
    /// whose embedding gets replaced (expanded) with the N projected audio
    /// embeddings. This preserves the chat-template structure around the audio.
    fn build_decoder_input(
        prompt_embeds: &Array3<f32>,
        projected_embeds: &Array3<f32>,
        prompt_token_ids: &[i64],
    ) -> Result<Array3<f32>> {
        let batch = prompt_embeds.shape()[0];
        let prompt_len = prompt_embeds.shape()[1];
        let audio_len = projected_embeds.shape()[1];
        let hidden = prompt_embeds.shape()[2];

        // Find the placeholder position
        let placeholder_pos = prompt_token_ids
            .iter()
            .position(|&id| id == AUDIO_PLACEHOLDER_TOKEN_ID)
            .ok_or_else(|| Error::Model("Audio placeholder token not found in prompt".into()))?;

        // Total length = prompt tokens - 1 placeholder + audio tokens
        let total_len = prompt_len - 1 + audio_len;
        let mut combined = Array3::<f32>::zeros((batch, total_len, hidden));

        for b in 0..batch {
            // Copy prompt embeddings BEFORE the placeholder
            for t in 0..placeholder_pos {
                for h in 0..hidden {
                    combined[[b, t, h]] = prompt_embeds[[b, t, h]];
                }
            }
            // Insert audio embeddings at the placeholder position
            for t in 0..audio_len {
                for h in 0..hidden {
                    combined[[b, placeholder_pos + t, h]] = projected_embeds[[b, t, h]];
                }
            }
            // Copy prompt embeddings AFTER the placeholder
            let after_placeholder = placeholder_pos + 1;
            let dest_offset = placeholder_pos + audio_len;
            for t in after_placeholder..prompt_len {
                for h in 0..hidden {
                    combined[[b, dest_offset + (t - after_placeholder), h]] = prompt_embeds[[b, t, h]];
                }
            }
        }

        Ok(combined)
    }

    /// Autoregressive greedy decoding with KV cache
    fn greedy_decode_cached(
        &mut self,
        inputs_embeds: &Array3<f32>,
    ) -> Result<Vec<u32>> {
        let mut kv_cache = QwenKVCache::new(
            self.config.num_decoder_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
        );

        let seq_len = inputs_embeds.shape()[1];
        let mut generated_tokens: Vec<u32> = Vec::new();

        // Step 0: Process all input embeddings (prefill)
        let first_token = {
            let attention_mask = Array2::from_elem((1, seq_len), 1i64);
            let position_ids = Array2::from_shape_vec(
                (1, seq_len),
                (0..seq_len as i64).collect(),
            ).map_err(|e| Error::Model(format!("Failed to create position_ids: {}", e)))?;

            let mut decoder_inputs = ort::inputs!(
                "inputs_embeds" => ort::value::Value::from_array(inputs_embeds.clone())?,
                "attention_mask" => ort::value::Value::from_array(attention_mask)?,
                "position_ids" => ort::value::Value::from_array(position_ids)?
            );

            // Add empty KV cache inputs
            for layer in 0..self.config.num_decoder_layers {
                let key_name = format!("past_key_values.{}.key", layer);
                let value_name = format!("past_key_values.{}.value", layer);
                decoder_inputs.push((
                    key_name.into(),
                    ort::value::Value::from_array(kv_cache.past_keys[layer].clone())?.into(),
                ));
                decoder_inputs.push((
                    value_name.into(),
                    ort::value::Value::from_array(kv_cache.past_values[layer].clone())?.into(),
                ));
            }

            let outputs = self.decoder.run(decoder_inputs)?;

            // Extract KV cache from present outputs
            let mut present_keys = Vec::new();
            let mut present_values = Vec::new();
            for layer in 0..self.config.num_decoder_layers {
                let key_name = format!("present.{}.key", layer);
                let value_name = format!("present.{}.value", layer);

                present_keys.push(Self::extract_4d_tensor(&outputs, &key_name)?);
                present_values.push(Self::extract_4d_tensor(&outputs, &value_name)?);
            }
            kv_cache.update(present_keys, present_values);

            // Extract next token from logits
            Self::extract_next_token_from_outputs(&outputs, &generated_tokens, self.config.repetition_penalty)?
        };

        if first_token == self.tokenizer.eos_token_id() {
            return Ok(generated_tokens);
        }
        generated_tokens.push(first_token);

        // Steps 1..N: process one token at a time with KV cache
        for _step in 1..self.config.max_sequence_length {
            let last_token = *generated_tokens.last().unwrap();
            let total_seq = seq_len + generated_tokens.len();

            // Embed the new token
            let token_embed = self.embed_prompt(&[last_token as i64])?;

            let attention_mask = Array2::from_elem((1, total_seq), 1i64);
            let position_ids = Array2::from_shape_vec(
                (1, 1),
                vec![(total_seq - 1) as i64],
            ).map_err(|e| Error::Model(format!("Failed to create position_ids: {}", e)))?;

            let mut decoder_inputs = ort::inputs!(
                "inputs_embeds" => ort::value::Value::from_array(token_embed)?,
                "attention_mask" => ort::value::Value::from_array(attention_mask)?,
                "position_ids" => ort::value::Value::from_array(position_ids)?
            );

            // Add cached KV
            for layer in 0..self.config.num_decoder_layers {
                let key_name = format!("past_key_values.{}.key", layer);
                let value_name = format!("past_key_values.{}.value", layer);
                decoder_inputs.push((
                    key_name.into(),
                    ort::value::Value::from_array(kv_cache.past_keys[layer].clone())?.into(),
                ));
                decoder_inputs.push((
                    value_name.into(),
                    ort::value::Value::from_array(kv_cache.past_values[layer].clone())?.into(),
                ));
            }

            let outputs = self.decoder.run(decoder_inputs)?;

            // Update KV cache
            let mut present_keys = Vec::new();
            let mut present_values = Vec::new();
            for layer in 0..self.config.num_decoder_layers {
                let key_name = format!("present.{}.key", layer);
                let value_name = format!("present.{}.value", layer);
                present_keys.push(Self::extract_4d_tensor(&outputs, &key_name)?);
                present_values.push(Self::extract_4d_tensor(&outputs, &value_name)?);
            }
            kv_cache.update(present_keys, present_values);

            let next_token = Self::extract_next_token_from_outputs(
                &outputs, &generated_tokens, self.config.repetition_penalty,
            )?;

            if next_token == self.tokenizer.eos_token_id() {
                break;
            }
            generated_tokens.push(next_token);
        }

        Ok(generated_tokens)
    }

    /// Extract a 4D tensor from ONNX outputs by name
    fn extract_4d_tensor(
        outputs: &ort::session::SessionOutputs,
        name: &str,
    ) -> Result<ndarray::Array4<f32>> {
        let (shape, data) = outputs[name]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract {}: {}", name, e)))?;

        let dims = shape.as_ref();
        if dims.len() != 4 {
            return Err(Error::Model(format!(
                "Expected 4D tensor for {}, got {}D: {:?}",
                name, dims.len(), dims
            )));
        }

        ndarray::Array4::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to reshape {}: {}", name, e)))
    }

    /// Extract next token from decoder logits with repetition penalty
    fn extract_next_token_from_outputs(
        outputs: &ort::session::SessionOutputs,
        past_tokens: &[u32],
        repetition_penalty: f32,
    ) -> Result<u32> {
        let (logits_shape, logits_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {}", e)))?;

        let logits_dims = logits_shape.as_ref();

        // Handle both 2D [batch, vocab_size] (KV-cached single-token step)
        // and 3D [batch, seq_len, vocab_size] (multi-token / full decode step)
        let mut last_logits: Vec<f32> = match logits_dims.len() {
            2 => {
                let vocab_size = logits_dims[1] as usize;
                logits_data[..vocab_size].to_vec()
            }
            3 => {
                let vocab_size = logits_dims[2] as usize;
                let seq_len = logits_dims[1] as usize;
                let last_pos_start = (seq_len - 1) * vocab_size;
                logits_data[last_pos_start..last_pos_start + vocab_size].to_vec()
            }
            _ => {
                return Err(Error::Model(format!(
                    "Unexpected logits dimensions: {:?}", logits_dims
                )));
            }
        };

        // Apply repetition penalty
        if repetition_penalty != 1.0 {
            for &token_id in past_tokens {
                let idx = token_id as usize;
                if idx < last_logits.len() {
                    if last_logits[idx] > 0.0 {
                        last_logits[idx] /= repetition_penalty;
                    } else {
                        last_logits[idx] *= repetition_penalty;
                    }
                }
            }
        }

        // Greedy selection
        Ok(last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0))
    }

    /// Enforce audio length limit (truncate to last max_audio_secs)
    fn enforce_audio_limit<'a>(&self, samples: &'a [f32]) -> &'a [f32] {
        let max_samples = (self.config.max_audio_secs * self.config.sample_rate as f32) as usize;
        if samples.len() > max_samples {
            eprintln!(
                "[CanaryQwen] Audio too long ({:.1}s), truncating to last {:.1}s",
                samples.len() as f32 / self.config.sample_rate as f32,
                self.config.max_audio_secs
            );
            &samples[samples.len() - max_samples..]
        } else {
            samples
        }
    }

    /// Transcribe audio samples
    ///
    /// # Arguments
    /// * `samples` - Audio samples (16kHz, mono, f32 normalized to [-1, 1])
    ///
    /// # Returns
    /// Transcribed text
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Enforce audio length limit
        let samples = self.enforce_audio_limit(samples);

        // Extract mel features
        let features = self.extract_features(samples)?;

        // Run encoder
        let (encoder_embeddings, _encoder_length) = self.run_encoder(&features)?;

        // Run projection (encoder_dim → LLM dim)
        let projected_embeddings = self.run_projection(&encoder_embeddings)?;

        // Use SALM chat-template prompt with audio placeholder:
        //   <|im_start|>user\nTranscribe the following: <|audioplaceholder|><|im_end|>\n<|im_start|>assistant\n
        let prompt_ids = PROMPT_TOKEN_IDS;

        // Embed prompt tokens (including placeholder, which will be replaced)
        let prompt_embeds = self.embed_prompt(prompt_ids)?;

        // Replace audio placeholder with projected audio embeddings
        let inputs_embeds = Self::build_decoder_input(&prompt_embeds, &projected_embeddings, prompt_ids)?;

        // Autoregressive decoding
        let generated_tokens = self.greedy_decode_cached(&inputs_embeds)?;

        // Decode tokens to text
        let text = self.tokenizer.decode(&generated_tokens)?;

        Ok(text.trim().to_string())
    }

    /// Get the tokenizer reference
    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }

    /// Get the config
    pub fn config(&self) -> &CanaryQwenConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CanaryQwenConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.language, "en");
        assert_eq!(config.max_audio_secs, 40.0);
        assert_eq!(config.num_decoder_layers, 28);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.max_sequence_length, 2048);
    }

    #[test]
    fn test_config_from_env_auto() {
        // Without env var set, should be None (auto)
        let config = CanaryQwenConfig::default();
        assert!(config.prefer_quantized.is_none());
    }

    #[test]
    fn test_audio_limit_enforcement() {
        let config = CanaryQwenConfig {
            max_audio_secs: 2.0,
            sample_rate: 16000,
            ..Default::default()
        };
        let max_samples = (config.max_audio_secs * config.sample_rate as f32) as usize;

        // Under limit: no truncation
        let short = vec![0.0f32; max_samples - 1000];
        assert_eq!(short.len(), max_samples - 1000);

        // Over limit: truncation to last max_audio_secs
        let long = vec![0.0f32; max_samples + 5000];
        let truncated_start = long.len() - max_samples;
        assert_eq!(truncated_start, 5000);
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = QwenKVCache::new(28, 8, 128);
        assert_eq!(cache.position, 0);
        assert_eq!(cache.past_keys.len(), 28);
        assert_eq!(cache.past_values.len(), 28);
        assert_eq!(cache.past_keys[0].shape(), &[1, 8, 0, 128]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = QwenKVCache::new(4, 8, 64);
        cache.position = 10;
        cache.reset();
        assert_eq!(cache.position, 0);
        assert_eq!(cache.past_keys[0].shape(), &[1, 8, 0, 64]);
    }

    #[test]
    fn test_build_decoder_input() {
        // Simulate 5 prompt tokens where token at index 2 is the audio placeholder
        let mut prompt = Array3::<f32>::zeros((1, 5, 8));
        // Mark each position with a unique value so we can verify placement
        for t in 0..5 {
            for h in 0..8 {
                prompt[[0, t, h]] = (t + 1) as f32 * 0.1; // 0.1, 0.2, 0.3, 0.4, 0.5
            }
        }
        let audio = Array3::<f32>::ones((1, 10, 8)); // audio embeddings are all 1.0
        let prompt_ids = &[100i64, 200, AUDIO_PLACEHOLDER_TOKEN_ID, 300, 400]; // placeholder at index 2

        let combined = CanaryQwenModel::build_decoder_input(&prompt, &audio, prompt_ids).unwrap();
        // Total = 5 - 1 (placeholder) + 10 (audio) = 14
        assert_eq!(combined.shape(), &[1, 14, 8]);
        // Position 0-1: prompt before placeholder (0.1, 0.2)
        assert!((combined[[0, 0, 0]] - 0.1).abs() < 1e-6);
        assert!((combined[[0, 1, 0]] - 0.2).abs() < 1e-6);
        // Position 2-11: audio embeddings (1.0)
        assert!((combined[[0, 2, 0]] - 1.0).abs() < 1e-6);
        assert!((combined[[0, 11, 0]] - 1.0).abs() < 1e-6);
        // Position 12-13: prompt after placeholder (0.4, 0.5)
        assert!((combined[[0, 12, 0]] - 0.4).abs() < 1e-6);
        assert!((combined[[0, 13, 0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_extract_next_token_repetition_penalty() {
        let mut logits = vec![0.0f32; 100];
        logits[42] = 10.0;

        // Without penalty
        let best = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap();
        assert_eq!(best, 42);

        // With penalty on token 42
        let penalty = 1.2f32;
        let past = vec![42u32];
        for &token_id in &past {
            let idx = token_id as usize;
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            }
        }
        assert!((logits[42] - 10.0 / 1.2).abs() < 1e-6);
    }
}
