//! Voxtral Mini 4B Realtime ASR model.
//!
//! Architecture: Audio Encoder (32L, 1280 hidden) → Projection → LLM Decoder (26L, 3072 hidden, GQA 32/8)
//! Uses tokenizer.json (HuggingFace Tokenizers format) instead of SentencePiece.
//! Mel spectrogram: 128 bins, n_fft=400, hop=160, 16kHz.

use crate::error::{Error, Result};
use crate::ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4, s};
use ort::session::Session;
use std::path::{Path, PathBuf};

const SAMPLE_RATE: usize = 16000;
const N_MELS: usize = 128;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const MAX_AUDIO_SECS: f32 = 30.0;
const GLOBAL_LOG_MEL_MAX: f32 = 1.5;

// Decoder token IDs
const BOS_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;
const PAD_TOKEN_ID: i64 = 11;

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct VoxtralConfig {
    pub n_mels: usize,
    pub sample_rate: usize,
    pub max_audio_secs: f32,
    pub max_sequence_length: usize,
    pub language: String,
    pub repetition_penalty: f32,
    pub num_decoder_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub prefer_quantized: Option<bool>,
}

impl Default for VoxtralConfig {
    fn default() -> Self {
        Self {
            n_mels: N_MELS,
            sample_rate: SAMPLE_RATE,
            max_audio_secs: MAX_AUDIO_SECS,
            max_sequence_length: 4096,
            language: "de".to_string(),
            repetition_penalty: 1.2,
            num_decoder_layers: 26,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            hidden_size: 3072,
            vocab_size: 131072,
            prefer_quantized: None,
        }
    }
}

// ============================================================================
// Tokenizer (HuggingFace Tokenizers format)
// ============================================================================

pub struct VoxtralTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl VoxtralTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| Error::Tokenizer(format!("Tokenizer encode error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self.tokenizer.decode(ids, true)
            .map_err(|e| Error::Tokenizer(format!("Tokenizer decode error: {}", e)))?;
        Ok(text)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

// ============================================================================
// KV Cache
// ============================================================================

struct KVCache {
    past_keys: Vec<Array4<f32>>,
    past_values: Vec<Array4<f32>>,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KVCache {
    fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let mut past_keys = Vec::with_capacity(num_layers);
        let mut past_values = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            past_keys.push(Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)));
            past_values.push(Array4::<f32>::zeros((1, num_kv_heads, 0, head_dim)));
        }
        Self { past_keys, past_values, num_layers, num_kv_heads, head_dim }
    }

    fn reset(&mut self) {
        for i in 0..self.num_layers {
            self.past_keys[i] = Array4::<f32>::zeros((1, self.num_kv_heads, 0, self.head_dim));
            self.past_values[i] = Array4::<f32>::zeros((1, self.num_kv_heads, 0, self.head_dim));
        }
    }

    fn update(&mut self, present_keys: Vec<Array4<f32>>, present_values: Vec<Array4<f32>>) {
        self.past_keys = present_keys;
        self.past_values = present_values;
    }
}

// ============================================================================
// Model
// ============================================================================

pub struct VoxtralModel {
    encoder_session: Session,
    decoder_session: Session,
    embed_tokens_session: Session,
    tokenizer: VoxtralTokenizer,
    kv_cache: KVCache,
    config: VoxtralConfig,
}

impl VoxtralModel {
    pub fn from_pretrained<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<VoxtralConfig>,
    ) -> Result<Self> {
        let dir = model_path.as_ref();
        let config = config.unwrap_or_default();
        let exec_config = exec_config.unwrap_or_default();

        // Find ONNX files
        let onnx_dir = dir.join("onnx");
        let base_dir = if onnx_dir.exists() { onnx_dir } else { dir.to_path_buf() };

        // Select quantization variant
        let use_quantized = config.prefer_quantized.unwrap_or(true); // Default Q4

        let (enc_name, dec_name, emb_name) = if use_quantized {
            ("audio_encoder_q4.onnx", "decoder_model_merged_q4.onnx", "embed_tokens_q4.onnx")
        } else {
            ("audio_encoder_fp16.onnx", "decoder_model_merged_fp16.onnx", "embed_tokens_fp16.onnx")
        };

        eprintln!("[Voxtral] Loading model from {:?} (quantized: {})", dir, use_quantized);

        let encoder_path = base_dir.join(enc_name);
        let decoder_path = base_dir.join(dec_name);
        let embed_path = base_dir.join(emb_name);

        // Load ONNX sessions using the standard builder pattern
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder_session = builder.commit_from_file(&encoder_path)?;
        eprintln!("[Voxtral] Encoder loaded: {:?}", encoder_path.file_name().unwrap());

        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_session = builder.commit_from_file(&decoder_path)?;
        eprintln!("[Voxtral] Decoder loaded: {:?}", decoder_path.file_name().unwrap());

        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let embed_tokens_session = builder.commit_from_file(&embed_path)?;
        eprintln!("[Voxtral] Embed tokens loaded: {:?}", embed_path.file_name().unwrap());

        // Load tokenizer
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = VoxtralTokenizer::from_file(&tokenizer_path)?;
        eprintln!("[Voxtral] Tokenizer loaded (vocab: {})", tokenizer.vocab_size());

        let kv_cache = KVCache::new(config.num_decoder_layers, config.num_kv_heads, config.head_dim);

        Ok(Self {
            encoder_session,
            decoder_session,
            embed_tokens_session,
            tokenizer,
            kv_cache,
            config,
        })
    }

    /// Transcribe audio samples to text
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        // Reset KV cache for fresh inference
        self.kv_cache.reset();

        // 1. Extract mel spectrogram features
        let t0 = std::time::Instant::now();
        let features = self.extract_features(samples)?;
        let feat_ms = t0.elapsed().as_millis();

        // 2. Run encoder
        let t1 = std::time::Instant::now();
        let encoder_output = self.run_encoder(&features)?;
        let enc_ms = t1.elapsed().as_millis();

        // 3. Run autoregressive decoder
        let t2 = std::time::Instant::now();
        let token_ids = self.greedy_decode(&encoder_output)?;
        let dec_ms = t2.elapsed().as_millis();

        // 4. Decode tokens to text
        let text = self.tokenizer.decode(&token_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())?;

        eprintln!(
            "[Voxtral] features={}ms encoder={}ms decoder={}ms ({} tokens) total={}ms",
            feat_ms, enc_ms, dec_ms, token_ids.len(), t0.elapsed().as_millis()
        );

        Ok(text)
    }

    // ========================================================================
    // Feature extraction (128-bin mel spectrogram)
    // ========================================================================

    fn extract_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        let max_samples = (self.config.max_audio_secs * SAMPLE_RATE as f32) as usize;
        let samples = if audio.len() > max_samples { &audio[..max_samples] } else { audio };

        // Compute STFT
        let n_frames = 1 + (samples.len() / HOP_LENGTH);
        let mut mel_spec = Array2::<f32>::zeros((N_MELS, n_frames));

        let mel_filterbank = self.create_mel_filterbank();
        let window = self.hann_window();

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            let mut frame = vec![0.0f32; N_FFT];
            let copy_len = N_FFT.min(samples.len().saturating_sub(start));
            for i in 0..copy_len {
                frame[i] = samples[start + i] * window[i];
            }

            // FFT (real-valued, use simple DFT for correctness)
            let n_fft_bins = N_FFT / 2 + 1;
            let mut power_spectrum = vec![0.0f32; n_fft_bins];
            for k in 0..n_fft_bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                for n in 0..N_FFT {
                    let angle = -2.0 * std::f32::consts::PI * k as f32 * n as f32 / N_FFT as f32;
                    re += frame[n] * angle.cos();
                    im += frame[n] * angle.sin();
                }
                power_spectrum[k] = re * re + im * im;
            }

            // Apply mel filterbank
            for mel_idx in 0..N_MELS {
                let mut energy = 0.0f32;
                for k in 0..n_fft_bins {
                    energy += mel_filterbank[[mel_idx, k]] * power_spectrum[k];
                }
                mel_spec[[mel_idx, frame_idx]] = energy.max(1e-10).ln();
            }
        }

        // Normalize: clamp to global_log_mel_max, then scale to [-1, 1]
        let max_val = GLOBAL_LOG_MEL_MAX;
        mel_spec.mapv_inplace(|v| {
            let clamped = v.min(max_val);
            clamped / max_val // Scale to [-inf/max, 1.0]
        });

        // Reshape to [batch=1, n_mels, n_frames]
        let shape = mel_spec.shape();
        Ok(mel_spec.into_shape_with_order((1, shape[0], shape[1]))
            .map_err(|e| Error::Config(format!("Reshape error: {}", e)))?)
    }

    fn create_mel_filterbank(&self) -> Array2<f32> {
        let n_fft_bins = N_FFT / 2 + 1;
        let mut filterbank = Array2::<f32>::zeros((N_MELS, n_fft_bins));

        let f_min = 0.0f32;
        let f_max = (SAMPLE_RATE / 2) as f32;
        let mel_min = Self::hz_to_mel(f_min);
        let mel_max = Self::hz_to_mel(f_max);

        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| Self::mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32))
            .collect();

        let bin_points: Vec<f32> = mel_points.iter()
            .map(|f| f * N_FFT as f32 / SAMPLE_RATE as f32)
            .collect();

        for m in 0..N_MELS {
            for k in 0..n_fft_bins {
                let kf = k as f32;
                if kf >= bin_points[m] && kf <= bin_points[m + 1] {
                    filterbank[[m, k]] = (kf - bin_points[m]) / (bin_points[m + 1] - bin_points[m]).max(1e-10);
                } else if kf >= bin_points[m + 1] && kf <= bin_points[m + 2] {
                    filterbank[[m, k]] = (bin_points[m + 2] - kf) / (bin_points[m + 2] - bin_points[m + 1]).max(1e-10);
                }
            }
        }

        filterbank
    }

    fn hann_window(&self) -> Vec<f32> {
        (0..N_FFT)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos()))
            .collect()
    }

    fn hz_to_mel(freq: f32) -> f32 { 2595.0 * (1.0 + freq / 700.0).log10() }
    fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) }

    // ========================================================================
    // Encoder
    // ========================================================================

    fn run_encoder(&mut self, features: &Array3<f32>) -> Result<Array3<f32>> {
        let input_value = ort::value::Value::from_array(features.clone())?;

        let outputs = self.encoder_session.run(ort::inputs!(
            "input_features" => input_value
        )?)?;

        let enc_out = &outputs[0];
        let (shape, data) = enc_out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Encoder extract: {}", e)))?;
        let dims = shape.as_ref();

        let output = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        ).map_err(|e| Error::Model(format!("Encoder reshape: {}", e)))?;

        Ok(output)
    }

    // ========================================================================
    // Autoregressive decoder
    // ========================================================================

    fn greedy_decode(&mut self, encoder_output: &Array3<f32>) -> Result<Vec<i64>> {
        let max_tokens = self.config.max_sequence_length.min(500); // Safety limit
        let mut tokens: Vec<i64> = vec![BOS_TOKEN_ID];
        let vocab_size = self.config.vocab_size;
        let rep_penalty = self.config.repetition_penalty;

        // Get embed tokens for BOS
        let mut current_ids = ndarray::Array2::<i64>::from_shape_vec(
            (1, 1), vec![BOS_TOKEN_ID]
        ).unwrap();

        // Feed encoder output as the first input
        let encoder_hidden = ort::value::Value::from_array(encoder_output.view())?;

        for step in 0..max_tokens {
            // Get token embeddings
            let input_ids_val = ort::value::Value::from_array(current_ids.view())?;
            let embed_output = self.embed_tokens_session.run(
                ort::inputs!["input_ids" => input_ids_val]?
            )?;
            let inputs_embeds = embed_output[0].try_extract_tensor::<f32>()?;

            // Build decoder inputs
            let use_cache = step > 0;
            let mut decoder_inputs = Vec::new();

            // inputs_embeds
            decoder_inputs.push(("inputs_embeds".to_string(),
                ort::value::Value::from_array(inputs_embeds.view())?));

            // encoder_hidden_states (only on first step, or always for cross-attention)
            decoder_inputs.push(("encoder_hidden_states".to_string(),
                encoder_hidden.view().try_clone()?));

            // KV cache
            for layer in 0..self.kv_cache.num_layers {
                let key_name = format!("past_key_values.{}.key", layer);
                let value_name = format!("past_key_values.{}.value", layer);
                decoder_inputs.push((key_name,
                    ort::value::Value::from_array(self.kv_cache.past_keys[layer].view())?));
                decoder_inputs.push((value_name,
                    ort::value::Value::from_array(self.kv_cache.past_values[layer].view())?));
            }

            // use_cache_branch
            let use_cache_val = ndarray::Array1::<bool>::from_vec(vec![use_cache]);
            decoder_inputs.push(("use_cache_branch".to_string(),
                ort::value::Value::from_array(use_cache_val.view())?));

            // Run decoder
            let outputs = self.decoder_session.run(
                decoder_inputs.into_iter().map(|(k, v)| (k.into(), v)).collect::<Vec<_>>()
            )?;

            // Extract logits (first output)
            let logits = outputs[0].try_extract_tensor::<f32>()?;
            let logits_shape = logits.shape();
            let seq_len = logits_shape[1];

            // Get logits for the last token
            let mut last_logits: Vec<f32> = (0..vocab_size)
                .map(|i| logits[[0, seq_len - 1, i]])
                .collect();

            // Apply repetition penalty
            if rep_penalty != 1.0 {
                for &tok in &tokens {
                    let idx = tok as usize;
                    if idx < last_logits.len() {
                        if last_logits[idx] > 0.0 {
                            last_logits[idx] /= rep_penalty;
                        } else {
                            last_logits[idx] *= rep_penalty;
                        }
                    }
                }
            }

            // Greedy selection
            let next_token = last_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(EOS_TOKEN_ID);

            // Check for EOS
            if next_token == EOS_TOKEN_ID {
                break;
            }

            tokens.push(next_token);

            // Update KV cache from decoder outputs
            let mut present_keys = Vec::new();
            let mut present_values = Vec::new();
            for layer in 0..self.kv_cache.num_layers {
                let key_idx = 1 + layer * 2;
                let val_idx = 2 + layer * 2;
                if key_idx < outputs.len() && val_idx < outputs.len() {
                    let key = outputs[key_idx].try_extract_tensor::<f32>()?.to_owned();
                    let val = outputs[val_idx].try_extract_tensor::<f32>()?.to_owned();
                    let k_shape = key.shape();
                    let v_shape = val.shape();
                    present_keys.push(key.into_shape_with_order((k_shape[0], k_shape[1], k_shape[2], k_shape[3]))
                        .map_err(|e| Error::Config(format!("KV key reshape: {}", e)))?);
                    present_values.push(val.into_shape_with_order((v_shape[0], v_shape[1], v_shape[2], v_shape[3]))
                        .map_err(|e| Error::Config(format!("KV val reshape: {}", e)))?);
                }
            }
            if present_keys.len() == self.kv_cache.num_layers {
                self.kv_cache.update(present_keys, present_values);
            }

            // Prepare next input
            current_ids = ndarray::Array2::<i64>::from_shape_vec(
                (1, 1), vec![next_token]
            ).unwrap();
        }

        // Remove BOS token
        if !tokens.is_empty() && tokens[0] == BOS_TOKEN_ID {
            tokens.remove(0);
        }

        Ok(tokens)
    }

    pub fn set_language(&mut self, lang: &str) {
        self.config.language = lang.to_string();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = VoxtralConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.num_decoder_layers, 26);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.vocab_size, 131072);
    }

    #[test]
    fn test_mel_conversion() {
        let freq = 1000.0;
        let mel = VoxtralModel::hz_to_mel(freq);
        let back = VoxtralModel::mel_to_hz(mel);
        assert!((freq - back).abs() < 0.1);
    }
}
