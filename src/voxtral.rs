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
    /// Pre-computed mel filterbank from Python reference [n_fft_bins=201, n_mels=128]
    mel_filterbank: Option<Array2<f32>>,
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

        // Select quantization variant: auto-detect based on available files
        // FP16 preferred when available (better quality), Q4 as fallback
        let fp16_exists = base_dir.join("audio_encoder_fp16.onnx").exists();
        let q4_exists = base_dir.join("audio_encoder_q4.onnx").exists();
        let use_quantized = config.prefer_quantized.unwrap_or(!fp16_exists && q4_exists);

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
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let encoder_session = builder.commit_from_file(&encoder_path)?;
        eprintln!("[Voxtral] Encoder loaded: {:?}", encoder_path.file_name().unwrap());

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let decoder_session = builder.commit_from_file(&decoder_path)?;
        eprintln!("[Voxtral] Decoder loaded: {:?}", decoder_path.file_name().unwrap());

        let builder = Session::builder()?;
        let mut builder = exec_config.apply_to_session_builder(builder)?;
        let embed_tokens_session = builder.commit_from_file(&embed_path)?;
        eprintln!("[Voxtral] Embed tokens loaded: {:?}", embed_path.file_name().unwrap());

        // Load tokenizer
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = VoxtralTokenizer::from_file(&tokenizer_path)?;
        eprintln!("[Voxtral] Tokenizer loaded (vocab: {})", tokenizer.vocab_size());

        let kv_cache = KVCache::new(config.num_decoder_layers, config.num_kv_heads, config.head_dim);

        // Try loading pre-computed mel filterbank
        let fb_path = dir.join("mel_filterbank.bin");
        let mel_filterbank = if fb_path.exists() {
            let data = std::fs::read(&fb_path)
                .map_err(|e| Error::Io(e))?;
            let n_bins = N_FFT / 2 + 1; // 201
            let floats: Vec<f32> = data.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let fb = Array2::from_shape_vec((n_bins, N_MELS), floats)
                .map_err(|e| Error::Config(format!("Filterbank reshape: {}", e)))?;
            eprintln!("[Voxtral] Loaded pre-computed mel filterbank: {:?}", fb.shape());
            Some(fb)
        } else {
            eprintln!("[Voxtral] No pre-computed filterbank, using computed one");
            None
        };

        Ok(Self {
            encoder_session,
            decoder_session,
            embed_tokens_session,
            tokenizer,
            kv_cache,
            config,
            mel_filterbank,
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
        // Clamp to encoder's max_position_embeddings (1500 frames ≈ 15s at hop=160)
        let max_frames = 1500;
        let max_samples_by_frames = (max_frames - 1) * HOP_LENGTH;
        let max_samples = (self.config.max_audio_secs * SAMPLE_RATE as f32) as usize;
        let max_samples = max_samples.min(max_samples_by_frames);
        let raw_samples = if audio.len() > max_samples { &audio[..max_samples] } else { audio };

        // Center-padding: pad n_fft/2 zeros on each side (matching torch.stft center=True)
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; pad + raw_samples.len() + pad];
        padded[pad..pad + raw_samples.len()].copy_from_slice(raw_samples);

        // Compute STFT frames (drop last frame to match torch.stft [..., :-1])
        let raw_frames = 1 + padded.len() / HOP_LENGTH;
        let n_frames_stft = raw_frames - 1; // Drop last frame
        let n_frames = (n_frames_stft / 8) * 8; // Round down to multiple of 8
        if n_frames == 0 {
            return Ok(Array3::<f32>::zeros((1, N_MELS, 0)));
        }

        let n_fft_bins = N_FFT / 2 + 1;
        let window = self.hann_window();

        // Use pre-computed filterbank or compute one
        let fb = self.mel_filterbank.as_ref().map(|f| f.clone())
            .unwrap_or_else(|| self.create_mel_filterbank());

        // Pre-allocate FFT planner
        use rustfft::{FftPlanner, num_complex::Complex};
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let mut mel_spec = Array2::<f32>::zeros((N_MELS, n_frames));

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            let mut complex_buf: Vec<Complex<f32>> = (0..N_FFT).map(|i| {
                let sample = if start + i < padded.len() { padded[start + i] } else { 0.0 };
                Complex { re: sample * window[i], im: 0.0 }
            }).collect();

            fft.process(&mut complex_buf);

            // Power spectrum
            let power: Vec<f32> = (0..n_fft_bins)
                .map(|k| complex_buf[k].re * complex_buf[k].re + complex_buf[k].im * complex_buf[k].im)
                .collect();

            // Apply mel filterbank: fb is [n_bins, n_mels], we want fb.T @ power
            for mel_idx in 0..N_MELS {
                let mut energy = 0.0f32;
                for k in 0..n_fft_bins {
                    energy += fb[[k, mel_idx]] * power[k];
                }
                mel_spec[[mel_idx, frame_idx]] = energy;
            }
        }

        // Normalize (matching VoxtralRealtimeFeatureExtractor exactly):
        // log10(clamp(mel, 1e-10)) → max(x, global_max - 8) → (x + 4) / 4
        let bottom_clamp = GLOBAL_LOG_MEL_MAX - 8.0;
        mel_spec.mapv_inplace(|v| {
            let log10_val = v.max(1e-10).log10();
            let clamped = log10_val.max(bottom_clamp);
            (clamped + 4.0) / 4.0
        });

        let (n_mels_dim, n_frames_dim) = (mel_spec.shape()[0], mel_spec.shape()[1]);
        let flat = mel_spec.into_raw_vec();
        Ok(Array3::from_shape_vec((1, n_mels_dim, n_frames_dim), flat)
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
            // Slaney normalization: divide by bandwidth in Hz (2 / (f_high - f_low))
            let f_low = mel_points[m];    // mel_points are already in Hz
            let f_high = mel_points[m + 2];
            let bandwidth_hz = f_high - f_low;
            if bandwidth_hz > 0.0 {
                let norm_factor = 2.0 / bandwidth_hz;
                for k in 0..n_fft_bins {
                    filterbank[[m, k]] *= norm_factor;
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
        // Voxtral encoder is a causal transformer with KV cache.
        // For non-streaming (full chunk), provide full attention mask and empty KV cache.
        let seq_len = features.shape()[2]; // [batch=1, n_mels=128, audio_seq_len]

        let input_value = ort::value::Value::from_array(features.clone())?;

        // The encoder has 2-stage downsampling:
        // 1. Conv downsample by 2 before transformer (mel_frames → mel_frames/2)
        // 2. Projector downsample by 4 after transformer (→ mel_frames/8 output tokens)
        // position_ids must match the post-conv, pre-transformer length = mel_frames / 2
        let transformer_seq_len = seq_len / 2;

        // Attention mask: all ones for the transformer sequence length
        let attention_mask = ndarray::Array2::<i64>::from_elem((1, transformer_seq_len), 1);

        // Position IDs: 0..transformer_seq_len (not mel frame count)
        let position_ids = ndarray::Array2::<i64>::from_shape_vec(
            (1, transformer_seq_len), (0..transformer_seq_len as i64).collect(),
        ).map_err(|e| Error::Model(format!("Encoder position_ids: {}", e)))?;

        // Past padding cache: zeros [batch=1, 1408, 2]
        let past_padding = ndarray::Array3::<f32>::zeros((1, 1408, 2));

        let num_encoder_layers = 32; // Voxtral encoder has 32 layers
        let num_heads = 32;
        let head_dim = 64;

        let mut inputs = ort::inputs!(
            "input_features" => input_value,
            "attention_mask" => ort::value::Value::from_array(attention_mask)?,
            "position_ids" => ort::value::Value::from_array(position_ids)?,
            "past_padding_cache" => ort::value::Value::from_array(past_padding)?
        );

        // Add empty KV cache for all 32 encoder layers
        for layer in 0..num_encoder_layers {
            let empty_key = Array4::<f32>::zeros((1, num_heads, 0, head_dim));
            let empty_val = Array4::<f32>::zeros((1, num_heads, 0, head_dim));
            inputs.push((
                format!("past_key_values.{}.key", layer).into(),
                ort::value::Value::from_array(empty_key)?.into(),
            ));
            inputs.push((
                format!("past_key_values.{}.value", layer).into(),
                ort::value::Value::from_array(empty_val)?.into(),
            ));
        }

        let outputs = self.encoder_session.run(inputs)?;

        // Extract audio_embeds [batch, num_audio_tokens, 3072]
        let (shape, data) = outputs["audio_embeds"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Encoder audio_embeds extract: {}", e)))?;
        let dims = shape.as_ref();

        let output = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        ).map_err(|e| Error::Model(format!("Encoder reshape: {}", e)))?;

        eprintln!("[Voxtral] Encoder output: [{}, {}, {}]", dims[0], dims[1], dims[2]);

        Ok(output)
    }

    // ========================================================================
    // Autoregressive decoder
    // ========================================================================

    /// Embed token IDs via embed_tokens ONNX session
    fn embed_tokens(&mut self, token_ids: &[i64]) -> Result<Array3<f32>> {
        let input = ndarray::Array2::from_shape_vec((1, token_ids.len()), token_ids.to_vec())
            .map_err(|e| Error::Model(format!("embed input: {}", e)))?;
        let input_value = ort::value::Value::from_array(input)?;
        let outputs = self.embed_tokens_session.run(ort::inputs!("input_ids" => input_value))?;

        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("embed extract: {}", e)))?;
        let dims = shape.as_ref();
        Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        ).map_err(|e| Error::Model(format!("embed reshape: {}", e)))
    }

    /// Extract a 4D tensor from ONNX outputs by name
    fn extract_4d(outputs: &ort::session::SessionOutputs, name: &str) -> Result<Array4<f32>> {
        let (shape, data) = outputs[name]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("extract {}: {}", name, e)))?;
        let dims = shape.as_ref();
        if dims.len() != 4 {
            return Err(Error::Model(format!("{}: expected 4D, got {}D", name, dims.len())));
        }
        Array4::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize, dims[3] as usize),
            data.to_vec(),
        ).map_err(|e| Error::Model(format!("{} reshape: {}", name, e)))
    }

    /// Extract next token from logits with repetition penalty
    fn extract_next_token(
        outputs: &ort::session::SessionOutputs,
        past_tokens: &[i64],
        rep_penalty: f32,
        vocab_size: usize,
    ) -> Result<i64> {
        let (logits_shape, logits_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("logits extract: {}", e)))?;
        let dims = logits_shape.as_ref();

        let mut last_logits: Vec<f32> = match dims.len() {
            2 => logits_data[..vocab_size].to_vec(),
            3 => {
                let seq_len = dims[1] as usize;
                let start = (seq_len - 1) * vocab_size;
                logits_data[start..start + vocab_size].to_vec()
            }
            _ => return Err(Error::Model(format!("logits: unexpected dims {:?}", dims))),
        };

        if rep_penalty != 1.0 {
            for &tok in past_tokens {
                let idx = tok as usize;
                if idx < last_logits.len() {
                    if last_logits[idx] > 0.0 { last_logits[idx] /= rep_penalty; }
                    else { last_logits[idx] *= rep_penalty; }
                }
            }
        }

        Ok(last_logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as i64)
            .unwrap_or(EOS_TOKEN_ID))
    }

    /// Autoregressive greedy decoding with KV cache.
    /// Voxtral uses a decoder-only architecture where audio_embeds from the encoder
    /// are concatenated with text token embeds as inputs_embeds (no cross-attention).
    fn greedy_decode(&mut self, encoder_output: &Array3<f32>) -> Result<Vec<i64>> {
        self.kv_cache.reset();
        let max_tokens = self.config.max_sequence_length.min(500);
        let mut generated: Vec<i64> = Vec::new();
        let audio_len = encoder_output.shape()[1]; // num_audio_tokens

        // Step 0: Prefill using Voxtral's streaming token protocol:
        // Token IDs: [BOS=1, AUDIO_TOKEN=24 × n_audio_tokens, DELAY_TOKEN=11 × 6]
        // Embed tokens → text_embeds, then ADD audio_embeds at AUDIO positions
        let first_token = {
            let audio_token_id: i64 = 24; // [AUDIO]
            let delay_token_id: i64 = PAD_TOKEN_ID; // PAD=11 used as delay token
            let num_delay = 6;

            // Build token sequence
            let mut token_ids: Vec<i64> = vec![BOS_TOKEN_ID];
            for _ in 0..audio_len { token_ids.push(audio_token_id); }
            for _ in 0..num_delay { token_ids.push(delay_token_id); }
            let total_len = token_ids.len();

            // Embed all tokens
            let text_embeds = self.embed_tokens(&token_ids)?; // [1, total_len, 3072]

            // ADD audio_embeds to text_embeds at AUDIO positions (indices 1..1+audio_len)
            let hidden = self.config.hidden_size;
            let mut combined = ndarray::Array3::<f32>::zeros((1, total_len, hidden));
            // Copy text embeds
            for t in 0..total_len {
                for h in 0..hidden {
                    combined[[0, t, h]] = text_embeds[[0, t, h]];
                }
            }
            // ADD audio embeds at AUDIO token positions
            for t in 0..audio_len {
                for h in 0..hidden {
                    combined[[0, 1 + t, h]] += encoder_output[[0, t, h]];
                }
            }

            // Attention mask: all ones
            let attention_mask = ndarray::Array2::<i64>::from_elem((1, total_len), 1);

            let mut inputs = ort::inputs!(
                "inputs_embeds" => ort::value::Value::from_array(combined)?,
                "attention_mask" => ort::value::Value::from_array(attention_mask)?
            );

            // Empty KV cache
            for layer in 0..self.kv_cache.num_layers {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    ort::value::Value::from_array(self.kv_cache.past_keys[layer].clone())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    ort::value::Value::from_array(self.kv_cache.past_values[layer].clone())?.into(),
                ));
            }

            let outputs = self.decoder_session.run(inputs)?;

            // Update KV cache
            let mut pk = Vec::new();
            let mut pv = Vec::new();
            for layer in 0..self.kv_cache.num_layers {
                pk.push(Self::extract_4d(&outputs, &format!("present.{}.key", layer))?);
                pv.push(Self::extract_4d(&outputs, &format!("present.{}.value", layer))?);
            }
            self.kv_cache.update(pk, pv);

            Self::extract_next_token(&outputs, &generated, self.config.repetition_penalty, self.config.vocab_size)?
        };

        if first_token == EOS_TOKEN_ID { return Ok(generated); }
        generated.push(first_token);

        // Steps 1..N: one token at a time with KV cache
        for _step in 1..max_tokens {
            let last_tok = *generated.last().unwrap();
            let tok_embed = self.embed_tokens(&[last_tok])?; // [1, 1, 3072]

            // Attention mask covers all previous tokens + current
            let total_seq = audio_len + generated.len() + 1; // audio + BOS + generated so far
            let attention_mask = ndarray::Array2::<i64>::from_elem((1, total_seq), 1);

            let mut inputs = ort::inputs!(
                "inputs_embeds" => ort::value::Value::from_array(tok_embed)?,
                "attention_mask" => ort::value::Value::from_array(attention_mask)?
            );

            for layer in 0..self.kv_cache.num_layers {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    ort::value::Value::from_array(self.kv_cache.past_keys[layer].clone())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    ort::value::Value::from_array(self.kv_cache.past_values[layer].clone())?.into(),
                ));
            }

            let outputs = self.decoder_session.run(inputs)?;

            let mut pk = Vec::new();
            let mut pv = Vec::new();
            for layer in 0..self.kv_cache.num_layers {
                pk.push(Self::extract_4d(&outputs, &format!("present.{}.key", layer))?);
                pv.push(Self::extract_4d(&outputs, &format!("present.{}.value", layer))?);
            }
            self.kv_cache.update(pk, pv);

            let next = Self::extract_next_token(&outputs, &generated, self.config.repetition_penalty, self.config.vocab_size)?;
            if next == EOS_TOKEN_ID { break; }
            generated.push(next);
        }

        Ok(generated)
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
