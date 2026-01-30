//! Canary 1B v2 ONNX model implementation
//!
//! Canary is NVIDIA's encoder-decoder ASR model supporting 25 languages.
//! This module provides ONNX-based inference using a separately exported
//! encoder and decoder model.
//!
//! Model source: https://huggingface.co/istupakov/canary-1b-v2-onnx

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array1, Array2, Array3, Array4};
use ort::session::Session;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
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

// Special token IDs (from vocab.txt)
const UNK_ID: i64 = 0;
const NOSPEECH_ID: i64 = 1;
const PAD_ID: i64 = 2;
const ENDOFTEXT_ID: i64 = 3;
const STARTOFTRANSCRIPT_ID: i64 = 4;
const PNC_ID: i64 = 5;           // <|pnc|> - punctuation
const STARTOFCONTEXT_ID: i64 = 7; // <|startofcontext|>
const NOITN_ID: i64 = 9;         // <|noitn|> - no ITN
const NOTIMESTAMP_ID: i64 = 11;  // <|notimestamp|>
const NODIARIZE_ID: i64 = 13;    // <|nodiarize|>
const EMO_UNDEFINED_ID: i64 = 16; // <|emo:undefined|>

// Language code (from vocab.txt)
const EN_LANG_ID: i64 = 64; // <|en|>

// ============================================================================
// Canary Configuration
// ============================================================================

/// Configuration for Canary model
#[derive(Debug, Clone)]
pub struct CanaryConfig {
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Maximum sequence length for decoder
    pub max_sequence_length: usize,
    /// Target language code (e.g., "en" for English)
    pub language: String,
    /// Repetition penalty applied during greedy decoding (default: 1.2)
    /// Values > 1.0 discourage repeating already-generated tokens.
    pub repetition_penalty: f32,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            n_mels: N_MELS,
            sample_rate: SAMPLE_RATE,
            max_sequence_length: 1024,
            language: "en".to_string(),
            repetition_penalty: 1.2,
        }
    }
}

// ============================================================================
// Canary Tokenizer
// ============================================================================

/// Tokenizer for Canary model using vocab.txt
#[derive(Debug, Clone)]
pub struct CanaryTokenizer {
    /// Token ID to string mapping
    id_to_token: Vec<String>,
    /// String to token ID mapping
    token_to_id: HashMap<String, i64>,
    /// Language code to token ID mapping
    lang_to_id: HashMap<String, i64>,
}

impl CanaryTokenizer {
    /// Load tokenizer from vocab.txt file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to open vocab file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut token_to_id = HashMap::new();
        let mut lang_to_id = HashMap::new();

        for line in reader.lines() {
            let line = line.map_err(|e| Error::Tokenizer(format!("Failed to read vocab: {}", e)))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Format: "token id" or just "token" (id = line number)
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            let (token, id) = if parts.len() == 2 {
                let id: i64 = parts[0]
                    .parse()
                    .map_err(|e| Error::Tokenizer(format!("Invalid token ID: {}", e)))?;
                (parts[1].to_string(), id)
            } else {
                (line.to_string(), id_to_token.len() as i64)
            };

            // Ensure vector is large enough
            if id as usize >= id_to_token.len() {
                id_to_token.resize(id as usize + 1, String::new());
            }
            id_to_token[id as usize] = token.clone();
            token_to_id.insert(token.clone(), id);

            // Track language codes (format: <|xx|> where xx is language code)
            if token.starts_with("<|") && token.ends_with("|>") && token.len() == 6 {
                let lang_code = &token[2..4];
                lang_to_id.insert(lang_code.to_string(), id);
            }
        }

        eprintln!("[CanaryTokenizer] Loaded {} tokens", id_to_token.len());
        eprintln!("[CanaryTokenizer] Found {} language codes", lang_to_id.len());

        Ok(Self {
            id_to_token,
            token_to_id,
            lang_to_id,
        })
    }

    /// Get token ID for a language code
    pub fn get_language_id(&self, lang: &str) -> i64 {
        self.lang_to_id.get(lang).copied().unwrap_or(EN_LANG_ID)
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[i64]) -> String {
        let mut result = String::new();

        for &id in token_ids {
            // Skip special tokens
            if id <= STARTOFTRANSCRIPT_ID || id == ENDOFTEXT_ID {
                continue;
            }

            if let Some(token) = self.id_to_token.get(id as usize) {
                // Skip control tokens
                if token.starts_with("<|") && token.ends_with("|>") {
                    continue;
                }

                // Handle SentencePiece-style word boundaries
                if token.starts_with('▁') {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(&token[3..]); // Skip the ▁ character (3 bytes in UTF-8)
                } else {
                    result.push_str(token);
                }
            }
        }

        result.trim().to_string()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

// ============================================================================
// Canary Model
// ============================================================================

/// Canary encoder-decoder model
pub struct CanaryModel {
    encoder: Session,
    decoder: Session,
    tokenizer: CanaryTokenizer,
    config: CanaryConfig,
}

impl CanaryModel {
    /// Load Canary model from directory
    ///
    /// Expected files:
    /// - encoder-model.onnx (or encoder-model.int8.onnx)
    /// - decoder-model.onnx (or decoder-model.int8.onnx)
    /// - vocab.txt
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<CanaryConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec_config = exec_config.unwrap_or_else(ExecutionConfig::from_env);
        let config = config.unwrap_or_default();

        // Find model files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_path = Self::find_decoder(model_dir)?;
        let vocab_path = model_dir.join("vocab.txt");

        eprintln!("[CanaryModel] Loading encoder from {:?}", encoder_path);
        eprintln!("[CanaryModel] Loading decoder from {:?}", decoder_path);

        // Load tokenizer
        let tokenizer = CanaryTokenizer::from_file(&vocab_path)?;

        // Load encoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder = builder.commit_from_file(&decoder_path)?;

        eprintln!("[CanaryModel] Models loaded successfully");
        eprintln!("[CanaryModel] Encoder inputs: {:?}", encoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        eprintln!("[CanaryModel] Decoder inputs: {:?}", decoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            config,
        })
    }

    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        // Prefer INT8 for smaller memory footprint
        let candidates = [
            "encoder-model.int8.onnx",
            "encoder-model.onnx",
            "encoder.onnx",
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

        // Apply pre-emphasis
        let audio = apply_preemphasis(audio, PREEMPHASIS);

        // Compute STFT
        let spectrogram = stft(&audio, N_FFT, HOP_LENGTH, WIN_LENGTH);

        // Create mel filterbank
        let mel_filterbank = self.create_mel_filterbank();

        // Apply mel filterbank
        let mel_spectrogram = mel_filterbank.dot(&spectrogram);

        // Log mel spectrogram
        let mel_spectrogram = mel_spectrogram.mapv(|x| (x.max(1e-10)).ln());

        // Transpose to (time, mels)
        let mel_spectrogram = mel_spectrogram.t().to_owned();

        // Normalize (mean=0, std=1 per feature)
        let mel_spectrogram = self.normalize_features(mel_spectrogram)?;

        // Add batch dimension: (1, time, mels)
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

        // Prepare input: (batch, time, mels) -> need to transpose to (batch, mels, time) for NeMo
        let batch_size = 1;
        let n_mels = features.shape()[2];

        // Transpose to (batch, mels, time)
        let features_transposed = features
            .clone()
            .into_shape((batch_size, time_steps, n_mels))
            .map_err(|e| Error::Model(format!("Failed to reshape: {}", e)))?;

        // Create transposed view: (batch, mels, time)
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

        // Extract encoder embeddings
        let embeddings = &outputs["encoder_embeddings"];
        let mask = &outputs["encoder_mask"];

        let (emb_shape, emb_data) = embeddings
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder embeddings: {}", e)))?;

        // The encoder mask is output as i64, keep it as i64 for the decoder
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

        // Keep mask as i64 (decoder expects int64)
        let encoder_mask = Array2::from_shape_vec(
            (mask_dims[0] as usize, mask_dims[1] as usize),
            mask_data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create mask array: {}", e)))?;

        Ok((encoder_out, encoder_mask))
    }

    /// Run greedy decoding
    fn greedy_decode(
        &mut self,
        encoder_embeddings: &Array3<f32>,
        encoder_mask: &Array2<i64>,
    ) -> Result<Vec<i64>> {
        let lang_id = self.tokenizer.get_language_id(&self.config.language);

        // Initialize with 9-token prompt (Canary format):
        // <|startofcontext|><|startoftranscript|><|emo:undefined|><|lang|><|lang|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>
        let initial_prompt: Vec<i64> = vec![
            STARTOFCONTEXT_ID,
            STARTOFTRANSCRIPT_ID,
            EMO_UNDEFINED_ID,
            lang_id,          // source language
            lang_id,          // target language (same for transcription)
            PNC_ID,           // punctuation enabled
            NOITN_ID,         // no inverse text normalization
            NOTIMESTAMP_ID,   // no timestamps
            NODIARIZE_ID,     // no diarization
        ];

        let mut tokens = initial_prompt.clone();

        // Note: This ONNX model doesn't have KV cache output, so we must pass ALL tokens each step
        // This is slower but necessary for correct decoding

        for step in 0..self.config.max_sequence_length {
            // Always pass all tokens - no KV caching available in this ONNX export
            let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.clone())
                .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?;

            // decoder_mems is just a dummy input with zero cache since the model ignores it anyway
            let decoder_mems = Array4::<f32>::zeros((10, 1, 0, 1024));

            let outputs = self.decoder.run(ort::inputs!(
                "input_ids" => ort::value::Value::from_array(input_ids)?,
                "encoder_embeddings" => ort::value::Value::from_array(encoder_embeddings.clone())?,
                "encoder_mask" => ort::value::Value::from_array(encoder_mask.clone())?,
                "decoder_mems" => ort::value::Value::from_array(decoder_mems)?
            ))?;

            // Extract logits
            let (logits_shape, logits_data) = outputs["logits"]
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract logits: {}", e)))?;

            let logits_dims = logits_shape.as_ref();
            let vocab_size = logits_dims[2] as usize;
            let seq_len = logits_dims[1] as usize;

            // Get logits for last position
            let last_pos_start = (seq_len - 1) * vocab_size;
            let mut last_logits: Vec<f32> = logits_data[last_pos_start..last_pos_start + vocab_size].to_vec();

            // Apply repetition penalty to already-generated tokens
            let penalty = self.config.repetition_penalty;
            if penalty != 1.0 {
                for &token_id in &tokens {
                    let idx = token_id as usize;
                    if idx < last_logits.len() {
                        if last_logits[idx] > 0.0 {
                            last_logits[idx] /= penalty;
                        } else {
                            last_logits[idx] *= penalty;
                        }
                    }
                }
            }

            // Greedy selection
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(ENDOFTEXT_ID);

            // Check for end of text
            if next_token == ENDOFTEXT_ID {
                break;
            }

            tokens.push(next_token);

            // Safety limit
            if step >= self.config.max_sequence_length - 1 {
                break;
            }
        }

        Ok(tokens)
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

        // Extract features
        let features = self.extract_features(samples)?;

        // Run encoder
        let (encoder_embeddings, encoder_mask) = self.run_encoder(&features)?;

        // Run decoder (greedy)
        let token_ids = self.greedy_decode(&encoder_embeddings, &encoder_mask)?;

        // Decode tokens to text
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CanaryConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.language, "en");
    }
}
