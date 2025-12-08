//! OpenAI Whisper ONNX model implementation
//!
//! Whisper is OpenAI's encoder-decoder ASR model supporting 99 languages.
//! This module provides ONNX-based inference using separately exported
//! encoder and decoder models.
//!
//! Supports: whisper-large-v2, whisper-large-v3

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use ndarray::{Array2, Array3};
use ort::session::Session;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

const SAMPLE_RATE: usize = 16000;
const N_MELS: usize = 128; // large-v2/v3 use 128 mels (v1 used 80)
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const CHUNK_LENGTH_SECS: f32 = 30.0;
const N_SAMPLES: usize = (CHUNK_LENGTH_SECS as usize) * SAMPLE_RATE; // 480000 samples = 30s

// Special token IDs (Whisper tokenizer)
const SOT_TOKEN: i64 = 50258;           // <|startoftranscript|>
const EOT_TOKEN: i64 = 50257;           // <|endoftext|>
const TRANSLATE_TOKEN: i64 = 50359;     // <|translate|>
const TRANSCRIBE_TOKEN: i64 = 50360;    // <|transcribe|>
const NO_TIMESTAMPS_TOKEN: i64 = 50364; // <|notimestamps|>
const TIMESTAMP_BEGIN: i64 = 50365;     // <|0.00|>
const NO_SPEECH_TOKEN: i64 = 50362;     // <|nospeech|>

// Language token offset (language codes start at 50259)
const LANG_TOKEN_OFFSET: i64 = 50259;

// ============================================================================
// Whisper Configuration
// ============================================================================

/// Model variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperVariant {
    LargeV2,
    LargeV3,
}

impl WhisperVariant {
    pub fn as_str(&self) -> &'static str {
        match self {
            WhisperVariant::LargeV2 => "whisper-large-v2",
            WhisperVariant::LargeV3 => "whisper-large-v3",
        }
    }

    pub fn from_path(path: &Path) -> Self {
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        if name.contains("v3") || name.contains("large-v3") {
            WhisperVariant::LargeV3
        } else {
            WhisperVariant::LargeV2
        }
    }
}

/// Configuration for Whisper model
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Model variant
    pub variant: WhisperVariant,
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// Sample rate in Hz
    pub sample_rate: usize,
    /// Maximum sequence length for decoder
    pub max_sequence_length: usize,
    /// Target language code (e.g., "en" for English, "auto" for detection)
    pub language: String,
    /// Task: "transcribe" or "translate"
    pub task: String,
    /// Whether to include timestamps
    pub timestamps: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            variant: WhisperVariant::LargeV3,
            n_mels: N_MELS,
            sample_rate: SAMPLE_RATE,
            max_sequence_length: 448,
            language: "en".to_string(),
            task: "transcribe".to_string(),
            timestamps: false,
        }
    }
}

// ============================================================================
// Whisper Tokenizer
// ============================================================================

/// Language codes supported by Whisper
const WHISPER_LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"), ("zh", "chinese"), ("de", "german"), ("es", "spanish"),
    ("ru", "russian"), ("ko", "korean"), ("fr", "french"), ("ja", "japanese"),
    ("pt", "portuguese"), ("tr", "turkish"), ("pl", "polish"), ("ca", "catalan"),
    ("nl", "dutch"), ("ar", "arabic"), ("sv", "swedish"), ("it", "italian"),
    ("id", "indonesian"), ("hi", "hindi"), ("fi", "finnish"), ("vi", "vietnamese"),
    ("he", "hebrew"), ("uk", "ukrainian"), ("el", "greek"), ("ms", "malay"),
    ("cs", "czech"), ("ro", "romanian"), ("da", "danish"), ("hu", "hungarian"),
    ("ta", "tamil"), ("no", "norwegian"), ("th", "thai"), ("ur", "urdu"),
    ("hr", "croatian"), ("bg", "bulgarian"), ("lt", "lithuanian"), ("la", "latin"),
    ("mi", "maori"), ("ml", "malayalam"), ("cy", "welsh"), ("sk", "slovak"),
    ("te", "telugu"), ("fa", "persian"), ("lv", "latvian"), ("bn", "bengali"),
    ("sr", "serbian"), ("az", "azerbaijani"), ("sl", "slovenian"), ("kn", "kannada"),
    ("et", "estonian"), ("mk", "macedonian"), ("br", "breton"), ("eu", "basque"),
    ("is", "icelandic"), ("hy", "armenian"), ("ne", "nepali"), ("mn", "mongolian"),
    ("bs", "bosnian"), ("kk", "kazakh"), ("sq", "albanian"), ("sw", "swahili"),
    ("gl", "galician"), ("mr", "marathi"), ("pa", "punjabi"), ("si", "sinhala"),
    ("km", "khmer"), ("sn", "shona"), ("yo", "yoruba"), ("so", "somali"),
    ("af", "afrikaans"), ("oc", "occitan"), ("ka", "georgian"), ("be", "belarusian"),
    ("tg", "tajik"), ("sd", "sindhi"), ("gu", "gujarati"), ("am", "amharic"),
    ("yi", "yiddish"), ("lo", "lao"), ("uz", "uzbek"), ("fo", "faroese"),
    ("ht", "haitian creole"), ("ps", "pashto"), ("tk", "turkmen"), ("nn", "nynorsk"),
    ("mt", "maltese"), ("sa", "sanskrit"), ("lb", "luxembourgish"), ("my", "myanmar"),
    ("bo", "tibetan"), ("tl", "tagalog"), ("mg", "malagasy"), ("as", "assamese"),
    ("tt", "tatar"), ("haw", "hawaiian"), ("ln", "lingala"), ("ha", "hausa"),
    ("ba", "bashkir"), ("jw", "javanese"), ("su", "sundanese"), ("yue", "cantonese"),
];

/// Tokenizer for Whisper model
#[derive(Debug, Clone)]
pub struct WhisperTokenizer {
    /// Token ID to string mapping
    vocab: HashMap<i64, String>,
    /// String to token ID mapping
    vocab_reverse: HashMap<String, i64>,
    /// Language code to token ID
    lang_to_id: HashMap<String, i64>,
    /// Byte encoder for BPE
    byte_encoder: HashMap<u8, char>,
    /// Byte decoder for BPE
    byte_decoder: HashMap<char, u8>,
}

impl WhisperTokenizer {
    /// Load tokenizer from tokenizer.json file (HuggingFace format)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Tokenizer(format!("Failed to open vocab file: {}", e)))?;

        let reader = BufReader::new(file);
        let vocab_json: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|e| Error::Tokenizer(format!("Failed to parse vocab JSON: {}", e)))?;

        let mut vocab = HashMap::new();
        let mut vocab_reverse = HashMap::new();

        // Try HuggingFace tokenizer.json format first: { "model": { "vocab": { "token": id } } }
        let vocab_obj = if let Some(model) = vocab_json.get("model") {
            model.get("vocab").and_then(|v| v.as_object())
        } else {
            // Fall back to direct format: { "token": id }
            vocab_json.as_object()
        };

        if let Some(obj) = vocab_obj {
            for (token, id) in obj {
                if let Some(id_num) = id.as_i64() {
                    vocab.insert(id_num, token.clone());
                    vocab_reverse.insert(token.clone(), id_num);
                }
            }
        }

        // Build language token mapping
        let mut lang_to_id = HashMap::new();
        for (i, (code, _name)) in WHISPER_LANGUAGES.iter().enumerate() {
            lang_to_id.insert(code.to_string(), LANG_TOKEN_OFFSET + i as i64);
        }

        // Build byte encoder/decoder for BPE
        let (byte_encoder, byte_decoder) = Self::build_byte_maps();

        eprintln!("[WhisperTokenizer] Loaded {} tokens", vocab.len());
        eprintln!("[WhisperTokenizer] Found {} language codes", lang_to_id.len());

        Ok(Self {
            vocab,
            vocab_reverse,
            lang_to_id,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Build byte-to-unicode and unicode-to-byte mappings for GPT-2 style BPE
    fn build_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        // Printable ASCII characters map to themselves
        let mut n = 0u32;
        for b in 0u8..=255 {
            let c = if (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b) {
                b as char
            } else {
                // Non-printable bytes get mapped to unicode characters starting at 256
                let c = char::from_u32(256 + n).unwrap_or('?');
                n += 1;
                c
            };
            byte_encoder.insert(b, c);
            byte_decoder.insert(c, b);
        }

        (byte_encoder, byte_decoder)
    }

    /// Get token ID for a language code
    pub fn get_language_id(&self, lang: &str) -> i64 {
        self.lang_to_id.get(lang).copied()
            .unwrap_or_else(|| self.lang_to_id.get("en").copied().unwrap_or(LANG_TOKEN_OFFSET))
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[i64]) -> String {
        let mut bytes = Vec::new();

        for &id in token_ids {
            // Skip special tokens
            if id >= SOT_TOKEN {
                continue;
            }

            if let Some(token) = self.vocab.get(&id) {
                // Convert BPE token back to bytes
                for c in token.chars() {
                    if let Some(&b) = self.byte_decoder.get(&c) {
                        bytes.push(b);
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Decode with timestamp extraction
    pub fn decode_with_timestamps(&self, token_ids: &[i64]) -> Vec<TimestampedSegment> {
        let mut segments = Vec::new();
        let mut current_text = Vec::new();
        let mut current_start: Option<f32> = None;

        for &id in token_ids {
            // Check for timestamp tokens
            if id >= TIMESTAMP_BEGIN {
                let time = (id - TIMESTAMP_BEGIN) as f32 * 0.02; // 20ms per timestamp token

                if current_start.is_none() {
                    current_start = Some(time);
                } else if !current_text.is_empty() {
                    // End of segment
                    let text = self.decode(&current_text);
                    if !text.trim().is_empty() {
                        segments.push(TimestampedSegment {
                            text: text.trim().to_string(),
                            start: current_start.unwrap(),
                            end: time,
                        });
                    }
                    current_text.clear();
                    current_start = Some(time);
                }
            } else if id < SOT_TOKEN && id != EOT_TOKEN {
                // Regular text token
                current_text.push(id);
            }
        }

        // Flush remaining text
        if !current_text.is_empty() {
            let text = self.decode(&current_text);
            if !text.trim().is_empty() {
                segments.push(TimestampedSegment {
                    text: text.trim().to_string(),
                    start: current_start.unwrap_or(0.0),
                    end: CHUNK_LENGTH_SECS,
                });
            }
        }

        segments
    }

    /// Build initial prompt tokens for transcription
    pub fn build_prompt(&self, language: &str, task: &str, timestamps: bool) -> Vec<i64> {
        let mut prompt = vec![SOT_TOKEN];

        // Add language token
        prompt.push(self.get_language_id(language));

        // Add task token
        if task == "translate" {
            prompt.push(TRANSLATE_TOKEN);
        } else {
            prompt.push(TRANSCRIBE_TOKEN);
        }

        // Add timestamp control
        if !timestamps {
            prompt.push(NO_TIMESTAMPS_TOKEN);
        }

        prompt
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// A timestamped transcription segment
#[derive(Debug, Clone)]
pub struct TimestampedSegment {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

// ============================================================================
// Mel Spectrogram Computation
// ============================================================================

/// Pre-computed mel filterbank weights
fn create_mel_filterbank(sample_rate: usize, n_fft: usize, n_mels: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    // Whisper uses HTK-style mel scale
    let min_mel = hz_to_mel_htk(0.0);
    let max_mel = hz_to_mel_htk(sample_rate as f32 / 2.0);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_htk(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32))
        .collect();

    let freq_bin_width = sample_rate as f32 / n_fft as f32;

    for mel_idx in 0..n_mels {
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

fn hz_to_mel_htk(freq: f32) -> f32 {
    2595.0 * (1.0 + freq / 700.0).log10()
}

fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Compute log-mel spectrogram from audio samples
fn compute_mel_spectrogram(
    samples: &[f32],
    sample_rate: usize,
    n_mels: usize,
) -> Result<Array2<f32>> {
    use crate::audio::stft;

    // Pad or trim to exactly 30 seconds
    let mut audio = vec![0.0f32; N_SAMPLES];
    let copy_len = samples.len().min(N_SAMPLES);
    audio[..copy_len].copy_from_slice(&samples[..copy_len]);

    // Compute STFT (no preemphasis for Whisper)
    let spectrogram = stft(&audio, N_FFT, HOP_LENGTH, N_FFT);

    // Create mel filterbank and apply
    let mel_filterbank = create_mel_filterbank(sample_rate, N_FFT, n_mels);
    let mel_spec = mel_filterbank.dot(&spectrogram);

    // Log scale with clamping (Whisper uses log10 not ln)
    let mel_spec = mel_spec.mapv(|x| (x.max(1e-10)).log10());

    // Normalize to max value and clamp
    let max_val = mel_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mel_spec = mel_spec.mapv(|x| {
        let normalized = (x - max_val).max(-8.0);
        (normalized + 4.0) / 4.0 // Scale to roughly [-1, 1]
    });

    Ok(mel_spec)
}

// ============================================================================
// Whisper Model
// ============================================================================

/// Whisper encoder-decoder model
pub struct WhisperModel {
    encoder: Session,
    decoder: Session,
    tokenizer: WhisperTokenizer,
    config: WhisperConfig,
    mel_filterbank: Array2<f32>,
    /// Cancellation token for responsive shutdown
    cancellation_token: Option<Arc<AtomicBool>>,
}

impl WhisperModel {
    /// Load Whisper model from directory
    ///
    /// Expected files:
    /// - encoder.onnx (or encoder.int8.onnx)
    /// - decoder.onnx (or decoder.int8.onnx or decoder_with_past.onnx)
    /// - vocab.json (or tokenizer.json)
    pub fn from_pretrained<P: AsRef<Path>>(
        model_dir: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<WhisperConfig>,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let exec_config = exec_config.unwrap_or_else(ExecutionConfig::from_env);
        let mut config = config.unwrap_or_default();

        // Detect variant from path
        config.variant = WhisperVariant::from_path(model_dir);

        // Find model files
        let encoder_path = Self::find_encoder(model_dir)?;
        let decoder_path = Self::find_decoder(model_dir)?;
        let vocab_path = Self::find_vocab(model_dir)?;

        eprintln!("[WhisperModel] Loading {} from {:?}", config.variant.as_str(), model_dir);
        eprintln!("[WhisperModel] Encoder: {:?}", encoder_path);
        eprintln!("[WhisperModel] Decoder: {:?}", decoder_path);

        // Load tokenizer
        let tokenizer = WhisperTokenizer::from_file(&vocab_path)?;

        // Load encoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let encoder = builder.commit_from_file(&encoder_path)?;

        // Load decoder
        let builder = Session::builder()?;
        let builder = exec_config.apply_to_session_builder(builder)?;
        let decoder = builder.commit_from_file(&decoder_path)?;

        // Pre-compute mel filterbank
        let mel_filterbank = create_mel_filterbank(config.sample_rate, N_FFT, config.n_mels);

        eprintln!("[WhisperModel] Models loaded successfully");
        eprintln!("[WhisperModel] Encoder inputs: {:?}", encoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
        eprintln!("[WhisperModel] Decoder inputs: {:?}", decoder.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            config,
            mel_filterbank,
            cancellation_token: None,
        })
    }

    fn find_encoder(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "encoder.int8.onnx",
            "encoder.onnx",
            "encoder_model.onnx",
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
            "decoder.int8.onnx",
            "decoder.onnx",
            "decoder_model.onnx",
            "decoder_with_past.onnx",
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

    fn find_vocab(dir: &Path) -> Result<PathBuf> {
        let candidates = [
            "vocab.json",
            "tokenizer.json",
            "added_tokens.json",
        ];
        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(Error::Config(format!(
            "No vocabulary file found in {}",
            dir.display()
        )))
    }

    /// Set cancellation token for responsive shutdown
    pub fn set_cancellation_token(&mut self, token: Arc<AtomicBool>) {
        self.cancellation_token = Some(token);
    }

    /// Check if transcription has been cancelled
    fn is_cancelled(&self) -> bool {
        self.cancellation_token
            .as_ref()
            .map(|t| !t.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    /// Extract mel spectrogram features from audio samples
    fn extract_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        let mel = compute_mel_spectrogram(audio, self.config.sample_rate, self.config.n_mels)?;

        let (n_mels, n_frames) = (mel.shape()[0], mel.shape()[1]);

        // Whisper expects exactly 3000 frames (30 seconds at 100 fps)
        const WHISPER_N_FRAMES: usize = 3000;

        // Pad or truncate to exactly 3000 frames
        let mel_padded = if n_frames < WHISPER_N_FRAMES {
            // Pad with zeros (silence) on the right
            let mut padded = Array2::<f32>::zeros((n_mels, WHISPER_N_FRAMES));
            padded.slice_mut(ndarray::s![.., ..n_frames]).assign(&mel);
            padded
        } else if n_frames > WHISPER_N_FRAMES {
            // Truncate to 3000 frames (take first 30 seconds)
            mel.slice(ndarray::s![.., ..WHISPER_N_FRAMES]).to_owned()
        } else {
            mel
        };

        // Reshape to (batch, n_mels, time)
        let mel_3d = mel_padded
            .into_shape((1, n_mels, WHISPER_N_FRAMES))
            .map_err(|e| Error::Audio(format!("Failed to reshape mel: {}", e)))?;

        Ok(mel_3d)
    }

    /// Run encoder on mel spectrogram
    fn run_encoder(&mut self, features: &Array3<f32>) -> Result<Array3<f32>> {
        let input_value = ort::value::Value::from_array(features.clone())?;

        let outputs = self.encoder.run(ort::inputs!(
            "input_features" => input_value
        ))?;

        // Extract encoder output (try different output names)
        let encoder_out = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("encoder_output"))
            .or_else(|| outputs.get("output"))
            .ok_or_else(|| Error::Model("No encoder output found".to_string()))?;

        let (shape, data) = encoder_out
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract encoder output: {}", e)))?;

        let dims = shape.as_ref();
        let encoder_output = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        )
        .map_err(|e| Error::Model(format!("Failed to create encoder array: {}", e)))?;

        Ok(encoder_output)
    }

    /// Run greedy decoding
    fn greedy_decode(&mut self, encoder_output: &Array3<f32>) -> Result<Vec<i64>> {
        // Build initial prompt
        let prompt = self.tokenizer.build_prompt(
            &self.config.language,
            &self.config.task,
            self.config.timestamps,
        );

        let mut tokens = prompt.clone();

        // Greedy autoregressive decoding
        for _step in 0..self.config.max_sequence_length {
            // Check for cancellation before each decoder step
            if self.is_cancelled() {
                break;
            }

            // Prepare input ids
            let seq_len = tokens.len();
            let input_ids = Array2::from_shape_vec((1, seq_len), tokens.clone())
                .map_err(|e| Error::Model(format!("Failed to create input_ids: {}", e)))?;

            // Run decoder with standard Whisper input names
            let outputs = self.decoder.run(ort::inputs!(
                "input_ids" => ort::value::Value::from_array(input_ids)?,
                "encoder_hidden_states" => ort::value::Value::from_array(encoder_output.clone())?
            ))?;

            // Extract logits
            let logits_out = outputs.get("logits")
                .or_else(|| outputs.get("output"))
                .ok_or_else(|| Error::Model("No decoder logits found".to_string()))?;

            let (logits_shape, logits_data) = logits_out
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Model(format!("Failed to extract logits: {}", e)))?;

            let logits_dims = logits_shape.as_ref();
            let vocab_size = logits_dims[2] as usize;
            let seq_len_out = logits_dims[1] as usize;

            // Get logits for last position
            let last_pos_start = (seq_len_out - 1) * vocab_size;
            let last_logits: Vec<f32> = logits_data[last_pos_start..last_pos_start + vocab_size].to_vec();

            // Greedy selection
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(EOT_TOKEN);

            // Check for end of text
            if next_token == EOT_TOKEN {
                break;
            }

            tokens.push(next_token);

            // Safety limit
            if tokens.len() >= self.config.max_sequence_length {
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
        let encoder_output = self.run_encoder(&features)?;

        // Run decoder (greedy)
        let token_ids = self.greedy_decode(&encoder_output)?;

        // Decode tokens to text
        let text = self.tokenizer.decode(&token_ids);

        Ok(text.trim().to_string())
    }

    /// Transcribe with timestamps
    pub fn transcribe_with_timestamps(&mut self, samples: &[f32]) -> Result<Vec<TimestampedSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        // Temporarily enable timestamps
        let original_timestamps = self.config.timestamps;
        self.config.timestamps = true;

        // Extract features
        let features = self.extract_features(samples)?;

        // Run encoder
        let encoder_output = self.run_encoder(&features)?;

        // Run decoder (greedy)
        let token_ids = self.greedy_decode(&encoder_output)?;

        // Restore timestamp setting
        self.config.timestamps = original_timestamps;

        // Decode with timestamps
        let segments = self.tokenizer.decode_with_timestamps(&token_ids);

        Ok(segments)
    }

    /// Set the target language
    pub fn set_language(&mut self, lang: &str) {
        self.config.language = lang.to_string();
    }

    /// Get the tokenizer reference
    pub fn tokenizer(&self) -> &WhisperTokenizer {
        &self.tokenizer
    }

    /// Get the model variant
    pub fn variant(&self) -> WhisperVariant {
        self.config.variant
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
        let config = WhisperConfig::default();
        assert_eq!(config.n_mels, 128);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.language, "en");
        assert_eq!(config.variant, WhisperVariant::LargeV3);
    }

    #[test]
    fn test_variant_detection() {
        assert_eq!(WhisperVariant::from_path(Path::new("whisper-large-v3")), WhisperVariant::LargeV3);
        assert_eq!(WhisperVariant::from_path(Path::new("whisper-large-v2")), WhisperVariant::LargeV2);
        assert_eq!(WhisperVariant::from_path(Path::new("whisper-large")), WhisperVariant::LargeV2);
    }

    #[test]
    fn test_language_tokens() {
        // Verify language code offset
        assert_eq!(LANG_TOKEN_OFFSET, 50259);
    }
}
