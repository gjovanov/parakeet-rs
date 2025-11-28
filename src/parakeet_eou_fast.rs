//! Low-latency streaming ASR with ParakeetEOU
//!
//! Optimized for real-time transcription with ~500ms-1s latency.
//! Key differences from standard ParakeetEOU:
//! - Reduced buffer size (2s instead of 4s)
//! - Smaller minimum buffer (500ms instead of 1s)
//! - Smaller chunk processing (100ms instead of 160ms)
//! - Partial result support with confidence tracking

use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_eou::{EncoderCache, ParakeetEOUModel};
use ndarray::{s, Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;
const FMAX: f32 = 8000.0;

// Constants tuned for low latency while maintaining model stability
// Note: Too aggressive reduction breaks the model. These values are tested.
const BUFFER_SIZE_SECS: f32 = 3.0; // Reduced from 4s, but not too low
const BUFFER_SIZE_SAMPLES: usize = (SAMPLE_RATE as f32 * BUFFER_SIZE_SECS) as usize; // 48000

const MIN_BUFFER_MS: usize = 800; // Need enough context for stable features
const MIN_BUFFER_SAMPLES: usize = SAMPLE_RATE * MIN_BUFFER_MS / 1000; // 12800

// Encoder cache - keep closer to original for model compatibility
const PRE_ENCODE_CACHE: usize = 9; // Same as original (required for model)
const FRAMES_PER_CHUNK: usize = 16; // Same as original (required for model)
const SLICE_LEN: usize = PRE_ENCODE_CACHE + FRAMES_PER_CHUNK; // 25 frames

/// Streaming transcription result with timing and confidence
#[derive(Debug, Clone)]
pub struct StreamingResult {
    /// Transcribed text
    pub text: String,
    /// Whether this is a final result (vs partial/interim)
    pub is_final: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Start time in seconds (relative to stream start)
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Whether end-of-utterance was detected
    pub is_eou: bool,
}

impl Default for StreamingResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            is_final: false,
            confidence: 0.0,
            start_time: 0.0,
            end_time: 0.0,
            is_eou: false,
        }
    }
}

/// Configuration for low-latency streaming
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Emit partial results (may change)
    pub emit_partials: bool,
    /// Minimum confidence to emit partial result
    pub partial_confidence_threshold: f32,
    /// Reset decoder state on end-of-utterance
    pub reset_on_eou: bool,
    /// Maximum symbols per frame (prevents runaway decoding)
    pub max_symbols_per_frame: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            emit_partials: true,
            partial_confidence_threshold: 0.3,
            reset_on_eou: false,
            max_symbols_per_frame: 5,
        }
    }
}

/// Low-latency streaming ASR model
pub struct ParakeetEOUFast {
    model: ParakeetEOUModel,
    tokenizer: tokenizers::Tokenizer,
    encoder_cache: EncoderCache,
    state_h: Array3<f32>,
    state_c: Array3<f32>,
    last_token: Array2<i32>,
    blank_id: i32,
    eou_id: i32,
    mel_basis: Array2<f32>,
    window: Vec<f32>,
    audio_buffer: VecDeque<f32>,
    config: StreamingConfig,

    // Timing and state tracking
    total_samples_processed: usize,
    current_utterance_start: f32,
    accumulated_text: String,
    accumulated_confidence: f32,
    token_count: usize,
}

impl ParakeetEOUFast {
    /// Load low-latency Parakeet EOU model
    ///
    /// # Arguments
    /// * `path` - Directory containing encoder.onnx, decoder_joint.onnx, and tokenizer.json
    /// * `exec_config` - Optional execution configuration (defaults to CPU)
    /// * `streaming_config` - Optional streaming configuration
    pub fn from_pretrained<P: AsRef<Path>>(
        path: P,
        exec_config: Option<ExecutionConfig>,
        streaming_config: Option<StreamingConfig>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer_path = path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Config(format!("Failed to load tokenizer: {e}")))?;

        let vocab_size = tokenizer.get_vocab_size(true);
        // In RNN-T models, blank token is typically at vocab_size or vocab_size+1
        // The EOU model uses 1026 as blank when vocab_size is around 1025
        let blank_id = if vocab_size <= 1026 { 1026 } else { vocab_size as i32 };
        let eou_id = tokenizer
            .token_to_id("<EOU>")
            .map(|id| id as i32)
            .unwrap_or(1024);

        let exec_config = exec_config.unwrap_or_default();
        let model = ParakeetEOUModel::from_pretrained(path, exec_config)?;

        Ok(Self {
            model,
            tokenizer,
            encoder_cache: EncoderCache::new(),
            state_h: Array3::zeros((1, 1, 640)),
            state_c: Array3::zeros((1, 1, 640)),
            last_token: Array2::from_elem((1, 1), blank_id),
            blank_id,
            eou_id,
            mel_basis: Self::create_mel_filterbank(),
            window: Self::create_window(),
            audio_buffer: VecDeque::with_capacity(BUFFER_SIZE_SAMPLES),
            config: streaming_config.unwrap_or_default(),
            total_samples_processed: 0,
            current_utterance_start: 0.0,
            accumulated_text: String::new(),
            accumulated_confidence: 0.0,
            token_count: 0,
        })
    }

    /// Transcribe a chunk of audio samples with low latency
    ///
    /// # Arguments
    /// * `chunk` - Audio chunk (recommended: 100ms / 1600 samples at 16kHz)
    ///
    /// # Returns
    /// StreamingResult with text, timing, and confidence information
    pub fn transcribe(&mut self, chunk: &[f32]) -> Result<StreamingResult> {
        let chunk_start_time =
            self.total_samples_processed as f32 / SAMPLE_RATE as f32;

        // Add new chunk to rolling buffer
        self.audio_buffer.extend(chunk.iter().copied());
        self.total_samples_processed += chunk.len();

        // Trim buffer to keep only recent samples
        while self.audio_buffer.len() > BUFFER_SIZE_SAMPLES {
            self.audio_buffer.pop_front();
        }

        // Wait for minimum buffer
        if self.audio_buffer.len() < MIN_BUFFER_SAMPLES {
            return Ok(StreamingResult {
                start_time: chunk_start_time,
                end_time: self.total_samples_processed as f32 / SAMPLE_RATE as f32,
                ..Default::default()
            });
        }

        // Extract features from buffer
        let buffer_slice: Vec<f32> = self.audio_buffer.iter().copied().collect();
        let full_features = self.extract_mel_features(&buffer_slice);
        let total_frames = full_features.shape()[2];

        let start_frame = total_frames.saturating_sub(SLICE_LEN);
        let features = full_features.slice(s![.., .., start_frame..]).to_owned();
        let time_steps = features.shape()[2];

        // Debug: print feature info occasionally
        static DEBUG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = DEBUG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count < 5 || count % 100 == 0 {
            eprintln!("  [EOU DEBUG] buffer={}, total_frames={}, slice_start={}, time_steps={}",
                self.audio_buffer.len(), total_frames, start_frame, time_steps);
        }

        // Encode
        let (encoder_out, new_cache) =
            self.model
                .run_encoder(&features, time_steps as i64, &self.encoder_cache)?;
        self.encoder_cache = new_cache;

        let output_frames = encoder_out.shape()[2];

        if count < 5 || count % 100 == 0 {
            eprintln!("  [EOU DEBUG] encoder_out frames={}", output_frames);
        }

        if output_frames == 0 {
            return Ok(StreamingResult {
                text: self.accumulated_text.clone(),
                is_final: false,
                confidence: self.current_confidence(),
                start_time: self.current_utterance_start,
                end_time: self.total_samples_processed as f32 / SAMPLE_RATE as f32,
                is_eou: false,
            });
        }

        // Decode
        let mut chunk_text = String::new();
        let mut chunk_confidence_sum = 0.0f32;
        let mut chunk_token_count = 0usize;
        let mut is_eou = false;

        for t in 0..output_frames {
            let current_frame = encoder_out.slice(s![.., .., t..t + 1]).to_owned();
            let mut syms_added = 0;

            while syms_added < self.config.max_symbols_per_frame {
                let (logits, new_h, new_c) = self.model.run_decoder(
                    &current_frame,
                    &self.last_token,
                    &self.state_h,
                    &self.state_c,
                )?;

                let vocab = logits.slice(s![0, 0, ..]);

                // Find max with softmax for confidence
                let mut max_idx = 0i32;
                let mut max_val = f32::NEG_INFINITY;
                let mut sum_exp = 0.0f32;

                for (i, &val) in vocab.iter().enumerate() {
                    if val.is_finite() {
                        let exp_val = (val - 10.0).exp(); // Shift for numerical stability
                        sum_exp += exp_val;
                        if val > max_val {
                            max_val = val;
                            max_idx = i as i32;
                        }
                    }
                }

                let confidence = if sum_exp > 0.0 {
                    ((max_val - 10.0).exp() / sum_exp).min(1.0)
                } else {
                    0.0
                };

                if max_idx == self.blank_id || max_idx == 0 {
                    break;
                }

                if max_idx == self.eou_id {
                    is_eou = true;
                    if self.config.reset_on_eou {
                        self.reset_states();
                    }
                    break;
                }

                if max_idx as usize >= self.tokenizer.get_vocab_size(true) {
                    break;
                }

                // Update states
                self.state_h = new_h;
                self.state_c = new_c;
                self.last_token.fill(max_idx);

                if let Some(token) = self.tokenizer.id_to_token(max_idx as u32) {
                    let clean = token.replace('▁', " ");
                    chunk_text.push_str(&clean);
                    chunk_confidence_sum += confidence;
                    chunk_token_count += 1;
                }
                syms_added += 1;
            }

            if is_eou {
                break;
            }
        }

        // Update accumulated state
        if !chunk_text.is_empty() {
            if self.accumulated_text.is_empty() {
                self.current_utterance_start = chunk_start_time;
            }
            self.accumulated_text.push_str(&chunk_text);
            self.accumulated_confidence += chunk_confidence_sum;
            self.token_count += chunk_token_count;
        }

        let end_time = self.total_samples_processed as f32 / SAMPLE_RATE as f32;

        // Determine if this should be final
        let is_final = is_eou;

        let result = StreamingResult {
            text: self.accumulated_text.clone(),
            is_final,
            confidence: self.current_confidence(),
            start_time: self.current_utterance_start,
            end_time,
            is_eou,
        };

        // Reset accumulated text on final/EOU
        if is_final {
            self.accumulated_text.clear();
            self.accumulated_confidence = 0.0;
            self.token_count = 0;
        }

        Ok(result)
    }

    /// Get the current utterance text without processing new audio
    pub fn current_text(&self) -> &str {
        &self.accumulated_text
    }

    /// Get current confidence score
    pub fn current_confidence(&self) -> f32 {
        if self.token_count > 0 {
            (self.accumulated_confidence / self.token_count as f32).min(1.0)
        } else {
            0.0
        }
    }

    /// Force finalize current utterance
    pub fn finalize(&mut self) -> StreamingResult {
        let result = StreamingResult {
            text: std::mem::take(&mut self.accumulated_text),
            is_final: true,
            confidence: self.current_confidence(),
            start_time: self.current_utterance_start,
            end_time: self.total_samples_processed as f32 / SAMPLE_RATE as f32,
            is_eou: false,
        };

        self.accumulated_confidence = 0.0;
        self.token_count = 0;
        self.current_utterance_start = result.end_time;

        result
    }

    /// Reset all states for new stream
    pub fn reset(&mut self) {
        self.encoder_cache = EncoderCache::new();
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.last_token.fill(self.blank_id);
        self.audio_buffer.clear();
        self.total_samples_processed = 0;
        self.current_utterance_start = 0.0;
        self.accumulated_text.clear();
        self.accumulated_confidence = 0.0;
        self.token_count = 0;
    }

    fn reset_states(&mut self) {
        // Soft reset: Only reset decoder states
        self.state_h.fill(0.0);
        self.state_c.fill(0.0);
        self.last_token.fill(self.blank_id);
    }

    fn extract_mel_features(&self, audio: &[f32]) -> Array3<f32> {
        let audio_pre = Self::apply_preemphasis(audio);
        let spec = self.stft(&audio_pre);
        let mel = self.mel_basis.dot(&spec);
        let mel_log = mel.mapv(|x| (x.max(0.0) + LOG_ZERO_GUARD).ln());
        mel_log.insert_axis(ndarray::Axis(0))
    }

    fn apply_preemphasis(audio: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(audio.len());
        if audio.is_empty() {
            return result;
        }

        let safe_x = |x: f32| if x.is_finite() { x } else { 0.0 };

        result.push(safe_x(audio[0]));
        for i in 1..audio.len() {
            result.push(safe_x(audio[i]) - PREEMPH * safe_x(audio[i - 1]));
        }
        result
    }

    fn stft(&self, audio: &[f32]) -> Array2<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let pad_amount = N_FFT / 2;
        let mut padded_audio = vec![0.0; pad_amount];
        padded_audio.extend_from_slice(audio);
        padded_audio.extend(std::iter::repeat_n(0.0, pad_amount));

        let num_frames = 1 + (padded_audio.len().saturating_sub(WIN_LENGTH)) / HOP_LENGTH;
        let freq_bins = N_FFT / 2 + 1;
        let mut spec = Array2::zeros((freq_bins, num_frames));

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            if start + WIN_LENGTH > padded_audio.len() {
                break;
            }

            let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
            for i in 0..WIN_LENGTH {
                buffer[i] = Complex::new(padded_audio[start + i] * self.window[i], 0.0);
            }
            fft.process(&mut buffer);
            for (i, val) in buffer.iter().take(freq_bins).enumerate() {
                let mag_sq = val.norm_sqr();
                spec[[i, frame_idx]] = if mag_sq.is_finite() { mag_sq } else { 0.0 };
            }
        }
        spec
    }

    fn create_window() -> Vec<f32> {
        (0..WIN_LENGTH)
            .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / ((WIN_LENGTH - 1) as f32)).cos())
            .collect()
    }

    fn create_mel_filterbank() -> Array2<f32> {
        let num_freqs = N_FFT / 2 + 1;

        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(FMAX);

        let mel_points: Vec<f32> = (0..=N_MELS + 1)
            .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32))
            .collect();

        let fft_freqs: Vec<f32> = (0..num_freqs)
            .map(|i| (SAMPLE_RATE as f32 / N_FFT as f32) * i as f32)
            .collect();

        let mut weights = Array2::zeros((N_MELS, num_freqs));

        for i in 0..N_MELS {
            let left = mel_points[i];
            let center = mel_points[i + 1];
            let right = mel_points[i + 2];
            for (j, &freq) in fft_freqs.iter().enumerate() {
                if freq >= left && freq <= center {
                    weights[[i, j]] = (freq - left) / (center - left);
                } else if freq > center && freq <= right {
                    weights[[i, j]] = (right - freq) / (right - center);
                }
            }
        }

        for i in 0..N_MELS {
            let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
            for j in 0..num_freqs {
                weights[[i, j]] *= enorm;
            }
        }

        weights
    }
}

/// Recommended chunk size in samples for streaming (160ms = 2560 samples)
/// This matches the original ParakeetEOU for model compatibility
pub const RECOMMENDED_CHUNK_SAMPLES: usize = 2560;

/// Recommended chunk duration in milliseconds
pub const RECOMMENDED_CHUNK_MS: usize = 160;
