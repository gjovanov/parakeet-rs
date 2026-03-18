//! Voxtral true streaming transcription mode.
//!
//! Processes audio incrementally using the encoder's KV cache.
//! Every ~160ms (2 audio tokens): encode new mel frames → feed to decoder → generate text.
//! Text appears with ~480ms latency (6 delay tokens × 80ms).

use crate::error::{Error, Result};
use crate::voxtral::{VoxtralModel, VoxtralConfig, EncoderState, DecoderState};
use crate::ExecutionConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use ndarray::Array3;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;
const N_MELS: usize = 128;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const FRAMES_PER_CHUNK: usize = 16;  // Process 16 mel frames at a time (2 audio tokens, ~160ms)
const SAMPLES_PER_CHUNK: usize = FRAMES_PER_CHUNK * HOP_LENGTH;  // 2560 samples
const AUDIO_TOKEN_ID: i64 = 24;
const DELAY_TOKEN_ID: i64 = 11;
const BOS_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;

#[derive(Debug, Clone)]
pub struct VoxtralStreamingConfig {
    /// Audio tokens to buffer before starting text generation (default: 10 = 800ms)
    pub min_audio_tokens: usize,
    /// Maximum text tokens per decoder step (default: 5)
    pub max_text_per_step: usize,
    /// Number of delay tokens (default: 6 = 480ms)
    pub num_delay_tokens: usize,
    /// Language
    pub language: String,
    /// Quantization
    pub prefer_quantized: Option<bool>,
}

impl Default for VoxtralStreamingConfig {
    fn default() -> Self {
        Self {
            min_audio_tokens: 10,
            max_text_per_step: 5,
            num_delay_tokens: 6,
            language: "de".to_string(),
            prefer_quantized: None,
        }
    }
}

pub struct VoxtralStreaming {
    model: VoxtralModel,
    config: VoxtralStreamingConfig,

    // Encoder state
    enc_state: EncoderState,

    // Decoder state
    dec_state: DecoderState,

    // Audio buffering
    sample_buffer: Vec<f32>,
    total_samples: usize,
    audio_token_count: usize,

    // Text state
    text_tokens: Vec<i64>,
    current_text: String,
    last_emitted_text_len: usize,

    // Mel extraction state
    mel_filterbank: Option<ndarray::Array2<f32>>,
    hann_window: Vec<f32>,

    // Timing
    initialized: bool,
    pending_segments: Vec<TranscriptionSegment>,
}

impl VoxtralStreaming {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<VoxtralStreamingConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let voxtral_config = VoxtralConfig {
            language: config.language.clone(),
            prefer_quantized: config.prefer_quantized,
            ..Default::default()
        };
        let model = VoxtralModel::from_pretrained(model_path.as_ref(), exec_config, Some(voxtral_config))?;

        let enc_state = model.new_encoder_state();
        let dec_state = model.new_decoder_state();

        // Load mel filterbank
        let fb_path = model_path.as_ref().join("mel_filterbank.bin");
        let mel_filterbank = if fb_path.exists() {
            let data = std::fs::read(&fb_path).map_err(|e| Error::Io(e))?;
            let floats: Vec<f32> = data.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Some(ndarray::Array2::from_shape_vec((N_FFT / 2 + 1, N_MELS), floats)
                .map_err(|e| Error::Config(format!("fb reshape: {}", e)))?)
        } else { None };

        let hann_window: Vec<f32> = (0..N_FFT)
            .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos()))
            .collect();

        Ok(Self {
            model,
            config,
            enc_state,
            dec_state,
            sample_buffer: Vec::with_capacity(SAMPLE_RATE * 2),
            total_samples: 0,
            audio_token_count: 0,
            text_tokens: Vec::new(),
            current_text: String::new(),
            last_emitted_text_len: 0,
            mel_filterbank,
            hann_window,
            initialized: false,
            pending_segments: Vec::new(),
        })
    }

    fn current_time(&self) -> f32 {
        self.total_samples as f32 / SAMPLE_RATE as f32
    }

    /// Extract mel features for a chunk of audio (with center padding on first chunk)
    fn extract_mel_chunk(&self, samples: &[f32]) -> Array3<f32> {
        use rustfft::{FftPlanner, num_complex::Complex};

        let n_frames = samples.len() / HOP_LENGTH;
        let n_frames = (n_frames / 2) * 2;  // Must be even (encoder downsamples by 2)
        if n_frames == 0 {
            return Array3::zeros((1, N_MELS, 0));
        }

        let n_fft_bins = N_FFT / 2 + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let fb = self.mel_filterbank.as_ref();
        let mut mel = ndarray::Array2::<f32>::zeros((N_MELS, n_frames));

        for f in 0..n_frames {
            let start = f * HOP_LENGTH;
            let mut buf: Vec<Complex<f32>> = (0..N_FFT).map(|i| {
                let s = if start + i < samples.len() { samples[start + i] } else { 0.0 };
                Complex { re: s * self.hann_window[i], im: 0.0 }
            }).collect();
            fft.process(&mut buf);

            let power: Vec<f32> = (0..n_fft_bins)
                .map(|k| buf[k].re * buf[k].re + buf[k].im * buf[k].im)
                .collect();

            if let Some(fb) = fb {
                for m in 0..N_MELS {
                    let mut e = 0.0f32;
                    for k in 0..n_fft_bins { e += fb[[k, m]] * power[k]; }
                    mel[[m, f]] = e;
                }
            }
        }

        // Normalize: log10 → clamp → scale
        mel.mapv_inplace(|v| {
            let l = v.max(1e-10).log10();
            let c = l.max(-6.5);  // global_log_mel_max - 8.0
            (c + 4.0) / 4.0
        });

        let flat = mel.into_raw_vec();
        Array3::from_shape_vec((1, N_MELS, n_frames), flat).unwrap()
    }

    /// Process buffered samples: extract mel → encode → feed decoder → generate text
    fn process_chunks(&mut self) -> Result<()> {
        while self.sample_buffer.len() >= SAMPLES_PER_CHUNK {
            let chunk_samples: Vec<f32> = self.sample_buffer.drain(..SAMPLES_PER_CHUNK).collect();

            // Extract mel for this chunk
            let mel = self.extract_mel_chunk(&chunk_samples);
            if mel.shape()[2] == 0 { continue; }

            // Encode incrementally
            let audio_embeds = self.model.encode_chunk(&mel, &mut self.enc_state)?;
            let n_new_tokens = audio_embeds.shape()[1];
            self.audio_token_count += n_new_tokens;

            // Build decoder input: embed AUDIO placeholder tokens, ADD audio embeds
            let audio_ids: Vec<i64> = vec![AUDIO_TOKEN_ID; n_new_tokens];

            // If not initialized, prepend BOS
            let embeds = if !self.initialized {
                let mut ids = vec![BOS_TOKEN_ID];
                ids.extend_from_slice(&audio_ids);
                let text_embeds = self.model.embed_tokens_public(&ids)?;
                let mut combined = text_embeds.clone();
                // Add audio embeds at positions 1..1+n_new_tokens
                let hidden = combined.shape()[2];
                for t in 0..n_new_tokens {
                    for h in 0..hidden {
                        combined[[0, 1 + t, h]] += audio_embeds[[0, t, h]];
                    }
                }
                self.initialized = true;
                combined
            } else {
                let text_embeds = self.model.embed_tokens_public(&audio_ids)?;
                let mut combined = text_embeds.clone();
                let hidden = combined.shape()[2];
                for t in 0..n_new_tokens {
                    for h in 0..hidden {
                        combined[[0, t, h]] += audio_embeds[[0, t, h]];
                    }
                }
                combined
            };

            // Feed to decoder (processes audio tokens, updates KV cache)
            let _token = self.model.decoder_step(&embeds, &mut self.dec_state)?;
            // The token from audio processing is usually not meaningful text

            // After enough audio, try generating text via delay tokens
            if self.audio_token_count >= self.config.min_audio_tokens {
                // Feed delay tokens
                let delay_ids: Vec<i64> = vec![DELAY_TOKEN_ID; self.config.num_delay_tokens];
                let delay_embeds = self.model.embed_tokens_public(&delay_ids)?;
                let _delay_token = self.model.decoder_step(&delay_embeds, &mut self.dec_state)?;

                // Generate text tokens
                for _ in 0..self.config.max_text_per_step {
                    // The last token from decoder is a candidate
                    let last_token = if self.text_tokens.is_empty() {
                        _delay_token
                    } else {
                        *self.text_tokens.last().unwrap()
                    };

                    if last_token == EOS_TOKEN_ID { break; }
                    if last_token > 10 && last_token != AUDIO_TOKEN_ID && last_token != DELAY_TOKEN_ID {
                        self.text_tokens.push(last_token);
                        self.dec_state.generated_tokens.push(last_token);
                    }

                    // Generate next
                    let tok_embed = self.model.embed_tokens_public(&[last_token])?;
                    let next = self.model.decoder_step(&tok_embed, &mut self.dec_state)?;

                    if next == EOS_TOKEN_ID { break; }
                    if next > 10 && next != AUDIO_TOKEN_ID && next != DELAY_TOKEN_ID {
                        self.text_tokens.push(next);
                        self.dec_state.generated_tokens.push(next);
                    } else {
                        break; // Not a text token, stop generating
                    }
                }

                // Decode accumulated text tokens
                if !self.text_tokens.is_empty() {
                    if let Ok(text) = self.model.decode_tokens(&self.text_tokens) {
                        self.current_text = text;
                    }
                }
            }
        }

        Ok(())
    }
}

impl StreamingTranscriber for VoxtralStreaming {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "voxtral-4b".to_string(),
            display_name: "Voxtral 4B (Streaming)".to_string(),
            description: "Voxtral 4B with true streaming inference (~480ms latency)".to_string(),
            supports_diarization: false,
            languages: vec!["en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string()],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        self.total_samples += samples.len();
        self.sample_buffer.extend_from_slice(samples);

        self.process_chunks()?;

        // Emit PARTIAL if text has changed
        let mut segments = Vec::new();
        if self.current_text.len() > self.last_emitted_text_len {
            let new_text = self.current_text[self.last_emitted_text_len..].to_string();
            segments.push(TranscriptionSegment {
                text: self.current_text.clone(),
                raw_text: None,
                start_time: 0.0,
                end_time: self.current_time(),
                speaker: None,
                confidence: None,
                is_final: false,
                inference_time_ms: None,
            });
            self.last_emitted_text_len = self.current_text.len();

            eprintln!(
                "[VoxtralStream] PARTIAL [{:.1}s] ({} audio_tok, {} text_tok) \"{}\"",
                self.current_time(), self.audio_token_count, self.text_tokens.len(),
                &self.current_text.chars().take(80).collect::<String>()
            );
        }

        // Check for sentence boundaries → emit FINAL
        if self.current_text.ends_with('.') || self.current_text.ends_with('!') || self.current_text.ends_with('?') {
            if self.current_text.split_whitespace().count() >= 3 {
                segments.push(TranscriptionSegment {
                    text: self.current_text.clone(),
                    raw_text: None,
                    start_time: 0.0,
                    end_time: self.current_time(),
                    speaker: None,
                    confidence: None,
                    is_final: true,
                    inference_time_ms: None,
                });
                eprintln!(
                    "[VoxtralStream] FINAL [{:.1}s] \"{}\"",
                    self.current_time(), &self.current_text.chars().take(80).collect::<String>()
                );
                self.current_text.clear();
                self.last_emitted_text_len = 0;
            }
        }

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.sample_buffer.len() as f32 / SAMPLE_RATE as f32,
            total_duration: self.current_time(),
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        // Process remaining samples
        if !self.sample_buffer.is_empty() {
            self.process_chunks()?;
        }

        // Emit remaining text as final FINAL
        let mut segments = Vec::new();
        if !self.current_text.is_empty() {
            segments.push(TranscriptionSegment {
                text: self.current_text.clone(),
                raw_text: None,
                start_time: 0.0,
                end_time: self.current_time(),
                speaker: None,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            });
        }

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.current_time(),
        })
    }

    fn reset(&mut self) {
        self.enc_state = self.model.new_encoder_state();
        self.dec_state = self.model.new_decoder_state();
        self.sample_buffer.clear();
        self.total_samples = 0;
        self.audio_token_count = 0;
        self.text_tokens.clear();
        self.current_text.clear();
        self.last_emitted_text_len = 0;
        self.initialized = false;
        self.pending_segments.clear();
    }

    fn buffer_duration(&self) -> f32 {
        self.sample_buffer.len() as f32 / SAMPLE_RATE as f32
    }

    fn total_duration(&self) -> f32 {
        self.current_time()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = VoxtralStreamingConfig::default();
        assert_eq!(config.min_audio_tokens, 10);
        assert_eq!(config.num_delay_tokens, 6);
    }

    #[test]
    fn test_samples_per_chunk() {
        assert_eq!(SAMPLES_PER_CHUNK, 2560);  // 16 frames × 160 hop
    }
}
