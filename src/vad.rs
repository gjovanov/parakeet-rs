//! Silero Voice Activity Detection (VAD) module
//!
//! Provides low-latency speech detection using the Silero VAD ONNX model.
//! The model processes audio in chunks and returns a probability of speech presence.
//!
//! ## Performance
//! - Model size: ~2.3 MB
//! - Latency: <1ms per 32ms audio chunk on CPU
//! - Supports 8kHz and 16kHz sample rates
//!
//! ## Usage
//! ```rust,ignore
//! let mut vad = SileroVad::new("silero_vad.onnx", None)?;
//! let probability = vad.process(&audio_chunk)?;
//! if probability > 0.5 {
//!     // Speech detected
//! }
//! ```

use crate::error::{Error, Result};
use crate::execution::ModelConfig;
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use std::path::Path;

/// Sample rate for VAD processing
pub const VAD_SAMPLE_RATE: usize = 16000;

/// Chunk size in samples for 16kHz (512 samples = 32ms)
pub const VAD_CHUNK_SIZE: usize = 512;

/// Configuration for VAD-triggered transcription
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Probability threshold for speech detection (0.0 - 1.0)
    pub speech_threshold: f32,

    /// Minimum silence duration (ms) before counting as a pause
    pub silence_trigger_ms: u32,

    /// Minimum speech duration (ms) before accepting as valid speech
    pub min_speech_ms: u32,

    /// Maximum speech duration (seconds) before forcing transcription
    pub max_speech_secs: f32,

    /// Padding (ms) to add before detected speech start
    pub speech_pad_start_ms: u32,

    /// Padding (ms) to add after detected speech end
    pub speech_pad_end_ms: u32,

    /// Number of consecutive speech frames required to confirm speech start
    pub speech_confirm_frames: u32,

    /// Number of consecutive silence frames required to confirm speech end
    pub silence_confirm_frames: u32,

    /// Maximum number of pauses before triggering transcription (0 = disabled)
    /// Transcription triggers on first pause if max_pauses == 1
    pub max_pauses: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            silence_trigger_ms: 500,
            min_speech_ms: 250,
            max_speech_secs: 30.0,
            speech_pad_start_ms: 300,  // 300ms to capture word starts
            speech_pad_end_ms: 250,    // 250ms to capture word ends
            speech_confirm_frames: 2,
            silence_confirm_frames: 8, // ~256ms at 32ms per frame
            max_pauses: 0, // Disabled by default (transcribe on first pause)
        }
    }
}

impl VadConfig {
    /// Create a speedy VAD config (faster response, still quality-focused)
    pub fn speedy() -> Self {
        Self {
            speech_threshold: 0.4,
            silence_trigger_ms: 300,
            min_speech_ms: 150,
            max_speech_secs: 20.0,
            speech_pad_start_ms: 300,  // 300ms to capture word starts
            speech_pad_end_ms: 250,    // 250ms to capture word ends like "zurück"
            speech_confirm_frames: 1,
            silence_confirm_frames: 5,
            max_pauses: 0,
        }
    }

    /// Create a pause-based VAD config for sentence-level segmentation
    /// Optimized for continuous news speech with natural sentence boundaries
    pub fn pause_based() -> Self {
        Self {
            speech_threshold: 0.35,    // Lower threshold to catch softer speech
            silence_trigger_ms: 350,   // 350ms pause = sentence boundary
            min_speech_ms: 200,        // Minimum speech duration
            max_speech_secs: 8.0,      // 8s max before hard cut
            speech_pad_start_ms: 300,  // 300ms padding to catch word starts
            speech_pad_end_ms: 200,    // 200ms padding to catch word ends
            speech_confirm_frames: 1,  // Fast speech confirmation
            silence_confirm_frames: 8, // ~256ms of silence confirmation
            max_pauses: 3,             // Allow 2 pauses within segment, emit on 3rd
        }
    }

    /// Create a low-latency VAD config
    pub fn low_latency() -> Self {
        Self {
            speech_threshold: 0.45,
            silence_trigger_ms: 400,
            min_speech_ms: 200,
            max_speech_secs: 25.0,
            speech_pad_start_ms: 250,  // 250ms to capture word starts
            speech_pad_end_ms: 200,    // 200ms to capture word ends
            speech_confirm_frames: 2,
            silence_confirm_frames: 6,
            max_pauses: 0,
        }
    }

    /// Create an ultra-low-latency VAD config (fastest, may cut words)
    pub fn ultra_low_latency() -> Self {
        Self {
            speech_threshold: 0.35,
            silence_trigger_ms: 250,
            min_speech_ms: 100,
            max_speech_secs: 15.0,
            speech_pad_start_ms: 200,  // 200ms minimum to capture word starts
            speech_pad_end_ms: 150,    // 150ms to capture word ends
            speech_confirm_frames: 1,
            silence_confirm_frames: 4,
            max_pauses: 0,
        }
    }

    /// Create an extreme-low-latency VAD config (fastest, least accurate)
    pub fn extreme_low_latency() -> Self {
        Self {
            speech_threshold: 0.3,
            silence_trigger_ms: 150,
            min_speech_ms: 50,
            max_speech_secs: 10.0,
            speech_pad_start_ms: 20,
            speech_pad_end_ms: 20,
            speech_confirm_frames: 1,
            silence_confirm_frames: 3,
            max_pauses: 0,
        }
    }

    /// Create VadConfig from mode string
    pub fn from_mode(mode: &str) -> Self {
        match mode {
            "speedy" => Self::speedy(),
            "pause_based" => Self::pause_based(),
            "low_latency" => Self::low_latency(),
            "ultra_low_latency" => Self::ultra_low_latency(),
            "extreme_low_latency" => Self::extreme_low_latency(),
            "lookahead" => Self::pause_based(), // Similar to pause-based
            _ => Self::speedy(), // Default
        }
    }

    /// Convert silence trigger to samples
    pub fn silence_trigger_samples(&self) -> usize {
        (self.silence_trigger_ms as usize * VAD_SAMPLE_RATE) / 1000
    }

    /// Convert min speech to samples
    pub fn min_speech_samples(&self) -> usize {
        (self.min_speech_ms as usize * VAD_SAMPLE_RATE) / 1000
    }

    /// Convert max speech to samples
    pub fn max_speech_samples(&self) -> usize {
        (self.max_speech_secs * VAD_SAMPLE_RATE as f32) as usize
    }

    /// Convert start padding to samples
    pub fn speech_pad_start_samples(&self) -> usize {
        (self.speech_pad_start_ms as usize * VAD_SAMPLE_RATE) / 1000
    }

    /// Convert end padding to samples
    pub fn speech_pad_end_samples(&self) -> usize {
        (self.speech_pad_end_ms as usize * VAD_SAMPLE_RATE) / 1000
    }
}

/// Context size for Silero VAD (last N samples from previous chunk)
const VAD_CONTEXT_SIZE: usize = 64;

/// Silero VAD model wrapper
///
/// Processes audio chunks and returns speech probability.
/// Maintains internal LSTM state across calls for streaming processing.
pub struct SileroVad {
    session: Session,
    /// LSTM state tensor [2, 1, 128]
    state: ndarray::Array3<f32>,
    /// Sample rate (8000 or 16000)
    sample_rate: i64,
    /// Total samples processed
    samples_processed: usize,
    /// Context buffer from previous chunk
    context: Vec<f32>,
}

impl SileroVad {
    /// Create a new Silero VAD instance
    ///
    /// # Arguments
    /// * `model_path` - Path to silero_vad.onnx model file
    /// * `config` - Optional execution config (CPU/GPU settings)
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: Option<ModelConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        // Use the project's apply_to_session_builder method
        let builder = SessionBuilder::new()?;
        let session = config.apply_to_session_builder(builder)?
            .commit_from_file(model_path)?;

        // Initialize LSTM state tensor [2, 1, 128] (2 layers for h and c, 1 batch, 128 hidden units)
        // Silero VAD v5 uses 128 hidden units
        let state = ndarray::Array3::<f32>::zeros((2, 1, 128));

        Ok(Self {
            session,
            state,
            sample_rate: 16000,
            samples_processed: 0,
            context: vec![0.0f32; VAD_CONTEXT_SIZE],
        })
    }

    /// Process an audio chunk and return speech probability
    ///
    /// # Arguments
    /// * `samples` - Audio samples (should be VAD_CHUNK_SIZE = 512 samples for 16kHz)
    ///
    /// # Returns
    /// Speech probability between 0.0 and 1.0
    pub fn process(&mut self, samples: &[f32]) -> Result<f32> {
        // Prepare chunk with context window prepended (as per C++ reference implementation)
        // Input to model is: [context_64_samples] + [current_512_samples] = 576 samples
        let input_size = VAD_CONTEXT_SIZE + VAD_CHUNK_SIZE;
        let mut chunk = vec![0.0f32; input_size];

        // Copy context from previous chunk
        chunk[..VAD_CONTEXT_SIZE].copy_from_slice(&self.context);

        // Copy current samples
        let copy_len = samples.len().min(VAD_CHUNK_SIZE);
        chunk[VAD_CONTEXT_SIZE..VAD_CONTEXT_SIZE + copy_len].copy_from_slice(&samples[..copy_len]);

        // Update context for next call (last 64 samples of current chunk)
        if samples.len() >= VAD_CONTEXT_SIZE {
            self.context.copy_from_slice(&samples[samples.len() - VAD_CONTEXT_SIZE..]);
        } else {
            // Shift context and add new samples
            let shift = VAD_CONTEXT_SIZE - samples.len();
            self.context.copy_within(samples.len().., 0);
            self.context[shift..].copy_from_slice(samples);
        }

        // Create input tensor: [1, context_size + chunk_size]
        let input = Array2::from_shape_vec((1, input_size), chunk)
            .map_err(|e| Error::Model(format!("Shape error: {}", e)))?;

        // Sample rate tensor
        let sr = Array1::from_vec(vec![self.sample_rate]);

        // Create ORT values
        let input_value = ort::value::Value::from_array(input)?;
        let sr_value = ort::value::Value::from_array(sr)?;
        let state_value = ort::value::Value::from_array(self.state.clone())?;

        // Run inference
        // Silero VAD inputs: input, state, sr
        // Outputs: output (probability), stateN (updated state)
        let outputs = self.session.run(ort::inputs!(
            "input" => input_value,
            "state" => state_value,
            "sr" => sr_value,
        ))?;

        // Extract output probability
        let (_, output_data) = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract VAD output: {e}")))?;
        let probability = output_data[0];

        // Update LSTM state
        let (_, state_data) = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract stateN: {e}")))?;

        self.state = ndarray::Array3::from_shape_vec(
            (2, 1, 128),
            state_data.to_vec(),
        ).map_err(|e| Error::Model(format!("State shape error: {}", e)))?;

        self.samples_processed += samples.len();

        Ok(probability)
    }

    /// Process multiple chunks and return probabilities
    pub fn process_batch(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut probs = Vec::new();
        for chunk in samples.chunks(VAD_CHUNK_SIZE) {
            probs.push(self.process(chunk)?);
        }
        Ok(probs)
    }

    /// Reset LSTM state (call between utterances or on new audio stream)
    pub fn reset(&mut self) {
        self.state = ndarray::Array3::<f32>::zeros((2, 1, 128));
        self.context = vec![0.0f32; VAD_CONTEXT_SIZE];
        self.samples_processed = 0;
    }

    /// Get total samples processed
    pub fn samples_processed(&self) -> usize {
        self.samples_processed
    }

    /// Get current time in seconds
    pub fn current_time(&self) -> f32 {
        self.samples_processed as f32 / VAD_SAMPLE_RATE as f32
    }
}

/// VAD state machine for detecting speech segments
#[derive(Debug, Clone)]
pub enum VadState {
    /// Waiting for speech to start
    Idle,

    /// Potential speech detected, confirming
    MaybeSpeech {
        /// Samples collected during confirmation
        samples: Vec<f32>,
        /// Number of consecutive speech frames
        speech_frames: u32,
        /// Start time of potential speech
        start_time: f32,
    },

    /// Speech confirmed, accumulating
    Speaking {
        /// All speech samples
        samples: Vec<f32>,
        /// Start time of speech
        start_time: f32,
        /// Number of pauses detected so far
        pause_count: u32,
    },

    /// Potential end of speech, counting silence
    MaybeSilence {
        /// All speech samples so far
        speech_samples: Vec<f32>,
        /// Silence samples (for padding)
        silence_samples: Vec<f32>,
        /// Number of consecutive silence frames
        silence_frames: u32,
        /// Start time of speech
        speech_start: f32,
        /// Number of pauses detected before this one
        pause_count: u32,
    },
}

/// Result from VAD processing
#[derive(Debug, Clone)]
pub struct VadSegment {
    /// Audio samples for this speech segment
    pub samples: Vec<f32>,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
}

/// VAD-based audio segmenter
///
/// Accumulates audio and detects speech segments based on voice activity.
pub struct VadSegmenter {
    vad: SileroVad,
    config: VadConfig,
    state: VadState,
    /// Pre-speech buffer for padding
    pre_buffer: Vec<f32>,
    /// Total samples received
    total_samples: usize,
    /// Pending segments ready for transcription
    pending_segments: Vec<VadSegment>,
}

impl VadSegmenter {
    /// Create a new VAD segmenter
    pub fn new<P: AsRef<Path>>(
        vad_model_path: P,
        config: VadConfig,
        exec_config: Option<ModelConfig>,
    ) -> Result<Self> {
        let vad = SileroVad::new(vad_model_path, exec_config)?;

        // Pre-buffer size based on start padding
        let pre_buffer_size = config.speech_pad_start_samples();

        Ok(Self {
            vad,
            config,
            state: VadState::Idle,
            pre_buffer: Vec::with_capacity(pre_buffer_size),
            total_samples: 0,
            pending_segments: Vec::new(),
        })
    }

    /// Push audio samples and detect speech segments
    ///
    /// Returns any completed speech segments ready for transcription.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>> {
        // Process in VAD_CHUNK_SIZE chunks
        for chunk in samples.chunks(VAD_CHUNK_SIZE) {
            let prob = self.vad.process(chunk)?;
            self.update_state(chunk, prob)?;
        }

        self.total_samples += samples.len();

        // Return and clear pending segments
        Ok(std::mem::take(&mut self.pending_segments))
    }

    /// Force finalization of any pending speech
    pub fn finalize(&mut self) -> Result<Option<VadSegment>> {
        match std::mem::replace(&mut self.state, VadState::Idle) {
            VadState::Speaking { samples, start_time, .. } => {
                if samples.len() >= self.config.min_speech_samples() {
                    let end_time = start_time + (samples.len() as f32 / VAD_SAMPLE_RATE as f32);
                    return Ok(Some(VadSegment {
                        samples,
                        start_time,
                        end_time,
                    }));
                }
            }
            VadState::MaybeSilence { speech_samples, silence_samples, speech_start, .. } => {
                let mut samples = speech_samples;
                // Add silence padding
                let pad_samples = silence_samples.len().min(self.config.speech_pad_end_samples());
                samples.extend_from_slice(&silence_samples[..pad_samples]);

                if samples.len() >= self.config.min_speech_samples() {
                    let end_time = speech_start + (samples.len() as f32 / VAD_SAMPLE_RATE as f32);
                    return Ok(Some(VadSegment {
                        samples,
                        start_time: speech_start,
                        end_time,
                    }));
                }
            }
            VadState::MaybeSpeech { samples, start_time, .. } => {
                // Not enough confirmation, but include if long enough
                if samples.len() >= self.config.min_speech_samples() {
                    let end_time = start_time + (samples.len() as f32 / VAD_SAMPLE_RATE as f32);
                    return Ok(Some(VadSegment {
                        samples,
                        start_time,
                        end_time,
                    }));
                }
            }
            VadState::Idle => {}
        }

        self.vad.reset();
        Ok(None)
    }

    /// Reset the segmenter state
    pub fn reset(&mut self) {
        self.state = VadState::Idle;
        self.pre_buffer.clear();
        self.total_samples = 0;
        self.pending_segments.clear();
        self.vad.reset();
    }

    /// Get current time in seconds
    pub fn current_time(&self) -> f32 {
        self.total_samples as f32 / VAD_SAMPLE_RATE as f32
    }

    fn update_state(&mut self, chunk: &[f32], prob: f32) -> Result<()> {
        let is_speech = prob > self.config.speech_threshold;
        let current_time = self.current_time();

        match &mut self.state {
            VadState::Idle => {
                // Maintain pre-buffer for padding
                self.pre_buffer.extend_from_slice(chunk);
                let max_pre = self.config.speech_pad_start_samples();
                if self.pre_buffer.len() > max_pre {
                    let drain_count = self.pre_buffer.len() - max_pre;
                    self.pre_buffer.drain(..drain_count);
                }

                if is_speech {
                    // Potential speech start
                    let mut samples = std::mem::take(&mut self.pre_buffer);
                    samples.extend_from_slice(chunk);

                    self.state = VadState::MaybeSpeech {
                        samples,
                        speech_frames: 1,
                        start_time: current_time - (self.config.speech_pad_start_ms as f32 / 1000.0),
                    };
                }
            }

            VadState::MaybeSpeech { samples, speech_frames, start_time } => {
                samples.extend_from_slice(chunk);

                if is_speech {
                    *speech_frames += 1;

                    if *speech_frames >= self.config.speech_confirm_frames {
                        // Speech confirmed
                        self.state = VadState::Speaking {
                            samples: std::mem::take(samples),
                            start_time: *start_time,
                            pause_count: 0,
                        };
                    }
                } else {
                    // False alarm, go back to idle
                    // Keep some samples in pre-buffer
                    let keep = samples.len().min(self.config.speech_pad_start_samples());
                    self.pre_buffer = samples[samples.len() - keep..].to_vec();
                    self.state = VadState::Idle;
                }
            }

            VadState::Speaking { samples, start_time, pause_count } => {
                samples.extend_from_slice(chunk);

                // Check for max speech duration
                if samples.len() >= self.config.max_speech_samples() {
                    // Force transcription
                    let segment_samples = std::mem::take(samples);
                    let end_time = *start_time + (segment_samples.len() as f32 / VAD_SAMPLE_RATE as f32);

                    self.pending_segments.push(VadSegment {
                        samples: segment_samples,
                        start_time: *start_time,
                        end_time,
                    });

                    self.state = VadState::Idle;
                    self.vad.reset(); // Reset LSTM state between segments
                } else if !is_speech {
                    // Potential end of speech
                    self.state = VadState::MaybeSilence {
                        speech_samples: std::mem::take(samples),
                        silence_samples: chunk.to_vec(),
                        silence_frames: 1,
                        speech_start: *start_time,
                        pause_count: *pause_count,
                    };
                }
            }

            VadState::MaybeSilence {
                speech_samples,
                silence_samples,
                silence_frames,
                speech_start,
                pause_count
            } => {
                let silence_duration_ms = (*silence_frames as f32 * VAD_CHUNK_SIZE as f32 / VAD_SAMPLE_RATE as f32) * 1000.0;
                let pause_detected = silence_duration_ms >= self.config.silence_trigger_ms as f32;

                if is_speech {
                    // Speech resumed - check if this silence counted as a pause
                    if pause_detected && self.config.max_pauses > 0 {
                        let new_pause_count = *pause_count + 1;

                        if new_pause_count >= self.config.max_pauses {
                            // Max pauses reached - transcribe
                            let mut segment_samples = std::mem::take(speech_samples);

                            // Add silence padding
                            let pad_samples = silence_samples.len().min(self.config.speech_pad_end_samples());
                            segment_samples.extend_from_slice(&silence_samples[..pad_samples]);

                            if segment_samples.len() >= self.config.min_speech_samples() {
                                let end_time = *speech_start + (segment_samples.len() as f32 / VAD_SAMPLE_RATE as f32);

                                self.pending_segments.push(VadSegment {
                                    samples: segment_samples,
                                    start_time: *speech_start,
                                    end_time,
                                });
                            }

                            // Start fresh with the new speech
                            let mut new_samples = std::mem::take(&mut self.pre_buffer);
                            new_samples.extend_from_slice(chunk);
                            self.state = VadState::Speaking {
                                samples: new_samples,
                                start_time: current_time,
                                pause_count: 0,
                            };
                            self.vad.reset();
                        } else {
                            // Continue accumulating speech with incremented pause count
                            speech_samples.extend_from_slice(silence_samples);
                            speech_samples.extend_from_slice(chunk);

                            self.state = VadState::Speaking {
                                samples: std::mem::take(speech_samples),
                                start_time: *speech_start,
                                pause_count: new_pause_count,
                            };
                        }
                    } else {
                        // No pause counting (max_pauses == 0) or short silence - just resume
                        speech_samples.extend_from_slice(silence_samples);
                        speech_samples.extend_from_slice(chunk);

                        self.state = VadState::Speaking {
                            samples: std::mem::take(speech_samples),
                            start_time: *speech_start,
                            pause_count: *pause_count,
                        };
                    }
                } else {
                    silence_samples.extend_from_slice(chunk);
                    *silence_frames += 1;

                    // When max_pauses is enabled, use longer silence for end-of-speech detection
                    // This allows pauses within speech without immediately transcribing
                    let end_of_speech_ms = if self.config.max_pauses > 0 {
                        // Use 2x silence_trigger for end-of-speech when pause counting is enabled
                        self.config.silence_trigger_ms as f32 * 2.0
                    } else {
                        // Original behavior: transcribe on first pause
                        self.config.silence_trigger_ms as f32
                    };

                    let updated_silence_ms = (*silence_frames as f32 * VAD_CHUNK_SIZE as f32 / VAD_SAMPLE_RATE as f32) * 1000.0;

                    if updated_silence_ms >= end_of_speech_ms {
                        // End of speech confirmed
                        let mut segment_samples = std::mem::take(speech_samples);

                        // Add silence padding
                        let pad_samples = silence_samples.len().min(self.config.speech_pad_end_samples());
                        segment_samples.extend_from_slice(&silence_samples[..pad_samples]);

                        // Only emit if long enough
                        if segment_samples.len() >= self.config.min_speech_samples() {
                            let end_time = *speech_start + (segment_samples.len() as f32 / VAD_SAMPLE_RATE as f32);

                            self.pending_segments.push(VadSegment {
                                samples: segment_samples,
                                start_time: *speech_start,
                                end_time,
                            });
                        }

                        self.state = VadState::Idle;
                        self.vad.reset(); // Reset LSTM state between segments
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if currently in speech
    pub fn is_speaking(&self) -> bool {
        matches!(self.state, VadState::Speaking { .. } | VadState::MaybeSilence { .. })
    }

    /// Get current state name for debugging
    pub fn state_name(&self) -> &'static str {
        match &self.state {
            VadState::Idle => "idle",
            VadState::MaybeSpeech { .. } => "maybe_speech",
            VadState::Speaking { .. } => "speaking",
            VadState::MaybeSilence { .. } => "maybe_silence",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_modes() {
        let speedy = VadConfig::speedy();
        let pause = VadConfig::pause_based();

        assert!(speedy.silence_trigger_ms < pause.silence_trigger_ms);
        assert!(speedy.speech_threshold < pause.speech_threshold);
    }

    #[test]
    fn test_vad_config_from_mode() {
        let config = VadConfig::from_mode("speedy");
        assert_eq!(config.silence_trigger_ms, 300);

        let config = VadConfig::from_mode("pause_based");
        assert_eq!(config.silence_trigger_ms, 350);
        assert_eq!(config.max_pauses, 3);
        assert_eq!(config.max_speech_secs, 8.0);
    }
}
