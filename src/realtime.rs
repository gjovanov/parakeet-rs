//! Real-time streaming transcription with speaker diarization
//!
//! Provides low-latency (~4-5 seconds) transcription with speaker attribution.
//!
//! # Architecture
//!
//! ```text
//! Audio Input -> Ring Buffer -> [ASR Thread] -> Transcription
//!                            -> [Diarization Thread] -> Speaker IDs
//!                                    |
//!                                    v
//!                              Output Merger -> Callbacks
//! ```
//!
//! # Example
//!
//! ```ignore
//! use parakeet_rs::realtime::{RealtimeTranscriber, RealtimeConfig, RealtimeCallback};
//!
//! struct MyCallback;
//! impl RealtimeCallback for MyCallback {
//!     fn on_partial(&self, result: RealtimeResult) {
//!         println!("[{}] {}", result.speaker_display(), result.text);
//!     }
//!     fn on_final(&self, result: RealtimeResult) {
//!         println!("FINAL [{}]: {}", result.speaker_display(), result.text);
//!     }
//!     fn on_speaker_update(&self, update: SpeakerUpdate) {
//!         println!("Speaker updated: {:?} -> {:?}", update.old_speaker, update.new_speaker);
//!     }
//! }
//!
//! let config = RealtimeConfig::default();
//! let mut transcriber = RealtimeTranscriber::new(config)?;
//! transcriber.start(MyCallback)?;
//!
//! // Push audio chunks as they arrive
//! transcriber.push_audio(&audio_samples)?;
//! ```

use crate::error::{Error, Result};
use crate::execution::ModelConfig;
use crate::parakeet_eou_fast::{ParakeetEOUFast, StreamingConfig, StreamingResult};
use crate::sortformer::DiarizationConfig;
use crate::sortformer_stream::SortformerStream;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Speaker identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    /// Speaker not yet identified (diarization pending)
    Unknown,
    /// Identified speaker (0-3)
    Id(usize),
}

impl Speaker {
    /// Display string for speaker
    pub fn display(&self) -> String {
        match self {
            Speaker::Unknown => "Speaker ?".to_string(),
            Speaker::Id(id) => format!("Speaker {}", id),
        }
    }
}

/// Real-time transcription result
#[derive(Debug, Clone)]
pub struct RealtimeResult {
    /// Transcribed text
    pub text: String,
    /// Speaker (may be Unknown initially)
    pub speaker: Speaker,
    /// Start time in seconds (relative to stream start)
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Whether this is a final result (vs partial/interim)
    pub is_final: bool,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Unique ID for this utterance (for tracking updates)
    pub utterance_id: u64,
}

impl RealtimeResult {
    /// Get display string for speaker
    pub fn speaker_display(&self) -> String {
        self.speaker.display()
    }
}

/// Speaker update notification (for retroactive attribution)
#[derive(Debug, Clone)]
pub struct SpeakerUpdate {
    /// Time range affected
    pub time_range: (f32, f32),
    /// Previous speaker
    pub old_speaker: Speaker,
    /// New speaker
    pub new_speaker: Speaker,
    /// Utterance ID being updated
    pub utterance_id: u64,
}

/// Callback trait for receiving real-time results
pub trait RealtimeCallback: Send + Sync {
    /// Called for partial (interim) results - may change
    fn on_partial(&self, result: RealtimeResult);

    /// Called for final results - stable, won't change
    fn on_final(&self, result: RealtimeResult);

    /// Called when speaker is retroactively identified
    fn on_speaker_update(&self, update: SpeakerUpdate);
}

/// Configuration for real-time transcription
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Path to ASR model directory (ParakeetEOU)
    pub asr_model_path: String,
    /// Path to diarization model (Sortformer ONNX)
    pub diarization_model_path: String,
    /// Execution provider configuration
    pub execution_config: Option<ModelConfig>,
    /// ASR streaming configuration
    pub streaming_config: StreamingConfig,
    /// Diarization configuration
    pub diarization_config: DiarizationConfig,
    /// Emit partial results
    pub emit_partials: bool,
    /// Chunk size in samples (recommended: 2560 = 160ms)
    pub chunk_size_samples: usize,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            asr_model_path: "./eou".to_string(),
            diarization_model_path: "diar_streaming_sortformer_4spk-v2.onnx".to_string(),
            execution_config: None,
            streaming_config: StreamingConfig::default(),
            diarization_config: DiarizationConfig::callhome(),
            emit_partials: true,
            chunk_size_samples: 2560, // 160ms at 16kHz
        }
    }
}

impl RealtimeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn asr_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.asr_model_path = path.as_ref().to_string_lossy().to_string();
        self
    }

    pub fn diarization_model_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.diarization_model_path = path.as_ref().to_string_lossy().to_string();
        self
    }

    pub fn execution_config(mut self, config: ModelConfig) -> Self {
        self.execution_config = Some(config);
        self
    }

    pub fn emit_partials(mut self, emit: bool) -> Self {
        self.emit_partials = emit;
        self
    }
}

/// Internal message types for thread communication
#[derive(Debug)]
enum AsrMessage {
    Result(StreamingResult),
    Stop,
}

#[derive(Debug)]
enum DiarMessage {
    SpeakerUpdate {
        timestamp: f32,
        speaker: Option<usize>,
    },
    Stop,
}

/// Pending utterance waiting for speaker attribution
struct PendingUtterance {
    utterance_id: u64,
    text: String,
    start_time: f32,
    end_time: f32,
    confidence: f32,
    speaker: Speaker,
    is_final: bool,
    emitted: bool,
}

/// Real-time transcriber with speaker diarization
pub struct RealtimeTranscriber {
    config: RealtimeConfig,

    // Shared state
    audio_buffer: Arc<Mutex<VecDeque<f32>>>,
    running: Arc<AtomicBool>,

    // Processing components (wrapped for thread safety)
    asr: Arc<Mutex<Option<ParakeetEOUFast>>>,
    diarization: Arc<Mutex<Option<SortformerStream>>>,

    // Worker threads
    asr_thread: Option<JoinHandle<()>>,
    diar_thread: Option<JoinHandle<()>>,

    // Pending utterances for speaker attribution
    pending_utterances: Vec<PendingUtterance>,
    next_utterance_id: u64,

    // Timing
    total_samples: usize,
}

impl RealtimeTranscriber {
    /// Create a new real-time transcriber
    pub fn new(config: RealtimeConfig) -> Result<Self> {
        // Initialize ASR model
        let asr = ParakeetEOUFast::from_pretrained(
            &config.asr_model_path,
            config.execution_config.clone(),
            Some(config.streaming_config.clone()),
        )?;

        // Initialize diarization model
        let diarization = SortformerStream::with_config(
            &config.diarization_model_path,
            config.execution_config.clone(),
            config.diarization_config.clone(),
        )?;

        Ok(Self {
            config,
            audio_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(32000))),
            running: Arc::new(AtomicBool::new(false)),
            asr: Arc::new(Mutex::new(Some(asr))),
            diarization: Arc::new(Mutex::new(Some(diarization))),
            asr_thread: None,
            diar_thread: None,
            pending_utterances: Vec::new(),
            next_utterance_id: 0,
            total_samples: 0,
        })
    }

    /// Start processing with callback
    pub fn start<C: RealtimeCallback + 'static>(&mut self, callback: C) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(Error::Config("Already running".to_string()));
        }

        self.running.store(true, Ordering::SeqCst);

        let callback = Arc::new(callback);

        // Start ASR processing thread
        let asr = self.asr.clone();
        let audio_buffer = self.audio_buffer.clone();
        let running = self.running.clone();
        let emit_partials = self.config.emit_partials;
        let chunk_size = self.config.chunk_size_samples;
        let callback_asr = callback.clone();

        self.asr_thread = Some(thread::spawn(move || {
            Self::asr_thread_fn(asr, audio_buffer, running, emit_partials, chunk_size, callback_asr);
        }));

        // Start diarization processing thread
        let diar = self.diarization.clone();
        let audio_buffer = self.audio_buffer.clone();
        let running = self.running.clone();
        let callback_diar = callback;

        self.diar_thread = Some(thread::spawn(move || {
            Self::diar_thread_fn(diar, audio_buffer, running, callback_diar);
        }));

        Ok(())
    }

    /// Push audio samples for processing
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Err(Error::Config("Not running".to_string()));
        }

        let mut buffer = self.audio_buffer.lock().map_err(|_| Error::Config("Lock error".to_string()))?;
        buffer.extend(samples.iter().copied());
        self.total_samples += samples.len();

        Ok(())
    }

    /// Stop processing
    pub fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.asr_thread.take() {
            let _ = handle.join();
        }

        if let Some(handle) = self.diar_thread.take() {
            let _ = handle.join();
        }

        Ok(())
    }

    /// Check if processing is active
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get total audio duration processed
    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / 16000.0
    }

    /// Reset for new stream
    pub fn reset(&mut self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            self.stop()?;
        }

        // Reset ASR
        if let Ok(mut asr_guard) = self.asr.lock() {
            if let Some(ref mut asr) = *asr_guard {
                asr.reset();
            }
        }

        // Reset diarization
        if let Ok(mut diar_guard) = self.diarization.lock() {
            if let Some(ref mut diar) = *diar_guard {
                diar.reset();
            }
        }

        // Clear state
        if let Ok(mut buffer) = self.audio_buffer.lock() {
            buffer.clear();
        }
        self.pending_utterances.clear();
        self.next_utterance_id = 0;
        self.total_samples = 0;

        Ok(())
    }

    /// ASR processing thread function
    fn asr_thread_fn<C: RealtimeCallback>(
        asr: Arc<Mutex<Option<ParakeetEOUFast>>>,
        audio_buffer: Arc<Mutex<VecDeque<f32>>>,
        running: Arc<AtomicBool>,
        emit_partials: bool,
        chunk_size: usize,
        callback: Arc<C>,
    ) {
        let mut utterance_id = 0u64;
        let mut last_text = String::new();

        while running.load(Ordering::SeqCst) {
            // Get audio chunk from buffer
            let chunk: Option<Vec<f32>> = {
                let mut buffer = match audio_buffer.lock() {
                    Ok(b) => b,
                    Err(_) => continue,
                };

                if buffer.len() >= chunk_size {
                    Some(buffer.drain(..chunk_size).collect())
                } else {
                    None
                }
            };

            if let Some(chunk) = chunk {
                // Process through ASR
                if let Ok(mut asr_guard) = asr.lock() {
                    if let Some(ref mut asr_model) = *asr_guard {
                        match asr_model.transcribe(&chunk) {
                            Ok(result) => {
                                if !result.text.is_empty() {
                                    // Check if text changed (for partial updates)
                                    let text_changed = result.text != last_text;

                                    if result.is_final {
                                        // Emit final result
                                        let realtime_result = RealtimeResult {
                                            text: result.text.clone(),
                                            speaker: Speaker::Unknown, // Will be updated by diarization
                                            start_time: result.start_time,
                                            end_time: result.end_time,
                                            is_final: true,
                                            confidence: result.confidence,
                                            utterance_id,
                                        };
                                        callback.on_final(realtime_result);
                                        utterance_id += 1;
                                        last_text.clear();
                                    } else if emit_partials && text_changed {
                                        // Emit partial result
                                        let realtime_result = RealtimeResult {
                                            text: result.text.clone(),
                                            speaker: Speaker::Unknown,
                                            start_time: result.start_time,
                                            end_time: result.end_time,
                                            is_final: false,
                                            confidence: result.confidence,
                                            utterance_id,
                                        };
                                        callback.on_partial(realtime_result);
                                        last_text = result.text;
                                    }
                                }
                            }
                            Err(_) => continue,
                        }
                    }
                }
            } else {
                // No audio available, sleep briefly
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    /// Diarization processing thread function
    fn diar_thread_fn<C: RealtimeCallback>(
        diar: Arc<Mutex<Option<SortformerStream>>>,
        audio_buffer: Arc<Mutex<VecDeque<f32>>>,
        running: Arc<AtomicBool>,
        callback: Arc<C>,
    ) {
        // Diarization needs more audio, so we use larger chunks
        let diar_chunk_size = 3200; // 200ms chunks for diarization
        let mut accumulated_samples = Vec::new();
        let mut last_speaker: Option<usize> = None;

        while running.load(Ordering::SeqCst) {
            // Peek at audio buffer (don't consume - ASR needs it too)
            let samples: Vec<f32> = {
                let buffer = match audio_buffer.lock() {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                buffer.iter().copied().collect()
            };

            // Accumulate samples for diarization
            if samples.len() > accumulated_samples.len() {
                let new_samples = &samples[accumulated_samples.len()..];
                accumulated_samples.extend_from_slice(new_samples);
            }

            // Process when we have enough new samples
            if accumulated_samples.len() >= diar_chunk_size * 50 {
                // ~10 seconds
                if let Ok(mut diar_guard) = diar.lock() {
                    if let Some(ref mut diar_model) = *diar_guard {
                        // Push accumulated audio
                        if let Ok(state) = diar_model.push_audio(&accumulated_samples) {
                            // Check for speaker change
                            if state.current_speaker != last_speaker {
                                let new_speaker = state.current_speaker.map(Speaker::Id).unwrap_or(Speaker::Unknown);
                                let old_speaker = last_speaker.map(Speaker::Id).unwrap_or(Speaker::Unknown);

                                if last_speaker.is_some() || state.current_speaker.is_some() {
                                    callback.on_speaker_update(SpeakerUpdate {
                                        time_range: (state.speaker_start_time, state.last_update_time),
                                        old_speaker,
                                        new_speaker,
                                        utterance_id: 0, // Would need better tracking
                                    });
                                }

                                last_speaker = state.current_speaker;
                            }
                        }
                    }
                }
            }

            // Sleep to avoid busy-waiting
            thread::sleep(Duration::from_millis(100));
        }
    }
}

impl Drop for RealtimeTranscriber {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Simple synchronous transcriber for cases where threading is not needed
pub struct SimplifiedRealtimeTranscriber {
    asr: ParakeetEOUFast,
    diarization: SortformerStream,
    total_samples: usize,
    current_utterance_id: u64,
}

impl SimplifiedRealtimeTranscriber {
    /// Create a new simplified transcriber
    pub fn new(config: RealtimeConfig) -> Result<Self> {
        let asr = ParakeetEOUFast::from_pretrained(
            &config.asr_model_path,
            config.execution_config.clone(),
            Some(config.streaming_config.clone()),
        )?;

        let diarization = SortformerStream::with_config(
            &config.diarization_model_path,
            config.execution_config.clone(),
            config.diarization_config.clone(),
        )?;

        Ok(Self {
            asr,
            diarization,
            total_samples: 0,
            current_utterance_id: 0,
        })
    }

    /// Process audio chunk and return result
    ///
    /// This is a synchronous method that processes the audio
    /// and returns the transcription with speaker attribution.
    pub fn process(&mut self, samples: &[f32]) -> Result<RealtimeResult> {
        self.total_samples += samples.len();

        // Process ASR
        let asr_result = self.asr.transcribe(samples)?;

        // Process diarization
        let _ = self.diarization.push_audio(samples)?;

        // Get current speaker
        let current_time = self.total_samples as f32 / 16000.0;
        let speaker = self
            .diarization
            .get_speaker_at(current_time)
            .map(Speaker::Id)
            .unwrap_or(Speaker::Unknown);

        // Build result
        let result = RealtimeResult {
            text: asr_result.text,
            speaker,
            start_time: asr_result.start_time,
            end_time: asr_result.end_time,
            is_final: asr_result.is_final,
            confidence: asr_result.confidence,
            utterance_id: self.current_utterance_id,
        };

        if asr_result.is_final {
            self.current_utterance_id += 1;
        }

        Ok(result)
    }

    /// Force finalize current utterance
    pub fn finalize(&mut self) -> RealtimeResult {
        let asr_result = self.asr.finalize();
        let current_time = self.total_samples as f32 / 16000.0;
        let speaker = self
            .diarization
            .get_speaker_at(current_time)
            .map(Speaker::Id)
            .unwrap_or(Speaker::Unknown);

        let result = RealtimeResult {
            text: asr_result.text,
            speaker,
            start_time: asr_result.start_time,
            end_time: asr_result.end_time,
            is_final: true,
            confidence: asr_result.confidence,
            utterance_id: self.current_utterance_id,
        };

        self.current_utterance_id += 1;
        result
    }

    /// Reset for new stream
    pub fn reset(&mut self) {
        self.asr.reset();
        self.diarization.reset();
        self.total_samples = 0;
        self.current_utterance_id = 0;
    }

    /// Get total audio duration processed
    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / 16000.0
    }
}
