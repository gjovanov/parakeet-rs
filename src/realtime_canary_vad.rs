//! VAD-triggered Canary transcription with optional segment buffering and diarization
//!
//! Uses Silero VAD to detect speech segments, then transcribes complete utterances
//! with the Canary model. Supports two modes:
//!
//! ## Immediate Mode (default)
//! - Transcribes each VAD segment immediately after pause detection
//! - Lower latency, but may produce fragmented short transcripts
//!
//! ## Buffered Mode
//! - Buffers VAD segments until minimum duration is reached
//! - Better quality by giving ASR more context
//! - Transcribes when: buffer >= min_duration OR long pause OR max_duration reached
//!
//! ## Benefits
//! - Lower CPU usage (transcription only on pauses)
//! - Better accuracy (complete utterances, no mid-word cuts)
//! - Natural sentence boundaries
//! - Optional speaker diarization per segment

use crate::canary::{CanaryConfig, CanaryModel};
use crate::error::Result;
use crate::execution::ModelConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::vad::{VadConfig, VadSegmenter, VadSegment, VAD_SAMPLE_RATE};
use std::path::Path;

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

/// Buffer mode for VAD transcription
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadBufferMode {
    /// Immediate: transcribe each VAD segment as it arrives
    Immediate,
    /// Buffered: accumulate segments until min/max duration reached
    Buffered,
    /// Sliding window: keep N segments, transcribe window, slide forward
    SlidingWindow,
}

impl Default for VadBufferMode {
    fn default() -> Self {
        VadBufferMode::Immediate
    }
}

/// Configuration for VAD-triggered Canary
#[derive(Debug, Clone)]
pub struct RealtimeCanaryVadConfig {
    /// VAD configuration
    pub vad: VadConfig,
    /// Target language code
    pub language: String,
    /// Buffer mode
    pub buffer_mode: VadBufferMode,
    /// Minimum buffer duration (seconds) before transcribing (0 = immediate mode)
    pub min_buffer_duration: f32,
    /// Maximum buffer duration (seconds) before forcing transcription
    pub max_buffer_duration: f32,
    /// Long pause threshold (seconds) - forces transcription even if min_buffer not reached
    pub long_pause_threshold: f32,
    /// Whether to enable diarization
    pub enable_diarization: bool,
    /// Maximum segments in sliding window (only for SlidingWindow mode)
    pub max_window_segments: usize,
    /// Overlap segments to keep after sliding (only for SlidingWindow mode)
    pub overlap_segments: usize,
}

impl Default for RealtimeCanaryVadConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            language: "en".to_string(),
            buffer_mode: VadBufferMode::Immediate,
            min_buffer_duration: 0.0, // Immediate mode by default
            max_buffer_duration: 15.0,
            long_pause_threshold: 1.5,
            enable_diarization: true,
            max_window_segments: 10,
            overlap_segments: 2,
        }
    }
}

impl RealtimeCanaryVadConfig {
    /// Create config from mode string
    pub fn from_mode(mode: &str, language: String) -> Self {
        Self {
            vad: VadConfig::from_mode(mode),
            language,
            ..Default::default()
        }
    }

    /// Create buffered mode config for sentence-level transcription
    /// Balanced for German: allows complete sentences with subordinate clauses
    pub fn buffered(language: String) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            buffer_mode: VadBufferMode::Buffered,
            min_buffer_duration: 1.5,  // Minimum 1.5s of speech for quality
            max_buffer_duration: 6.0,  // Allow up to 6s for complete German sentences
            long_pause_threshold: 1.0, // 1s pause = sentence boundary
            enable_diarization: true,
            max_window_segments: 10,
            overlap_segments: 2,
        }
    }

    /// Create buffered mode with custom durations
    pub fn buffered_custom(language: String, min_secs: f32, max_secs: f32, long_pause_secs: f32) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            buffer_mode: VadBufferMode::Buffered,
            min_buffer_duration: min_secs,
            max_buffer_duration: max_secs,
            long_pause_threshold: long_pause_secs,
            enable_diarization: true,
            max_window_segments: 10,
            overlap_segments: 2,
        }
    }

    /// Create sliding window mode config
    ///
    /// Uses buffered mode with aggressive limits to prevent segment accumulation:
    /// - Max 2 segments or 5 seconds before transcription
    /// - Transcribes on any pause >= 350ms
    /// - No overlap (clear buffer after each transcription)
    pub fn sliding_window(language: String) -> Self {
        Self {
            vad: VadConfig::pause_based(),  // Better sentence boundaries (350ms pause)
            language,
            buffer_mode: VadBufferMode::Buffered,
            min_buffer_duration: 1.0,   // At least 1s before transcribing
            max_buffer_duration: 5.0,   // Max 5s to keep inference fast
            long_pause_threshold: 0.35, // Transcribe on 350ms pause
            enable_diarization: true,
            max_window_segments: 2,     // Max 2 segments
            overlap_segments: 0,        // No overlap - clear buffer after transcription
        }
    }

    /// Create sliding window mode with custom parameters
    pub fn sliding_window_custom(
        language: String,
        max_segments: usize,
        max_duration: f32,
        overlap: usize,
    ) -> Self {
        Self {
            vad: VadConfig::pause_based(),
            language,
            buffer_mode: VadBufferMode::SlidingWindow,
            min_buffer_duration: 0.0,
            max_buffer_duration: max_duration,
            long_pause_threshold: 0.5,
            enable_diarization: true,
            max_window_segments: max_segments,
            overlap_segments: overlap,
        }
    }
}

/// Buffered VAD segment with audio samples
#[derive(Debug, Clone)]
struct BufferedSegment {
    samples: Vec<f32>,
    start_time: f32,
    end_time: f32,
}

/// VAD-triggered Canary transcriber with optional diarization
///
/// Uses voice activity detection to segment audio into utterances,
/// then transcribes each complete utterance with the Canary model.
/// Supports buffered mode for better quality transcription.
#[cfg(feature = "sortformer")]
pub struct RealtimeCanaryVad {
    model: CanaryModel,
    diarizer: Option<SortformerStream>,
    segmenter: VadSegmenter,
    config: RealtimeCanaryVadConfig,

    /// Total samples received
    total_samples: usize,

    /// Pending transcription segments (output)
    pending_segments: Vec<TranscriptionSegment>,

    /// Buffered VAD segments waiting to be transcribed
    segment_buffer: Vec<BufferedSegment>,

    /// Total duration of buffered audio (seconds)
    buffer_duration: f32,

    /// Time since last speech ended (for detecting long pauses)
    silence_duration: f32,

    /// Buffer mode from config
    buffer_mode: VadBufferMode,

    /// Last transcription text for sliding window deduplication
    last_window_text: String,

    /// Number of segments already "committed" in sliding window
    committed_segment_count: usize,

    /// End time of last emitted segment (for monotonic timestamps)
    last_emitted_end_time: f32,
}

/// VAD-triggered Canary transcriber (without sortformer)
#[cfg(not(feature = "sortformer"))]
pub struct RealtimeCanaryVad {
    model: CanaryModel,
    segmenter: VadSegmenter,
    config: RealtimeCanaryVadConfig,

    /// Total samples received
    total_samples: usize,

    /// Pending transcription segments (output)
    pending_segments: Vec<TranscriptionSegment>,

    /// Buffered VAD segments waiting to be transcribed
    segment_buffer: Vec<BufferedSegment>,

    /// Total duration of buffered audio (seconds)
    buffer_duration: f32,

    /// Time since last speech ended (for detecting long pauses)
    silence_duration: f32,

    /// Buffer mode from config
    buffer_mode: VadBufferMode,

    /// Last transcription text for sliding window deduplication
    last_window_text: String,

    /// Number of segments already "committed" in sliding window
    committed_segment_count: usize,

    /// End time of last emitted segment (for monotonic timestamps)
    last_emitted_end_time: f32,
}

#[cfg(feature = "sortformer")]
impl RealtimeCanaryVad {
    /// Create a new VAD-triggered Canary transcriber with optional diarization
    ///
    /// # Arguments
    /// * `canary_model_path` - Path to Canary ONNX model directory
    /// * `diar_model_path` - Optional path to diarization model
    /// * `vad_model_path` - Path to silero_vad.onnx
    /// * `exec_config` - Optional execution config (CPU/GPU)
    /// * `config` - Optional transcriber config
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>, P3: AsRef<Path>>(
        canary_model_path: P1,
        diar_model_path: Option<P2>,
        vad_model_path: P3,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeCanaryVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffer_mode = config.buffer_mode;

        // Create Canary model
        let canary_config = CanaryConfig {
            language: config.language.clone(),
            ..Default::default()
        };
        let model = CanaryModel::from_pretrained(
            canary_model_path,
            exec_config.clone(),
            Some(canary_config),
        )?;

        // Create diarizer if path provided and enabled
        let diarizer = if config.enable_diarization {
            if let Some(diar_path) = diar_model_path {
                eprintln!("[CanaryVAD] Creating diarizer from {:?}", diar_path.as_ref());
                Some(SortformerStream::with_config(
                    diar_path,
                    exec_config.clone(),
                    Default::default(),
                )?)
            } else {
                eprintln!("[CanaryVAD] Diarization enabled but no model path provided");
                None
            }
        } else {
            None
        };

        // Create VAD segmenter
        let segmenter = VadSegmenter::new(
            vad_model_path,
            config.vad.clone(),
            exec_config,
        )?;

        match buffer_mode {
            VadBufferMode::Buffered => {
                eprintln!(
                    "[CanaryVAD] Buffered mode: min={:.1}s, max={:.1}s, long_pause={:.1}s, diarization={}",
                    config.min_buffer_duration,
                    config.max_buffer_duration,
                    config.long_pause_threshold,
                    diarizer.is_some()
                );
            }
            VadBufferMode::SlidingWindow => {
                eprintln!(
                    "[CanaryVAD] Sliding window mode: max_segments={}, max_duration={:.1}s, overlap={}, diarization={}",
                    config.max_window_segments,
                    config.max_buffer_duration,
                    config.overlap_segments,
                    diarizer.is_some()
                );
            }
            VadBufferMode::Immediate => {
                eprintln!("[CanaryVAD] Immediate mode: diarization={}", diarizer.is_some());
            }
        }

        Ok(Self {
            model,
            diarizer,
            segmenter,
            config,
            total_samples: 0,
            pending_segments: Vec::new(),
            segment_buffer: Vec::new(),
            buffer_duration: 0.0,
            silence_duration: 0.0,
            buffer_mode,
            last_window_text: String::new(),
            committed_segment_count: 0,
            last_emitted_end_time: 0.0,
        })
    }

    /// Check if diarization is available
    pub fn has_diarization(&self) -> bool {
        self.diarizer.is_some()
    }
}

#[cfg(not(feature = "sortformer"))]
impl RealtimeCanaryVad {
    /// Create a new VAD-triggered Canary transcriber (without diarization)
    ///
    /// # Arguments
    /// * `canary_model_path` - Path to Canary ONNX model directory
    /// * `vad_model_path` - Path to silero_vad.onnx
    /// * `exec_config` - Optional execution config (CPU/GPU)
    /// * `config` - Optional transcriber config
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        canary_model_path: P1,
        vad_model_path: P2,
        exec_config: Option<ModelConfig>,
        config: Option<RealtimeCanaryVadConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let buffer_mode = config.buffer_mode;

        // Create Canary model
        let canary_config = CanaryConfig {
            language: config.language.clone(),
            ..Default::default()
        };
        let model = CanaryModel::from_pretrained(
            canary_model_path,
            exec_config.clone(),
            Some(canary_config),
        )?;

        // Create VAD segmenter
        let segmenter = VadSegmenter::new(
            vad_model_path,
            config.vad.clone(),
            exec_config,
        )?;

        match buffer_mode {
            VadBufferMode::Buffered => {
                eprintln!(
                    "[CanaryVAD] Buffered mode: min={:.1}s, max={:.1}s, long_pause={:.1}s",
                    config.min_buffer_duration,
                    config.max_buffer_duration,
                    config.long_pause_threshold
                );
            }
            VadBufferMode::SlidingWindow => {
                eprintln!(
                    "[CanaryVAD] Sliding window mode: max_segments={}, max_duration={:.1}s, overlap={}",
                    config.max_window_segments,
                    config.max_buffer_duration,
                    config.overlap_segments
                );
            }
            VadBufferMode::Immediate => {
                eprintln!("[CanaryVAD] Immediate mode");
            }
        }

        Ok(Self {
            model,
            segmenter,
            config,
            total_samples: 0,
            pending_segments: Vec::new(),
            segment_buffer: Vec::new(),
            buffer_duration: 0.0,
            silence_duration: 0.0,
            buffer_mode,
            last_window_text: String::new(),
            committed_segment_count: 0,
            last_emitted_end_time: 0.0,
        })
    }

    /// Check if diarization is available (always false without sortformer)
    pub fn has_diarization(&self) -> bool {
        false
    }
}

impl RealtimeCanaryVad {

    /// Push audio samples and get transcription results
    ///
    /// Audio is accumulated until a speech pause is detected,
    /// then the complete utterance is transcribed.
    ///
    /// In buffered mode, segments are accumulated until:
    /// - Buffer duration >= min_buffer_duration, OR
    /// - A long pause (>= long_pause_threshold) is detected, OR
    /// - Buffer duration >= max_buffer_duration
    ///
    /// In sliding window mode:
    /// - Accumulates up to max_window_segments OR max_buffer_duration
    /// - Transcribes entire window, then slides forward keeping overlap_segments
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<CanaryVadResult> {
        self.total_samples += samples.len();
        let chunk_duration = samples.len() as f32 / VAD_SAMPLE_RATE as f32;

        // Push audio to diarizer for speaker tracking (sortformer feature)
        #[cfg(feature = "sortformer")]
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        // Get any completed speech segments from VAD
        let vad_segments = self.segmenter.push_audio(samples)?;

        match self.buffer_mode {
            VadBufferMode::Buffered => {
                self.process_buffered_mode(vad_segments, chunk_duration)?;
            }
            VadBufferMode::SlidingWindow => {
                self.process_sliding_window_mode(vad_segments, chunk_duration)?;
            }
            VadBufferMode::Immediate => {
                // Immediate mode: transcribe each segment as it arrives
                for segment in vad_segments {
                    self.transcribe_segment_immediate(segment)?;
                }
            }
        }

        // Return and clear pending segments
        let segments = std::mem::take(&mut self.pending_segments);

        Ok(CanaryVadResult {
            segments,
            is_speaking: self.segmenter.is_speaking(),
            total_duration: self.total_duration(),
        })
    }

    /// Process audio in buffered mode
    fn process_buffered_mode(&mut self, vad_segments: Vec<VadSegment>, chunk_duration: f32) -> Result<()> {
        let is_speaking = self.segmenter.is_speaking();

        // Track silence duration when not speaking
        if !is_speaking && self.segment_buffer.is_empty() {
            // No buffered speech, just track silence
            self.silence_duration += chunk_duration;
        } else if !is_speaking && !self.segment_buffer.is_empty() {
            // We have buffered speech and now silence
            self.silence_duration += chunk_duration;
        } else {
            // Speaking - reset silence counter
            self.silence_duration = 0.0;
        }

        // Add new segments to buffer
        for segment in vad_segments {
            let seg_duration = segment.end_time - segment.start_time;
            self.buffer_duration += seg_duration;
            self.segment_buffer.push(BufferedSegment {
                samples: segment.samples,
                start_time: segment.start_time,
                end_time: segment.end_time,
            });
            // Reset silence counter when we get a new segment
            self.silence_duration = 0.0;
        }

        // Decide whether to transcribe the buffer
        // Minimum speech required before we'll consider transcribing (avoid single words)
        const MIN_SPEECH_FOR_TRANSCRIPTION: f32 = 1.0; // At least 1 second of speech

        let should_transcribe = if self.segment_buffer.is_empty() {
            false
        } else if self.buffer_duration >= self.config.max_buffer_duration {
            // Max buffer reached - force transcription
            eprintln!("[CanaryVAD] Max buffer reached ({:.1}s), transcribing", self.buffer_duration);
            true
        } else if self.silence_duration >= self.config.long_pause_threshold
                  && self.buffer_duration >= MIN_SPEECH_FOR_TRANSCRIPTION {
            // Long pause detected AND we have enough speech to transcribe
            eprintln!(
                "[CanaryVAD] Long pause ({:.1}s >= {:.1}s), transcribing {:.1}s buffer",
                self.silence_duration,
                self.config.long_pause_threshold,
                self.buffer_duration
            );
            true
        } else if self.buffer_duration >= self.config.min_buffer_duration && !is_speaking {
            // Min buffer reached and not currently speaking
            eprintln!(
                "[CanaryVAD] Min buffer reached ({:.1}s), transcribing",
                self.buffer_duration
            );
            true
        } else {
            false
        };

        if should_transcribe {
            self.transcribe_buffer()?;
        }

        Ok(())
    }

    /// Process audio in sliding window mode
    fn process_sliding_window_mode(&mut self, vad_segments: Vec<VadSegment>, chunk_duration: f32) -> Result<()> {
        let is_speaking = self.segmenter.is_speaking();

        // Track silence duration
        if !is_speaking {
            self.silence_duration += chunk_duration;
        } else {
            self.silence_duration = 0.0;
        }

        // Add new segments to buffer
        for segment in vad_segments {
            let seg_duration = segment.end_time - segment.start_time;
            self.buffer_duration += seg_duration;
            self.segment_buffer.push(BufferedSegment {
                samples: segment.samples,
                start_time: segment.start_time,
                end_time: segment.end_time,
            });
            self.silence_duration = 0.0;
        }

        // Check if window is full (by segment count OR duration)
        let window_full_by_segments = self.segment_buffer.len() >= self.config.max_window_segments;
        let window_full_by_duration = self.buffer_duration >= self.config.max_buffer_duration;

        // Also trigger on pause - don't wait for window to fill
        let pause_trigger = self.silence_duration >= self.config.long_pause_threshold;

        // Transcribe when: (window is full OR pause detected) AND not speaking AND have segments
        let should_transcribe = !self.segment_buffer.is_empty()
            && !is_speaking
            && (window_full_by_segments || window_full_by_duration || pause_trigger);

        if should_transcribe {
            let reason = if window_full_by_segments {
                "segments full"
            } else if window_full_by_duration {
                "duration full"
            } else {
                "pause detected"
            };
            eprintln!(
                "[CanaryVAD] Sliding window transcribe ({}): {} segments, {:.1}s duration",
                reason,
                self.segment_buffer.len(),
                self.buffer_duration
            );
            self.transcribe_sliding_window()?;
        }

        Ok(())
    }

    /// Transcribe the sliding window and emit only new content
    fn transcribe_sliding_window(&mut self) -> Result<()> {
        if self.segment_buffer.is_empty() {
            return Ok(());
        }

        // Combine all buffered segments into one audio stream
        let start_time = self.segment_buffer.first().unwrap().start_time;
        let end_time = self.segment_buffer.last().unwrap().end_time;

        let total_samples: usize = self.segment_buffer.iter().map(|s| s.samples.len()).sum();
        let mut combined_samples = Vec::with_capacity(total_samples);

        for segment in &self.segment_buffer {
            combined_samples.extend_from_slice(&segment.samples);
        }

        eprintln!(
            "[CanaryVAD] Transcribing sliding window: {} segments, {:.2}s - {:.2}s ({} samples)",
            self.segment_buffer.len(),
            start_time,
            end_time,
            combined_samples.len()
        );

        // Transcribe the combined audio
        let text = self.model.transcribe(&combined_samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            // Get speaker from diarizer if available
            #[cfg(feature = "sortformer")]
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (start_time + end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };
            #[cfg(not(feature = "sortformer"))]
            let speaker: Option<usize> = None;

            // Find new text by comparing with previous window transcription
            let new_text = self.extract_new_text(&text);

            if !new_text.is_empty() {
                // Use the actual segment timing from the newest segment in the buffer
                // This ensures monotonically increasing timestamps
                let newest_segment = self.segment_buffer.last().unwrap();
                let new_start = self.last_emitted_end_time.max(newest_segment.start_time);
                let new_end = newest_segment.end_time;

                eprintln!(
                    "[CanaryVAD] Sliding window NEW text [{:.2}s - {:.2}s] Speaker {:?}: \"{}\"",
                    new_start, new_end, speaker, new_text
                );

                self.pending_segments.push(TranscriptionSegment {
                    text: new_text,
                    start_time: new_start,
                    end_time: new_end,
                    speaker,
                    confidence: None,
                    is_final: true,
                    inference_time_ms: None,  // Set by transcription handler
                });

                // Update last emitted time for monotonic sequence
                self.last_emitted_end_time = new_end;
            }

            // Update last window text for next comparison
            self.last_window_text = text;
        }

        // Slide the window: keep overlap_segments for context
        self.slide_window();

        Ok(())
    }

    /// Extract new text that wasn't in the previous transcription
    fn extract_new_text(&self, current_text: &str) -> String {
        if self.last_window_text.is_empty() {
            return current_text.to_string();
        }

        // Find the longest common suffix of last_window_text that matches
        // a prefix of current_text, then return what's after that
        let last_words: Vec<&str> = self.last_window_text.split_whitespace().collect();
        let current_words: Vec<&str> = current_text.split_whitespace().collect();

        // Try to find overlap by looking for last_window words at the start of current
        let mut best_overlap = 0;
        for overlap_len in 1..=last_words.len().min(current_words.len()) {
            let last_suffix = &last_words[last_words.len() - overlap_len..];
            let current_prefix = &current_words[..overlap_len];

            if last_suffix == current_prefix {
                best_overlap = overlap_len;
            }
        }

        // Return everything after the overlap
        if best_overlap > 0 && best_overlap < current_words.len() {
            current_words[best_overlap..].join(" ")
        } else if best_overlap == 0 {
            // No overlap found, return all current text
            current_text.to_string()
        } else {
            // Complete overlap (no new text)
            String::new()
        }
    }

    /// Slide the window forward, keeping overlap_segments for context
    fn slide_window(&mut self) {
        let overlap = self.config.overlap_segments;
        let to_remove = self.segment_buffer.len().saturating_sub(overlap);

        if to_remove > 0 {
            // Remove older segments
            let removed: Vec<_> = self.segment_buffer.drain(..to_remove).collect();

            // Update buffer duration
            let removed_duration: f32 = removed.iter()
                .map(|s| s.end_time - s.start_time)
                .sum();
            self.buffer_duration = (self.buffer_duration - removed_duration).max(0.0);

            // Track committed segments
            self.committed_segment_count += to_remove;

            eprintln!(
                "[CanaryVAD] Slid window: removed {} segments, keeping {} for context, buffer now {:.1}s",
                to_remove,
                self.segment_buffer.len(),
                self.buffer_duration
            );
        }
    }

    /// Transcribe a single VAD segment immediately (non-buffered mode)
    fn transcribe_segment_immediate(&mut self, segment: VadSegment) -> Result<()> {
        // Transcribe the speech segment
        let text = self.model.transcribe(&segment.samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            // Get speaker from diarizer if available (sortformer feature)
            #[cfg(feature = "sortformer")]
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (segment.start_time + segment.end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };
            #[cfg(not(feature = "sortformer"))]
            let speaker: Option<usize> = None;

            eprintln!(
                "[CanaryVAD] Transcribed segment [{:.2}s - {:.2}s] Speaker {:?}: \"{}\"",
                segment.start_time, segment.end_time, speaker, text
            );

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time: segment.start_time,
                end_time: segment.end_time,
                speaker,
                confidence: None,
                is_final: true, // VAD segments are always final (complete utterances)
                inference_time_ms: None,
            });
        }

        Ok(())
    }

    /// Transcribe all buffered segments together (buffered mode)
    fn transcribe_buffer(&mut self) -> Result<()> {
        if self.segment_buffer.is_empty() {
            return Ok(());
        }

        // Combine all buffered segments into one audio stream
        let start_time = self.segment_buffer.first().unwrap().start_time;
        let end_time = self.segment_buffer.last().unwrap().end_time;

        // Calculate total samples and combine
        let total_samples: usize = self.segment_buffer.iter().map(|s| s.samples.len()).sum();
        let mut combined_samples = Vec::with_capacity(total_samples);

        for segment in &self.segment_buffer {
            combined_samples.extend_from_slice(&segment.samples);
        }

        eprintln!(
            "[CanaryVAD] Transcribing buffer: {} segments, {:.2}s - {:.2}s ({} samples)",
            self.segment_buffer.len(),
            start_time,
            end_time,
            combined_samples.len()
        );

        // Transcribe the combined audio
        let text = self.model.transcribe(&combined_samples)?;
        let text = text.trim().to_string();

        if !text.is_empty() {
            // Get speaker from diarizer if available (sortformer feature)
            #[cfg(feature = "sortformer")]
            let speaker = if let Some(diarizer) = &self.diarizer {
                let mid_time = (start_time + end_time) / 2.0;
                diarizer.get_speaker_at(mid_time)
            } else {
                None
            };
            #[cfg(not(feature = "sortformer"))]
            let speaker: Option<usize> = None;

            eprintln!(
                "[CanaryVAD] Buffered transcription [{:.2}s - {:.2}s] Speaker {:?}: \"{}\"",
                start_time, end_time, speaker, text
            );

            self.pending_segments.push(TranscriptionSegment {
                text,
                start_time,
                end_time,
                speaker,
                confidence: None,
                is_final: true,
                inference_time_ms: None,
            });
        }

        // Clear buffer
        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;

        Ok(())
    }

    /// Finalize any pending speech
    pub fn finalize(&mut self) -> Result<CanaryVadResult> {
        // Get any remaining speech from VAD
        if let Some(segment) = self.segmenter.finalize()? {
            match self.buffer_mode {
                VadBufferMode::Buffered | VadBufferMode::SlidingWindow => {
                    // Add to buffer
                    let seg_duration = segment.end_time - segment.start_time;
                    self.buffer_duration += seg_duration;
                    self.segment_buffer.push(BufferedSegment {
                        samples: segment.samples,
                        start_time: segment.start_time,
                        end_time: segment.end_time,
                    });
                }
                VadBufferMode::Immediate => {
                    self.transcribe_segment_immediate(segment)?;
                }
            }
        }

        // Transcribe any remaining buffer
        if !self.segment_buffer.is_empty() {
            eprintln!(
                "[CanaryVAD] Finalizing: transcribing remaining {:.1}s buffer ({} segments)",
                self.buffer_duration,
                self.segment_buffer.len()
            );
            match self.buffer_mode {
                VadBufferMode::SlidingWindow => {
                    self.transcribe_sliding_window()?;
                }
                VadBufferMode::Buffered => {
                    self.transcribe_buffer()?;
                }
                VadBufferMode::Immediate => {}
            }
        }

        let segments = std::mem::take(&mut self.pending_segments);

        Ok(CanaryVadResult {
            segments,
            is_speaking: false,
            total_duration: self.total_duration(),
        })
    }

    /// Reset the transcriber state
    pub fn reset(&mut self) {
        self.segmenter.reset();
        self.total_samples = 0;
        self.pending_segments.clear();
        self.segment_buffer.clear();
        self.buffer_duration = 0.0;
        self.silence_duration = 0.0;
        self.last_window_text.clear();
        self.committed_segment_count = 0;
        self.last_emitted_end_time = 0.0;
    }

    /// Get total audio duration in seconds
    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / VAD_SAMPLE_RATE as f32
    }

    /// Check if currently detecting speech
    pub fn is_speaking(&self) -> bool {
        self.segmenter.is_speaking()
    }

    /// Set the target language
    pub fn set_language(&mut self, lang: &str) {
        self.model.set_language(lang);
        self.config.language = lang.to_string();
    }

    /// Get VAD state name for debugging
    pub fn vad_state(&self) -> &'static str {
        self.segmenter.state_name()
    }
}

/// Result from VAD-triggered Canary processing
#[derive(Debug, Clone)]
pub struct CanaryVadResult {
    /// Completed transcription segments
    pub segments: Vec<TranscriptionSegment>,
    /// Whether speech is currently being detected
    pub is_speaking: bool,
    /// Total audio duration processed
    pub total_duration: f32,
}

// ============================================================================
// StreamingTranscriber implementation
// ============================================================================

impl StreamingTranscriber for RealtimeCanaryVad {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "canary-1b-vad".to_string(),
            display_name: "Canary 1B v2 (VAD)".to_string(),
            description: "NVIDIA Canary with Silero VAD for utterance detection".to_string(),
            supports_diarization: self.has_diarization(),
            languages: vec![
                "en".to_string(), "de".to_string(), "fr".to_string(), "es".to_string(),
                "it".to_string(), "pt".to_string(), "nl".to_string(), "pl".to_string(),
            ],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryVad::push_audio(self, samples)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0, // VAD doesn't have a fixed buffer
            total_duration: result.total_duration,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        let result = RealtimeCanaryVad::finalize(self)?;

        Ok(StreamingChunkResult {
            segments: result.segments,
            buffer_duration: 0.0,
            total_duration: result.total_duration,
        })
    }

    fn reset(&mut self) {
        RealtimeCanaryVad::reset(self);
    }

    fn buffer_duration(&self) -> f32 {
        0.0 // VAD doesn't use a fixed buffer
    }

    fn total_duration(&self) -> f32 {
        RealtimeCanaryVad::total_duration(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_mode() {
        let config = RealtimeCanaryVadConfig::from_mode("speedy", "de".to_string());
        assert_eq!(config.language, "de");
        assert_eq!(config.vad.silence_trigger_ms, 300);
    }
}
