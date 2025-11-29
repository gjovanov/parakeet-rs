//! Quasi-realtime transcription using ParakeetTDT with ring buffer.
//!
//! Inspired by the EOU streaming approach:
//! - Maintains a ring buffer for audio context (default 15 seconds)
//! - Processes buffer periodically (every ~2 seconds of new audio)
//! - Extracts full context from buffer for accurate transcription
//! - Only emits "confirmed" tokens that won't change
//!
//! ## Key differences from EOU streaming:
//! - TDT has no encoder cache, so we reprocess the buffer each time
//! - Higher latency (~5-10s) but much better transcription quality
//! - Word-level timestamps for accurate speaker attribution

use crate::decoder::TimedToken;
use crate::error::Result;
use crate::execution::ModelConfig as ExecutionConfig;
use crate::parakeet_tdt::ParakeetTDT;
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use std::collections::VecDeque;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Configuration for quasi-realtime TDT processing
#[derive(Debug, Clone)]
pub struct RealtimeTDTConfig {
    /// Ring buffer size in seconds (default: 15.0)
    /// Larger buffer = more context = better quality but more processing
    pub buffer_size_secs: f32,

    /// How often to process (in seconds of new audio, default: 2.0)
    /// Smaller = lower latency but more CPU usage
    pub process_interval_secs: f32,

    /// Confirmed zone: tokens older than this won't change (default: 3.0)
    /// This is the "safe zone" before the buffer end
    pub confirm_threshold_secs: f32,
}

impl Default for RealtimeTDTConfig {
    fn default() -> Self {
        Self {
            buffer_size_secs: 15.0,
            process_interval_secs: 2.0,
            confirm_threshold_secs: 3.0,
        }
    }
}

impl RealtimeTDTConfig {
    /// Low latency mode: faster updates, smaller buffer
    pub fn low_latency() -> Self {
        Self {
            buffer_size_secs: 10.0,
            process_interval_secs: 1.5,
            confirm_threshold_secs: 2.0,
        }
    }

    /// High quality mode: larger buffer, more context
    pub fn high_quality() -> Self {
        Self {
            buffer_size_secs: 20.0,
            process_interval_secs: 3.0,
            confirm_threshold_secs: 4.0,
        }
    }

    // Keep old API for compatibility
    pub fn chunk_size_secs(&self) -> f32 {
        self.buffer_size_secs
    }

    pub fn overlap_secs(&self) -> f32 {
        self.confirm_threshold_secs
    }
}

/// A confirmed segment with timestamps
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub tokens: Vec<TimedToken>,
    pub is_final: bool,
}

/// Result from processing
#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub segments: Vec<Segment>,
    pub full_text: String,
    pub buffer_time: f32,
    pub needs_more_audio: bool,
}

/// Quasi-realtime TDT transcriber with ring buffer
pub struct RealtimeTDT {
    model: ParakeetTDT,
    config: RealtimeTDTConfig,

    /// Ring buffer for audio (like EOU's 4s buffer, but larger for TDT)
    audio_buffer: VecDeque<f32>,
    buffer_size_samples: usize,

    /// Tracking state
    total_samples_received: usize,
    samples_since_last_process: usize,
    process_interval_samples: usize,

    /// Time tracking for confirmed zone
    confirmed_until: f32,

    /// Accumulated results
    finalized_segments: Vec<Segment>,
    pending_tokens: Vec<TimedToken>,
}

impl RealtimeTDT {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeTDTConfig>,
    ) -> Result<Self> {
        let model = ParakeetTDT::from_pretrained(model_path, exec_config)?;
        let config = config.unwrap_or_default();

        let buffer_size_samples = (config.buffer_size_secs * SAMPLE_RATE as f32) as usize;
        let process_interval_samples = (config.process_interval_secs * SAMPLE_RATE as f32) as usize;

        Ok(Self {
            model,
            config,
            audio_buffer: VecDeque::with_capacity(buffer_size_samples),
            buffer_size_samples,
            total_samples_received: 0,
            samples_since_last_process: 0,
            process_interval_samples,
            confirmed_until: 0.0,
            finalized_segments: Vec::new(),
            pending_tokens: Vec::new(),
        })
    }

    /// Push audio samples (like EOU's transcribe method)
    ///
    /// Ring buffer approach:
    /// 1. Add samples to ring buffer
    /// 2. Trim buffer to max size (keeping most recent)
    /// 3. Process when enough new audio accumulated
    /// 4. Only emit tokens in "confirmed zone"
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<ChunkResult> {
        // Add to ring buffer
        self.audio_buffer.extend(samples.iter().copied());
        self.total_samples_received += samples.len();
        self.samples_since_last_process += samples.len();

        // Trim buffer to keep only most recent samples (ring buffer behavior)
        while self.audio_buffer.len() > self.buffer_size_samples {
            self.audio_buffer.pop_front();
        }

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Check if we should process
        // Need: minimum buffer AND enough new audio since last process
        let min_buffer_samples = (self.config.confirm_threshold_secs * 2.0 * SAMPLE_RATE as f32) as usize;

        if self.audio_buffer.len() < min_buffer_samples {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: buffer_secs,
                needs_more_audio: true,
            });
        }

        if self.samples_since_last_process < self.process_interval_samples {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: buffer_secs,
                needs_more_audio: true,
            });
        }

        // Process the buffer
        self.samples_since_last_process = 0;
        self.process_buffer()
    }

    /// Process the ring buffer and emit confirmed tokens
    fn process_buffer(&mut self) -> Result<ChunkResult> {
        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;

        // Convert ring buffer to vec for processing
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();

        // Calculate global time offset
        // The buffer starts at: total_received - buffer_length
        let buffer_start_time = (self.total_samples_received - self.audio_buffer.len()) as f32 / SAMPLE_RATE as f32;

        // Process through TDT
        let result = self.model.transcribe_samples(
            audio,
            SAMPLE_RATE as u32,
            1,
            Some(TimestampMode::Words),
        )?;

        // Adjust timestamps to global time
        let adjusted_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += buffer_start_time;
                t.end += buffer_start_time;
                t
            })
            .collect();

        // Determine confirmed zone
        // Tokens are "confirmed" if they end before (buffer_end - confirm_threshold)
        let buffer_end_time = buffer_start_time + buffer_secs;
        let confirm_until = buffer_end_time - self.config.confirm_threshold_secs;

        // Split tokens into confirmed and pending
        // Use a small margin to prevent edge duplications due to timestamp variance
        let margin = 0.1; // 100ms margin
        let mut confirmed_tokens: Vec<TimedToken> = Vec::new();
        let mut new_pending: Vec<TimedToken> = Vec::new();

        for token in adjusted_tokens {
            if token.end <= self.confirmed_until + margin {
                // Already emitted or too close to boundary, skip
                continue;
            } else if token.end < confirm_until {
                // In confirmed zone
                confirmed_tokens.push(token);
            } else {
                // In pending zone (might change with more context)
                new_pending.push(token);
            }
        }

        let mut new_segments = Vec::new();

        // Find sentence boundary for clean output
        if !confirmed_tokens.is_empty() {
            let sentence_end_idx = find_last_sentence_boundary(&confirmed_tokens, confirm_until);

            let (final_tokens, extra_pending) = if let Some(idx) = sentence_end_idx {
                let (final_part, extra) = confirmed_tokens.split_at(idx + 1);
                (final_part.to_vec(), extra.to_vec())
            } else {
                // No sentence boundary - emit all to avoid infinite buffering
                (confirmed_tokens, Vec::new())
            };

            if !final_tokens.is_empty() {
                // Deduplicate: check if last emitted word matches first new word
                let mut tokens_to_use = final_tokens.clone();
                if let Some(last_seg) = self.finalized_segments.last() {
                    if let Some(last_emitted) = last_seg.tokens.last() {
                        if let Some(first_new) = tokens_to_use.first() {
                            if last_emitted.text.trim() == first_new.text.trim() {
                                // Remove duplicate first token
                                tokens_to_use.remove(0);
                            }
                        }
                    }
                }

                if !tokens_to_use.is_empty() {
                    let segment_text: String = tokens_to_use
                        .iter()
                        .map(|t| t.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");

                    let segment = Segment {
                        text: segment_text,
                        start_time: tokens_to_use.first().map(|t| t.start).unwrap_or(0.0),
                        end_time: tokens_to_use.last().map(|t| t.end).unwrap_or(0.0),
                        tokens: tokens_to_use.clone(),
                        is_final: true,
                    };

                    // Update confirmed_until
                    if let Some(last) = tokens_to_use.last() {
                        self.confirmed_until = last.end;
                    }

                    self.finalized_segments.push(segment.clone());
                    new_segments.push(segment);
                }
            }

            // Add extra to pending
            new_pending = [extra_pending, new_pending].concat();
        }

        self.pending_tokens = new_pending;

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: buffer_secs,
            needs_more_audio: true,
        })
    }

    /// Finalize and get remaining text
    pub fn finalize(&mut self) -> Result<ChunkResult> {
        if self.audio_buffer.is_empty() && self.pending_tokens.is_empty() {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: false,
            });
        }

        // Process remaining buffer with no confirm threshold
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();
        if audio.is_empty() {
            // Just emit pending tokens
            let mut new_segments = Vec::new();
            if !self.pending_tokens.is_empty() {
                let segment_text: String = self.pending_tokens
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let segment = Segment {
                    text: segment_text,
                    start_time: self.pending_tokens.first().map(|t| t.start).unwrap_or(0.0),
                    end_time: self.pending_tokens.last().map(|t| t.end).unwrap_or(0.0),
                    tokens: std::mem::take(&mut self.pending_tokens),
                    is_final: true,
                };

                self.finalized_segments.push(segment.clone());
                new_segments.push(segment);
            }

            return Ok(ChunkResult {
                segments: new_segments,
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: false,
            });
        }

        let buffer_start_time = (self.total_samples_received - self.audio_buffer.len()) as f32 / SAMPLE_RATE as f32;

        let result = self.model.transcribe_samples(
            audio,
            SAMPLE_RATE as u32,
            1,
            Some(TimestampMode::Words),
        )?;

        // Get all remaining tokens
        let final_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += buffer_start_time;
                t.end += buffer_start_time;
                t
            })
            .filter(|t| t.end > self.confirmed_until)
            .collect();

        let mut new_segments = Vec::new();

        if !final_tokens.is_empty() {
            let segment_text: String = final_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let segment = Segment {
                text: segment_text,
                start_time: final_tokens.first().map(|t| t.start).unwrap_or(0.0),
                end_time: final_tokens.last().map(|t| t.end).unwrap_or(0.0),
                tokens: final_tokens,
                is_final: true,
            };

            self.finalized_segments.push(segment.clone());
            new_segments.push(segment);
        }

        self.audio_buffer.clear();
        self.pending_tokens.clear();

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: 0.0,
            needs_more_audio: false,
        })
    }

    pub fn segments(&self) -> &[Segment] {
        &self.finalized_segments
    }

    pub fn get_full_text(&self) -> String {
        let mut text: String = self.finalized_segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        if !self.pending_tokens.is_empty() {
            let pending: String = self.pending_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            if !text.is_empty() && !pending.is_empty() {
                text.push(' ');
            }
            text.push_str(&pending);
        }

        text
    }

    pub fn current_time(&self) -> f32 {
        self.total_samples_received as f32 / SAMPLE_RATE as f32
    }

    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.total_samples_received = 0;
        self.samples_since_last_process = 0;
        self.confirmed_until = 0.0;
        self.finalized_segments.clear();
        self.pending_tokens.clear();
    }
}

/// Find the last sentence boundary (., !, ?) before a given time
fn find_last_sentence_boundary(tokens: &[TimedToken], before_time: f32) -> Option<usize> {
    let mut last_boundary_idx: Option<usize> = None;

    for (i, token) in tokens.iter().enumerate() {
        if token.end >= before_time {
            break;
        }
        let text = token.text.trim();
        if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
            last_boundary_idx = Some(i);
        }
    }

    // Fallback: if no sentence boundary, return last token before time
    if last_boundary_idx.is_none() {
        for (i, token) in tokens.iter().enumerate().rev() {
            if token.end < before_time {
                return Some(i);
            }
        }
    }

    last_boundary_idx
}

// ============================================================================
// Diarized version
// ============================================================================

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

#[cfg(feature = "sortformer")]
pub struct RealtimeTDTDiarized {
    tdt: RealtimeTDT,
    diarization: SortformerStream,
}

#[cfg(feature = "sortformer")]
impl RealtimeTDTDiarized {
    pub fn new<P1: AsRef<Path>, P2: AsRef<Path>>(
        tdt_model_path: P1,
        diar_model_path: P2,
        exec_config: Option<ExecutionConfig>,
        tdt_config: Option<RealtimeTDTConfig>,
    ) -> Result<Self> {
        let tdt = RealtimeTDT::new(tdt_model_path, exec_config.clone(), tdt_config)?;
        let diarization = SortformerStream::with_config(
            diar_model_path,
            exec_config,
            crate::sortformer::DiarizationConfig::callhome(),
        )?;

        Ok(Self { tdt, diarization })
    }

    pub fn push_audio(&mut self, samples: &[f32]) -> Result<DiarizedChunkResult> {
        let tdt_result = self.tdt.push_audio(samples)?;
        let _ = self.diarization.push_audio(samples)?;

        let diarized_segments: Vec<DiarizedSegment> = tdt_result.segments
            .into_iter()
            .map(|seg| {
                let mid_time = (seg.start_time + seg.end_time) / 2.0;
                let speaker = self.diarization.get_speaker_at(mid_time);

                DiarizedSegment {
                    text: seg.text,
                    start_time: seg.start_time,
                    end_time: seg.end_time,
                    speaker,
                    tokens: seg.tokens,
                    is_final: seg.is_final,
                }
            })
            .collect();

        Ok(DiarizedChunkResult {
            segments: diarized_segments,
            full_text: tdt_result.full_text,
            buffer_time: tdt_result.buffer_time,
            needs_more_audio: tdt_result.needs_more_audio,
        })
    }

    pub fn finalize(&mut self) -> Result<DiarizedChunkResult> {
        let tdt_result = self.tdt.finalize()?;

        let diarized_segments: Vec<DiarizedSegment> = tdt_result.segments
            .into_iter()
            .map(|seg| {
                let mid_time = (seg.start_time + seg.end_time) / 2.0;
                let speaker = self.diarization.get_speaker_at(mid_time);

                DiarizedSegment {
                    text: seg.text,
                    start_time: seg.start_time,
                    end_time: seg.end_time,
                    speaker,
                    tokens: seg.tokens,
                    is_final: seg.is_final,
                }
            })
            .collect();

        Ok(DiarizedChunkResult {
            segments: diarized_segments,
            full_text: tdt_result.full_text,
            buffer_time: 0.0,
            needs_more_audio: false,
        })
    }

    pub fn reset(&mut self) {
        self.tdt.reset();
        self.diarization.reset();
    }
}

#[cfg(feature = "sortformer")]
#[derive(Debug, Clone)]
pub struct DiarizedSegment {
    pub text: String,
    pub start_time: f32,
    pub end_time: f32,
    pub speaker: Option<usize>,
    pub tokens: Vec<TimedToken>,
    pub is_final: bool,
}

#[cfg(feature = "sortformer")]
impl DiarizedSegment {
    pub fn speaker_display(&self) -> String {
        match self.speaker {
            Some(id) => format!("Speaker {}", id),
            None => "Speaker ?".to_string(),
        }
    }
}

#[cfg(feature = "sortformer")]
#[derive(Debug, Clone)]
pub struct DiarizedChunkResult {
    pub segments: Vec<DiarizedSegment>,
    pub full_text: String,
    pub buffer_time: f32,
    pub needs_more_audio: bool,
}
