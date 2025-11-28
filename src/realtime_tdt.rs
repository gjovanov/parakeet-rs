//! Quasi-realtime transcription using ParakeetTDT with overlapping chunks.
//!
//! This module provides near-realtime transcription by processing audio in
//! overlapping chunks with the high-quality TDT model. Unlike true streaming,
//! this approach introduces ~5-10 second latency but produces much better
//! quality than the EOU streaming model.
//!
//! ## How it works
//!
//! 1. Audio is accumulated in a buffer
//! 2. When buffer reaches chunk_size (default 10s), process the chunk
//! 3. Overlap regions (default 2s) are used to align and deduplicate text
//! 4. Results include word-level timestamps for accurate speaker attribution
//!
//! ## Latency vs Quality tradeoff
//!
//! - chunk_size=5s, overlap=1s: ~5-6s latency, good quality
//! - chunk_size=10s, overlap=2s: ~10-12s latency, excellent quality (default)
//! - chunk_size=15s, overlap=3s: ~15-18s latency, best quality

use crate::decoder::TimedToken;
use crate::error::Result;
use crate::execution::ModelConfig as ExecutionConfig;
use crate::parakeet_tdt::ParakeetTDT;
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

/// Find the last sentence boundary (., !, ?) before the overlap start time.
/// Returns the index of the last token that ends a sentence, or None if no boundary found.
fn find_last_sentence_boundary(tokens: &[TimedToken], overlap_start_time: f32) -> Option<usize> {
    let mut last_boundary_idx: Option<usize> = None;

    for (i, token) in tokens.iter().enumerate() {
        // Only consider tokens that end before the overlap zone
        if token.end >= overlap_start_time {
            break;
        }

        // Check if this token ends a sentence
        let text = token.text.trim();
        if text.ends_with('.') || text.ends_with('!') || text.ends_with('?') {
            last_boundary_idx = Some(i);
        }
    }

    // If no sentence boundary found but we have tokens before overlap,
    // fall back to the last token before overlap (better than nothing)
    if last_boundary_idx.is_none() {
        for (i, token) in tokens.iter().enumerate().rev() {
            if token.end < overlap_start_time {
                return Some(i);
            }
        }
    }

    last_boundary_idx
}

/// Configuration for quasi-realtime TDT processing
#[derive(Debug, Clone)]
pub struct RealtimeTDTConfig {
    /// Chunk size in seconds (default: 10.0)
    pub chunk_size_secs: f32,
    /// Overlap between chunks in seconds (default: 2.0)
    pub overlap_secs: f32,
    /// Minimum buffer before first processing (default: chunk_size)
    pub min_buffer_secs: f32,
    /// Emit partial results during overlap processing
    pub emit_partials: bool,
}

impl Default for RealtimeTDTConfig {
    fn default() -> Self {
        Self {
            chunk_size_secs: 10.0,
            overlap_secs: 2.0,
            min_buffer_secs: 10.0,
            emit_partials: true,
        }
    }
}

impl RealtimeTDTConfig {
    /// Create config optimized for low latency (~5s)
    pub fn low_latency() -> Self {
        Self {
            chunk_size_secs: 5.0,
            overlap_secs: 1.0,
            min_buffer_secs: 5.0,
            emit_partials: true,
        }
    }

    /// Create config optimized for quality (~15s latency)
    pub fn high_quality() -> Self {
        Self {
            chunk_size_secs: 15.0,
            overlap_secs: 3.0,
            min_buffer_secs: 15.0,
            emit_partials: true,
        }
    }
}

/// A token segment with global timestamps
#[derive(Debug, Clone)]
pub struct Segment {
    /// The transcribed text
    pub text: String,
    /// Start time in seconds (from stream start)
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Individual tokens with timestamps
    pub tokens: Vec<TimedToken>,
    /// Whether this is a final result (won't change)
    pub is_final: bool,
}

/// Result from processing a chunk
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// New segments from this chunk
    pub segments: Vec<Segment>,
    /// Full accumulated text so far
    pub full_text: String,
    /// Current buffer position in seconds
    pub buffer_time: f32,
    /// Whether more audio is needed
    pub needs_more_audio: bool,
}

/// Quasi-realtime transcriber using TDT model
pub struct RealtimeTDT {
    model: ParakeetTDT,
    config: RealtimeTDTConfig,

    // Audio buffer
    audio_buffer: Vec<f32>,

    // State tracking
    total_samples_processed: usize,
    /// Time up to which we've confirmed and emitted final text
    confirmed_until: f32,
    /// Tokens from the overlap zone, waiting to be re-transcribed
    pending_tokens: Vec<TimedToken>,

    // Accumulated results
    finalized_segments: Vec<Segment>,
}

impl RealtimeTDT {
    /// Create a new quasi-realtime TDT transcriber
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<RealtimeTDTConfig>,
    ) -> Result<Self> {
        let model = ParakeetTDT::from_pretrained(model_path, exec_config)?;
        let config = config.unwrap_or_default();

        Ok(Self {
            model,
            config,
            audio_buffer: Vec::new(),
            total_samples_processed: 0,
            confirmed_until: 0.0,
            pending_tokens: Vec::new(),
            finalized_segments: Vec::new(),
        })
    }

    /// Push audio samples and get transcription results
    ///
    /// Uses a sliding window approach: processes accumulated audio and emits
    /// confirmed segments while keeping a buffer for context.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<ChunkResult> {
        self.audio_buffer.extend_from_slice(samples);

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;
        let min_samples = (self.config.min_buffer_secs * SAMPLE_RATE as f32) as usize;
        let overlap_samples = (self.config.overlap_secs * SAMPLE_RATE as f32) as usize;

        // Check if we have enough audio to process
        if self.audio_buffer.len() < min_samples {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: buffer_secs,
                needs_more_audio: true,
            });
        }

        // Process the entire buffer to get full context
        let result = self.model.transcribe_samples(
            self.audio_buffer.clone(),
            SAMPLE_RATE as u32,
            1, // mono
            Some(TimestampMode::Words),
        )?;

        // Adjust timestamps to global time (from stream start)
        let global_offset = self.total_samples_processed as f32 / SAMPLE_RATE as f32;
        let adjusted_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += global_offset;
                t.end += global_offset;
                t
            })
            .collect();

        // Determine safe zone: everything except the last overlap_secs
        let buffer_end_time = global_offset + buffer_secs;
        let safe_until = buffer_end_time - self.config.overlap_secs;

        // Split tokens: safe zone goes to segments, overlap stays pending
        let mut safe_tokens: Vec<TimedToken> = Vec::new();
        let mut new_pending: Vec<TimedToken> = Vec::new();

        for token in adjusted_tokens {
            if token.end <= self.confirmed_until {
                // Already emitted, skip
                continue;
            } else if token.end < safe_until {
                safe_tokens.push(token);
            } else {
                new_pending.push(token);
            }
        }

        let mut new_segments = Vec::new();

        // Find sentence boundary for clean output
        let sentence_end_idx = find_last_sentence_boundary(&safe_tokens, safe_until);

        let (final_tokens, extra_pending) = if let Some(idx) = sentence_end_idx {
            let (final_part, extra) = safe_tokens.split_at(idx + 1);
            (final_part.to_vec(), extra.to_vec())
        } else if !safe_tokens.is_empty() {
            // No sentence boundary - emit all safe tokens to avoid infinite buffering
            (safe_tokens, Vec::new())
        } else {
            (Vec::new(), Vec::new())
        };

        // Create segment from final tokens
        if !final_tokens.is_empty() {
            let segment_text: String = final_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let segment = Segment {
                text: segment_text,
                start_time: final_tokens.first().map(|t| t.start).unwrap_or(global_offset),
                end_time: final_tokens.last().map(|t| t.end).unwrap_or(safe_until),
                tokens: final_tokens.clone(),
                is_final: true,
            };

            // Update confirmed_until
            if let Some(last) = final_tokens.last() {
                self.confirmed_until = last.end;
            }

            self.finalized_segments.push(segment.clone());
            new_segments.push(segment);
        }

        // Update pending tokens
        self.pending_tokens = extra_pending;
        self.pending_tokens.extend(new_pending);

        // Trim buffer to keep only overlap region (to limit memory/reprocessing)
        // Keep enough audio so next iteration has full context
        let max_buffer = (self.config.chunk_size_secs * SAMPLE_RATE as f32) as usize;
        if self.audio_buffer.len() > max_buffer {
            let drain_amount = self.audio_buffer.len() - overlap_samples;
            self.audio_buffer.drain(..drain_amount);
            self.total_samples_processed += drain_amount;
        }

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: self.audio_buffer.len() as f32 / SAMPLE_RATE as f32,
            needs_more_audio: true,
        })
    }

    /// Finalize processing and get any remaining text
    pub fn finalize(&mut self) -> Result<ChunkResult> {
        if self.audio_buffer.is_empty() {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: 0.0,
                needs_more_audio: false,
            });
        }

        // Process remaining audio
        let chunk_start_time = self.total_samples_processed as f32 / SAMPLE_RATE as f32;
        let chunk = std::mem::take(&mut self.audio_buffer);

        let result = self.model.transcribe_samples(
            chunk,
            SAMPLE_RATE as u32,
            1,
            Some(TimestampMode::Words),
        )?;

        // Adjust timestamps and create final segment
        let adjusted_tokens: Vec<TimedToken> = result.tokens
            .into_iter()
            .map(|mut t| {
                t.start += chunk_start_time;
                t.end += chunk_start_time;
                t
            })
            .collect();

        let mut new_segments = Vec::new();

        // Filter out tokens we've already confirmed
        let final_tokens: Vec<TimedToken> = adjusted_tokens
            .into_iter()
            .filter(|t| t.end > self.confirmed_until)
            .collect();

        if !final_tokens.is_empty() {
            let segment_text: String = final_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let segment = Segment {
                text: segment_text,
                start_time: final_tokens.first().map(|t| t.start).unwrap_or(chunk_start_time),
                end_time: final_tokens.last().map(|t| t.end).unwrap_or(chunk_start_time),
                tokens: final_tokens,
                is_final: true,
            };

            self.finalized_segments.push(segment.clone());
            new_segments.push(segment);
        }

        self.pending_tokens.clear();

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: 0.0,
            needs_more_audio: false,
        })
    }

    /// Get all finalized segments
    pub fn segments(&self) -> &[Segment] {
        &self.finalized_segments
    }

    /// Get full accumulated text
    pub fn get_full_text(&self) -> String {
        let mut text: String = self.finalized_segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Append pending tokens
        if !self.pending_tokens.is_empty() {
            let pending_text: String = self.pending_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            if !text.is_empty() && !pending_text.is_empty() {
                text.push(' ');
            }
            text.push_str(&pending_text);
        }

        text
    }

    /// Get current processing position in seconds
    pub fn current_time(&self) -> f32 {
        self.total_samples_processed as f32 / SAMPLE_RATE as f32
    }

    /// Reset the transcriber state
    pub fn reset(&mut self) {
        self.audio_buffer.clear();
        self.total_samples_processed = 0;
        self.confirmed_until = 0.0;
        self.pending_tokens.clear();
        self.finalized_segments.clear();
    }
}

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

/// Quasi-realtime transcriber with speaker diarization
#[cfg(feature = "sortformer")]
pub struct RealtimeTDTDiarized {
    tdt: RealtimeTDT,
    diarization: SortformerStream,
}

#[cfg(feature = "sortformer")]
impl RealtimeTDTDiarized {
    /// Create a new quasi-realtime transcriber with diarization
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

    /// Push audio and get transcription with speaker labels
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<DiarizedChunkResult> {
        // Process through TDT
        let tdt_result = self.tdt.push_audio(samples)?;

        // Process through diarization
        let _ = self.diarization.push_audio(samples)?;

        // Attribute speakers to segments
        let diarized_segments: Vec<DiarizedSegment> = tdt_result.segments
            .into_iter()
            .map(|seg| {
                // Use midpoint of segment for speaker lookup
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

    /// Finalize and get remaining text with speakers
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

    /// Reset both transcription and diarization
    pub fn reset(&mut self) {
        self.tdt.reset();
        self.diarization.reset();
    }
}

/// A segment with speaker attribution
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
    /// Get display string for speaker
    pub fn speaker_display(&self) -> String {
        match self.speaker {
            Some(id) => format!("Speaker {}", id),
            None => "Speaker ?".to_string(),
        }
    }
}

/// Result from diarized chunk processing
#[cfg(feature = "sortformer")]
#[derive(Debug, Clone)]
pub struct DiarizedChunkResult {
    pub segments: Vec<DiarizedSegment>,
    pub full_text: String,
    pub buffer_time: f32,
    pub needs_more_audio: bool,
}
