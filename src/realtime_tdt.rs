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
    last_processed_end: f32,

    // Accumulated results
    finalized_segments: Vec<Segment>,
    pending_text: String,
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
            last_processed_end: 0.0,
            finalized_segments: Vec::new(),
            pending_text: String::new(),
        })
    }

    /// Push audio samples and get transcription results
    ///
    /// Returns results when enough audio has accumulated for processing.
    /// Call this repeatedly with audio chunks (any size).
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<ChunkResult> {
        self.audio_buffer.extend_from_slice(samples);

        let buffer_secs = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;
        let chunk_samples = (self.config.chunk_size_secs * SAMPLE_RATE as f32) as usize;
        let overlap_samples = (self.config.overlap_secs * SAMPLE_RATE as f32) as usize;
        let min_samples = (self.config.min_buffer_secs * SAMPLE_RATE as f32) as usize;

        // Check if we have enough audio to process
        if self.audio_buffer.len() < min_samples {
            return Ok(ChunkResult {
                segments: Vec::new(),
                full_text: self.get_full_text(),
                buffer_time: buffer_secs,
                needs_more_audio: true,
            });
        }

        // Process chunks while we have enough audio
        let mut new_segments = Vec::new();

        while self.audio_buffer.len() >= chunk_samples {
            // Extract chunk for processing
            let chunk: Vec<f32> = self.audio_buffer[..chunk_samples].to_vec();

            // Calculate global time offset for this chunk
            let chunk_start_time = self.total_samples_processed as f32 / SAMPLE_RATE as f32;

            // Process through TDT model
            let result = self.model.transcribe_samples(
                chunk,
                SAMPLE_RATE as u32,
                1, // mono
                Some(TimestampMode::Words),
            )?;

            // Adjust timestamps to global time
            let adjusted_tokens: Vec<TimedToken> = result.tokens
                .into_iter()
                .map(|mut t| {
                    t.start += chunk_start_time;
                    t.end += chunk_start_time;
                    t
                })
                .collect();

            // Determine which tokens are in the "final" zone vs overlap zone
            let overlap_start_time = chunk_start_time + self.config.chunk_size_secs - self.config.overlap_secs;

            // Split tokens into final and pending
            let (final_tokens, pending_tokens): (Vec<_>, Vec<_>) = adjusted_tokens
                .into_iter()
                .partition(|t| t.end < overlap_start_time);

            // Create segment from final tokens
            if !final_tokens.is_empty() {
                let segment_text: String = final_tokens
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                let segment = Segment {
                    text: segment_text,
                    start_time: final_tokens.first().map(|t| t.start).unwrap_or(chunk_start_time),
                    end_time: final_tokens.last().map(|t| t.end).unwrap_or(overlap_start_time),
                    tokens: final_tokens,
                    is_final: true,
                };

                self.finalized_segments.push(segment.clone());
                new_segments.push(segment);
            }

            // Store pending tokens for overlap handling
            self.pending_text = pending_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            // Advance buffer, keeping overlap for next chunk
            let advance_samples = chunk_samples - overlap_samples;
            self.audio_buffer.drain(..advance_samples);
            self.total_samples_processed += advance_samples;
            self.last_processed_end = chunk_start_time + self.config.chunk_size_secs - self.config.overlap_secs;
        }

        Ok(ChunkResult {
            segments: new_segments,
            full_text: self.get_full_text(),
            buffer_time: self.audio_buffer.len() as f32 / SAMPLE_RATE as f32,
            needs_more_audio: self.audio_buffer.len() < chunk_samples,
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

        if !adjusted_tokens.is_empty() {
            let segment_text: String = adjusted_tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let segment = Segment {
                text: segment_text,
                start_time: adjusted_tokens.first().map(|t| t.start).unwrap_or(chunk_start_time),
                end_time: adjusted_tokens.last().map(|t| t.end).unwrap_or(chunk_start_time),
                tokens: adjusted_tokens,
                is_final: true,
            };

            self.finalized_segments.push(segment.clone());
            new_segments.push(segment);
        }

        self.pending_text.clear();

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

        if !self.pending_text.is_empty() {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&self.pending_text);
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
        self.last_processed_end = 0.0;
        self.finalized_segments.clear();
        self.pending_text.clear();
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
