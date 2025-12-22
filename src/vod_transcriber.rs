//! VoD (Video on Demand) Batch Transcription
//!
//! This module provides batch transcription for media files, processing them in
//! large chunks (10 minutes) with overlap (1 minute) for sentence-level deduplication.
//!
//! The output is a structured transcript with segments and word-level timing.

use crate::audio;
use crate::canary::{CanaryConfig, CanaryModel};
use crate::config::PreprocessorConfig;
use crate::decoder::TimedToken;
use crate::decoder_tdt::ParakeetTDTDecoder;
use crate::error::Result;
use crate::execution::ModelConfig as ExecutionConfig;
use crate::model_tdt::ParakeetTDTModel;
use crate::timestamps::{process_timestamps, TimestampMode};
use crate::vocab::Vocabulary;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for VoD transcription
#[derive(Debug, Clone)]
pub struct VodConfig {
    /// Duration of each chunk in seconds (default: 600.0 = 10 minutes)
    pub chunk_duration_secs: f32,
    /// Overlap between chunks in seconds (default: 60.0 = 1 minute)
    pub overlap_duration_secs: f32,
    /// Number of parallel workers (default: 4)
    pub num_workers: usize,
    /// Jaccard similarity threshold for deduplication (default: 0.8)
    pub dedup_threshold: f32,
    /// Target language code
    pub language: String,
}

impl Default for VodConfig {
    fn default() -> Self {
        Self {
            chunk_duration_secs: 600.0,  // 10 minutes
            overlap_duration_secs: 60.0, // 1 minute
            num_workers: 4,
            dedup_threshold: 0.8,
            language: "en".to_string(),
        }
    }
}

// ============================================================================
// Output Types
// ============================================================================

/// A single word with timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VodWord {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

/// A transcription segment (typically a sentence) with words
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VodSegment {
    pub id: usize,
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub speaker: Option<usize>,
    pub words: Vec<VodWord>,
}

/// Complete transcript output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VodTranscript {
    pub session_id: String,
    pub model: String,
    pub language: String,
    pub duration_secs: f32,
    pub created_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub segments: Vec<VodSegment>,
}

/// Progress callback type
pub type ProgressCallback = Box<dyn Fn(VodProgress) + Send + Sync>;

/// Progress information for VoD transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VodProgress {
    pub total_chunks: usize,
    pub completed_chunks: usize,
    pub current_chunk: usize,
    pub percent: f32,
}

// ============================================================================
// Internal Types
// ============================================================================

/// Audio chunk for processing
struct AudioChunk {
    /// Chunk index (0-based)
    index: usize,
    /// Audio samples
    samples: Vec<f32>,
    /// Start time in the original audio (seconds)
    start_time: f32,
    /// End time in the original audio (seconds)
    end_time: f32,
}

/// Result from processing a chunk
struct ChunkResult {
    /// Chunk index
    index: usize,
    /// Transcription segments (sentences)
    segments: Vec<VodSegment>,
    /// Start time of chunk
    start_time: f32,
    /// End time of chunk
    end_time: f32,
}

/// Job for worker threads
struct TranscriptionJob {
    chunk: AudioChunk,
}

// ============================================================================
// VoD Transcriber - TDT
// ============================================================================

/// VoD transcriber using TDT model
pub struct VodTranscriberTDT {
    config: VodConfig,
    model_path: PathBuf,
    exec_config: ExecutionConfig,
    preprocessor_config: PreprocessorConfig,
}

impl VodTranscriberTDT {
    /// Create a new VoD transcriber for TDT model
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: VodConfig,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let exec_config = exec_config.unwrap_or_default();

        // TDT-specific preprocessor config
        let preprocessor_config = PreprocessorConfig {
            feature_extractor_type: "ParakeetFeatureExtractor".to_string(),
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            processor_class: "ParakeetProcessor".to_string(),
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        };

        Ok(Self {
            config,
            model_path,
            exec_config,
            preprocessor_config,
        })
    }

    /// Transcribe an audio file
    pub fn transcribe_file<P: AsRef<Path>>(
        &self,
        wav_path: P,
        session_id: &str,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<VodTranscript> {
        let created_at = Utc::now();
        let wav_path = wav_path.as_ref();

        // Load audio
        let (samples, spec) = audio::load_audio(wav_path)?;
        let sample_rate = spec.sample_rate as usize;
        let channels = spec.channels;

        // Convert to mono if needed
        let samples = if channels > 1 {
            samples
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };

        let duration_secs = samples.len() as f32 / sample_rate as f32;

        // Create chunks
        let chunks = self.create_chunks(&samples, sample_rate);
        let total_chunks = chunks.len();

        eprintln!(
            "[VoD] Processing {} chunks ({:.1} min file, {:.0}s chunks, {:.0}s overlap)",
            total_chunks,
            duration_secs / 60.0,
            self.config.chunk_duration_secs,
            self.config.overlap_duration_secs
        );

        // Process chunks in parallel
        let chunk_results = self.process_chunks_parallel(chunks, progress_callback)?;

        // Merge and deduplicate results
        let segments = self.merge_and_deduplicate(chunk_results);

        let completed_at = Utc::now();

        Ok(VodTranscript {
            session_id: session_id.to_string(),
            model: "parakeet-tdt".to_string(),
            language: self.config.language.clone(),
            duration_secs,
            created_at,
            completed_at,
            segments,
        })
    }

    /// Create audio chunks with overlap
    fn create_chunks(&self, samples: &[f32], sample_rate: usize) -> Vec<AudioChunk> {
        let chunk_samples = (self.config.chunk_duration_secs * sample_rate as f32) as usize;
        let overlap_samples = (self.config.overlap_duration_secs * sample_rate as f32) as usize;
        let step_samples = chunk_samples - overlap_samples;

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < samples.len() {
            let end = (start + chunk_samples).min(samples.len());
            let chunk_samples_vec = samples[start..end].to_vec();

            let start_time = start as f32 / sample_rate as f32;
            let end_time = end as f32 / sample_rate as f32;

            chunks.push(AudioChunk {
                index,
                samples: chunk_samples_vec,
                start_time,
                end_time,
            });

            start += step_samples;
            index += 1;

            // Avoid creating tiny final chunks
            if samples.len() - start < overlap_samples && start < samples.len() {
                // Include remaining in previous chunk is better, but for simplicity
                // we'll process it as a small final chunk
                break;
            }
        }

        // Handle remaining samples if any
        if start < samples.len() {
            let chunk_samples_vec = samples[start..].to_vec();
            let start_time = start as f32 / sample_rate as f32;
            let end_time = samples.len() as f32 / sample_rate as f32;

            chunks.push(AudioChunk {
                index,
                samples: chunk_samples_vec,
                start_time,
                end_time,
            });
        }

        chunks
    }

    /// Process chunks in parallel using worker threads
    fn process_chunks_parallel(
        &self,
        chunks: Vec<AudioChunk>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<ChunkResult>> {
        let total_chunks = chunks.len();
        let num_workers = self.config.num_workers.min(total_chunks);

        if total_chunks == 0 {
            return Ok(Vec::new());
        }

        // Channels for job distribution and result collection
        let (job_tx, job_rx) = mpsc::channel::<TranscriptionJob>();
        let (result_tx, result_rx) = mpsc::channel::<Result<ChunkResult>>();

        let job_rx = Arc::new(Mutex::new(job_rx));
        let completed_count = Arc::new(AtomicUsize::new(0));

        // Clone configs for workers
        let model_path = self.model_path.clone();
        let exec_config = self.exec_config.clone();
        let preprocessor_config = self.preprocessor_config.clone();

        // Spawn worker threads
        let mut handles = Vec::new();
        for worker_id in 0..num_workers {
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let model_path = model_path.clone();
            let exec_config = exec_config.clone();
            let preprocessor_config = preprocessor_config.clone();

            let handle = thread::spawn(move || {
                // Each worker loads its own model instance
                let vocab_path = model_path.join("vocab.txt");
                let model = match ParakeetTDTModel::from_pretrained(&model_path, exec_config) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[VoD Worker {}] Failed to load model: {}", worker_id, e);
                        return;
                    }
                };
                let vocab = match Vocabulary::from_file(&vocab_path) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[VoD Worker {}] Failed to load vocab: {}", worker_id, e);
                        return;
                    }
                };
                let decoder = ParakeetTDTDecoder::from_vocab(vocab);
                let mut model = model;

                eprintln!("[VoD Worker {}] Ready", worker_id);

                loop {
                    // Get next job
                    let job = {
                        let rx = job_rx.lock().unwrap();
                        rx.recv()
                    };

                    match job {
                        Ok(job) => {
                            let result = Self::process_chunk_tdt(
                                &mut model,
                                &decoder,
                                &preprocessor_config,
                                job.chunk,
                            );
                            let _ = result_tx.send(result);
                        }
                        Err(_) => {
                            // Channel closed, exit
                            break;
                        }
                    }
                }
            });
            handles.push(handle);
        }

        // Drop our copy of result_tx so channel closes when workers finish
        drop(result_tx);

        // Send jobs
        for chunk in chunks {
            let job = TranscriptionJob { chunk };
            if job_tx.send(job).is_err() {
                break;
            }
        }
        drop(job_tx); // Signal no more jobs

        // Collect results
        let mut results = Vec::new();
        for result in result_rx {
            match result {
                Ok(chunk_result) => {
                    let completed = completed_count.fetch_add(1, Ordering::SeqCst) + 1;

                    if let Some(ref callback) = progress_callback {
                        callback(VodProgress {
                            total_chunks,
                            completed_chunks: completed,
                            current_chunk: chunk_result.index,
                            percent: (completed as f32 / total_chunks as f32) * 100.0,
                        });
                    }

                    eprintln!(
                        "[VoD] Completed chunk {}/{} ({:.1}s - {:.1}s)",
                        completed,
                        total_chunks,
                        chunk_result.start_time,
                        chunk_result.end_time
                    );

                    results.push(chunk_result);
                }
                Err(e) => {
                    eprintln!("[VoD] Chunk processing error: {}", e);
                }
            }
        }

        // Wait for all workers
        for handle in handles {
            let _ = handle.join();
        }

        // Sort by chunk index
        results.sort_by_key(|r| r.index);

        Ok(results)
    }

    /// Process a single chunk with TDT model
    fn process_chunk_tdt(
        model: &mut ParakeetTDTModel,
        decoder: &ParakeetTDTDecoder,
        preprocessor_config: &PreprocessorConfig,
        chunk: AudioChunk,
    ) -> Result<ChunkResult> {
        let samples = chunk.samples;
        let chunk_offset = chunk.start_time;

        // Extract features
        let features = audio::extract_features_raw(
            samples,
            preprocessor_config.sampling_rate as u32,
            1, // mono
            preprocessor_config,
        )?;

        // Run inference
        let (tokens, frame_indices, durations) = model.forward(features)?;

        // Decode with timestamps
        let mut result = decoder.decode_with_timestamps(
            &tokens,
            &frame_indices,
            &durations,
            preprocessor_config.hop_length,
            preprocessor_config.sampling_rate,
        )?;

        // Convert to Words mode for grouping
        result.tokens = process_timestamps(&result.tokens, TimestampMode::Words);

        // Group words into sentences
        let segments = Self::group_into_sentences(&result.tokens, chunk_offset, chunk.index * 1000);

        Ok(ChunkResult {
            index: chunk.index,
            segments,
            start_time: chunk.start_time,
            end_time: chunk.end_time,
        })
    }

    /// Group words into sentences based on punctuation
    fn group_into_sentences(
        words: &[TimedToken],
        time_offset: f32,
        base_id: usize,
    ) -> Vec<VodSegment> {
        if words.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut current_words: Vec<VodWord> = Vec::new();
        let mut current_text = String::new();
        let mut segment_start = 0.0;
        let mut segment_id = base_id;

        for word in words {
            let adjusted_start = word.start + time_offset;
            let adjusted_end = word.end + time_offset;

            if current_words.is_empty() {
                segment_start = adjusted_start;
            }

            current_words.push(VodWord {
                word: word.text.clone(),
                start: adjusted_start,
                end: adjusted_end,
            });

            if !current_text.is_empty() {
                current_text.push(' ');
            }
            current_text.push_str(&word.text);

            // Check for sentence boundary
            let is_sentence_end = word.text.contains('.')
                || word.text.contains('?')
                || word.text.contains('!');

            if is_sentence_end {
                segments.push(VodSegment {
                    id: segment_id,
                    text: current_text.trim().to_string(),
                    start: segment_start,
                    end: adjusted_end,
                    speaker: None,
                    words: current_words.clone(),
                });

                segment_id += 1;
                current_words.clear();
                current_text.clear();
            }
        }

        // Add remaining words as final segment
        if !current_words.is_empty() {
            let end = current_words.last().map(|w| w.end).unwrap_or(segment_start);
            segments.push(VodSegment {
                id: segment_id,
                text: current_text.trim().to_string(),
                start: segment_start,
                end,
                speaker: None,
                words: current_words,
            });
        }

        segments
    }

    /// Merge results from all chunks and deduplicate overlapping sentences
    fn merge_and_deduplicate(&self, chunk_results: Vec<ChunkResult>) -> Vec<VodSegment> {
        if chunk_results.is_empty() {
            return Vec::new();
        }

        let mut all_segments: Vec<VodSegment> = Vec::new();
        let mut seen_sentences: HashSet<String> = HashSet::new();

        for (i, result) in chunk_results.iter().enumerate() {
            let overlap_start = if i == 0 {
                0.0
            } else {
                result.start_time
            };
            let overlap_end = result.start_time + self.config.overlap_duration_secs;

            for segment in &result.segments {
                // For segments in the overlap region (not the first chunk),
                // check for duplicates
                if i > 0 && segment.start < overlap_end {
                    let normalized = self.normalize_text(&segment.text);

                    // Check if we've seen a similar sentence
                    let is_duplicate = seen_sentences.iter().any(|seen| {
                        self.jaccard_similarity(&normalized, seen) >= self.config.dedup_threshold
                    });

                    if is_duplicate {
                        continue; // Skip duplicate
                    }
                }

                // Add to seen set and output
                let normalized = self.normalize_text(&segment.text);
                seen_sentences.insert(normalized);
                all_segments.push(segment.clone());
            }
        }

        // Re-number segment IDs
        for (i, segment) in all_segments.iter_mut().enumerate() {
            segment.id = i;
        }

        // Sort by start time
        all_segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());

        all_segments
    }

    /// Normalize text for comparison
    fn normalize_text(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Calculate Jaccard similarity between two normalized texts
    fn jaccard_similarity(&self, a: &str, b: &str) -> f32 {
        let words_a: HashSet<&str> = a.split_whitespace().collect();
        let words_b: HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            return 0.0;
        }

        intersection as f32 / union as f32
    }
}

// ============================================================================
// VoD Transcriber - Canary
// ============================================================================

/// VoD transcriber using Canary model
pub struct VodTranscriberCanary {
    config: VodConfig,
    model_path: PathBuf,
    exec_config: ExecutionConfig,
}

impl VodTranscriberCanary {
    /// Create a new VoD transcriber for Canary model
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config: VodConfig,
        exec_config: Option<ExecutionConfig>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            model_path: model_path.as_ref().to_path_buf(),
            exec_config: exec_config.unwrap_or_default(),
        })
    }

    /// Transcribe an audio file
    pub fn transcribe_file<P: AsRef<Path>>(
        &self,
        wav_path: P,
        session_id: &str,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<VodTranscript> {
        let created_at = Utc::now();
        let wav_path = wav_path.as_ref();

        // Load audio
        let (samples, spec) = audio::load_audio(wav_path)?;
        let sample_rate = spec.sample_rate as usize;
        let channels = spec.channels;

        // Convert to mono if needed
        let samples = if channels > 1 {
            samples
                .chunks(channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        } else {
            samples
        };

        let duration_secs = samples.len() as f32 / sample_rate as f32;

        // Create chunks
        let chunks = self.create_chunks(&samples, sample_rate);
        let total_chunks = chunks.len();

        eprintln!(
            "[VoD Canary] Processing {} chunks ({:.1} min file)",
            total_chunks,
            duration_secs / 60.0
        );

        // Process chunks in parallel
        let chunk_results = self.process_chunks_parallel(chunks, progress_callback)?;

        // Merge and deduplicate results
        let segments = self.merge_and_deduplicate(chunk_results);

        let completed_at = Utc::now();

        Ok(VodTranscript {
            session_id: session_id.to_string(),
            model: "canary-1b".to_string(),
            language: self.config.language.clone(),
            duration_secs,
            created_at,
            completed_at,
            segments,
        })
    }

    /// Create audio chunks with overlap (same as TDT)
    fn create_chunks(&self, samples: &[f32], sample_rate: usize) -> Vec<AudioChunk> {
        let chunk_samples = (self.config.chunk_duration_secs * sample_rate as f32) as usize;
        let overlap_samples = (self.config.overlap_duration_secs * sample_rate as f32) as usize;
        let step_samples = chunk_samples - overlap_samples;

        let mut chunks = Vec::new();
        let mut start = 0;
        let mut index = 0;

        while start < samples.len() {
            let end = (start + chunk_samples).min(samples.len());
            let chunk_samples_vec = samples[start..end].to_vec();

            let start_time = start as f32 / sample_rate as f32;
            let end_time = end as f32 / sample_rate as f32;

            chunks.push(AudioChunk {
                index,
                samples: chunk_samples_vec,
                start_time,
                end_time,
            });

            start += step_samples;
            index += 1;
        }

        chunks
    }

    /// Process chunks in parallel
    fn process_chunks_parallel(
        &self,
        chunks: Vec<AudioChunk>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<ChunkResult>> {
        let total_chunks = chunks.len();
        let num_workers = self.config.num_workers.min(total_chunks);

        if total_chunks == 0 {
            return Ok(Vec::new());
        }

        let (job_tx, job_rx) = mpsc::channel::<TranscriptionJob>();
        let (result_tx, result_rx) = mpsc::channel::<Result<ChunkResult>>();

        let job_rx = Arc::new(Mutex::new(job_rx));
        let completed_count = Arc::new(AtomicUsize::new(0));

        let model_path = self.model_path.clone();
        let exec_config = self.exec_config.clone();
        let language = self.config.language.clone();

        // Spawn workers
        let mut handles = Vec::new();
        for worker_id in 0..num_workers {
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let model_path = model_path.clone();
            let exec_config = exec_config.clone();
            let language = language.clone();

            let handle = thread::spawn(move || {
                // Load Canary model
                let canary_config = CanaryConfig {
                    language: language.clone(),
                    ..Default::default()
                };

                let mut model = match CanaryModel::from_pretrained(
                    &model_path,
                    Some(exec_config),
                    Some(canary_config),
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[VoD Canary Worker {}] Failed to load model: {}", worker_id, e);
                        return;
                    }
                };

                model.set_language(&language);
                eprintln!("[VoD Canary Worker {}] Ready", worker_id);

                loop {
                    let job = {
                        let rx = job_rx.lock().unwrap();
                        rx.recv()
                    };

                    match job {
                        Ok(job) => {
                            let result = Self::process_chunk_canary(&mut model, job.chunk);
                            let _ = result_tx.send(result);
                        }
                        Err(_) => break,
                    }
                }
            });
            handles.push(handle);
        }

        drop(result_tx);

        // Send jobs
        for chunk in chunks {
            let _ = job_tx.send(TranscriptionJob { chunk });
        }
        drop(job_tx);

        // Collect results
        let mut results = Vec::new();
        for result in result_rx {
            match result {
                Ok(chunk_result) => {
                    let completed = completed_count.fetch_add(1, Ordering::SeqCst) + 1;

                    if let Some(ref callback) = progress_callback {
                        callback(VodProgress {
                            total_chunks,
                            completed_chunks: completed,
                            current_chunk: chunk_result.index,
                            percent: (completed as f32 / total_chunks as f32) * 100.0,
                        });
                    }

                    eprintln!(
                        "[VoD Canary] Completed chunk {}/{}",
                        completed, total_chunks
                    );

                    results.push(chunk_result);
                }
                Err(e) => {
                    eprintln!("[VoD Canary] Chunk error: {}", e);
                }
            }
        }

        for handle in handles {
            let _ = handle.join();
        }

        results.sort_by_key(|r| r.index);
        Ok(results)
    }

    /// Process a single chunk with Canary model
    fn process_chunk_canary(model: &mut CanaryModel, chunk: AudioChunk) -> Result<ChunkResult> {
        let chunk_offset = chunk.start_time;
        let chunk_duration = chunk.end_time - chunk.start_time;

        // Transcribe
        let text = model.transcribe(&chunk.samples)?;

        if text.is_empty() {
            return Ok(ChunkResult {
                index: chunk.index,
                segments: Vec::new(),
                start_time: chunk.start_time,
                end_time: chunk.end_time,
            });
        }

        // Split into sentences and estimate word timings
        let segments = Self::text_to_segments_with_estimated_timing(
            &text,
            chunk_offset,
            chunk_duration,
            chunk.index * 1000,
        );

        Ok(ChunkResult {
            index: chunk.index,
            segments,
            start_time: chunk.start_time,
            end_time: chunk.end_time,
        })
    }

    /// Convert text to segments with estimated timing
    /// Since Canary doesn't provide word-level timestamps, we estimate based on word length
    fn text_to_segments_with_estimated_timing(
        text: &str,
        time_offset: f32,
        duration: f32,
        base_id: usize,
    ) -> Vec<VodSegment> {
        // Split into sentences
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '?' || c == '!')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.is_empty() {
            return Vec::new();
        }

        // Count total characters to distribute time
        let total_chars: usize = sentences.iter().map(|s| s.trim().len()).sum();
        if total_chars == 0 {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut current_time = time_offset;

        for (i, sentence) in sentences.iter().enumerate() {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            let sentence_chars = sentence.len();
            let sentence_duration = duration * (sentence_chars as f32 / total_chars as f32);
            let sentence_start = current_time;
            let sentence_end = current_time + sentence_duration;

            // Split into words and estimate their timing
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let total_word_chars: usize = words.iter().map(|w| w.len()).sum();

            let mut word_time = sentence_start;
            let mut vod_words = Vec::new();

            for word in &words {
                let word_duration =
                    sentence_duration * (word.len() as f32 / total_word_chars.max(1) as f32);
                vod_words.push(VodWord {
                    word: word.to_string(),
                    start: word_time,
                    end: word_time + word_duration,
                });
                word_time += word_duration;
            }

            // Add punctuation back to text
            let punct_char = if i < sentences.len() - 1 || text.ends_with('.') {
                '.'
            } else if text.ends_with('?') {
                '?'
            } else if text.ends_with('!') {
                '!'
            } else {
                '.'
            };

            segments.push(VodSegment {
                id: base_id + i,
                text: format!("{}{}", sentence, punct_char),
                start: sentence_start,
                end: sentence_end,
                speaker: None,
                words: vod_words,
            });

            current_time = sentence_end;
        }

        segments
    }

    /// Merge and deduplicate (same logic as TDT)
    fn merge_and_deduplicate(&self, chunk_results: Vec<ChunkResult>) -> Vec<VodSegment> {
        if chunk_results.is_empty() {
            return Vec::new();
        }

        let mut all_segments: Vec<VodSegment> = Vec::new();
        let mut seen_sentences: HashSet<String> = HashSet::new();

        for (i, result) in chunk_results.iter().enumerate() {
            let overlap_end = result.start_time + self.config.overlap_duration_secs;

            for segment in &result.segments {
                if i > 0 && segment.start < overlap_end {
                    let normalized = self.normalize_text(&segment.text);

                    let is_duplicate = seen_sentences.iter().any(|seen| {
                        self.jaccard_similarity(&normalized, seen) >= self.config.dedup_threshold
                    });

                    if is_duplicate {
                        continue;
                    }
                }

                let normalized = self.normalize_text(&segment.text);
                seen_sentences.insert(normalized);
                all_segments.push(segment.clone());
            }
        }

        for (i, segment) in all_segments.iter_mut().enumerate() {
            segment.id = i;
        }

        all_segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
        all_segments
    }

    fn normalize_text(&self, text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn jaccard_similarity(&self, a: &str, b: &str) -> f32 {
        let words_a: HashSet<&str> = a.split_whitespace().collect();
        let words_b: HashSet<&str> = b.split_whitespace().collect();

        if words_a.is_empty() && words_b.is_empty() {
            return 1.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let transcriber = VodTranscriberTDT {
            config: VodConfig::default(),
            model_path: PathBuf::from("."),
            exec_config: ExecutionConfig::default(),
            preprocessor_config: PreprocessorConfig::default(),
        };

        // Identical sentences
        assert_eq!(
            transcriber.jaccard_similarity("hello world", "hello world"),
            1.0
        );

        // Completely different
        assert_eq!(transcriber.jaccard_similarity("hello", "world"), 0.0);

        // Partial overlap
        let sim = transcriber.jaccard_similarity("hello world foo", "hello world bar");
        assert!(sim > 0.4 && sim < 0.8);
    }

    #[test]
    fn test_normalize_text() {
        let transcriber = VodTranscriberTDT {
            config: VodConfig::default(),
            model_path: PathBuf::from("."),
            exec_config: ExecutionConfig::default(),
            preprocessor_config: PreprocessorConfig::default(),
        };

        assert_eq!(
            transcriber.normalize_text("Hello, World!"),
            "hello world"
        );
        assert_eq!(
            transcriber.normalize_text("  Multiple   spaces  "),
            "multiple spaces"
        );
    }

    #[test]
    fn test_vod_config_default() {
        let config = VodConfig::default();
        assert_eq!(config.chunk_duration_secs, 600.0);
        assert_eq!(config.overlap_duration_secs, 60.0);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.dedup_threshold, 0.8);
    }
}
