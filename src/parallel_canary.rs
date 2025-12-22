//! Parallel sliding window transcription using multiple Canary model instances
//!
//! This module implements a parallel transcription strategy where multiple
//! Canary model instances process overlapping audio windows concurrently.
//! This reduces effective latency from ~5s to ~1s on an 8-core CPU.
//!
//! ## Algorithm
//!
//! With N threads and buffer size B chunks:
//! - Thread 1 processes [c1]
//! - Thread 2 processes [c1, c2]
//! - ...
//! - Thread B processes [c1, c2, ..., cB]
//! - Thread B+1 processes [c2, c3, ..., cB+1] (sliding window)
//! - ...
//!
//! Results are merged using timestamp-based alignment, preferring
//! transcriptions with more context (larger windows).

use crate::canary::{CanaryConfig, CanaryModel};
use crate::error::Result;
use crate::execution::ModelConfig;
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[cfg(feature = "sortformer")]
use crate::sortformer_stream::SortformerStream;

/// Configuration for parallel transcription
#[derive(Debug, Clone)]
pub struct ParallelCanaryConfig {
    /// Number of worker threads (default: 8)
    pub num_threads: usize,
    /// Maximum buffer size in 1-second chunks (default: 6)
    pub buffer_size_chunks: usize,
    /// Duration of each chunk in seconds (default: 1.0)
    pub chunk_duration_secs: f32,
    /// Language code for transcription
    pub language: String,
    /// Intra-op threads per model (default: 1 for parallelism)
    pub intra_threads: usize,
}

impl Default for ParallelCanaryConfig {
    fn default() -> Self {
        Self {
            num_threads: 8,
            buffer_size_chunks: 6,
            chunk_duration_secs: 1.0,
            language: "de".to_string(),
            intra_threads: 1,
        }
    }
}

impl ParallelCanaryConfig {
    /// Create config with specified threads and buffer size
    pub fn new(num_threads: usize, buffer_size_secs: usize, language: String) -> Self {
        Self {
            num_threads,
            buffer_size_chunks: buffer_size_secs,
            language,
            ..Default::default()
        }
    }

    /// Samples per chunk at 16kHz
    pub fn samples_per_chunk(&self) -> usize {
        (self.chunk_duration_secs * 16000.0) as usize
    }
}

/// A single audio chunk (1 second of audio)
#[derive(Clone)]
struct AudioChunk {
    /// Chunk index (0-based, increments each second)
    index: u64,
    /// Audio samples (16kHz mono f32)
    samples: Vec<f32>,
    /// Start time in seconds
    start_time: f32,
}

/// Job sent to a worker thread
struct TranscriptionJob {
    /// Unique job ID
    job_id: u64,
    /// Index of first chunk in window
    start_chunk_idx: u64,
    /// Index of last chunk in window
    end_chunk_idx: u64,
    /// Combined audio samples for the window
    audio: Vec<f32>,
    /// Start time of window in seconds
    start_time: f32,
    /// End time of window in seconds
    end_time: f32,
}

/// Result from a worker thread
struct TranscriptionResult {
    /// Job ID this result corresponds to
    job_id: u64,
    /// Index of first chunk in window
    start_chunk_idx: u64,
    /// Index of last chunk in window
    end_chunk_idx: u64,
    /// Transcribed text
    text: String,
    /// Start time of transcription
    start_time: f32,
    /// End time of transcription
    end_time: f32,
    /// Inference duration in milliseconds
    inference_time_ms: u32,
    /// Number of chunks in window (context size)
    window_size: usize,
}

/// Merges overlapping transcription results
///
/// Strategy: Emit results as soon as they arrive, but avoid emitting
/// overlapping time regions. During warmup (first buffer_size chunks),
/// emit partial results. After warmup, prefer full-context results.
struct ResultMerger {
    /// Last emitted end time (in chunks)
    last_emitted_end_chunk: i64,
    /// Buffer size (max window)
    buffer_size: usize,
    /// Minimum window size to consider (smaller windows may hallucinate)
    min_window_size: usize,
}

impl ResultMerger {
    fn new(buffer_size: usize) -> Self {
        Self {
            last_emitted_end_chunk: -1,
            buffer_size,
            // Require at least 3 seconds of context to avoid hallucinations
            min_window_size: 3.min(buffer_size),
        }
    }

    /// Process pending results and emit segments
    ///
    /// Emits results for new time regions, preferring those with more context.
    fn process(
        &mut self,
        pending: &mut BTreeMap<u64, TranscriptionResult>,
        _buffer_size: usize,
    ) -> Vec<TranscriptionSegment> {
        let mut segments = Vec::new();
        let mut to_remove = Vec::new();

        // Group results by their end_chunk_idx (each end_chunk can have multiple results)
        // Find the best result that covers a NEW time region
        let mut best_result: Option<(u64, &TranscriptionResult)> = None;

        for (&job_id, result) in pending.iter() {
            let end_chunk = result.end_chunk_idx as i64;

            // Skip results we've already processed (covers time we already emitted)
            if end_chunk <= self.last_emitted_end_chunk {
                to_remove.push(job_id);
                continue;
            }

            // Skip results with too small window (prone to hallucination)
            if result.window_size < self.min_window_size {
                to_remove.push(job_id);
                continue;
            }

            // Find the best result for the EARLIEST new end_chunk
            match &best_result {
                None => best_result = Some((job_id, result)),
                Some((_, best)) => {
                    // Prefer earlier end_chunk (to emit as soon as possible)
                    // If same end_chunk, prefer larger window (more context)
                    if result.end_chunk_idx < best.end_chunk_idx
                        || (result.end_chunk_idx == best.end_chunk_idx
                            && result.window_size > best.window_size)
                    {
                        best_result = Some((job_id, result));
                    }
                }
            }
        }

        // Emit the best result if found
        if let Some((job_id, result)) = best_result {
            let text = truncate_hallucination(&result.text);

            if !text.is_empty() {
                segments.push(TranscriptionSegment {
                    text,
                    start_time: result.start_time,
                    end_time: result.end_time,
                    speaker: None,
                    confidence: Some(1.0),
                    is_final: true,
                    inference_time_ms: Some(result.inference_time_ms),
                });

                eprintln!(
                    "[ParallelCanary] Emitting segment [{:.1}s-{:.1}s] end_chunk={} (window: {}, inference: {}ms)",
                    result.start_time, result.end_time, result.end_chunk_idx, result.window_size, result.inference_time_ms
                );
            }

            self.last_emitted_end_chunk = result.end_chunk_idx as i64;
            to_remove.push(job_id);

            // Remove all results with end_chunk <= the one we just emitted
            for (&jid, r) in pending.iter() {
                if r.end_chunk_idx <= result.end_chunk_idx && jid != job_id {
                    to_remove.push(jid);
                }
            }
        }

        // Remove processed results
        for job_id in to_remove {
            pending.remove(&job_id);
        }

        segments
    }
}

/// Truncate text at first detected hallucination (3+ consecutive repeated words)
fn truncate_hallucination(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return text.to_string();
    }

    let mut consecutive_count = 1;

    for i in 1..words.len() {
        if words[i].to_lowercase() == words[i - 1].to_lowercase() && words[i].len() > 1 {
            consecutive_count += 1;
            if consecutive_count >= 3 {
                // Truncate before the repetition started
                let truncate_at = i - consecutive_count + 1;
                if truncate_at > 0 {
                    eprintln!(
                        "[ParallelCanary] Truncating hallucination at word {}: '{}'",
                        truncate_at,
                        words[i]
                    );
                    return words[..truncate_at].join(" ");
                }
                return String::new();
            }
        } else {
            consecutive_count = 1;
        }
    }

    // Also check for repeated phrases (2-3 word patterns)
    for pattern_len in 2..=3 {
        if words.len() < pattern_len * 3 {
            continue;
        }

        for i in 0..=(words.len() - pattern_len * 3) {
            let pattern: Vec<&str> = words[i..i + pattern_len].to_vec();
            let mut pattern_count = 1;

            let mut j = i + pattern_len;
            while j + pattern_len <= words.len() {
                let candidate: Vec<&str> = words[j..j + pattern_len].to_vec();
                if candidate
                    .iter()
                    .zip(pattern.iter())
                    .all(|(a, b)| a.to_lowercase() == b.to_lowercase())
                {
                    pattern_count += 1;
                    if pattern_count >= 3 {
                        eprintln!(
                            "[ParallelCanary] Truncating repeated phrase at {}: '{}'",
                            i,
                            pattern.join(" ")
                        );
                        if i > 0 {
                            return words[..i].join(" ");
                        }
                        return String::new();
                    }
                    j += pattern_len;
                } else {
                    break;
                }
            }
        }
    }

    text.to_string()
}

/// Parallel Canary transcriber with multiple model instances
pub struct ParallelCanary {
    /// Configuration
    config: ParallelCanaryConfig,

    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,

    /// Channel to send jobs to workers
    job_tx: Sender<TranscriptionJob>,

    /// Channel to receive results from workers
    result_rx: Receiver<TranscriptionResult>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Audio chunk buffer
    chunk_buffer: VecDeque<AudioChunk>,

    /// Current chunk being accumulated
    current_chunk_samples: Vec<f32>,

    /// Next chunk index
    next_chunk_idx: u64,

    /// Next job ID
    next_job_id: AtomicU64,

    /// Next worker index (round-robin)
    next_worker_idx: usize,

    /// Pending results awaiting merge
    pending_results: BTreeMap<u64, TranscriptionResult>,

    /// Result merger
    merger: ResultMerger,

    /// Total samples received
    total_samples: usize,

    /// Diarizer for speaker identification
    #[cfg(feature = "sortformer")]
    diarizer: Option<SortformerStream>,
}

impl ParallelCanary {
    /// Create a new parallel transcriber
    ///
    /// # Arguments
    /// * `model_dir` - Path to Canary model directory
    /// * `exec_config` - Optional execution configuration
    /// * `config` - Parallel transcription configuration
    pub fn new(
        model_dir: &Path,
        exec_config: Option<ModelConfig>,
        config: Option<ParallelCanaryConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        eprintln!(
            "[ParallelCanary] Initializing with {} threads, {} chunk buffer, language: {}",
            config.num_threads, config.buffer_size_chunks, config.language
        );

        // Create channels
        let (job_tx, job_rx) = mpsc::channel::<TranscriptionJob>();
        let (result_tx, result_rx) = mpsc::channel::<TranscriptionResult>();

        // Wrap job receiver in Arc for sharing across workers
        let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

        let shutdown = Arc::new(AtomicBool::new(false));

        // Spawn worker threads, each with its own model instance
        let mut workers = Vec::with_capacity(config.num_threads);

        eprintln!("[ParallelCanary] Loading {} model instances...", config.num_threads);
        let load_start = Instant::now();

        for i in 0..config.num_threads {
            let model_dir = model_dir.to_path_buf();
            let exec_config = exec_config.clone().unwrap_or_else(|| {
                ModelConfig::new()
                    .with_intra_threads(config.intra_threads)
                    .with_inter_threads(1)
            });
            let language = config.language.clone();
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let shutdown = Arc::clone(&shutdown);

            let handle = thread::spawn(move || {
                // Load model for this worker with the specified language
                eprintln!("[ParallelCanary] Worker {} loading model...", i);
                let canary_config = CanaryConfig {
                    language: language.clone(),
                    ..Default::default()
                };
                let mut model = match CanaryModel::from_pretrained(&model_dir, Some(exec_config), Some(canary_config)) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[ParallelCanary] Worker {} failed to load model: {}", i, e);
                        return;
                    }
                };
                eprintln!("[ParallelCanary] Worker {} ready (language: {})", i, language);

                // Process jobs until shutdown
                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

                    // Try to get a job
                    let job = {
                        let rx = job_rx.lock().unwrap();
                        rx.recv_timeout(std::time::Duration::from_millis(100))
                    };

                    match job {
                        Ok(job) => {
                            let inference_start = Instant::now();

                            // Run transcription
                            let text = match model.transcribe(&job.audio) {
                                Ok(t) => t,
                                Err(e) => {
                                    eprintln!(
                                        "[ParallelCanary] Worker {} inference error: {}",
                                        i, e
                                    );
                                    continue;
                                }
                            };

                            let inference_time_ms = inference_start.elapsed().as_millis() as u32;

                            // Send result
                            let result = TranscriptionResult {
                                job_id: job.job_id,
                                start_chunk_idx: job.start_chunk_idx,
                                end_chunk_idx: job.end_chunk_idx,
                                text,
                                start_time: job.start_time,
                                end_time: job.end_time,
                                inference_time_ms,
                                window_size: (job.end_chunk_idx - job.start_chunk_idx + 1) as usize,
                            };

                            if result_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }

                eprintln!("[ParallelCanary] Worker {} shutting down", i);
            });

            workers.push(handle);
        }

        eprintln!(
            "[ParallelCanary] All {} workers initialized in {:.2}s",
            config.num_threads,
            load_start.elapsed().as_secs_f32()
        );

        Ok(Self {
            config: config.clone(),
            workers,
            job_tx,
            result_rx,
            shutdown,
            chunk_buffer: VecDeque::new(),
            current_chunk_samples: Vec::new(),
            next_chunk_idx: 0,
            next_job_id: AtomicU64::new(0),
            next_worker_idx: 0,
            pending_results: BTreeMap::new(),
            merger: ResultMerger::new(config.buffer_size_chunks),
            total_samples: 0,
            #[cfg(feature = "sortformer")]
            diarizer: None,
        })
    }

    /// Create a new parallel transcriber with optional diarization
    #[cfg(feature = "sortformer")]
    pub fn new_with_diarization<P1: AsRef<Path>, P2: AsRef<Path>>(
        model_dir: P1,
        diar_model_path: Option<P2>,
        exec_config: Option<ModelConfig>,
        config: Option<ParallelCanaryConfig>,
    ) -> Result<Self> {
        let mut transcriber = Self::new(model_dir.as_ref(), exec_config.clone(), config)?;

        if let Some(diar_path) = diar_model_path {
            eprintln!("[ParallelCanary] Creating diarizer from {:?}", diar_path.as_ref());
            transcriber.diarizer = Some(SortformerStream::new(diar_path)?);
        }

        Ok(transcriber)
    }

    /// Check if diarization is available
    pub fn has_diarization(&self) -> bool {
        #[cfg(feature = "sortformer")]
        return self.diarizer.is_some();
        #[cfg(not(feature = "sortformer"))]
        return false;
    }

    /// Get speaker at a given time
    fn get_speaker_at(&self, time: f32) -> Option<usize> {
        #[cfg(feature = "sortformer")]
        if let Some(diarizer) = &self.diarizer {
            return diarizer.get_speaker_at(time);
        }
        None
    }

    /// Dispatch a job to the next worker
    fn dispatch_job(&mut self) {
        if self.chunk_buffer.is_empty() {
            return;
        }

        // Build sliding window
        let window_end = self.chunk_buffer.len();
        let window_start = if window_end > self.config.buffer_size_chunks {
            window_end - self.config.buffer_size_chunks
        } else {
            0
        };

        // Collect audio from window
        let mut audio = Vec::new();
        let mut start_chunk_idx = 0;
        let mut end_chunk_idx = 0;
        let mut start_time = 0.0f32;
        let mut end_time = 0.0f32;

        for (i, chunk) in self.chunk_buffer.iter().enumerate() {
            if i >= window_start && i < window_end {
                if i == window_start {
                    start_chunk_idx = chunk.index;
                    start_time = chunk.start_time;
                }
                audio.extend(&chunk.samples);
                end_chunk_idx = chunk.index;
                end_time = chunk.start_time + self.config.chunk_duration_secs;
            }
        }

        if audio.is_empty() {
            return;
        }

        let job_id = self.next_job_id.fetch_add(1, Ordering::Relaxed);

        let job = TranscriptionJob {
            job_id,
            start_chunk_idx,
            end_chunk_idx,
            audio,
            start_time,
            end_time,
        };

        // Send to workers (they compete for jobs)
        if let Err(e) = self.job_tx.send(job) {
            eprintln!("[ParallelCanary] Failed to dispatch job: {}", e);
        }

        self.next_worker_idx = (self.next_worker_idx + 1) % self.config.num_threads;
    }

    /// Collect completed results from workers
    fn collect_results(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.pending_results.insert(result.job_id, result);
        }
    }

    /// Trim old chunks from buffer
    fn trim_buffer(&mut self) {
        // Keep at most 2x buffer_size chunks
        let max_chunks = self.config.buffer_size_chunks * 2;
        while self.chunk_buffer.len() > max_chunks {
            self.chunk_buffer.pop_front();
        }
    }
}

impl StreamingTranscriber for ParallelCanary {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "parallel-canary".to_string(),
            display_name: format!("Parallel Canary ({}x{})", self.config.num_threads, self.config.buffer_size_chunks),
            description: format!(
                "Parallel sliding window transcription with {} threads and {} second buffer",
                self.config.num_threads, self.config.buffer_size_chunks
            ),
            supports_diarization: self.has_diarization(),
            languages: vec![self.config.language.clone()],
            is_loaded: true,
        }
    }

    fn buffer_duration(&self) -> f32 {
        self.chunk_buffer.len() as f32 * self.config.chunk_duration_secs
    }

    fn total_duration(&self) -> f32 {
        self.total_samples as f32 / 16000.0
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        // Push audio to diarizer if available
        #[cfg(feature = "sortformer")]
        if let Some(diarizer) = &mut self.diarizer {
            let _ = diarizer.push_audio(samples);
        }

        self.total_samples += samples.len();

        // Accumulate samples into current chunk
        self.current_chunk_samples.extend_from_slice(samples);

        let samples_per_chunk = self.config.samples_per_chunk();
        let mut segments = Vec::new();

        // Process complete chunks
        while self.current_chunk_samples.len() >= samples_per_chunk {
            // Extract chunk
            let chunk_samples: Vec<f32> = self.current_chunk_samples.drain(..samples_per_chunk).collect();

            let chunk = AudioChunk {
                index: self.next_chunk_idx,
                start_time: self.next_chunk_idx as f32 * self.config.chunk_duration_secs,
                samples: chunk_samples,
            };

            self.chunk_buffer.push_back(chunk);
            self.next_chunk_idx += 1;

            // Dispatch job for this chunk
            self.dispatch_job();

            // Trim old chunks
            self.trim_buffer();
        }

        // Collect any completed results
        self.collect_results();

        // Merge and emit segments
        let merged = self.merger.process(
            &mut self.pending_results,
            self.config.buffer_size_chunks,
        );
        segments.extend(merged);

        // Add speaker info to segments if diarization is available
        for segment in &mut segments {
            let mid_time = (segment.start_time + segment.end_time) / 2.0;
            segment.speaker = self.get_speaker_at(mid_time);
        }

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.chunk_buffer.len() as f32 * self.config.chunk_duration_secs,
            total_duration: self.total_samples as f32 / 16000.0,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        eprintln!("[ParallelCanary] Finalizing...");

        // Process any remaining samples as final chunk
        if !self.current_chunk_samples.is_empty() {
            let chunk = AudioChunk {
                index: self.next_chunk_idx,
                start_time: self.next_chunk_idx as f32 * self.config.chunk_duration_secs,
                samples: std::mem::take(&mut self.current_chunk_samples),
            };
            self.chunk_buffer.push_back(chunk);
            self.next_chunk_idx += 1;
            self.dispatch_job();
        }

        // Wait for pending jobs to complete (with timeout)
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while Instant::now() < deadline {
            self.collect_results();

            // Check if we have results for all dispatched jobs
            let total_jobs = self.next_job_id.load(Ordering::Relaxed);
            let completed = self.pending_results.len() as u64;
            if completed >= total_jobs.saturating_sub(self.config.num_threads as u64) {
                break;
            }

            thread::sleep(std::time::Duration::from_millis(100));
        }

        // Collect final results
        self.collect_results();

        // Force emit all pending as final
        let mut segments = Vec::new();
        let pending = std::mem::take(&mut self.pending_results);
        for (_, result) in pending {
            let text = truncate_hallucination(&result.text);
            if !text.is_empty() {
                let mid_time = (result.start_time + result.end_time) / 2.0;
                segments.push(TranscriptionSegment {
                    text,
                    start_time: result.start_time,
                    end_time: result.end_time,
                    speaker: self.get_speaker_at(mid_time),
                    confidence: Some(1.0),
                    is_final: true,
                    inference_time_ms: Some(result.inference_time_ms),
                });
            }
        }

        // Sort by start time
        segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

        // Add speaker info to any segments that don't have it
        for segment in &mut segments {
            if segment.speaker.is_none() {
                let mid_time = (segment.start_time + segment.end_time) / 2.0;
                segment.speaker = self.get_speaker_at(mid_time);
            }
        }

        eprintln!("[ParallelCanary] Finalized with {} segments", segments.len());

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.total_samples as f32 / 16000.0,
        })
    }

    fn reset(&mut self) {
        self.chunk_buffer.clear();
        self.current_chunk_samples.clear();
        self.next_chunk_idx = 0;
        self.next_job_id.store(0, Ordering::Relaxed);
        self.next_worker_idx = 0;
        self.pending_results.clear();
        self.total_samples = 0;
    }
}

impl Drop for ParallelCanary {
    fn drop(&mut self) {
        eprintln!("[ParallelCanary] Shutting down {} workers...", self.workers.len());
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for workers to finish
        for (i, worker) in self.workers.drain(..).enumerate() {
            if worker.join().is_err() {
                eprintln!("[ParallelCanary] Worker {} panicked", i);
            }
        }

        eprintln!("[ParallelCanary] Shutdown complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_hallucination_single_word() {
        let text = "hello world world world test";
        let result = truncate_hallucination(text);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_hallucination_phrase() {
        let text = "hello world how are how are how are you";
        let result = truncate_hallucination(text);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_truncate_hallucination_no_repetition() {
        let text = "hello world how are you today";
        let result = truncate_hallucination(text);
        assert_eq!(result, text);
    }

    #[test]
    fn test_config_samples_per_chunk() {
        let config = ParallelCanaryConfig::default();
        assert_eq!(config.samples_per_chunk(), 16000);
    }
}
