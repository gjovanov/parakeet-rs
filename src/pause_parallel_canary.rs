//! Pause-based parallel transcription using multiple Canary model instances
//!
//! This module combines the pause-detection approach from Speedy mode with
//! the parallel worker pool from Parallel mode. Jobs are dispatched only
//! when speech pauses are detected, resulting in:
//! - Natural segment boundaries (better transcription quality)
//! - Lower job volume (~0.3-0.5 jobs/sec vs 1 job/sec)
//! - Ordered output emission (simpler merging)
//!
//! ## Algorithm
//!
//! 1. Accumulate audio, detecting speech vs silence using energy threshold
//! 2. When pause detected (silence > threshold), package segment as job
//! 3. Dispatch job to next available worker (round-robin)
//! 4. Workers return results tagged with segment index
//! 5. Merger emits results in order as they complete

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

const SAMPLE_RATE: usize = 16000;

/// Configuration for pause-based parallel transcription
#[derive(Debug, Clone)]
pub struct PauseParallelConfig {
    /// Number of worker threads (default: 8)
    pub num_threads: usize,
    /// Language code for transcription
    pub language: String,
    /// Intra-op threads per model (default: 1 for parallelism)
    pub intra_threads: usize,
    /// Pause threshold in seconds (default: 0.3)
    /// Silence longer than this triggers segment boundary
    pub pause_threshold_secs: f32,
    /// Energy threshold for silence detection (default: 0.008)
    pub silence_energy_threshold: f32,
    /// Maximum segment duration in seconds (default: 5.0)
    /// Forces segment break even without pause to limit latency
    pub max_segment_duration_secs: f32,
    /// Context buffer size in seconds (default: 3.0)
    /// Past audio prepended to each segment for better ASR context
    pub context_buffer_secs: f32,
}

impl Default for PauseParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 8,
            language: "de".to_string(),
            intra_threads: 1,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            max_segment_duration_secs: 5.0,
            context_buffer_secs: 3.0,
        }
    }
}

impl PauseParallelConfig {
    /// Create config with specified threads and language
    pub fn new(num_threads: usize, language: String) -> Self {
        Self {
            num_threads,
            language,
            ..Default::default()
        }
    }

    /// Samples per second
    pub fn samples_per_second(&self) -> usize {
        SAMPLE_RATE
    }
}

/// Calculate RMS (root mean square) energy of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Job sent to a worker thread
struct TranscriptionJob {
    /// Segment index (sequential, for ordered emission)
    segment_idx: u64,
    /// Audio samples (context + segment)
    audio: Vec<f32>,
    /// Start time of actual segment (excluding context)
    segment_start_time: f32,
    /// End time of segment
    segment_end_time: f32,
    /// Duration of prepended context
    context_duration: f32,
}

/// Result from a worker thread
struct TranscriptionResult {
    /// Segment index
    segment_idx: u64,
    /// Transcribed text
    text: String,
    /// Start time of segment
    start_time: f32,
    /// End time of segment
    end_time: f32,
    /// Inference duration in milliseconds
    inference_time_ms: u32,
}

/// Manages ordered emission of results
struct OrderedMerger {
    /// Next segment index to emit
    next_emit_idx: u64,
    /// Last emitted end time for deduplication
    last_emitted_end_time: f32,
}

impl OrderedMerger {
    fn new() -> Self {
        Self {
            next_emit_idx: 0,
            last_emitted_end_time: 0.0,
        }
    }

    /// Process pending results and emit any that are ready (in order)
    fn process(
        &mut self,
        pending: &mut BTreeMap<u64, TranscriptionResult>,
    ) -> Vec<TranscriptionSegment> {
        let mut segments = Vec::new();

        // Emit results in order
        while let Some(result) = pending.remove(&self.next_emit_idx) {
            let text = truncate_hallucination(&result.text);

            if !text.is_empty() {
                // Skip if this segment overlaps with already emitted content
                if result.start_time >= self.last_emitted_end_time - 0.1 {
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
                        "[PauseParallelCanary] Emitting segment {} [{:.1}s-{:.1}s] (inference: {}ms)",
                        result.segment_idx, result.start_time, result.end_time, result.inference_time_ms
                    );

                    self.last_emitted_end_time = result.end_time;
                }
            }

            self.next_emit_idx += 1;
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
                let truncate_at = i - consecutive_count + 1;
                if truncate_at > 0 {
                    eprintln!(
                        "[PauseParallelCanary] Truncating hallucination at word {}: '{}'",
                        truncate_at, words[i]
                    );
                    return words[..truncate_at].join(" ");
                }
                return String::new();
            }
        } else {
            consecutive_count = 1;
        }
    }

    // Check for repeated phrases (2-3 word patterns)
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
                            "[PauseParallelCanary] Truncating repeated phrase at {}: '{}'",
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

/// Pause-based parallel Canary transcriber
pub struct PauseParallelCanary {
    /// Configuration
    config: PauseParallelConfig,

    /// Worker thread handles
    workers: Vec<JoinHandle<()>>,

    /// Channel to send jobs to workers
    job_tx: Sender<TranscriptionJob>,

    /// Channel to receive results from workers
    result_rx: Receiver<TranscriptionResult>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Ring buffer for context (past audio)
    context_buffer: VecDeque<f32>,
    context_buffer_samples: usize,

    /// Current segment being accumulated
    current_segment_audio: Vec<f32>,
    current_segment_start: f32,

    /// Pause detection state
    in_speech: bool,
    silence_start_time: Option<f32>,

    /// Segment tracking
    next_segment_idx: AtomicU64,

    /// Pending results awaiting ordered emission
    pending_results: BTreeMap<u64, TranscriptionResult>,

    /// Ordered merger
    merger: OrderedMerger,

    /// Total samples received
    total_samples: usize,

    /// Jobs in flight counter
    jobs_in_flight: Arc<AtomicU64>,
}

impl PauseParallelCanary {
    /// Create a new pause-based parallel transcriber
    pub fn new(
        model_dir: &Path,
        exec_config: Option<ModelConfig>,
        config: Option<PauseParallelConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        eprintln!(
            "[PauseParallelCanary] Initializing with {} threads, language: {}, pause: {}ms",
            config.num_threads, config.language, (config.pause_threshold_secs * 1000.0) as u32
        );

        // Create channels
        let (job_tx, job_rx) = mpsc::channel::<TranscriptionJob>();
        let (result_tx, result_rx) = mpsc::channel::<TranscriptionResult>();

        // Wrap job receiver in Arc for sharing across workers
        let job_rx = Arc::new(std::sync::Mutex::new(job_rx));

        let shutdown = Arc::new(AtomicBool::new(false));
        let jobs_in_flight = Arc::new(AtomicU64::new(0));

        // Spawn worker threads
        let mut workers = Vec::with_capacity(config.num_threads);

        eprintln!("[PauseParallelCanary] Loading {} model instances...", config.num_threads);
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
            let jobs_counter = Arc::clone(&jobs_in_flight);

            let handle = thread::spawn(move || {
                eprintln!("[PauseParallelCanary] Worker {} loading model...", i);
                let canary_config = CanaryConfig {
                    language: language.clone(),
                    ..Default::default()
                };
                let mut model = match CanaryModel::from_pretrained(&model_dir, Some(exec_config), Some(canary_config)) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[PauseParallelCanary] Worker {} failed to load model: {}", i, e);
                        return;
                    }
                };
                eprintln!("[PauseParallelCanary] Worker {} ready (language: {})", i, language);

                loop {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }

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
                                        "[PauseParallelCanary] Worker {} inference error: {}",
                                        i, e
                                    );
                                    jobs_counter.fetch_sub(1, Ordering::Relaxed);
                                    continue;
                                }
                            };

                            let inference_time_ms = inference_start.elapsed().as_millis() as u32;

                            // Send result
                            let result = TranscriptionResult {
                                segment_idx: job.segment_idx,
                                text,
                                start_time: job.segment_start_time,
                                end_time: job.segment_end_time,
                                inference_time_ms,
                            };

                            jobs_counter.fetch_sub(1, Ordering::Relaxed);

                            if result_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }

                eprintln!("[PauseParallelCanary] Worker {} shutting down", i);
            });

            workers.push(handle);
        }

        eprintln!(
            "[PauseParallelCanary] All {} workers initialized in {:.2}s",
            config.num_threads,
            load_start.elapsed().as_secs_f32()
        );

        let context_buffer_samples = (config.context_buffer_secs * SAMPLE_RATE as f32) as usize;

        Ok(Self {
            config,
            workers,
            job_tx,
            result_rx,
            shutdown,
            context_buffer: VecDeque::with_capacity(context_buffer_samples),
            context_buffer_samples,
            current_segment_audio: Vec::new(),
            current_segment_start: 0.0,
            in_speech: false,
            silence_start_time: None,
            next_segment_idx: AtomicU64::new(0),
            pending_results: BTreeMap::new(),
            merger: OrderedMerger::new(),
            total_samples: 0,
            jobs_in_flight,
        })
    }

    /// Dispatch a segment as a job to workers
    fn dispatch_segment(&mut self, segment_audio: Vec<f32>, start_time: f32, end_time: f32) {
        if segment_audio.is_empty() {
            return;
        }

        // Prepend context from ring buffer
        let context: Vec<f32> = self.context_buffer.iter().copied().collect();
        let context_duration = context.len() as f32 / SAMPLE_RATE as f32;

        let mut audio = context;
        audio.extend(&segment_audio);

        let segment_idx = self.next_segment_idx.fetch_add(1, Ordering::Relaxed);

        let job = TranscriptionJob {
            segment_idx,
            audio,
            segment_start_time: start_time,
            segment_end_time: end_time,
            context_duration,
        };

        eprintln!(
            "[PauseParallelCanary] Dispatching segment {} [{:.1}s-{:.1}s] ({:.1}s audio + {:.1}s context)",
            segment_idx, start_time, end_time,
            segment_audio.len() as f32 / SAMPLE_RATE as f32,
            context_duration
        );

        self.jobs_in_flight.fetch_add(1, Ordering::Relaxed);

        if let Err(e) = self.job_tx.send(job) {
            eprintln!("[PauseParallelCanary] Failed to dispatch job: {}", e);
            self.jobs_in_flight.fetch_sub(1, Ordering::Relaxed);
        }

        // Update context buffer with the segment we just sent
        for sample in &segment_audio {
            self.context_buffer.push_back(*sample);
        }
        while self.context_buffer.len() > self.context_buffer_samples {
            self.context_buffer.pop_front();
        }
    }

    /// Collect completed results from workers
    fn collect_results(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.pending_results.insert(result.segment_idx, result);
        }
    }
}

impl StreamingTranscriber for PauseParallelCanary {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "pause-parallel-canary".to_string(),
            display_name: format!("Pause-Parallel Canary ({}T)", self.config.num_threads),
            description: format!(
                "Pause-based parallel transcription with {} threads, {} pause threshold",
                self.config.num_threads,
                format!("{}ms", (self.config.pause_threshold_secs * 1000.0) as u32)
            ),
            supports_diarization: false,
            languages: vec![self.config.language.clone()],
            is_loaded: true,
        }
    }

    fn buffer_duration(&self) -> f32 {
        self.current_segment_audio.len() as f32 / SAMPLE_RATE as f32
    }

    fn total_duration(&self) -> f32 {
        self.total_samples as f32 / SAMPLE_RATE as f32
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        let current_time = self.total_samples as f32 / SAMPLE_RATE as f32;
        let chunk_duration = samples.len() as f32 / SAMPLE_RATE as f32;
        let chunk_end_time = current_time + chunk_duration;

        self.total_samples += samples.len();

        // Calculate energy for pause detection
        let rms = calculate_rms(samples);
        let is_silence = rms < self.config.silence_energy_threshold;

        // State machine for segment accumulation
        if is_silence {
            if self.in_speech {
                // Still accumulate during brief silences
                self.current_segment_audio.extend_from_slice(samples);

                // Track silence start
                if self.silence_start_time.is_none() {
                    self.silence_start_time = Some(current_time);
                }

                let silence_duration = chunk_end_time - self.silence_start_time.unwrap();

                // Check for pause
                if silence_duration >= self.config.pause_threshold_secs {
                    // Pause detected - dispatch segment
                    if !self.current_segment_audio.is_empty() {
                        let segment_audio = std::mem::take(&mut self.current_segment_audio);
                        self.dispatch_segment(
                            segment_audio,
                            self.current_segment_start,
                            chunk_end_time,
                        );
                    }
                    self.in_speech = false;
                    self.silence_start_time = None;
                }
            }
            // If not in_speech and silence, just wait
        } else {
            // Speech detected
            if !self.in_speech {
                // Transition: silence -> speech
                self.in_speech = true;
                self.current_segment_start = current_time;
            }
            self.silence_start_time = None;
            self.current_segment_audio.extend_from_slice(samples);
        }

        // Force segment break if too long (limit latency)
        let current_segment_duration = self.current_segment_audio.len() as f32 / SAMPLE_RATE as f32;
        if current_segment_duration > self.config.max_segment_duration_secs {
            eprintln!(
                "[PauseParallelCanary] Forcing segment break at {:.1}s (max duration reached)",
                chunk_end_time
            );
            let segment_audio = std::mem::take(&mut self.current_segment_audio);
            self.dispatch_segment(segment_audio, self.current_segment_start, chunk_end_time);
            self.current_segment_start = chunk_end_time;
        }

        // Collect results from workers
        self.collect_results();

        // Process and emit ordered segments
        let segments = self.merger.process(&mut self.pending_results);

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.current_segment_audio.len() as f32 / SAMPLE_RATE as f32,
            total_duration: self.total_samples as f32 / SAMPLE_RATE as f32,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        eprintln!("[PauseParallelCanary] Finalizing...");

        // Dispatch any remaining audio
        if !self.current_segment_audio.is_empty() {
            let segment_audio = std::mem::take(&mut self.current_segment_audio);
            let end_time = self.total_samples as f32 / SAMPLE_RATE as f32;
            self.dispatch_segment(segment_audio, self.current_segment_start, end_time);
        }

        // Wait for all jobs to complete (with timeout)
        let deadline = Instant::now() + std::time::Duration::from_secs(30);
        while Instant::now() < deadline {
            self.collect_results();

            // Check if all jobs completed
            let jobs_pending = self.jobs_in_flight.load(Ordering::Relaxed);
            let results_pending = self.pending_results.len() as u64;
            let total_segments = self.next_segment_idx.load(Ordering::Relaxed);
            let emitted = self.merger.next_emit_idx;

            if jobs_pending == 0 && emitted + results_pending >= total_segments {
                break;
            }

            thread::sleep(std::time::Duration::from_millis(100));
        }

        // Collect final results
        self.collect_results();

        // Emit all remaining results in order
        let mut segments = self.merger.process(&mut self.pending_results);

        // Force emit any remaining out-of-order results
        let remaining: Vec<_> = std::mem::take(&mut self.pending_results).into_iter().collect();
        for (_, result) in remaining {
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
            }
        }

        // Sort by start time
        segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

        eprintln!("[PauseParallelCanary] Finalized with {} segments", segments.len());

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.total_samples as f32 / SAMPLE_RATE as f32,
        })
    }

    fn reset(&mut self) {
        self.context_buffer.clear();
        self.current_segment_audio.clear();
        self.current_segment_start = 0.0;
        self.in_speech = false;
        self.silence_start_time = None;
        self.next_segment_idx.store(0, Ordering::Relaxed);
        self.pending_results.clear();
        self.merger = OrderedMerger::new();
        self.total_samples = 0;
    }
}

impl Drop for PauseParallelCanary {
    fn drop(&mut self) {
        eprintln!("[PauseParallelCanary] Shutting down {} workers...", self.workers.len());
        self.shutdown.store(true, Ordering::Relaxed);

        for (i, worker) in self.workers.drain(..).enumerate() {
            if worker.join().is_err() {
                eprintln!("[PauseParallelCanary] Worker {} panicked", i);
            }
        }

        eprintln!("[PauseParallelCanary] Shutdown complete");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_hallucination() {
        let text = "hello world world world test";
        let result = truncate_hallucination(text);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_calculate_rms() {
        let samples = vec![0.1, -0.1, 0.1, -0.1];
        let rms = calculate_rms(&samples);
        assert!((rms - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_config_default() {
        let config = PauseParallelConfig::default();
        assert_eq!(config.num_threads, 8);
        assert_eq!(config.pause_threshold_secs, 0.3);
    }
}
