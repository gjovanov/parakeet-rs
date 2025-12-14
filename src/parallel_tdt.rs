//! Parallel sliding window transcription using multiple ParakeetTDT model instances
//!
//! Similar to parallel_canary.rs but for the faster TDT model.
//! Since TDT is already fast (~100-200ms), parallelization provides
//! throughput benefits rather than latency reduction.

use crate::error::Result;
use crate::execution::ModelConfig;
use crate::parakeet_tdt::ParakeetTDT;
use crate::streaming_transcriber::{
    ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment,
};
use crate::timestamps::TimestampMode;
use crate::transcriber::Transcriber;
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

const SAMPLE_RATE: usize = 16000;

/// Configuration for parallel TDT transcription
#[derive(Debug, Clone)]
pub struct ParallelTDTConfig {
    /// Number of worker threads (default: 4, TDT is fast so fewer needed)
    pub num_threads: usize,
    /// Buffer size in chunks (default: 6)
    pub buffer_size_chunks: usize,
    /// Chunk duration in seconds (default: 1.0)
    pub chunk_duration_secs: f32,
    /// Intra-op threads per model (default: 2)
    pub intra_threads: usize,
}

impl Default for ParallelTDTConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            buffer_size_chunks: 6,
            chunk_duration_secs: 1.0,
            intra_threads: 2,
        }
    }
}

impl ParallelTDTConfig {
    pub fn new(num_threads: usize, buffer_size_chunks: usize) -> Self {
        Self {
            num_threads,
            buffer_size_chunks,
            ..Default::default()
        }
    }

    fn chunk_samples(&self) -> usize {
        (self.chunk_duration_secs * SAMPLE_RATE as f32) as usize
    }
}

/// Job sent to worker thread
struct TranscriptionJob {
    job_id: u64,
    audio: Vec<f32>,
    start_chunk_idx: u64,
    end_chunk_idx: u64,
    window_size: usize,
}

/// Result from worker thread
struct TranscriptionResult {
    job_id: u64,
    text: String,
    start_chunk_idx: u64,
    end_chunk_idx: u64,
    window_size: usize,
    inference_time_ms: u32,
}

/// Merges overlapping transcription results
struct ResultMerger {
    last_emitted_end_chunk: i64,
    buffer_size: usize,
    min_window_size: usize,
}

impl ResultMerger {
    fn new(buffer_size: usize) -> Self {
        Self {
            last_emitted_end_chunk: -1,
            buffer_size,
            min_window_size: 3.min(buffer_size),
        }
    }

    fn process(
        &mut self,
        pending: &mut BTreeMap<u64, TranscriptionResult>,
        chunk_duration_secs: f32,
    ) -> Vec<TranscriptionSegment> {
        let mut segments = Vec::new();
        let mut to_remove = Vec::new();

        let mut best_result: Option<(u64, &TranscriptionResult)> = None;

        for (&job_id, result) in pending.iter() {
            let end_chunk = result.end_chunk_idx as i64;

            if end_chunk <= self.last_emitted_end_chunk {
                to_remove.push(job_id);
                continue;
            }

            if result.window_size < self.min_window_size {
                to_remove.push(job_id);
                continue;
            }

            match &best_result {
                None => best_result = Some((job_id, result)),
                Some((_, best)) => {
                    if result.end_chunk_idx < best.end_chunk_idx
                        || (result.end_chunk_idx == best.end_chunk_idx
                            && result.window_size > best.window_size)
                    {
                        best_result = Some((job_id, result));
                    }
                }
            }
        }

        if let Some((job_id, result)) = best_result {
            let text = truncate_hallucination(&result.text);

            if !text.is_empty() {
                let start_time = result.start_chunk_idx as f32 * chunk_duration_secs;
                let end_time = result.end_chunk_idx as f32 * chunk_duration_secs;

                segments.push(TranscriptionSegment {
                    text,
                    start_time,
                    end_time,
                    speaker: None,
                    confidence: Some(1.0),
                    is_final: true,
                    inference_time_ms: Some(result.inference_time_ms),
                });

                eprintln!(
                    "[ParallelTDT] Emitting segment [{:.1}s-{:.1}s] end_chunk={} (window: {}, inference: {}ms)",
                    start_time, end_time, result.end_chunk_idx, result.window_size, result.inference_time_ms
                );
            }

            self.last_emitted_end_chunk = result.end_chunk_idx as i64;
            to_remove.push(job_id);

            for (&jid, r) in pending.iter() {
                if r.end_chunk_idx <= result.end_chunk_idx && jid != job_id {
                    to_remove.push(jid);
                }
            }
        }

        for job_id in to_remove {
            pending.remove(&job_id);
        }

        segments
    }
}

/// Truncate text at first detected hallucination
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
                    return words[..truncate_at].join(" ");
                }
                return String::new();
            }
        } else {
            consecutive_count = 1;
        }
    }
    text.to_string()
}

/// Parallel TDT transcriber using multiple model instances
pub struct ParallelTDT {
    config: ParallelTDTConfig,
    workers: Vec<JoinHandle<()>>,
    job_tx: Sender<TranscriptionJob>,
    result_rx: Receiver<TranscriptionResult>,
    shutdown: Arc<AtomicBool>,
    chunk_buffer: VecDeque<Vec<f32>>,
    current_chunk: Vec<f32>,
    next_job_id: AtomicU64,
    next_chunk_idx: u64,
    pending_results: BTreeMap<u64, TranscriptionResult>,
    merger: ResultMerger,
    total_samples: usize,
}

impl ParallelTDT {
    pub fn new(
        model_dir: &Path,
        exec_config: Option<ModelConfig>,
        config: Option<ParallelTDTConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        eprintln!(
            "[ParallelTDT] Initializing with {} threads, {} chunk buffer",
            config.num_threads, config.buffer_size_chunks
        );

        let (job_tx, job_rx) = mpsc::channel::<TranscriptionJob>();
        let (result_tx, result_rx) = mpsc::channel::<TranscriptionResult>();
        let job_rx = Arc::new(std::sync::Mutex::new(job_rx));
        let shutdown = Arc::new(AtomicBool::new(false));

        let mut workers = Vec::with_capacity(config.num_threads);

        eprintln!("[ParallelTDT] Loading {} model instances...", config.num_threads);
        let load_start = Instant::now();

        for i in 0..config.num_threads {
            let model_dir = model_dir.to_path_buf();
            let exec_config = exec_config.clone().unwrap_or_else(|| {
                ModelConfig::new()
                    .with_intra_threads(config.intra_threads)
                    .with_inter_threads(1)
            });
            let job_rx = Arc::clone(&job_rx);
            let result_tx = result_tx.clone();
            let shutdown = Arc::clone(&shutdown);

            let handle = thread::spawn(move || {
                eprintln!("[ParallelTDT] Worker {} loading model...", i);
                let mut model = match ParakeetTDT::from_pretrained(&model_dir, Some(exec_config)) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[ParallelTDT] Worker {} failed to load: {}", i, e);
                        return;
                    }
                };
                eprintln!("[ParallelTDT] Worker {} ready", i);

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

                            let transcription = match model.transcribe_samples(
                                job.audio,
                                SAMPLE_RATE as u32,
                                1,
                                Some(TimestampMode::Words),
                            ) {
                                Ok(t) => t,
                                Err(e) => {
                                    eprintln!("[ParallelTDT] Worker {} error: {}", i, e);
                                    continue;
                                }
                            };

                            let inference_time_ms = inference_start.elapsed().as_millis() as u32;

                            let result = TranscriptionResult {
                                job_id: job.job_id,
                                text: transcription.text,
                                start_chunk_idx: job.start_chunk_idx,
                                end_chunk_idx: job.end_chunk_idx,
                                window_size: job.window_size,
                                inference_time_ms,
                            };

                            if result_tx.send(result).is_err() {
                                break;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                eprintln!("[ParallelTDT] Worker {} shutting down", i);
            });

            workers.push(handle);
        }

        eprintln!(
            "[ParallelTDT] All {} workers initialized in {:.2}s",
            config.num_threads,
            load_start.elapsed().as_secs_f32()
        );

        Ok(Self {
            config: config.clone(),
            workers,
            job_tx,
            result_rx,
            shutdown,
            chunk_buffer: VecDeque::with_capacity(config.buffer_size_chunks),
            current_chunk: Vec::new(),
            next_job_id: AtomicU64::new(0),
            next_chunk_idx: 0,
            pending_results: BTreeMap::new(),
            merger: ResultMerger::new(config.buffer_size_chunks),
            total_samples: 0,
        })
    }

    fn dispatch_job(&mut self) {
        if self.chunk_buffer.is_empty() {
            return;
        }

        let window_size = self.chunk_buffer.len();
        let start_chunk_idx = self.next_chunk_idx.saturating_sub(window_size as u64);
        let end_chunk_idx = self.next_chunk_idx - 1;

        let audio: Vec<f32> = self.chunk_buffer.iter().flatten().copied().collect();
        let job_id = self.next_job_id.fetch_add(1, Ordering::Relaxed);

        let job = TranscriptionJob {
            job_id,
            audio,
            start_chunk_idx,
            end_chunk_idx,
            window_size,
        };

        if let Err(e) = self.job_tx.send(job) {
            eprintln!("[ParallelTDT] Failed to dispatch job: {}", e);
        }
    }

    fn collect_results(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.pending_results.insert(result.job_id, result);
        }
    }
}

impl StreamingTranscriber for ParallelTDT {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "parallel-tdt".to_string(),
            display_name: format!("Parallel TDT ({}T)", self.config.num_threads),
            description: format!(
                "Parallel sliding window with {} threads, {} chunk buffer",
                self.config.num_threads, self.config.buffer_size_chunks
            ),
            supports_diarization: false,
            languages: vec!["en".to_string()],
            is_loaded: true,
        }
    }

    fn buffer_duration(&self) -> f32 {
        self.current_chunk.len() as f32 / SAMPLE_RATE as f32
    }

    fn total_duration(&self) -> f32 {
        self.total_samples as f32 / SAMPLE_RATE as f32
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        self.total_samples += samples.len();
        self.current_chunk.extend_from_slice(samples);

        let chunk_samples = self.config.chunk_samples();

        while self.current_chunk.len() >= chunk_samples {
            let chunk: Vec<f32> = self.current_chunk.drain(..chunk_samples).collect();
            self.chunk_buffer.push_back(chunk);
            self.next_chunk_idx += 1;

            while self.chunk_buffer.len() > self.config.buffer_size_chunks {
                self.chunk_buffer.pop_front();
            }

            self.dispatch_job();
        }

        self.collect_results();
        let segments = self
            .merger
            .process(&mut self.pending_results, self.config.chunk_duration_secs);

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.current_chunk.len() as f32 / SAMPLE_RATE as f32,
            total_duration: self.total_samples as f32 / SAMPLE_RATE as f32,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        eprintln!("[ParallelTDT] Finalizing...");

        if !self.current_chunk.is_empty() {
            self.chunk_buffer.push_back(std::mem::take(&mut self.current_chunk));
            self.next_chunk_idx += 1;
            self.dispatch_job();
        }

        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while Instant::now() < deadline && !self.pending_results.is_empty() {
            self.collect_results();
            thread::sleep(std::time::Duration::from_millis(50));
        }

        self.collect_results();
        let mut segments = self
            .merger
            .process(&mut self.pending_results, self.config.chunk_duration_secs);

        let remaining: Vec<_> = std::mem::take(&mut self.pending_results).into_iter().collect();
        for (_, result) in remaining {
            let text = truncate_hallucination(&result.text);
            if !text.is_empty() {
                let start_time = result.start_chunk_idx as f32 * self.config.chunk_duration_secs;
                let end_time = result.end_chunk_idx as f32 * self.config.chunk_duration_secs;
                segments.push(TranscriptionSegment {
                    text,
                    start_time,
                    end_time,
                    speaker: None,
                    confidence: Some(1.0),
                    is_final: true,
                    inference_time_ms: Some(result.inference_time_ms),
                });
            }
        }

        segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
        eprintln!("[ParallelTDT] Finalized with {} segments", segments.len());

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: 0.0,
            total_duration: self.total_samples as f32 / SAMPLE_RATE as f32,
        })
    }

    fn reset(&mut self) {
        self.chunk_buffer.clear();
        self.current_chunk.clear();
        self.next_chunk_idx = 0;
        self.pending_results.clear();
        self.merger = ResultMerger::new(self.config.buffer_size_chunks);
        self.total_samples = 0;
    }
}

impl Drop for ParallelTDT {
    fn drop(&mut self) {
        eprintln!("[ParallelTDT] Shutting down {} workers...", self.workers.len());
        self.shutdown.store(true, Ordering::Relaxed);
        for (i, worker) in self.workers.drain(..).enumerate() {
            if worker.join().is_err() {
                eprintln!("[ParallelTDT] Worker {} panicked", i);
            }
        }
        eprintln!("[ParallelTDT] Shutdown complete");
    }
}
