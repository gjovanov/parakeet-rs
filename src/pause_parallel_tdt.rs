//! Pause-based parallel transcription using multiple ParakeetTDT model instances
//!
//! Similar to pause_parallel_canary.rs but for the faster TDT model.
//! Jobs are dispatched when speech pauses are detected, resulting in
//! natural segment boundaries and ordered output.

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

/// Configuration for pause-based parallel TDT transcription
#[derive(Debug, Clone)]
pub struct PauseParallelTDTConfig {
    /// Number of worker threads (default: 4)
    pub num_threads: usize,
    /// Intra-op threads per model (default: 2)
    pub intra_threads: usize,
    /// Pause threshold in seconds (default: 0.3)
    pub pause_threshold_secs: f32,
    /// Energy threshold for silence detection (default: 0.008)
    pub silence_energy_threshold: f32,
    /// Maximum segment duration in seconds (default: 5.0)
    pub max_segment_duration_secs: f32,
    /// Context buffer size in seconds (default: 2.0)
    pub context_buffer_secs: f32,
}

impl Default for PauseParallelTDTConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            intra_threads: 2,
            pause_threshold_secs: 0.3,
            silence_energy_threshold: 0.008,
            max_segment_duration_secs: 5.0,
            context_buffer_secs: 2.0,
        }
    }
}

impl PauseParallelTDTConfig {
    pub fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            ..Default::default()
        }
    }
}

/// Calculate RMS energy of audio samples
fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Job sent to worker thread
struct TranscriptionJob {
    segment_idx: u64,
    audio: Vec<f32>,
    segment_start_time: f32,
    segment_end_time: f32,
    context_duration: f32,
}

/// Result from worker thread
struct TranscriptionResult {
    segment_idx: u64,
    text: String,
    start_time: f32,
    end_time: f32,
    inference_time_ms: u32,
}

/// Manages ordered emission of results
struct OrderedMerger {
    next_emit_idx: u64,
    last_emitted_end_time: f32,
}

impl OrderedMerger {
    fn new() -> Self {
        Self {
            next_emit_idx: 0,
            last_emitted_end_time: 0.0,
        }
    }

    fn process(
        &mut self,
        pending: &mut BTreeMap<u64, TranscriptionResult>,
    ) -> Vec<TranscriptionSegment> {
        let mut segments = Vec::new();

        while let Some(result) = pending.remove(&self.next_emit_idx) {
            let text = truncate_hallucination(&result.text);

            if !text.is_empty() && result.start_time >= self.last_emitted_end_time - 0.1 {
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
                    "[PauseParallelTDT] Emitting segment {} [{:.1}s-{:.1}s] (inference: {}ms)",
                    result.segment_idx, result.start_time, result.end_time, result.inference_time_ms
                );

                self.last_emitted_end_time = result.end_time;
            }

            self.next_emit_idx += 1;
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

/// Pause-based parallel TDT transcriber
pub struct PauseParallelTDT {
    config: PauseParallelTDTConfig,
    workers: Vec<JoinHandle<()>>,
    job_tx: Sender<TranscriptionJob>,
    result_rx: Receiver<TranscriptionResult>,
    shutdown: Arc<AtomicBool>,
    context_buffer: VecDeque<f32>,
    context_buffer_samples: usize,
    current_segment_audio: Vec<f32>,
    current_segment_start: f32,
    in_speech: bool,
    silence_start_time: Option<f32>,
    next_segment_idx: AtomicU64,
    pending_results: BTreeMap<u64, TranscriptionResult>,
    merger: OrderedMerger,
    total_samples: usize,
    jobs_in_flight: Arc<AtomicU64>,
}

impl PauseParallelTDT {
    pub fn new(
        model_dir: &Path,
        exec_config: Option<ModelConfig>,
        config: Option<PauseParallelTDTConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();

        eprintln!(
            "[PauseParallelTDT] Initializing with {} threads, pause: {}ms",
            config.num_threads,
            (config.pause_threshold_secs * 1000.0) as u32
        );

        let (job_tx, job_rx) = mpsc::channel::<TranscriptionJob>();
        let (result_tx, result_rx) = mpsc::channel::<TranscriptionResult>();
        let job_rx = Arc::new(std::sync::Mutex::new(job_rx));
        let shutdown = Arc::new(AtomicBool::new(false));
        let jobs_in_flight = Arc::new(AtomicU64::new(0));

        let mut workers = Vec::with_capacity(config.num_threads);

        eprintln!("[PauseParallelTDT] Loading {} model instances...", config.num_threads);
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
            let jobs_counter = Arc::clone(&jobs_in_flight);

            let handle = thread::spawn(move || {
                eprintln!("[PauseParallelTDT] Worker {} loading model...", i);
                let mut model = match ParakeetTDT::from_pretrained(&model_dir, Some(exec_config)) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[PauseParallelTDT] Worker {} failed to load: {}", i, e);
                        return;
                    }
                };
                eprintln!("[PauseParallelTDT] Worker {} ready", i);

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
                                    eprintln!("[PauseParallelTDT] Worker {} error: {}", i, e);
                                    jobs_counter.fetch_sub(1, Ordering::Relaxed);
                                    continue;
                                }
                            };

                            let inference_time_ms = inference_start.elapsed().as_millis() as u32;

                            let result = TranscriptionResult {
                                segment_idx: job.segment_idx,
                                text: transcription.text,
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
                eprintln!("[PauseParallelTDT] Worker {} shutting down", i);
            });

            workers.push(handle);
        }

        eprintln!(
            "[PauseParallelTDT] All {} workers initialized in {:.2}s",
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

    fn dispatch_segment(&mut self, segment_audio: Vec<f32>, start_time: f32, end_time: f32) {
        if segment_audio.is_empty() {
            return;
        }

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
            "[PauseParallelTDT] Dispatching segment {} [{:.1}s-{:.1}s] ({:.1}s audio + {:.1}s context)",
            segment_idx, start_time, end_time,
            segment_audio.len() as f32 / SAMPLE_RATE as f32,
            context_duration
        );

        self.jobs_in_flight.fetch_add(1, Ordering::Relaxed);

        if let Err(e) = self.job_tx.send(job) {
            eprintln!("[PauseParallelTDT] Failed to dispatch job: {}", e);
            self.jobs_in_flight.fetch_sub(1, Ordering::Relaxed);
        }

        for sample in &segment_audio {
            self.context_buffer.push_back(*sample);
        }
        while self.context_buffer.len() > self.context_buffer_samples {
            self.context_buffer.pop_front();
        }
    }

    fn collect_results(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.pending_results.insert(result.segment_idx, result);
        }
    }
}

impl StreamingTranscriber for PauseParallelTDT {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "pause-parallel-tdt".to_string(),
            display_name: format!("Pause-Parallel TDT ({}T)", self.config.num_threads),
            description: format!(
                "Pause-based parallel TDT with {} threads, {}ms pause threshold",
                self.config.num_threads,
                (self.config.pause_threshold_secs * 1000.0) as u32
            ),
            supports_diarization: false,
            languages: vec!["en".to_string()],
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

        let rms = calculate_rms(samples);
        let is_silence = rms < self.config.silence_energy_threshold;

        if is_silence {
            if self.in_speech {
                self.current_segment_audio.extend_from_slice(samples);

                if self.silence_start_time.is_none() {
                    self.silence_start_time = Some(current_time);
                }

                let silence_duration = chunk_end_time - self.silence_start_time.unwrap();

                if silence_duration >= self.config.pause_threshold_secs {
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
        } else {
            if !self.in_speech {
                self.in_speech = true;
                self.current_segment_start = current_time;
            }
            self.silence_start_time = None;
            self.current_segment_audio.extend_from_slice(samples);
        }

        let current_segment_duration = self.current_segment_audio.len() as f32 / SAMPLE_RATE as f32;
        if current_segment_duration > self.config.max_segment_duration_secs {
            eprintln!(
                "[PauseParallelTDT] Forcing segment break at {:.1}s (max duration)",
                chunk_end_time
            );
            let segment_audio = std::mem::take(&mut self.current_segment_audio);
            self.dispatch_segment(segment_audio, self.current_segment_start, chunk_end_time);
            self.current_segment_start = chunk_end_time;
        }

        self.collect_results();
        let segments = self.merger.process(&mut self.pending_results);

        Ok(StreamingChunkResult {
            segments,
            buffer_duration: self.current_segment_audio.len() as f32 / SAMPLE_RATE as f32,
            total_duration: self.total_samples as f32 / SAMPLE_RATE as f32,
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        eprintln!("[PauseParallelTDT] Finalizing...");

        if !self.current_segment_audio.is_empty() {
            let segment_audio = std::mem::take(&mut self.current_segment_audio);
            let end_time = self.total_samples as f32 / SAMPLE_RATE as f32;
            self.dispatch_segment(segment_audio, self.current_segment_start, end_time);
        }

        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while Instant::now() < deadline {
            self.collect_results();
            let jobs_pending = self.jobs_in_flight.load(Ordering::Relaxed);
            let total_segments = self.next_segment_idx.load(Ordering::Relaxed);
            let emitted = self.merger.next_emit_idx;

            if jobs_pending == 0 && emitted >= total_segments {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(50));
        }

        self.collect_results();
        let mut segments = self.merger.process(&mut self.pending_results);

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

        segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
        eprintln!("[PauseParallelTDT] Finalized with {} segments", segments.len());

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

impl Drop for PauseParallelTDT {
    fn drop(&mut self) {
        eprintln!("[PauseParallelTDT] Shutting down {} workers...", self.workers.len());
        self.shutdown.store(true, Ordering::Relaxed);
        for (i, worker) in self.workers.drain(..).enumerate() {
            if worker.join().is_err() {
                eprintln!("[PauseParallelTDT] Worker {} panicked", i);
            }
        }
        eprintln!("[PauseParallelTDT] Shutdown complete");
    }
}
