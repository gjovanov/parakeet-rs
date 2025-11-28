//! Streaming wrapper for Sortformer diarization
//!
//! Provides continuous speaker tracking with incremental updates.
//! Designed for real-time applications where audio arrives in small chunks.

use crate::error::Result;
use crate::execution::ModelConfig;
use crate::sortformer::{DiarizationConfig, SpeakerSegment, Sortformer};
use std::collections::VecDeque;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;
const FRAME_DURATION: f32 = 0.08; // 80ms per model frame
const CHUNK_FRAMES: usize = 124; // Frames per Sortformer chunk
const SUBSAMPLING: usize = 8;

// Minimum audio needed before first diarization (samples)
// ~10 seconds = CHUNK_FRAMES * SUBSAMPLING * HOP_LENGTH / SAMPLE_RATE
const MIN_SAMPLES_FOR_DIARIZATION: usize = 160000; // 10 seconds

// How often to update diarization (samples)
// Update every ~2 seconds for more responsive speaker changes
const UPDATE_INTERVAL_SAMPLES: usize = 32000; // 2 seconds

/// Speaker prediction at a specific timestamp
#[derive(Debug, Clone)]
pub struct SpeakerPrediction {
    /// Timestamp in seconds
    pub timestamp: f32,
    /// Speaker ID (0-3) or None if no speaker detected
    pub speaker_id: Option<usize>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Current speaker state
#[derive(Debug, Clone, Default)]
pub struct SpeakerState {
    /// Current dominant speaker
    pub current_speaker: Option<usize>,
    /// Time when current speaker started
    pub speaker_start_time: f32,
    /// All active speaker segments
    pub segments: Vec<SpeakerSegment>,
    /// Timestamp of last update
    pub last_update_time: f32,
}

/// Streaming Sortformer wrapper for real-time diarization
pub struct SortformerStream {
    sortformer: Sortformer,
    /// Accumulated audio samples (mono, 16kHz)
    audio_buffer: VecDeque<f32>,
    /// Total samples processed
    total_samples: usize,
    /// Samples since last diarization update
    samples_since_update: usize,
    /// Current speaker state
    state: SpeakerState,
    /// Historical speaker segments
    history: Vec<SpeakerSegment>,
    /// Speaker predictions per timestamp (for interpolation)
    predictions: VecDeque<SpeakerPrediction>,
    /// Maximum history to keep (in seconds)
    max_history_secs: f32,
}

impl SortformerStream {
    /// Create a new streaming Sortformer
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::with_config(model_path, None, DiarizationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config<P: AsRef<Path>>(
        model_path: P,
        execution_config: Option<ModelConfig>,
        diar_config: DiarizationConfig,
    ) -> Result<Self> {
        let sortformer = Sortformer::with_config(model_path, execution_config, diar_config)?;

        Ok(Self {
            sortformer,
            audio_buffer: VecDeque::with_capacity(MIN_SAMPLES_FOR_DIARIZATION * 2),
            total_samples: 0,
            samples_since_update: 0,
            state: SpeakerState::default(),
            history: Vec::new(),
            predictions: VecDeque::new(),
            max_history_secs: 300.0, // 5 minutes
        })
    }

    /// Push audio samples and get updated speaker state
    ///
    /// Returns the current speaker state after processing.
    /// May return unchanged state if not enough audio for update.
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<&SpeakerState> {
        // Add samples to buffer
        self.audio_buffer.extend(samples.iter().copied());
        self.total_samples += samples.len();
        self.samples_since_update += samples.len();

        // Check if we should update diarization
        let should_update = self.audio_buffer.len() >= MIN_SAMPLES_FOR_DIARIZATION
            && self.samples_since_update >= UPDATE_INTERVAL_SAMPLES;

        if should_update {
            self.update_diarization()?;
            self.samples_since_update = 0;
        }

        Ok(&self.state)
    }

    /// Get the current speaker at a specific timestamp
    pub fn get_speaker_at(&self, timestamp: f32) -> Option<usize> {
        // Check current segments first
        for segment in self.state.segments.iter().rev() {
            if timestamp >= segment.start && timestamp <= segment.end {
                return Some(segment.speaker_id);
            }
        }

        // Check history
        for segment in self.history.iter().rev() {
            if timestamp >= segment.start && timestamp <= segment.end {
                return Some(segment.speaker_id);
            }
        }

        None
    }

    /// Get the current dominant speaker
    pub fn current_speaker(&self) -> Option<usize> {
        self.state.current_speaker
    }

    /// Get current speaker state
    pub fn state(&self) -> &SpeakerState {
        &self.state
    }

    /// Get all historical segments
    pub fn history(&self) -> &[SpeakerSegment] {
        &self.history
    }

    /// Get total audio duration processed
    pub fn total_duration(&self) -> f32 {
        self.total_samples as f32 / SAMPLE_RATE as f32
    }

    /// Check if diarization has produced any results yet
    pub fn has_results(&self) -> bool {
        !self.state.segments.is_empty() || !self.history.is_empty()
    }

    /// Reset state for new stream
    pub fn reset(&mut self) {
        self.sortformer.reset_state();
        self.audio_buffer.clear();
        self.total_samples = 0;
        self.samples_since_update = 0;
        self.state = SpeakerState::default();
        self.history.clear();
        self.predictions.clear();
    }

    /// Force an immediate diarization update (if enough audio)
    pub fn force_update(&mut self) -> Result<&SpeakerState> {
        if self.audio_buffer.len() >= MIN_SAMPLES_FOR_DIARIZATION {
            self.update_diarization()?;
        }
        Ok(&self.state)
    }

    /// Internal: Update diarization with current audio buffer
    fn update_diarization(&mut self) -> Result<()> {
        // Convert buffer to vec for processing
        let audio: Vec<f32> = self.audio_buffer.iter().copied().collect();

        // Run diarization on full buffer
        // Note: Sortformer handles its own streaming state internally
        let segments = self.sortformer.diarize(audio, SAMPLE_RATE as u32, 1)?;

        // Calculate time offset (buffer start time in the overall stream)
        let buffer_duration = self.audio_buffer.len() as f32 / SAMPLE_RATE as f32;
        let stream_time = self.total_samples as f32 / SAMPLE_RATE as f32;
        let buffer_start_time = stream_time - buffer_duration;

        // Update segments with proper time offset
        let mut adjusted_segments: Vec<SpeakerSegment> = segments
            .into_iter()
            .map(|mut seg| {
                seg.start += buffer_start_time;
                seg.end += buffer_start_time;
                seg
            })
            .collect();

        // Find current dominant speaker (most recent active segment)
        let current_time = stream_time;
        let current_speaker = adjusted_segments
            .iter()
            .rev()
            .find(|seg| seg.start <= current_time && seg.end >= current_time - 0.5)
            .map(|seg| seg.speaker_id);

        // Update state
        self.state = SpeakerState {
            current_speaker,
            speaker_start_time: adjusted_segments
                .iter()
                .rev()
                .find(|seg| Some(seg.speaker_id) == current_speaker)
                .map(|seg| seg.start)
                .unwrap_or(current_time),
            segments: adjusted_segments.clone(),
            last_update_time: current_time,
        };

        // Move older segments to history
        let cutoff_time = current_time - self.max_history_secs;
        self.history.retain(|seg| seg.end > cutoff_time);

        // Add completed segments to history (segments that ended before recent window)
        let recent_window = 10.0; // Keep last 10 seconds in active state
        for seg in adjusted_segments.drain(..) {
            if seg.end < current_time - recent_window {
                // Check if this segment is already in history
                let exists = self.history.iter().any(|h| {
                    (h.start - seg.start).abs() < 0.1
                        && (h.end - seg.end).abs() < 0.1
                        && h.speaker_id == seg.speaker_id
                });
                if !exists {
                    self.history.push(seg);
                }
            }
        }

        // Sort history by start time
        self.history
            .sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());

        // Trim audio buffer to keep only recent audio
        // Keep 2x the minimum for overlap
        let max_buffer_samples = MIN_SAMPLES_FOR_DIARIZATION * 2;
        while self.audio_buffer.len() > max_buffer_samples {
            self.audio_buffer.pop_front();
        }

        // Add prediction for current timestamp
        self.predictions.push_back(SpeakerPrediction {
            timestamp: current_time,
            speaker_id: current_speaker,
            confidence: if current_speaker.is_some() {
                0.8
            } else {
                0.0
            },
        });

        // Trim old predictions
        let prediction_cutoff = current_time - 60.0; // Keep 1 minute
        while self
            .predictions
            .front()
            .map(|p| p.timestamp < prediction_cutoff)
            .unwrap_or(false)
        {
            self.predictions.pop_front();
        }

        Ok(())
    }

    /// Interpolate speaker for timestamps between updates
    pub fn interpolate_speaker(&self, timestamp: f32) -> SpeakerPrediction {
        // Find nearest predictions
        let mut before: Option<&SpeakerPrediction> = None;
        let mut after: Option<&SpeakerPrediction> = None;

        for pred in &self.predictions {
            if pred.timestamp <= timestamp {
                before = Some(pred);
            } else if after.is_none() {
                after = Some(pred);
                break;
            }
        }

        // If we have a direct hit or only one side, use that
        if let Some(b) = before {
            if after.is_none() || (timestamp - b.timestamp).abs() < 0.1 {
                return b.clone();
            }
        }

        if let Some(a) = after {
            if before.is_none() || (timestamp - a.timestamp).abs() < 0.1 {
                return a.clone();
            }
        }

        // Interpolate based on which is closer
        match (before, after) {
            (Some(b), Some(a)) => {
                if (timestamp - b.timestamp) < (a.timestamp - timestamp) {
                    b.clone()
                } else {
                    a.clone()
                }
            }
            (Some(b), None) => b.clone(),
            (None, Some(a)) => a.clone(),
            (None, None) => SpeakerPrediction {
                timestamp,
                speaker_id: None,
                confidence: 0.0,
            },
        }
    }
}

/// Builder for SortformerStream with customization options
pub struct SortformerStreamBuilder {
    model_path: String,
    execution_config: Option<ModelConfig>,
    diar_config: DiarizationConfig,
    max_history_secs: f32,
}

impl SortformerStreamBuilder {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        Self {
            model_path: model_path.as_ref().to_string_lossy().to_string(),
            execution_config: None,
            diar_config: DiarizationConfig::default(),
            max_history_secs: 300.0,
        }
    }

    pub fn execution_config(mut self, config: ModelConfig) -> Self {
        self.execution_config = Some(config);
        self
    }

    pub fn diarization_config(mut self, config: DiarizationConfig) -> Self {
        self.diar_config = config;
        self
    }

    pub fn max_history_secs(mut self, secs: f32) -> Self {
        self.max_history_secs = secs;
        self
    }

    pub fn build(self) -> Result<SortformerStream> {
        let mut stream =
            SortformerStream::with_config(&self.model_path, self.execution_config, self.diar_config)?;
        stream.max_history_secs = self.max_history_secs;
        Ok(stream)
    }
}
