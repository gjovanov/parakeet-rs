//! Pause-segmented transcription for TDT (Parakeet) model.
//! Same architecture as pause_segmented.rs but uses ParakeetTDT's
//! transcribe_samples() API instead of CanaryModel::transcribe().

use crate::error::Result;
use crate::ExecutionConfig;
use crate::pause_segmented::{calculate_rms, truncate_hallucination, strip_context_prefix, PauseSegmentedConfig};
use crate::streaming_transcriber::{ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment};
use crate::parakeet_tdt::ParakeetTDT;
use crate::transcriber::Transcriber;
use std::path::Path;

const SAMPLE_RATE: usize = 16000;

pub struct PauseSegmentedTDT {
    model: ParakeetTDT,
    config: PauseSegmentedConfig,
    speech_buffer: Vec<f32>,
    total_samples: usize,
    speech_start_sample: Option<usize>,
    silence_start_time: Option<f32>,
    is_speaking: bool,
    samples_since_partial: usize,
    consecutive_silence_frames: usize,
    pending_segments: Vec<TranscriptionSegment>,
    last_final_end_time: f32,
    context_ring: std::collections::VecDeque<(Vec<f32>, String)>,
    #[cfg(feature = "sortformer")]
    diarizer: Option<crate::sortformer_stream::SortformerStream>,
}

impl PauseSegmentedTDT {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        exec_config: Option<ExecutionConfig>,
        config: Option<PauseSegmentedConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let model = ParakeetTDT::from_pretrained(model_path, exec_config)?;

        Ok(Self {
            model,
            config,
            speech_buffer: Vec::with_capacity(SAMPLE_RATE * 15),
            total_samples: 0,
            speech_start_sample: None,
            silence_start_time: None,
            is_speaking: false,
            samples_since_partial: 0,
            consecutive_silence_frames: 0,
            pending_segments: Vec::new(),
            last_final_end_time: 0.0,
            context_ring: std::collections::VecDeque::new(),
            #[cfg(feature = "sortformer")]
            diarizer: None,
        })
    }

    #[cfg(feature = "sortformer")]
    pub fn new_with_diarization<P: AsRef<Path>>(
        model_path: P,
        diar_path: Option<P>,
        exec_config: Option<ExecutionConfig>,
        config: Option<PauseSegmentedConfig>,
    ) -> Result<Self> {
        let mut instance = Self::new(model_path, exec_config, config)?;
        if let Some(diar) = diar_path {
            instance.diarizer = Some(crate::sortformer_stream::SortformerStream::new(diar)?);
        }
        Ok(instance)
    }

    fn current_time(&self) -> f32 { self.total_samples as f32 / SAMPLE_RATE as f32 }
    fn speech_duration(&self) -> f32 { self.speech_buffer.len() as f32 / SAMPLE_RATE as f32 }
    fn speech_start_time(&self) -> f32 {
        self.speech_start_sample.map(|s| s as f32 / SAMPLE_RATE as f32).unwrap_or(self.current_time())
    }

    #[cfg(feature = "sortformer")]
    fn get_speaker_at(&mut self, time: f32) -> Option<usize> {
        self.diarizer.as_mut().and_then(|d| d.get_speaker_at(time))
    }
    #[cfg(not(feature = "sortformer"))]
    fn get_speaker_at(&self, _time: f32) -> Option<usize> { None }

    fn transcribe_audio(&mut self, audio: &[f32]) -> Result<String> {
        let result = self.model.transcribe_samples(audio.to_vec(), SAMPLE_RATE as u32, 1, None)?;
        Ok(result.text.trim().to_string())
    }

    fn transcribe_and_emit_final(&mut self) -> Result<()> {
        if self.speech_buffer.is_empty() { return Ok(()); }

        let duration = self.speech_duration();
        if duration < self.config.min_segment_secs {
            self.speech_buffer.clear();
            self.speech_start_sample = None;
            self.samples_since_partial = 0;
            return Ok(());
        }

        let start_time = self.speech_start_time();
        let end_time = start_time + duration;
        let ctx = self.config.context_segments;

        let (audio, context_text) = if ctx > 1 && !self.context_ring.is_empty() {
            let mut combined: Vec<f32> = Vec::new();
            let mut ctx_text = String::new();
            let count = (ctx - 1).min(self.context_ring.len());
            let start_idx = self.context_ring.len() - count;
            for (a, t) in self.context_ring.iter().skip(start_idx) {
                combined.extend_from_slice(a);
                if !ctx_text.is_empty() { ctx_text.push(' '); }
                ctx_text.push_str(t);
            }
            combined.extend_from_slice(&self.speech_buffer);
            (combined, ctx_text)
        } else {
            (self.speech_buffer.clone(), String::new())
        };

        let full_text = self.transcribe_audio(&audio)?;
        let text = if !context_text.is_empty() && !full_text.is_empty() {
            strip_context_prefix(&full_text, &context_text)
        } else { full_text };
        let text = text.trim().to_string();

        if !text.is_empty() {
            let text = truncate_hallucination(&text);
            let mid_time = (start_time + end_time) / 2.0;
            let speaker = self.get_speaker_at(mid_time);

            eprintln!("[PauseSegmentedTDT] FINAL [{:.2}s-{:.2}s] ({:.1}s) \"{}\"",
                start_time, end_time, duration, &text.chars().take(80).collect::<String>());

            self.context_ring.push_back((self.speech_buffer.clone(), text.clone()));
            while self.context_ring.len() > self.config.context_segments.max(1) {
                self.context_ring.pop_front();
            }

            self.pending_segments.push(TranscriptionSegment {
                text, raw_text: None, start_time, end_time, speaker,
                confidence: None, is_final: true, inference_time_ms: None,
            });
            self.last_final_end_time = end_time;
        } else {
            self.context_ring.push_back((self.speech_buffer.clone(), String::new()));
            while self.context_ring.len() > self.config.context_segments.max(1) {
                self.context_ring.pop_front();
            }
        }

        self.speech_buffer.clear();
        self.speech_start_sample = None;
        self.samples_since_partial = 0;
        Ok(())
    }

    fn emit_partial(&mut self) -> Result<()> {
        if self.speech_buffer.is_empty() { return Ok(()); }
        let start_time = self.speech_start_time();
        let end_time = start_time + self.speech_duration();
        let buf = self.speech_buffer.clone();
        let text = self.transcribe_audio(&buf)?;
        let text = text.trim().to_string();
        if !text.is_empty() {
            let text = truncate_hallucination(&text);
            let speaker = self.get_speaker_at((start_time + end_time) / 2.0);
            self.pending_segments.push(TranscriptionSegment {
                text, raw_text: None, start_time, end_time, speaker,
                confidence: None, is_final: false, inference_time_ms: None,
            });
        }
        self.samples_since_partial = 0;
        Ok(())
    }

    fn take_segments(&mut self) -> Vec<TranscriptionSegment> { std::mem::take(&mut self.pending_segments) }
}

impl StreamingTranscriber for PauseSegmentedTDT {
    fn model_info(&self) -> ModelInfo {
        ModelInfo {
            id: "parakeet-tdt".to_string(),
            display_name: "Parakeet TDT (Pause-Segmented)".to_string(),
            description: "NVIDIA Parakeet TDT with pause-based segmentation".to_string(),
            supports_diarization: { #[cfg(feature = "sortformer")] { self.diarizer.is_some() } #[cfg(not(feature = "sortformer"))] { false } },
            languages: vec!["en".to_string()],
            is_loaded: true,
        }
    }

    fn push_audio(&mut self, samples: &[f32]) -> Result<StreamingChunkResult> {
        #[cfg(feature = "sortformer")]
        if let Some(ref mut d) = self.diarizer { let _ = d.push_audio(samples); }

        use crate::pause_segmented::FRAME_SAMPLES;

        let threshold = self.config.silence_energy_threshold;
        let pause_frames_needed = (self.config.pause_threshold_secs * SAMPLE_RATE as f32 / FRAME_SAMPLES as f32).ceil() as usize;

        let mut offset = 0;
        while offset < samples.len() {
            let end = (offset + FRAME_SAMPLES).min(samples.len());
            let frame = &samples[offset..end];
            offset = end;

            if frame.len() < FRAME_SAMPLES / 2 {
                self.total_samples += frame.len();
                break;
            }

            let rms = calculate_rms(frame);
            let is_speech = rms >= threshold;
            self.total_samples += frame.len();

            if is_speech {
                self.consecutive_silence_frames = 0;
                if !self.is_speaking {
                    self.is_speaking = true;
                    if self.speech_start_sample.is_none() {
                        self.speech_start_sample = Some(self.total_samples - frame.len());
                    }
                }
                self.speech_buffer.extend_from_slice(frame);
                self.samples_since_partial += frame.len();

                if self.speech_duration() >= self.config.max_segment_secs {
                    self.transcribe_and_emit_final()?;
                } else if self.samples_since_partial as f32 / SAMPLE_RATE as f32 >= self.config.partial_interval_secs {
                    self.emit_partial()?;
                }
            } else {
                self.consecutive_silence_frames += 1;
                if self.is_speaking { self.speech_buffer.extend_from_slice(frame); }
                if self.consecutive_silence_frames >= pause_frames_needed && self.is_speaking {
                    self.is_speaking = false;
                    self.transcribe_and_emit_final()?;
                }
            }
        }

        let segments = self.take_segments();
        Ok(StreamingChunkResult {
            segments, buffer_duration: self.speech_duration(), total_duration: self.current_time(),
        })
    }

    fn finalize(&mut self) -> Result<StreamingChunkResult> {
        if !self.speech_buffer.is_empty() { self.transcribe_and_emit_final()?; }
        Ok(StreamingChunkResult { segments: self.take_segments(), buffer_duration: 0.0, total_duration: self.current_time() })
    }

    fn reset(&mut self) {
        self.speech_buffer.clear(); self.total_samples = 0; self.speech_start_sample = None;
        self.silence_start_time = None; self.is_speaking = false; self.samples_since_partial = 0;
        self.pending_segments.clear(); self.last_final_end_time = 0.0; self.context_ring.clear();
    }

    fn buffer_duration(&self) -> f32 { self.speech_duration() }
    fn total_duration(&self) -> f32 { self.current_time() }
}
