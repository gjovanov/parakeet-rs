//! Noise cancellation module for audio preprocessing
//!
//! Supports RNNoise for real-time noise suppression. Both models work at 48kHz,
//! so this module handles resampling from 16kHz (parakeet's sample rate) to 48kHz and back.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Type of noise cancellation to apply
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NoiseCancellationType {
    /// No noise cancellation
    #[default]
    None,
    /// RNNoise - lightweight neural network noise suppression (48kHz)
    #[serde(rename = "rnnoise")]
    RNNoise,
    /// DeepFilterNet3 - high-quality deep filtering (48kHz)
    #[serde(rename = "deepfilternet3")]
    DeepFilterNet3,
}

impl fmt::Display for NoiseCancellationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NoiseCancellationType::None => write!(f, "none"),
            NoiseCancellationType::RNNoise => write!(f, "rnnoise"),
            NoiseCancellationType::DeepFilterNet3 => write!(f, "deepfilternet3"),
        }
    }
}

impl NoiseCancellationType {
    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "rnnoise" => NoiseCancellationType::RNNoise,
            "deepfilternet3" | "deepfilter" => NoiseCancellationType::DeepFilterNet3,
            _ => NoiseCancellationType::None,
        }
    }
}

/// Trait for noise cancellation processors
pub trait NoiseCanceller: Send + Sync {
    /// Process 16kHz audio samples and return denoised 16kHz samples
    fn process(&mut self, samples_16k: &[f32]) -> Vec<f32>;

    /// Reset internal state
    fn reset(&mut self);

    /// Get the name of this noise canceller
    fn name(&self) -> &'static str;
}

/// RNNoise processor with 16kHz↔48kHz resampling
pub struct RNNoiseProcessor {
    denoiser: Box<nnnoiseless::DenoiseState<'static>>,
    /// Buffer for 48kHz samples waiting to be processed
    buffer_48k: Vec<f32>,
    /// Buffer for 16kHz output samples
    output_buffer_16k: Vec<f32>,
    /// Residual samples from upsampling (less than 3 samples)
    upsample_residual: Vec<f32>,
}

impl RNNoiseProcessor {
    /// Create a new RNNoise processor
    pub fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Self {
            denoiser: nnnoiseless::DenoiseState::new(),
            buffer_48k: Vec::with_capacity(4800), // 100ms at 48kHz
            output_buffer_16k: Vec::with_capacity(1600), // 100ms at 16kHz
            upsample_residual: Vec::new(),
        })
    }

    /// Simple linear interpolation upsample 16kHz → 48kHz (3x)
    fn upsample_3x(&mut self, samples_16k: &[f32]) -> Vec<f32> {
        if samples_16k.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(samples_16k.len() * 3);

        for i in 0..samples_16k.len() {
            let current = samples_16k[i];
            let next = if i + 1 < samples_16k.len() {
                samples_16k[i + 1]
            } else {
                current // Repeat last sample
            };

            // Linear interpolation: insert current, then two interpolated values
            result.push(current);
            result.push(current + (next - current) / 3.0);
            result.push(current + 2.0 * (next - current) / 3.0);
        }

        result
    }

    /// Simple averaging downsample 48kHz → 16kHz (1/3)
    fn downsample_3x(&mut self, samples_48k: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(samples_48k.len() / 3 + 1);

        let mut i = 0;
        while i + 2 < samples_48k.len() {
            // Average 3 samples into 1
            let avg = (samples_48k[i] + samples_48k[i + 1] + samples_48k[i + 2]) / 3.0;
            result.push(avg);
            i += 3;
        }

        result
    }
}

impl Default for RNNoiseProcessor {
    fn default() -> Self {
        Self::new().expect("Failed to create RNNoise processor")
    }
}

impl NoiseCanceller for RNNoiseProcessor {
    fn process(&mut self, samples_16k: &[f32]) -> Vec<f32> {
        if samples_16k.is_empty() {
            return Vec::new();
        }

        // 1. Upsample 16kHz → 48kHz
        let upsampled = self.upsample_3x(samples_16k);
        self.buffer_48k.extend_from_slice(&upsampled);

        // 2. Process in 480-sample frames (10ms at 48kHz) - RNNoise frame size
        let frame_size = nnnoiseless::FRAME_SIZE; // 480
        let mut denoised_48k = Vec::new();

        while self.buffer_48k.len() >= frame_size {
            let frame: Vec<f32> = self.buffer_48k.drain(..frame_size).collect();
            let mut output = vec![0.0f32; frame_size];
            self.denoiser.process_frame(&mut output, &frame);
            denoised_48k.extend_from_slice(&output);
        }

        // 3. Downsample 48kHz → 16kHz
        if !denoised_48k.is_empty() {
            let downsampled = self.downsample_3x(&denoised_48k);
            self.output_buffer_16k.extend_from_slice(&downsampled);
        }

        // Return accumulated output
        std::mem::take(&mut self.output_buffer_16k)
    }

    fn reset(&mut self) {
        self.buffer_48k.clear();
        self.output_buffer_16k.clear();
        self.upsample_residual.clear();
        // Note: nnnoiseless::DenoiseState doesn't have a reset method,
        // but it maintains minimal state between frames
    }

    fn name(&self) -> &'static str {
        "RNNoise"
    }
}

/// Factory function to create a noise canceller
///
/// # Arguments
/// * `noise_type` - Type of noise cancellation to use
/// * `_model_path` - Optional path to model (for DeepFilterNet3)
///
/// # Returns
/// Some(Box<dyn NoiseCanceller>) if noise cancellation is enabled, None otherwise
pub fn create_noise_canceller(
    noise_type: NoiseCancellationType,
    _model_path: Option<&std::path::Path>,
) -> Option<Box<dyn NoiseCanceller>> {
    match noise_type {
        NoiseCancellationType::None => None,
        NoiseCancellationType::RNNoise => {
            match RNNoiseProcessor::new() {
                Ok(processor) => {
                    eprintln!("[NoiseCancellation] RNNoise enabled (16kHz↔48kHz resampling)");
                    Some(Box::new(processor))
                }
                Err(e) => {
                    eprintln!("[NoiseCancellation] Failed to create RNNoise processor: {}", e);
                    None
                }
            }
        }
        NoiseCancellationType::DeepFilterNet3 => {
            // DeepFilterNet3 requires the deep_filter crate which is not currently
            // available as a simple dependency. For now, we'll fall back to RNNoise.
            eprintln!("[NoiseCancellation] DeepFilterNet3 not yet implemented, falling back to RNNoise");
            match RNNoiseProcessor::new() {
                Ok(processor) => Some(Box::new(processor)),
                Err(e) => {
                    eprintln!("[NoiseCancellation] Failed to create RNNoise processor: {}", e);
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_cancellation_type_display() {
        assert_eq!(NoiseCancellationType::None.to_string(), "none");
        assert_eq!(NoiseCancellationType::RNNoise.to_string(), "rnnoise");
        assert_eq!(NoiseCancellationType::DeepFilterNet3.to_string(), "deepfilternet3");
    }

    #[test]
    fn test_noise_cancellation_type_from_str() {
        assert_eq!(NoiseCancellationType::from_str("none"), NoiseCancellationType::None);
        assert_eq!(NoiseCancellationType::from_str("rnnoise"), NoiseCancellationType::RNNoise);
        assert_eq!(NoiseCancellationType::from_str("RNNoise"), NoiseCancellationType::RNNoise);
        assert_eq!(NoiseCancellationType::from_str("deepfilternet3"), NoiseCancellationType::DeepFilterNet3);
        assert_eq!(NoiseCancellationType::from_str("deepfilter"), NoiseCancellationType::DeepFilterNet3);
        assert_eq!(NoiseCancellationType::from_str("invalid"), NoiseCancellationType::None);
    }

    #[test]
    fn test_rnnoise_processor_creation() {
        let processor = RNNoiseProcessor::new();
        assert!(processor.is_ok());
    }

    #[test]
    fn test_rnnoise_processor_process() {
        let mut processor = RNNoiseProcessor::new().unwrap();

        // Create some test samples (320 samples = 20ms at 16kHz)
        let samples: Vec<f32> = (0..320).map(|i| (i as f32 * 0.01).sin()).collect();

        // Process the samples
        let output = processor.process(&samples);

        // Output should be similar length (may vary slightly due to buffering)
        // RNNoise processes 480 samples at 48kHz (10ms), so with 320 16kHz samples (20ms)
        // we should get approximately 2 frames worth of output
        assert!(!output.is_empty() || processor.buffer_48k.len() > 0);
    }

    #[test]
    fn test_create_noise_canceller_none() {
        let canceller = create_noise_canceller(NoiseCancellationType::None, None);
        assert!(canceller.is_none());
    }

    #[test]
    fn test_create_noise_canceller_rnnoise() {
        let canceller = create_noise_canceller(NoiseCancellationType::RNNoise, None);
        assert!(canceller.is_some());
        assert_eq!(canceller.unwrap().name(), "RNNoise");
    }
}
