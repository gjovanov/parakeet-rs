//! # parakeet-rs
//!
//! Rust bindings for NVIDIA's Parakeet speech recognition model using ONNX Runtime.
//!
//! Parakeet is a state-of-the-art automatic speech recognition (ASR) model developed by NVIDIA,
//! based on the FastConformer-TDT architecture with 600 million parameters.
//!
//! ## Features
//!
//! - Easy-to-use API for speech-to-text transcription
//! - Support for ONNX format models
//! - 16kHz mono audio input
//! - Punctuation and capitalization included in output
//! - Fast inference using ONNX Runtime
//!
//! ## Quick Start
//!
//! ```ignore
//! use parakeet_rs::Parakeet;
//!
//! // Load the model
//! let parakeet = Parakeet::from_pretrained(".")?;
//!
//! // Transcribe audio file
//! let text = parakeet.transcribe_file("audio.wav")?;
//! println!("Transcription: {}", text);
//! ```
//!
//! ## Model Requirements
//!
//! Your model directory should contain:
//! - `model.onnx` - The ONNX model file
//! - `model.onnx_data` - External model weights
//! - `config.json` - Model configuration
//! - `preprocessor_config.json` - Audio preprocessing configuration
//! - `tokenizer.json` - Tokenizer vocabulary
//! - `tokenizer_config.json` - Tokenizer configuration
//!
//! ## Audio Requirements
//!
//! - Format: WAV
//! - Sample Rate: 16kHz
//! - Channels: Mono (stereo will be converted automatically)
//! - Bit Depth: 16-bit PCM or 32-bit float

mod audio;
pub mod canary;
pub mod canary_flash;
mod config;
mod decoder;
mod decoder_tdt;
mod error;
mod execution;
mod model;
mod model_eou;
mod model_tdt;
mod parakeet;
mod parakeet_eou;
pub mod parakeet_eou_fast;
mod parakeet_tdt;
#[cfg(feature = "sortformer")]
pub mod sortformer;
#[cfg(feature = "sortformer")]
pub mod sortformer_stream;
#[cfg(feature = "sortformer")]
pub mod realtime;
pub mod realtime_canary;
pub mod realtime_canary_flash;
pub mod realtime_tdt;
pub mod parallel_canary;
pub mod pause_parallel_canary;
pub mod parallel_tdt;
pub mod pause_parallel_tdt;
pub mod streaming_transcriber;
pub mod model_registry;
pub mod media_manager;
pub mod session;
pub mod vad;
pub mod realtime_canary_vad;
#[cfg(feature = "sortformer")]
pub mod realtime_tdt_vad;
pub mod noise_cancellation;
pub mod sentence_buffer;
pub mod growing_text;
pub mod vod_transcriber;
mod timestamps;
mod transcriber;
mod vocab;

pub use error::{Error, Result};
pub use execution::{ExecutionProvider, ModelConfig as ExecutionConfig};

/// Initialize ONNX Runtime. Required when using the `load-dynamic` feature.
/// This must be called before any model loading operations.
/// Safe to call multiple times - subsequent calls are no-ops.
pub fn init_ort() -> Result<()> {
    use std::sync::Once;
    static INIT: Once = Once::new();
    static mut INIT_RESULT: Option<String> = None;

    INIT.call_once(|| {
        match ort::init().commit() {
            Ok(_) => {
                eprintln!("[ORT] ONNX Runtime initialized successfully");
            }
            Err(e) => {
                unsafe { INIT_RESULT = Some(e.to_string()); }
            }
        }
    });

    unsafe {
        if let Some(ref err) = INIT_RESULT {
            return Err(Error::Model(format!("Failed to initialize ONNX Runtime: {}", err)));
        }
    }
    Ok(())
}
pub use parakeet::Parakeet;
pub use parakeet_tdt::ParakeetTDT;
pub use timestamps::TimestampMode;
pub use transcriber::*;

pub use config::{ModelConfig as ModelConfigJson, PreprocessorConfig};

pub use decoder::{ParakeetDecoder, TimedToken, TranscriptionResult};
pub use model::ParakeetModel;
pub use model_eou::ParakeetEOUModel;
pub use parakeet_eou::ParakeetEOU;
pub use parakeet_eou_fast::{
    ParakeetEOUFast, StreamingConfig, StreamingResult, RECOMMENDED_CHUNK_MS,
    RECOMMENDED_CHUNK_SAMPLES,
};

#[cfg(feature = "sortformer")]
pub use sortformer_stream::{SortformerStream, SortformerStreamBuilder, SpeakerPrediction, SpeakerState};

#[cfg(feature = "sortformer")]
pub use realtime::{
    RealtimeCallback, RealtimeConfig, RealtimeResult, RealtimeTranscriber,
    SimplifiedRealtimeTranscriber, Speaker, SpeakerUpdate,
};

pub use realtime_tdt::{ChunkResult, RealtimeTDT, RealtimeTDTConfig, Segment};

#[cfg(feature = "sortformer")]
pub use realtime_tdt::{DiarizedChunkResult, DiarizedSegment, RealtimeTDTDiarized};

pub use streaming_transcriber::{
    ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment, TranscriberFactory,
};

pub use model_registry::{ModelRegistry, ModelType, RegisteredModel, SharedModelRegistry};

pub use media_manager::{
    MediaFile, MediaFormat, MediaManager, MediaManagerConfig, SharedMediaManager,
};

pub use session::{
    MediaSourceType, SessionInfo, SessionManager, SessionState, SharedSessionManager,
    TranscriptionSession, VodProgressInfo,
};

pub use canary::{CanaryConfig, CanaryModel, CanaryTokenizer};
pub use canary_flash::{CanaryFlashConfig, CanaryFlashModel, DecoderKVCache};
pub use realtime_canary::{CanaryChunkResult, RealtimeCanary, RealtimeCanaryConfig};
pub use realtime_canary_flash::{CanaryFlashChunkResult, RealtimeCanaryFlash, RealtimeCanaryFlashConfig};
pub use parallel_canary::{ParallelCanary, ParallelCanaryConfig};
pub use pause_parallel_canary::{PauseParallelCanary, PauseParallelConfig};
pub use parallel_tdt::{ParallelTDT, ParallelTDTConfig};
pub use pause_parallel_tdt::{PauseParallelTDT, PauseParallelTDTConfig};

// VAD exports
pub use vad::{SileroVad, VadConfig, VadSegment, VadSegmenter, VadState, VAD_CHUNK_SIZE, VAD_SAMPLE_RATE};
pub use realtime_canary_vad::{CanaryVadResult, RealtimeCanaryVad, RealtimeCanaryVadConfig};

#[cfg(feature = "sortformer")]
pub use realtime_tdt_vad::{RealtimeTdtVad, RealtimeTdtVadConfig, TdtVadResult};

// Sentence buffer exports
pub use sentence_buffer::{SentenceBuffer, SentenceBufferConfig, SentenceBufferMode};

// Growing text merger exports
pub use growing_text::{
    FinalizedSentence, GrowingTextConfig, GrowingTextMerger, GrowingTextResult,
};

// VoD transcriber exports
pub use vod_transcriber::{
    SegmentCallback, VodConfig, VodProgress, VodSegment, VodTranscript, VodTranscriberCanary,
    VodTranscriberTDT, VodWord,
};

// Noise cancellation exports
pub use noise_cancellation::{
    create_noise_canceller, NoiseCancellationType, NoiseCanceller, RNNoiseProcessor,
};

