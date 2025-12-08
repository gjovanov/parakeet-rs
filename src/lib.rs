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
pub mod realtime_tdt;
pub mod streaming_transcriber;
pub mod model_registry;
pub mod media_manager;
pub mod session;
pub mod vad;
pub mod realtime_canary_vad;
#[cfg(feature = "sortformer")]
pub mod realtime_tdt_vad;
#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(feature = "whisper")]
pub mod realtime_whisper;
#[cfg(feature = "whisper")]
pub mod realtime_whisper_vad;
mod timestamps;
mod transcriber;
mod vocab;

pub use error::{Error, Result};
pub use execution::{ExecutionProvider, ModelConfig as ExecutionConfig};
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
    SessionInfo, SessionManager, SessionState, SharedSessionManager, TranscriptionSession,
};

pub use canary::{CanaryConfig, CanaryModel, CanaryTokenizer};
pub use realtime_canary::{CanaryChunkResult, RealtimeCanary, RealtimeCanaryConfig};

// VAD exports
pub use vad::{SileroVad, VadConfig, VadSegment, VadSegmenter, VadState, VAD_CHUNK_SIZE, VAD_SAMPLE_RATE};
pub use realtime_canary_vad::{CanaryVadResult, RealtimeCanaryVad, RealtimeCanaryVadConfig};

#[cfg(feature = "sortformer")]
pub use realtime_tdt_vad::{RealtimeTdtVad, RealtimeTdtVadConfig, TdtVadResult};

// Whisper exports
#[cfg(feature = "whisper")]
pub use whisper::{WhisperConfig, WhisperModel, WhisperTokenizer, WhisperVariant};
#[cfg(feature = "whisper")]
pub use realtime_whisper::{RealtimeWhisper, RealtimeWhisperConfig, WhisperChunkResult};
#[cfg(feature = "whisper")]
pub use realtime_whisper_vad::{RealtimeWhisperVad, RealtimeWhisperVadConfig, WhisperVadResult};
