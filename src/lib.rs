//! # parakeet-rs
//!
//! Rust ASR server supporting Parakeet TDT and Canary 1B models via ONNX Runtime.
//!
//! ## Supported Models
//! - **Parakeet TDT** (0.6B) — English, CTC/TDT architecture
//! - **Canary 1B** — Multilingual (en, de, fr, es), encoder-decoder
//!
//! ## Supported Modes
//! - **speedy** — Low-latency sliding buffer, real-time feedback
//! - **growing_segments** — Live subtitles, word-by-word PARTIAL → FINAL
//! - **pause_segmented** — Acoustic pause detection, one transcription per pause, precise timestamps

// Core modules
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
mod timestamps;
mod transcriber;
mod vocab;

// Streaming/realtime modules
#[cfg(feature = "sortformer")]
pub mod sortformer;
#[cfg(feature = "sortformer")]
pub mod sortformer_stream;
#[cfg(feature = "sortformer")]
pub mod realtime;
pub mod realtime_canary;
pub mod realtime_tdt;

// Transcription modes
pub mod streaming_transcriber;
pub mod pause_segmented;
pub mod pause_segmented_tdt;
pub mod growing_text;
pub mod sentence_buffer;

// Infrastructure
pub mod model_registry;
pub mod media_manager;
pub mod session;
pub mod vad;
pub mod noise_cancellation;
pub mod german_normalizer;

// Core exports
pub use error::{Error, Result};
pub use execution::{
    ExecutionProvider, GpuOptimizationLevel, ModelConfig as ExecutionConfig, ModelRole,
};

/// Initialize ONNX Runtime. Required when using the `load-dynamic` feature.
pub fn init_ort() -> Result<()> {
    use std::sync::Once;
    static INIT: Once = Once::new();
    static mut INIT_RESULT: Option<String> = None;

    INIT.call_once(|| {
        let ok = ort::init().commit();
        if ok {
            eprintln!("[ORT] ONNX Runtime initialized successfully");
        } else {
            unsafe { INIT_RESULT = Some("ort::init().commit() returned false".to_string()); }
        }
    });

    unsafe {
        if let Some(ref err) = INIT_RESULT {
            return Err(Error::Model(format!("Failed to initialize ONNX Runtime: {}", err)));
        }
    }
    Ok(())
}

// Parakeet TDT exports
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
pub use realtime_tdt::{ChunkResult, RealtimeTDT, RealtimeTDTConfig, Segment};
#[cfg(feature = "sortformer")]
pub use realtime_tdt::{DiarizedChunkResult, DiarizedSegment, RealtimeTDTDiarized};

// Canary 1B exports
pub use canary::{CanaryConfig, CanaryModel, CanaryTokenizer};
pub use realtime_canary::{CanaryChunkResult, RealtimeCanary, RealtimeCanaryConfig};

// Sortformer/diarization exports
#[cfg(feature = "sortformer")]
pub use sortformer_stream::{SortformerStream, SortformerStreamBuilder, SpeakerPrediction, SpeakerState};
#[cfg(feature = "sortformer")]
pub use realtime::{
    RealtimeCallback, RealtimeConfig, RealtimeResult, RealtimeTranscriber,
    SimplifiedRealtimeTranscriber, Speaker, SpeakerUpdate,
};

// Streaming transcriber exports
pub use streaming_transcriber::{
    ModelInfo, StreamingChunkResult, StreamingTranscriber, TranscriptionSegment, TranscriberFactory,
};

// Model registry exports
pub use model_registry::{ModelRegistry, ModelType, RegisteredModel, SharedModelRegistry};

// Media manager exports
pub use media_manager::{
    MediaFile, MediaFormat, MediaManager, MediaManagerConfig, SharedMediaManager,
};

// Session exports
pub use session::{
    MediaSourceType, SessionInfo, SessionManager, SessionState, SharedSessionManager,
    TranscriptionSession,
};

// VAD exports
pub use vad::{SileroVad, VadConfig, VadSegment, VadSegmenter, VadState, VAD_CHUNK_SIZE, VAD_SAMPLE_RATE};

// Sentence buffer exports
pub use sentence_buffer::{SentenceBuffer, SentenceBufferConfig, SentenceBufferMode};

// Growing text merger exports
pub use growing_text::{
    FinalizedSentence, GrowingTextConfig, GrowingTextMerger, GrowingTextResult,
};

// German normalizer exports
pub use german_normalizer::{GermanTextNormalizer, normalize_german};

// Noise cancellation exports
pub use noise_cancellation::{
    create_noise_canceller, NoiseCancellationType, NoiseCanceller, RNNoiseProcessor,
};

// Pause segmented exports
pub use pause_segmented::{PauseSegmentedCanary, PauseSegmentedConfig};
pub use pause_segmented_tdt::PauseSegmentedTDT;
