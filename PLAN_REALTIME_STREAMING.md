# Real-Time Streaming Transcription with Diarization

## Goal
Implement streaming transcription with speaker diarization achieving **4-5 second maximum latency** between spoken audio and displayed transcription with speaker attribution.

---

## Current State Analysis

### Available Models

| Model | Type | Latency | Streaming | Notes |
|-------|------|---------|-----------|-------|
| **ParakeetEOU** | ASR | 1-4s | Quasi | 4s buffer, "not work very well" |
| **Sortformer** | Diarization | ~100ms/chunk | Yes | Outputs every ~10s audio |
| **TDT** | ASR | Full file | No | Batch only, NOT usable |

### Key Constraints
1. **ParakeetEOU** uses 4-second buffer - too large for low latency
2. **Sortformer** requires ~10 seconds of audio before meaningful speaker output
3. **TDT cannot stream** - requires full audio for encoder
4. Frame alignment: Both use 80ms frames (8x subsampling of 10ms mel frames)

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Audio Input Stream                                │
│                    (Microphone / Real-time source)                       │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Audio Buffer      │
                    │   (Ring Buffer)     │
                    │   ~2-3 seconds      │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Mel Feature    │  │  Mel Feature    │  │   Timestamp     │
│  Extractor      │  │  Extractor      │  │   Synchronizer  │
│  (for ASR)      │  │  (for Diar)     │  │                 │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    │
┌─────────────────┐  ┌─────────────────┐           │
│  ParakeetEOU    │  │   Sortformer    │           │
│  (Modified)     │  │   (Streaming)   │           │
│  ~500ms chunks  │  │   ~10s chunks   │           │
└────────┬────────┘  └────────┬────────┘           │
         │                    │                    │
         │  Transcription     │  Speaker           │
         │  + Timestamps      │  Segments          │
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Output Merger   │
                    │   & Attributor    │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Final Output     │
                    │  [Speaker]: Text  │
                    │  with timestamps  │
                    └───────────────────┘
```

---

## Implementation Plan

### Phase 1: Optimized ParakeetEOU for Low Latency
**Target: Reduce ASR latency from 1-4s to ~500ms-1s**

#### 1.1 Create `ParakeetEOUFast` variant
- New file: `src/parakeet_eou_fast.rs`
- Modifications:
  ```rust
  // Reduce buffer from 4s to 2s
  const BUFFER_SIZE_SECS: f32 = 2.0;
  const BUFFER_SIZE_SAMPLES: usize = 32000;  // 2s @ 16kHz

  // Reduce minimum buffer from 1s to 500ms
  const MIN_BUFFER_SAMPLES: usize = 8000;  // 500ms

  // Smaller chunks for lower latency
  const CHUNK_SIZE_MS: usize = 100;  // 100ms instead of 160ms
  const CHUNK_SIZE_SAMPLES: usize = 1600;

  // Reduced encoder cache
  const PRE_ENCODE_CACHE: usize = 5;  // 50ms instead of 90ms
  ```

#### 1.2 Implement sliding window with overlap
- Process 100ms chunks with 50ms overlap
- Use speculative decoding: output partial results, refine on next chunk
- Implement confidence-based output smoothing

### Phase 2: Streaming Sortformer Integration
**Target: Continuous speaker updates with minimal delay**

#### 2.1 Create `SortformerStream` wrapper
- New file: `src/sortformer_stream.rs`
- Features:
  - Accumulates audio continuously
  - Outputs speaker predictions as soon as available
  - Maintains speaker history for consistency
  - Interpolates speaker assignments between updates

#### 2.2 Speaker prediction buffering
```rust
struct SpeakerBuffer {
    // Recent speaker predictions (rolling window)
    predictions: VecDeque<SpeakerPrediction>,
    // Smoothed speaker assignments
    smoothed: HashMap<TimeRange, SpeakerId>,
    // Last update timestamp
    last_update: f32,
}
```

### Phase 3: Real-Time Synchronization
**Target: Merge ASR and diarization with <100ms alignment error**

#### 3.1 Create `RealtimeTranscriber` coordinator
- New file: `src/realtime.rs`
- Responsibilities:
  - Manages shared audio buffer
  - Coordinates ASR and diarization threads
  - Merges outputs with proper timestamps
  - Handles speaker attribution for partial results

#### 3.2 Output pipeline
```rust
pub struct RealtimeOutput {
    pub text: String,
    pub speaker: Option<SpeakerId>,
    pub start_time: f32,
    pub end_time: f32,
    pub is_final: bool,  // false = partial, subject to change
    pub confidence: f32,
}

pub trait RealtimeCallback: Send + Sync {
    fn on_partial(&self, output: RealtimeOutput);
    fn on_final(&self, output: RealtimeOutput);
    fn on_speaker_change(&self, speaker: SpeakerId, timestamp: f32);
}
```

### Phase 4: Thread-Safe Architecture
**Target: Efficient concurrent processing**

#### 4.1 Architecture
```rust
pub struct RealtimeTranscriber {
    // Shared state
    audio_buffer: Arc<RwLock<RingBuffer>>,
    speaker_state: Arc<RwLock<SpeakerBuffer>>,

    // Processing threads
    asr_thread: JoinHandle<()>,
    diar_thread: JoinHandle<()>,

    // Communication channels
    asr_tx: Sender<AsrChunk>,
    diar_tx: Sender<DiarChunk>,
    output_rx: Receiver<RealtimeOutput>,
}
```

#### 4.2 Channel-based communication
- Use `crossbeam-channel` for lock-free message passing
- ASR thread: Sends transcription segments
- Diarization thread: Sends speaker predictions
- Merger thread: Combines and outputs results

---

## Latency Budget (5 seconds total)

| Component | Latency | Cumulative |
|-----------|---------|------------|
| Audio capture | 50ms | 50ms |
| Buffer accumulation (minimum) | 500ms | 550ms |
| Feature extraction | 50ms | 600ms |
| ASR inference (ParakeetEOU) | 200ms | 800ms |
| ASR decoding | 100ms | 900ms |
| **First ASR output** | - | **~1s** |
| Diarization (first update) | 2-3s | 3-4s |
| Speaker attribution | 50ms | 3.5-4.5s |
| Output formatting | 10ms | **~4-5s** |

**Note**: After warmup, continuous latency drops to ~500-800ms for ASR, with speaker updates every ~2-3 seconds.

---

## File Changes Summary

### New Files
1. `src/parakeet_eou_fast.rs` - Low-latency ASR variant
2. `src/sortformer_stream.rs` - Streaming diarization wrapper
3. `src/realtime.rs` - Coordinator and public API
4. `examples/realtime_streaming.rs` - Demo with microphone input

### Modified Files
1. `src/lib.rs` - Export new modules
2. `Cargo.toml` - Add dependencies:
   - `crossbeam-channel` - Lock-free channels
   - `cpal` - Audio capture (for example)
   - `parking_lot` - Fast RwLock

### Dependencies to Add
```toml
[dependencies]
crossbeam-channel = "0.5"
parking_lot = "0.12"

[dev-dependencies]
cpal = "0.15"  # For microphone example
```

---

## Alternative Approaches Considered

### Option A: WebSocket-based streaming server
- **Pros**: Language-agnostic clients, standard protocol
- **Cons**: Additional latency, more complex deployment
- **Decision**: Can be added later as wrapper around core implementation

### Option B: Use Whisper instead of Parakeet
- **Pros**: Whisper has native streaming variants (whisper.cpp)
- **Cons**: Different model ecosystem, lose Parakeet's advantages
- **Decision**: Stick with Parakeet for consistency

### Option C: Wait for full diarization before output
- **Pros**: Simpler, always accurate speaker
- **Cons**: 10+ second latency, defeats purpose
- **Decision**: Rejected - use speculative attribution

---

## Implementation Order

1. **Week 1**: `ParakeetEOUFast` - Reduce ASR latency
   - Modify buffer sizes
   - Implement 100ms chunking
   - Test latency improvements

2. **Week 2**: `SortformerStream` - Streaming diarization
   - Wrapper for continuous updates
   - Speaker history smoothing
   - Interpolation between updates

3. **Week 3**: `RealtimeTranscriber` - Integration
   - Thread coordination
   - Channel-based merging
   - Output callback system

4. **Week 4**: Example & Polish
   - Microphone input example
   - Latency measurements
   - Documentation

---

## Success Criteria

1. First transcription output within **1 second** of speech
2. Speaker attribution within **5 seconds** of speech
3. Continuous operation without memory leaks
4. Real-time factor (RTF) < 0.3 on modern CPU
5. GPU acceleration reduces latency by 2-3x

---

## Design Decisions (Confirmed)

### 1. Speaker Attribution Before Diarization Ready
**Decision**: Show "Speaker ?" for first ~10 seconds
- Output transcription immediately with placeholder speaker
- When diarization catches up, emit retroactive speaker updates
- Clients can choose to display immediately or buffer

### 2. Partial Result Handling
**Decision**: Hybrid approach
- Stream partial results marked `is_final: false` for real-time display
- Emit final results marked `is_final: true` for storage/processing
- Clients can subscribe to either or both streams

### 3. Deployment Targets
**Decision**: Modular system supporting all modes
- Core: Rust library with callback API
- CLI: Command-line tool with microphone input
- Server: WebSocket server for network clients

---

## API Design

### Core Library API
```rust
// Callback trait for receiving results
pub trait RealtimeCallback: Send + Sync {
    /// Called for partial (interim) results - may change
    fn on_partial(&self, result: RealtimeResult);

    /// Called for final results - stable, won't change
    fn on_final(&self, result: RealtimeResult);

    /// Called when speaker is retroactively identified
    fn on_speaker_update(&self, update: SpeakerUpdate);
}

pub struct RealtimeResult {
    pub text: String,
    pub speaker: Speaker,  // Speaker::Unknown or Speaker::Id(n)
    pub start_time: f32,
    pub end_time: f32,
    pub is_final: bool,
    pub confidence: f32,
}

pub struct SpeakerUpdate {
    pub time_range: (f32, f32),
    pub old_speaker: Speaker,
    pub new_speaker: Speaker,
}

// Main API
pub struct RealtimeTranscriber { /* ... */ }

impl RealtimeTranscriber {
    pub fn new(config: RealtimeConfig) -> Result<Self>;
    pub fn start<C: RealtimeCallback>(&mut self, callback: C) -> Result<()>;
    pub fn push_audio(&mut self, samples: &[f32]) -> Result<()>;
    pub fn stop(&mut self) -> Result<()>;
}
```

### CLI Interface
```bash
# Basic usage (reads from default microphone)
parakeet-realtime

# With options
parakeet-realtime --device "USB Microphone" --format json --output transcript.jsonl

# Pipe mode (reads PCM from stdin)
ffmpeg -i stream.mp3 -f s16le -ar 16000 -ac 1 - | parakeet-realtime --stdin
```

### WebSocket Server
```bash
# Start server
parakeet-server --port 8080 --max-connections 10

# Client sends: Binary PCM audio frames (16kHz, mono, f32le)
# Server sends: JSON messages
{
  "type": "partial" | "final" | "speaker_update",
  "text": "Hello world",
  "speaker": "Speaker 0" | "Speaker ?",
  "start": 1.23,
  "end": 2.45,
  "confidence": 0.95
}
```

---

## Open Technical Questions

1. **GPU sharing**: Should ASR and diarization share GPU or use separate CUDA streams?
   - Tentative: Use single GPU with sequential inference to avoid memory contention
