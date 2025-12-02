# Multi-Stream Parallel Transcription + GPU Support

## Current Architecture Analysis

### What We Have Now:
1. **Single transcriber instance** shared across all WebSocket clients
2. **Single stdin audio source** - all clients receive the same audio/transcription
3. **Architecture**: `stdin → single transcriber thread → broadcast to all clients`
4. **ONNX Runtime** with CPU execution provider (GPU support exists but requires feature flags)

### Key Limitation:
Current design is a **broadcast model** - one audio stream, many viewers. NOT multi-stream.

---

## Part 1: Multi-Stream Architecture

### Option A: Per-Client Transcriber (Simpler, More Memory)

```
┌─────────────────────────────────────────────────────────────┐
│                    WebRTC Server                             │
├─────────────────────────────────────────────────────────────┤
│  Client 1 ──► RealtimeTDT #1 ──► Subtitles ──► Client 1     │
│  Client 2 ──► RealtimeTDT #2 ──► Subtitles ──► Client 2     │
│  Client 3 ──► RealtimeTDT #3 ──► Subtitles ──► Client 3     │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Complete isolation between streams
- Simple conceptually - each client owns its transcriber
- No shared state/mutex contention

**Cons:**
- Memory: ~300-500MB per model instance × N clients
- Model loading time per new client (~2-5 seconds)

### Option B: Shared Model, Separate State (Complex, Less Memory)

```
┌─────────────────────────────────────────────────────────────┐
│               Shared Model Pool (Arc<Mutex>)                 │
├─────────────────────────────────────────────────────────────┤
│  Client 1 State ──┐                                          │
│  Client 2 State ──┼──► Model Pool ──► Thread Pool            │
│  Client 3 State ──┘     (1-N models)                         │
└─────────────────────────────────────────────────────────────┘
```

**Pros:**
- Less memory (shared model weights if ONNX supports)
- Better GPU utilization (batching)

**Cons:**
- ONNX Runtime sessions are NOT thread-safe for concurrent inference
- Would need a pool of sessions with mutex coordination
- Complex state management

### Recommended: Option A with Model Pool

Create a **bounded pool** of pre-loaded transcriber instances:

```rust
struct TranscriberPool {
    available: Arc<Mutex<Vec<RealtimeTDT>>>,
    max_size: usize,
    model_path: PathBuf,
    exec_config: ExecutionConfig,
}

impl TranscriberPool {
    fn acquire(&self) -> Option<RealtimeTDT> {
        // Try to get from pool, or create new if under max_size
    }

    fn release(&self, transcriber: RealtimeTDT) {
        // Reset state and return to pool
    }
}
```

---

## Part 2: GPU Support

### Current Infrastructure:
- Feature flags exist: `cuda`, `tensorrt` in Cargo.toml
- `ExecutionProvider` enum already supports CUDA/TensorRT
- `ModelConfig::apply_to_session_builder()` configures providers

### What Needs to Change:

#### 1. Runtime Detection of GPU
```rust
fn detect_gpu() -> Option<ExecutionProvider> {
    if std::env::var("USE_GPU").unwrap_or_default() == "true" {
        #[cfg(feature = "cuda")]
        {
            // Check if CUDA is actually available
            if cuda_available() {
                return Some(ExecutionProvider::Cuda);
            }
        }
    }
    None
}
```

#### 2. Dockerfile Update for CUDA
```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2-cudnn8-runtime-ubuntu22.04

# Install ONNX Runtime with CUDA support
# Build with: cargo build --release --features cuda,sortformer
```

#### 3. docker-compose.yml for GPU
```yaml
services:
  transcriber:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - USE_GPU=true
```

---

## Part 3: GPU + Multi-Threading Analysis

### Does GPU with Multiple Threads Make Sense?

**Short answer: It depends on the workload.**

#### Scenario 1: Single GPU, Multiple Streams
```
Stream 1 ──┐
Stream 2 ──┼──► GPU Queue ──► Single CUDA Session ──► Sequential Processing
Stream 3 ──┘
```

**Analysis:**
- ONNX Runtime CUDA sessions process one inference at a time
- GPU parallelism is WITHIN a single inference (matrix ops, attention)
- Multiple streams would queue and process sequentially
- **Result:** GPU helps each stream go faster, but streams wait for each other

#### Scenario 2: Multiple GPU Sessions (Same GPU)
```
Stream 1 ──► CUDA Session 1 ──┐
Stream 2 ──► CUDA Session 2 ──┼──► Shared GPU Memory
Stream 3 ──► CUDA Session 3 ──┘
```

**Analysis:**
- Each session loads model into GPU memory (~500MB-1GB per model)
- Limited by GPU memory (e.g., 8GB GPU = max ~8-16 sessions)
- CUDA streams can run concurrently if memory allows
- **Result:** Better parallelism but memory-bound

#### Scenario 3: Batched Inference (Optimal for GPU)
```
Stream 1 ──┐
Stream 2 ──┼──► Batch Collector ──► Single Inference ──► Demux Results
Stream 3 ──┘     (wait for N or timeout)
```

**Analysis:**
- Collect audio segments from multiple streams
- Process as single batched inference
- GPU excels at batched operations (better memory coalescing)
- **Result:** Maximum GPU utilization, but adds latency for batching

### Recommendation Matrix:

| # Streams | Hardware | Best Approach |
|-----------|----------|---------------|
| 1-2 | CPU (8+ cores) | Separate threads, no batching |
| 1-2 | GPU | Single CUDA session, serialize |
| 3-10 | CPU | Thread pool, separate instances |
| 3-10 | GPU | 2-3 CUDA sessions, batch within |
| 10+ | GPU | Dynamic batching with queue |
| 10+ | Multi-GPU | Session per GPU + load balancing |

---

## Part 4: Implementation Plan

### Phase 1: Multi-Stream Foundation
1. Add `stream_id` to `RealtimeTDT` for context isolation
2. Create `TranscriberPool` with configurable max instances
3. Modify WebSocket handler to acquire/release from pool
4. Add per-client audio ingestion (not just stdin broadcast)

### Phase 2: GPU Support
1. Add `--use-gpu` CLI flag and `USE_GPU` env var
2. Runtime GPU detection with graceful fallback
3. Create `Dockerfile.cuda` with NVIDIA base
4. Update docker-compose for GPU deployment

### Phase 3: Optimized Multi-Stream + GPU
1. Implement request batching for GPU mode
2. Add inference queue with configurable batch size/timeout
3. Implement memory monitoring to limit concurrent sessions
4. Add metrics (inference time, queue depth, GPU utilization)

### Files to Modify:

| File | Changes |
|------|---------|
| `src/execution.rs` | Add runtime GPU detection |
| `src/realtime_tdt.rs` | Add `reset()` method for pool reuse |
| `examples/webrtc_transcriber.rs` | Per-client transcription, pool |
| `Cargo.toml` | New feature flag combinations |
| `docker/Dockerfile.cuda` | New CUDA-enabled image |
| `docker-compose.yml` | GPU resource reservation |

---

## Key Trade-offs Summary

| Approach | Memory | Latency | Throughput | Complexity |
|----------|--------|---------|------------|------------|
| Per-client CPU | High | Low | Medium | Low |
| Pool + CPU | Medium | Low | Medium | Medium |
| Single GPU | Low | Low | High (sequential) | Low |
| Multi-session GPU | High | Low | High (parallel) | Medium |
| Batched GPU | Low | Medium | Highest | High |

**Recommended starting point:** Pool-based CPU with optional single-GPU acceleration. Add batching later if throughput becomes an issue.

---

## Context and Thread Safety Notes

### ParakeetTDT Model Characteristics:
- **Stateless inference**: Each `transcribe_samples()` call is independent
- **No hidden state**: No RNN states, caches, or persistent context between calls
- **Thread safety**: `&mut self` required - not safe for concurrent access to same instance

### ONNX Runtime Thread Safety:
- Sessions are NOT thread-safe for concurrent `run()` calls
- Multiple sessions can run in parallel (separate memory)
- GPU sessions share CUDA context but can run on different streams

### RealtimeTDT State:
- Ring buffer, lookahead segments, confirmed tokens - all per-instance
- Must be reset or recreated for new streams
- No cross-contamination if each stream has its own instance
