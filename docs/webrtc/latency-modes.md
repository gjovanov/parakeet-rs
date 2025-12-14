# Latency Modes

> **Navigation**: [Index](./README.md) | [Architecture](./architecture.md) | [API Reference](./api-reference.md) | Latency Modes | [Frontend](./frontend.md) | [Deployment](./deployment.md)

The server supports **12 latency modes**, each optimized for different use cases. Modes are selected per-session via the `mode` parameter when creating a session.

## Mode Overview

### Standard Modes (Continuous Processing)

| Mode | Buffer | Interval | Confirm | Expected Latency | Pause Detection |
|------|--------|----------|---------|------------------|-----------------|
| `speedy` | 8.0s | 0.2s | 0.4s | ~0.3-1.5s | ✅ Yes |
| `pause_based` | 10.0s | 0.3s | 0.5s | ~0.5-2.0s | ✅ Yes |
| `low_latency` | 10.0s | 1.5s | 2.0s | ~3.5s | ❌ No |
| `ultra_low_latency` | 8.0s | 1.0s | 1.5s | ~2.5s | ❌ No |
| `extreme_low_latency` | 5.0s | 0.5s | 0.8s | ~1.3s | ❌ No |
| `lookahead` | 10.0s | 0.3s | 0.5s | ~1.0-3.0s | ✅ Yes |

### VAD-Triggered Modes (Silero VAD)

| Mode | Description | Pause Threshold |
|------|-------------|-----------------|
| `vad_speedy` | Fast VAD-triggered transcription | ~0.3s pause |
| `vad_pause_based` | Accurate VAD with longer pauses | ~0.7s pause |
| `vad_sliding_window` | Buffers 5 segments / 10s with context overlap | N/A |

### Pure Streaming Mode

| Mode | Description |
|------|-------------|
| `asr` | Pure streaming ASR without VAD, continuous processing |

### Parallel Modes (Multi-threaded CPU Optimization)

| Mode | Threads | Description |
|------|---------|-------------|
| `parallel` | 8 (Canary) / 4 (TDT) | Multi-threaded sliding window inference |
| `pause_parallel` | 8 (Canary) / 4 (TDT) | Pause-triggered parallel with ordered output |

---

## Mode Details

### `speedy` (Default, Recommended)

Best balance of latency and quality.

- Uses pause detection to confirm at natural speech boundaries
- Aggressive timings (0.2s interval, 0.4s confirm threshold)
- **Ideal for:** Live captioning, real-time subtitles, interactive applications

```json
{
  "buffer_size_secs": 8.0,
  "process_interval_secs": 0.2,
  "confirm_threshold_secs": 0.4,
  "pause_based_confirm": true,
  "pause_threshold_secs": 0.35
}
```

### `pause_based`

Similar to speedy but with more conservative timings.

- Better accuracy at the cost of slightly higher latency
- **Ideal for:** High-quality transcription where accuracy > speed

```json
{
  "buffer_size_secs": 10.0,
  "process_interval_secs": 0.3,
  "confirm_threshold_secs": 0.5,
  "pause_based_confirm": true,
  "pause_threshold_secs": 0.3
}
```

### `low_latency`

Time-based confirmation without pause detection.

- Predictable, fixed latency
- **Ideal for:** Broadcast scenarios with consistent delay requirements

```json
{
  "buffer_size_secs": 10.0,
  "process_interval_secs": 1.5,
  "confirm_threshold_secs": 2.0,
  "pause_based_confirm": false
}
```

### `ultra_low_latency`

Faster than low-latency with smaller buffer.

- Good for applications needing faster response
- **Ideal for:** Live interviews, Q&A sessions

```json
{
  "buffer_size_secs": 8.0,
  "process_interval_secs": 1.0,
  "confirm_threshold_secs": 1.5,
  "pause_based_confirm": false
}
```

### `extreme_low_latency`

Fastest possible response time.

- May sacrifice some accuracy for speed
- **Ideal for:** Real-time voice assistants, gaming, interactive voice applications

```json
{
  "buffer_size_secs": 5.0,
  "process_interval_secs": 0.5,
  "confirm_threshold_secs": 0.8,
  "pause_based_confirm": false
}
```

### `lookahead`

Best transcription quality.

- Uses future context for better accuracy
- Processes segments with knowledge of subsequent audio
- **Ideal for:** Post-processing, archival transcription, highest quality requirements

```json
{
  "buffer_size_secs": 10.0,
  "process_interval_secs": 0.3,
  "confirm_threshold_secs": 0.5,
  "pause_based_confirm": true,
  "lookahead_mode": true,
  "lookahead_segments": 2
}
```

### `vad_speedy`

Silero VAD triggered with short pause detection.

- Transcribes complete utterances after ~0.3s pauses
- **Ideal for:** Conversational audio with clear speaker turns

### `vad_pause_based`

Silero VAD with longer pause detection (~0.7s).

- More accurate segmentation
- **Ideal for:** Presentations, lectures with natural pauses

### `vad_sliding_window`

Sliding window VAD mode.

- Buffers multiple VAD segments (5 segments or 10s)
- Transcribes with context overlap for better accuracy
- **Ideal for:** Complex multi-speaker audio

### `asr`

Pure streaming without VAD.

- Continuous sliding window processing
- **Ideal for:** Continuous audio streams without natural pauses

### `parallel`

Multi-threaded parallel inference with sliding window.

- Uses 8 threads for Canary model, 4 threads for TDT (TDT is faster)
- Each thread processes overlapping audio windows in round-robin
- Results are merged and deduplicated
- **Ideal for:** CPU-bound scenarios where you want maximum throughput

```json
{
  "num_threads": 8,
  "buffer_size_chunks": 6,
  "chunk_duration_secs": 1.0,
  "intra_threads": 1
}
```

### `pause_parallel`

Pause-triggered parallel inference with ordered output.

- Dispatches transcription jobs on speech pauses (natural boundaries)
- Multiple model instances process segments in parallel
- Maintains chronological order of output
- **Ideal for:** Conversational audio with multiple speakers

```json
{
  "num_threads": 8,
  "pause_threshold_secs": 0.3,
  "silence_energy_threshold": 0.008,
  "max_segment_duration_secs": 5.0,
  "context_buffer_secs": 3.0
}
```

---

## Choosing the Right Mode

### Quality vs Latency Spectrum

```
  Lower Latency                                    Higher Quality
  ◄─────────────────────────────────────────────────────────────────────►

  extreme    ultra      speedy     vad_      pause_    low_      lookahead
  _low_      _low_                 speedy    based     latency
  latency    latency

  ~1.3s      ~2.5s      ~0.3-1.5s  ~0.3s     ~0.5-2s   ~3.5s     ~1.0-3.0s
```

### VAD Modes (Complete Utterance Based)

```
  Fast Response                                    Better Accuracy
  ◄─────────────────────────────────────────────────────────────────────►

  vad_speedy          vad_pause_based         vad_sliding_window
  (~0.3s pause)       (~0.7s pause)           (5 seg / 10s context)
```

### Use Case Recommendations

| Use Case | Recommended Mode |
|----------|------------------|
| Live captioning | `speedy` |
| Real-time subtitles | `speedy` |
| Interactive applications | `speedy` or `extreme_low_latency` |
| Broadcast with fixed delay | `low_latency` |
| Live interviews | `ultra_low_latency` |
| Voice assistants | `extreme_low_latency` |
| High-quality transcription | `pause_based` or `lookahead` |
| Conversational audio | `vad_speedy` |
| Presentations/lectures | `vad_pause_based` |
| Multi-speaker meetings | `vad_sliding_window` |
| Continuous audio streams | `asr` |
| High-throughput CPU transcription | `parallel` |
| Multi-speaker with CPU optimization | `pause_parallel` |

---

## Token Confirmation Strategy

The key challenge in streaming ASR is deciding when to "confirm" (finalize) tokens.

### Time-Based Confirmation

```
Time:    0s        2s        4s        6s        8s       10s
Audio:   [=========================================]
         ↑                                        ↑
    Buffer Start                            Buffer End

         [CONFIRMED ZONE]  [PENDING ZONE]
         ◀───────────────▶ ◀────────────▶
              Emit these    May change
```

Tokens in the **confirmed zone** are emitted immediately and won't change. Tokens in the **pending zone** may be revised as more audio arrives.

### Pause-Based Confirmation

Detects natural speech pauses using RMS energy:

```rust
fn detect_silence(&mut self, samples: &[f32]) -> bool {
    // Calculate RMS energy
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_squares / samples.len() as f32).sqrt();

    // Compare against threshold (configurable)
    rms < self.config.silence_energy_threshold // e.g., 0.008
}
```

**Advantages of pause-based:**
- Respects natural speech boundaries
- Variable latency based on content
- Better sentence-level accuracy

**Advantages of time-based:**
- Predictable, fixed latency
- Works for continuous speech
- Simpler to reason about

---

## API Usage

### Specifying Mode in Session Creation

```bash
curl -X POST http://localhost:8080/api/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "tdt-en",
    "media_id": "media-uuid",
    "mode": "vad_speedy",
    "language": "en"
  }'
```

### Listing Available Modes

```bash
curl http://localhost:8080/api/modes
```

Response:

```json
{
  "success": true,
  "data": [
    {
      "id": "speedy",
      "name": "Speedy (~0.3-1.5s)",
      "description": "Best balance of latency and quality. Uses pause detection."
    },
    {
      "id": "vad_speedy",
      "name": "VAD Speedy (~0.3s pause)",
      "description": "Silero VAD triggered. Transcribes complete utterances after short pauses."
    }
    // ... more modes
  ]
}
```

---

## Related Documentation

- [Architecture](./architecture.md) - How modes affect the audio pipeline
- [API Reference](./api-reference.md) - Complete API for mode selection
- [Deployment](./deployment.md) - Server-side mode configuration
