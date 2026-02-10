# Testing

[Architecture](architecture.md) | [API Reference](api.md) | [Transcription Modes](transcription-modes.md) | [Frontend](frontend.md) | [FAB Teletext](fab-teletext.md) | [Testing](testing.md) | [Deployment](deployment.md)

---

## Test Summary

| Type | Count | Location | Description |
|------|-------|----------|-------------|
| **Library unit tests** | 109 | `src/*.rs` | Core library logic (models, text processing, configs) |
| **Server unit tests** | 71 | `src/bin/server/**/*.rs` | API handlers, FAB forwarder (31 tests), transcription configs |
| **Integration tests** | 17 | `tests/integration_transcription.rs` | End-to-end transcription with real models |
| **Doc tests** | 2 | `src/lib.rs` | Documentation examples |
| **Playwright E2E** | 34 | `tests/e2e/*.spec.ts` | Browser-based UI and API tests |
| **Total** | **233** | | |

## Running Tests

### Rust Tests (199 total)

```bash
# Run all tests (unit + integration, excluding ignored)
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  cargo test --features "server,sortformer"

# Run only library tests
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  cargo test --lib --features "server,sortformer"

# Run only server binary tests
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  cargo test --bin parakeet-server --features "server,sortformer"

# Run a specific test
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  cargo test test_split_for_teletext --features "server,sortformer"

# Run integration tests (requires model files)
ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
  cargo test --test integration_transcription --features "server,sortformer" -- --ignored
```

### Playwright E2E Tests (34 total)

```bash
# Install dependencies
cd tests/e2e && bun install

# Run all E2E tests (requires running server on port 80)
BASE_URL=http://localhost:80 bunx playwright test

# Run a specific spec file
BASE_URL=http://localhost:80 bunx playwright test session-lifecycle.spec.ts

# Run with headed browser (visible)
BASE_URL=http://localhost:80 bunx playwright test --headed

# Run with debug mode
BASE_URL=http://localhost:80 bunx playwright test --debug
```

**Prerequisites**: The server must be running on the specified `BASE_URL` with models loaded.

## Rust Test Breakdown

### Library Tests (109)

Key test areas:

| Module | Tests | Description |
|--------|-------|-------------|
| `growing_text` | ~20 | GrowingTextMerger anchor-based merge, finalization, dedup |
| `sentence_buffer` | ~15 | Sentence boundary detection, completion modes |
| `canary` / `canary_flash` | ~10 | Tokenizer, decoder, KV cache shape handling |
| `vad` | ~8 | Silero VAD state machine, segment detection |
| `model_registry` | ~5 | Model discovery, type mapping |
| Various configs | ~50 | Config creation, validation, defaults |

### Server Tests (71)

| Module | Tests | Description |
|--------|-------|-------------|
| `fab_forwarder` | 31 | Teletext splitting, dedup, containment coefficient, debounce |
| `transcription/configs` | 16 | Per-mode config factories (Canary + TDT) |
| `transcription/factory` | 9 | Model type name mapping |
| Other | 15 | API response format, config, SRT |

### Integration Tests (17)

Located in `tests/integration_transcription.rs`. These test full transcription pipelines but require model files to be present.

**7 ignored tests** (require running server or model files):

| Test | Requirement |
|------|-------------|
| `test_canary_speedy_*` | Canary model files |
| `test_canary_pause_based_*` | Canary model files |
| `test_canary_vod_*` | Canary model files |
| `test_canary_low_latency_*` | Canary model files |
| `test_kv_cache_quality` | Canary model files |

## Playwright E2E Tests (34)

6 spec files covering all major features:

| Spec File | Tests | Description |
|-----------|-------|-------------|
| `session-lifecycle.spec.ts` | ~8 | Create, start, stop, delete sessions via API |
| `webrtc.spec.ts` | ~5 | WebRTC connection, ICE negotiation, audio playback |
| `transcription.spec.ts` | ~7 | Live transcription, subtitle display, partial/final messages |
| `vod-transcription.spec.ts` | ~5 | VoD batch processing, transcript download |
| `export.spec.ts` | ~4 | Transcript export functionality |
| `media-upload.spec.ts` | ~5 | File upload, media listing, deletion |

### E2E Test Gotchas

- **WebSocket URL**: `/ws/:session_id` (not `/api/sessions/:id/ws`)
- **Sessions need explicit start**: `POST /api/sessions/{id}/start` after creation
- **VoD race condition**: Connect WebSocket *before* starting session
- **Clear button**: Uses `confirm()` dialog — handle with `page.on('dialog')`
- **API envelope**: Always parse `json.data` from `{success, data, error}`

## Test Fixtures

Located in `tests/fixtures/`:

| File | Description |
|------|-------------|
| `de_short.wav` | Short German TTS audio clip |
| `de_medium.wav` | Medium German TTS audio clip |
| `de_long.wav` | Long German TTS audio clip |
| `de_numbers.wav` | German numbers/dates TTS clip |
| `references.json` | Expected transcriptions for WER/CER evaluation |

### Quality Metrics

Integration tests evaluate transcription quality using:

- **WER** (Word Error Rate): Percentage of word-level errors
- **CER** (Character Error Rate): Percentage of character-level errors
- **Key phrase recall**: Percentage of expected phrases found in output

Reference results from `references.json` are used to ensure models maintain expected quality.
