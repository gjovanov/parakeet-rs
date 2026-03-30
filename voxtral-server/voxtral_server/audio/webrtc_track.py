"""Custom aiortc AudioStreamTrack that feeds PCM frames from the audio pipeline."""

from __future__ import annotations

import asyncio
import sys
import time
from fractions import Fraction

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame


SAMPLE_RATE = 48000  # WebRTC expects 48kHz
FRAME_SAMPLES = 960  # 20ms at 48kHz
CHANNELS = 1

# Log audio health every N seconds
_HEALTH_LOG_INTERVAL = 30.0


class PcmAudioTrack(MediaStreamTrack):
    """
    Audio track that reads PCM samples from an asyncio queue and produces
    av.AudioFrame objects at 48kHz mono, 20ms per frame.

    Uses a numpy ring buffer for zero-copy frame extraction.
    Pacing is driven by a simple wall-clock schedule (start_time + pts).
    The audio pump task (session_runner._audio_pump) feeds this track
    independently from vLLM, so audio delivery is never blocked.
    """

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)
        # Ring buffer: 4 seconds capacity
        self._ring = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
        self._ring_write = 0
        self._ring_read = 0
        self._timestamp = 0
        self._start_time: float | None = None
        # Diagnostics
        self._drops = 0
        self._silence_frames = 0
        self._total_frames = 0
        self._last_health_log = 0.0
        self._max_buffered = 0

    @property
    def _buffered(self) -> int:
        return self._ring_write - self._ring_read

    def _ring_push(self, samples: np.ndarray) -> None:
        n = len(samples)
        cap = len(self._ring)
        # If push would overflow, drop oldest data
        if self._buffered + n > cap:
            overflow = (self._buffered + n) - cap
            self._ring_read += overflow
        start = self._ring_write % cap
        if start + n <= cap:
            self._ring[start:start + n] = samples
        else:
            first = cap - start
            self._ring[start:] = samples[:first]
            self._ring[:n - first] = samples[first:]
        self._ring_write += n

    def _ring_pop(self, n: int) -> np.ndarray:
        cap = len(self._ring)
        start = self._ring_read % cap
        if start + n <= cap:
            out = self._ring[start:start + n].copy()
        else:
            first = cap - start
            out = np.concatenate([self._ring[start:], self._ring[:n - first]])
        self._ring_read += n
        return out

    def push_samples(self, samples_48k: np.ndarray) -> None:
        """Push 48kHz samples into the queue (called from audio pipeline)."""
        try:
            self._queue.put_nowait(samples_48k)
        except asyncio.QueueFull:
            self._drops += 1
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._queue.put_nowait(samples_48k)
            except asyncio.QueueFull:
                pass

    def _drain_queue_to_ring(self) -> None:
        """Move all available chunks from the asyncio queue into the ring buffer."""
        while True:
            try:
                chunk = self._queue.get_nowait()
                self._ring_push(chunk)
            except asyncio.QueueEmpty:
                break

    async def recv(self) -> AudioFrame:
        """Called by aiortc to get the next audio frame (20ms)."""
        if self._start_time is None:
            self._start_time = time.monotonic()

        # Pace to real-time: frame N should be delivered at start + N*20ms
        target_time = self._start_time + (self._timestamp / SAMPLE_RATE)
        now = time.monotonic()
        if target_time > now:
            await asyncio.sleep(target_time - now)

        # Drain queued chunks into ring buffer
        self._drain_queue_to_ring()

        # If ring is empty, do one brief async wait for data
        if self._buffered < FRAME_SAMPLES:
            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                self._ring_push(chunk)
            except asyncio.TimeoutError:
                pass

        # Extract frame or emit silence
        is_silence = False
        if self._buffered < FRAME_SAMPLES:
            frame_data = np.zeros(FRAME_SAMPLES, dtype=np.float32)
            is_silence = True
            self._silence_frames += 1
        else:
            frame_data = self._ring_pop(FRAME_SAMPLES)

        self._total_frames += 1
        buffered = self._buffered
        if buffered > self._max_buffered:
            self._max_buffered = buffered

        # Health log
        now = time.monotonic()
        if now - self._last_health_log >= _HEALTH_LOG_INTERVAL:
            self._last_health_log = now
            elapsed = now - (self._start_time or now)
            buf_ms = buffered / SAMPLE_RATE * 1000
            print(
                f"[AudioTrack] {elapsed:.0f}s: "
                f"buf={buf_ms:.0f}ms (max={self._max_buffered / SAMPLE_RATE * 1000:.0f}ms) "
                f"q={self._queue.qsize()}/{self._queue.maxsize} "
                f"silence={self._silence_frames}/{self._total_frames} "
                f"drops={self._drops}",
                file=sys.stderr,
            )
            self._max_buffered = 0

        # Build av.AudioFrame
        arr_s16 = np.clip(frame_data * 32767, -32768, 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(
            arr_s16.reshape(1, -1),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = SAMPLE_RATE
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, SAMPLE_RATE)
        self._timestamp += FRAME_SAMPLES

        return frame
