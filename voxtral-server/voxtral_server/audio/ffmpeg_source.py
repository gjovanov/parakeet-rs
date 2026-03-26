"""FFmpeg audio source — decode file to PCM s16le 16kHz mono in realtime."""

from __future__ import annotations

import asyncio
import struct
import sys
from pathlib import Path
from typing import AsyncIterator


SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le
CHUNK_SAMPLES = 8000  # 0.5s chunks
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


async def stream_pcm(
    filepath: Path,
    cancel_event: asyncio.Event,
) -> AsyncIterator[list[float]]:
    """
    Spawn FFmpeg to decode audio file to PCM and yield f32 sample chunks.

    Uses -re for realtime pacing (critical for playback sync).
    Yields lists of float samples in [-1.0, 1.0] range.
    """
    cmd = [
        "ffmpeg",
        "-re",  # realtime pacing
        "-i", str(filepath),
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-loglevel", "error",
        "-",  # stdout
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    print(f"[FFmpeg] Started PID={proc.pid} for {filepath.name}", file=sys.stderr)

    try:
        assert proc.stdout is not None
        while not cancel_event.is_set():
            try:
                raw = await asyncio.wait_for(
                    proc.stdout.read(CHUNK_BYTES),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                continue

            if not raw:
                break  # EOF

            # Convert s16le bytes to f32 samples in [-1.0, 1.0]
            n_samples = len(raw) // BYTES_PER_SAMPLE
            samples = struct.unpack(f"<{n_samples}h", raw[:n_samples * BYTES_PER_SAMPLE])
            yield [s / 32768.0 for s in samples]

    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        print(f"[FFmpeg] Finished PID={proc.pid}", file=sys.stderr)


def resample_16k_to_48k(samples_16k: list[float]) -> list[float]:
    """Simple 3x upsample from 16kHz to 48kHz (sample triplication)."""
    out = []
    for s in samples_16k:
        out.append(s)
        out.append(s)
        out.append(s)
    return out
