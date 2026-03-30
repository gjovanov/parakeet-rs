"""FFmpeg audio source — decode file to PCM s16le 16kHz mono in realtime."""

from __future__ import annotations

import asyncio
import logging
import struct
from pathlib import Path
from typing import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)


SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le
CHUNK_SAMPLES = 8000  # 0.5s chunks
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE


async def stream_pcm(
    source: Path | str,
    cancel_event: asyncio.Event,
) -> AsyncIterator[np.ndarray]:
    """
    Spawn FFmpeg to decode audio source to PCM and yield f32 sample chunks.

    source can be a file Path or a URL string (e.g. srt://host:port).
    Uses -re for realtime pacing on files; SRT streams are inherently realtime.
    Yields numpy float32 arrays of samples in [-1.0, 1.0] range.
    """
    source_str = str(source)
    is_srt = source_str.startswith("srt://")

    cmd = ["ffmpeg"]
    if not is_srt:
        cmd += ["-re"]  # realtime pacing (not needed for live SRT)
    cmd += [
        "-i", source_str,
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

    label = Path(source_str).name if not is_srt else source_str
    logger.info("FFmpeg started PID=%d for %s", proc.pid, label)

    try:
        if proc.stdout is None:
            raise RuntimeError("FFmpeg process has no stdout pipe")
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

            # Convert s16le bytes to f32 numpy array in [-1.0, 1.0]
            samples_s16 = np.frombuffer(raw, dtype=np.int16)
            yield samples_s16.astype(np.float32) / 32768.0

    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        logger.info("FFmpeg finished PID=%d", proc.pid)


def resample_16k_to_48k(samples_16k: np.ndarray) -> np.ndarray:
    """3x upsample from 16kHz to 48kHz via sample repetition (numpy)."""
    return np.repeat(samples_16k, 3)
