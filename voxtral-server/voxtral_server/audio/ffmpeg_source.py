"""FFmpeg audio source — decode file/SRT to PCM s16le 16kHz mono in realtime."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

import numpy as np

logger = logging.getLogger(__name__)


SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le
CHUNK_SAMPLES = 8000  # 0.5s chunks
CHUNK_BYTES = CHUNK_SAMPLES * BYTES_PER_SAMPLE

# SRT reconnect settings
SRT_MAX_RETRIES = 0  # 0 = unlimited retries (reconnect until cancel_event)
SRT_RETRY_DELAY = 3  # seconds between reconnect attempts


async def _run_ffmpeg(
    source_str: str,
    cancel_event: asyncio.Event,
) -> AsyncIterator[np.ndarray]:
    """Single FFmpeg run — yields PCM chunks until EOF or cancel."""
    is_srt = source_str.startswith("srt://")

    cmd = ["ffmpeg"]
    if not is_srt:
        cmd += ["-re"]
    cmd += [
        "-i", source_str,
        "-f", "s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-loglevel", "error",
        "-",
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

            samples_s16 = np.frombuffer(raw, dtype=np.int16)
            yield samples_s16.astype(np.float32) / 32768.0

    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
        stderr_out = ""
        if proc.stderr:
            try:
                stderr_out = (await proc.stderr.read()).decode(errors="replace").strip()
            except Exception:
                pass
        logger.info("FFmpeg finished PID=%d (exit=%s)%s",
                     proc.pid, proc.returncode,
                     f" stderr: {stderr_out[:200]}" if stderr_out else "")


async def stream_pcm(
    source: Path | str,
    cancel_event: asyncio.Event,
) -> AsyncIterator[np.ndarray]:
    """
    Spawn FFmpeg to decode audio source to PCM and yield f32 sample chunks.

    For SRT sources, automatically reconnects on disconnect (encoder hiccup,
    network blip) until cancel_event is set.
    For file sources, runs once — no retry on EOF.
    """
    source_str = str(source)
    is_srt = source_str.startswith("srt://")

    attempt = 0
    while not cancel_event.is_set():
        async for chunk in _run_ffmpeg(source_str, cancel_event):
            yield chunk
            attempt = 0  # reset on successful data

        if not is_srt or cancel_event.is_set():
            break  # files don't reconnect

        attempt += 1
        if SRT_MAX_RETRIES and attempt > SRT_MAX_RETRIES:
            logger.warning("SRT reconnect limit reached (%d) for %s", SRT_MAX_RETRIES, source_str)
            break

        logger.warning("SRT disconnected for %s, reconnecting in %ds (attempt %d)...",
                        source_str, SRT_RETRY_DELAY, attempt)
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=SRT_RETRY_DELAY)
            break  # cancel_event was set during the wait
        except asyncio.TimeoutError:
            pass  # timeout expired, retry


def resample_16k_to_48k(samples_16k: np.ndarray) -> np.ndarray:
    """3x upsample from 16kHz to 48kHz via sample repetition (numpy)."""
    return np.repeat(samples_16k, 3)
