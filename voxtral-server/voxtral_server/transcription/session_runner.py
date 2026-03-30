"""Session runner — orchestrates FFmpeg, vLLM transcription, and subtitle emission."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

import numpy as np

from ..audio.ffmpeg_source import resample_16k_to_48k, stream_pcm, SAMPLE_RATE
from ..audio.webrtc_track import PcmAudioTrack
from ..state import SessionContext
from ..models import SessionState

logger = logging.getLogger(__name__)
from .vllm_client import VLLMClient


# How many seconds of audio to batch before sending to vLLM
BATCH_INTERVAL_SECS = 0.5
BATCH_SAMPLES = int(SAMPLE_RATE * BATCH_INTERVAL_SECS)

# Sentence boundary: period/exclamation/question followed by whitespace and uppercase.
# Must have a word (not just a number or abbreviation) before the punctuation.
# This avoids splitting on "19. November" or "Dr. Müller".
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[a-zäöüß][.!?])\s+(?=[A-ZÄÖÜ])'
)


def _extract_complete_sentences(text: str) -> tuple[list[str], str]:
    """
    Split text into complete sentences and a remaining fragment.

    Returns (sentences, remainder) where:
    - sentences: list of complete sentences (ending with . ! ?)
    - remainder: text after the last sentence boundary (may be empty)
    """
    parts = _SENTENCE_BOUNDARY.split(text)
    if not parts:
        return [], text

    sentences = []
    for part in parts[:-1]:
        s = part.strip()
        if s:
            sentences.append(s)

    # Last part: check if it ends with sentence punctuation
    last = parts[-1].strip()
    if last and last[-1] in '.!?':
        sentences.append(last)
        return sentences, ""
    else:
        return sentences, last


async def _audio_pump(
    ctx: SessionContext,
    media_path: Path | str,
    vllm_queue: asyncio.Queue,
) -> None:
    """Read audio from FFmpeg and push to WebRTC track + vLLM queue.

    This task runs independently so audio delivery to WebRTC is never
    blocked by vLLM processing, eliminating jitter.
    """
    session_id = ctx.info.id
    total_samples = 0

    async for samples_16k in stream_pcm(media_path, ctx.cancel_event):
        if ctx.cancel_event.is_set():
            break

        total_samples += len(samples_16k)
        ctx.info.progress_secs = total_samples / SAMPLE_RATE

        # Push to WebRTC track (48kHz) — never blocks
        samples_48k = resample_16k_to_48k(samples_16k)
        ctx.audio_track.push_samples(samples_48k)

        # Push to vLLM queue for transcription — drop-oldest on overflow
        if not ctx.info.without_transcription:
            try:
                vllm_queue.put_nowait(samples_16k)
            except asyncio.QueueFull:
                try:
                    vllm_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    vllm_queue.put_nowait(samples_16k)
                except asyncio.QueueFull:
                    pass

    # Signal EOF to vLLM task
    try:
        vllm_queue.put_nowait(None)
    except asyncio.QueueFull:
        pass


async def _vllm_task(
    ctx: SessionContext,
    vllm: VLLMClient,
    vllm_queue: asyncio.Queue,
    start_time: float,
) -> tuple[int, int]:
    """Process audio batches through vLLM and broadcast subtitles.

    Runs concurrently with the audio pump so vLLM latency never
    blocks audio delivery.
    """
    session_id = ctx.info.id
    growing_text = ""
    full_transcript = ""
    segment_count = 0
    audio_batch: list[float] = []
    total_samples = 0

    while not ctx.cancel_event.is_set():
        try:
            chunk = await asyncio.wait_for(vllm_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            break

        if chunk is None:
            break  # EOF from audio pump

        total_samples += len(chunk)

        # Accumulate into batch (convert numpy to list for vLLM)
        if isinstance(chunk, np.ndarray):
            audio_batch.extend(chunk.tolist())
        else:
            audio_batch.extend(chunk)

        if len(audio_batch) < BATCH_SAMPLES:
            continue

        # Send batch to vLLM
        t0 = time.monotonic()
        await vllm.send_audio(audio_batch)
        await vllm.commit()
        vllm_ms = (time.monotonic() - t0) * 1000
        audio_batch = []

        if vllm_ms > 200:
            logger.warning("Session %s: vLLM send+commit took %.0fms", session_id, vllm_ms)

        # Brief wait for vLLM to produce deltas
        await asyncio.sleep(0.1)

        batch_delta = vllm.drain_deltas()
        if not batch_delta:
            continue

        growing_text += batch_delta
        current_time = total_samples / SAMPLE_RATE

        # Extract complete sentences
        sentences, remainder = _extract_complete_sentences(growing_text)

        for sentence in sentences:
            segment_count += 1
            full_transcript = (full_transcript + " " + sentence).strip()
            await ctx.broadcast({
                "type": "subtitle",
                "text": sentence,
                "growing_text": None,
                "full_transcript": full_transcript,
                "delta": sentence,
                "tail_changed": False,
                "speaker": None,
                "start": max(0, current_time - 2.0),
                "end": current_time,
                "is_final": True,
                "inference_time_ms": int((time.monotonic() - start_time) * 1000) % 1000,
            })

        growing_text = remainder

        if growing_text.strip():
            await ctx.broadcast({
                "type": "subtitle",
                "text": growing_text.strip(),
                "growing_text": growing_text.strip(),
                "full_transcript": (full_transcript + " " + growing_text).strip(),
                "delta": batch_delta,
                "tail_changed": False,
                "speaker": None,
                "start": max(0, current_time - 2.0),
                "end": current_time,
                "is_final": False,
                "inference_time_ms": None,
            })

    # Flush remaining audio
    if audio_batch and vllm.is_connected:
        await vllm.send_audio(audio_batch)
        await vllm.commit()
        await asyncio.sleep(2.0)

        final_delta = vllm.drain_deltas()
        if final_delta:
            growing_text += final_delta

        if growing_text.strip():
            sentences, remainder = _extract_complete_sentences(growing_text)
            if remainder.strip():
                sentences.append(remainder.strip())

            current_time = total_samples / SAMPLE_RATE
            for sentence in sentences:
                segment_count += 1
                full_transcript = (full_transcript + " " + sentence).strip()
                await ctx.broadcast({
                    "type": "subtitle",
                    "text": sentence,
                    "growing_text": None,
                    "full_transcript": full_transcript,
                    "delta": sentence,
                    "tail_changed": False,
                    "speaker": None,
                    "start": max(0, current_time - 2.0),
                    "end": current_time,
                    "is_final": True,
                    "inference_time_ms": None,
                })

    return segment_count, total_samples


async def run_session(ctx: SessionContext, media_path: Path | str | None) -> None:
    """
    Main session loop:
    1. Start FFmpeg to decode audio (audio pump task)
    2. Run vLLM transcription concurrently (vLLM task)
    3. Audio pump pushes 48kHz to WebRTC + 16kHz to vLLM queue
    4. vLLM task processes batches and broadcasts subtitles

    Audio delivery and vLLM processing run in separate tasks so that
    vLLM latency never blocks audio, eliminating WebRTC jitter.
    """
    session_id = ctx.info.id
    vllm = VLLMClient(language=ctx.info.language)

    # Create audio track immediately so WS handler can attach it to WebRTC
    ctx.audio_track = PcmAudioTrack()

    try:
        # Connect to vLLM
        if not ctx.info.without_transcription:
            try:
                await vllm.connect()
            except Exception as e:
                logger.error("Session %s: vLLM connection failed: %s", session_id, e)
                ctx.info.state = SessionState.ERROR
                await ctx.broadcast({"type": "error", "message": f"vLLM connection failed: {e}"})
                return

        ctx.info.state = SessionState.RUNNING

        if media_path is None:
            logger.info("Session %s: no media file, waiting", session_id)
            return

        # Wait for a client to send "ready" before starting audio
        logger.info("Session %s: waiting for client...", session_id)
        try:
            await asyncio.wait_for(ctx.client_ready.wait(), timeout=10.0)
            logger.info("Session %s: client ready", session_id)
        except asyncio.TimeoutError:
            logger.warning("Session %s: no client after 10s, starting anyway", session_id)

        # Brief delay for WebRTC setup
        await asyncio.sleep(0.3)

        # Broadcast start
        await ctx.broadcast({"type": "start"})

        start_time = time.monotonic()

        # Shared queue between audio pump and vLLM task
        vllm_queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        # Run audio pump and vLLM processing concurrently
        audio_task = asyncio.create_task(
            _audio_pump(ctx, media_path, vllm_queue),
            name=f"audio-{session_id}",
        )

        vllm_result = None
        if not ctx.info.without_transcription:
            vllm_task_handle = asyncio.create_task(
                _vllm_task(ctx, vllm, vllm_queue, start_time),
                name=f"vllm-{session_id}",
            )
            results = await asyncio.gather(
                audio_task, vllm_task_handle, return_exceptions=True,
            )
            # vllm_task returns (segment_count, total_samples) or exception
            if isinstance(results[1], tuple):
                vllm_result = results[1]
        else:
            await audio_task

        # Broadcast end
        total_samples = int(ctx.info.progress_secs * SAMPLE_RATE)
        total_duration = total_samples / SAMPLE_RATE
        ctx.info.state = SessionState.COMPLETED
        await ctx.broadcast({
            "type": "end",
            "total_duration": round(total_duration, 2),
        })

        segment_count = vllm_result[0] if vllm_result else 0
        wall_time = time.monotonic() - start_time
        logger.info("Session %s: completed %.1fs audio, %d segments, %.1fs wall",
                     session_id, total_duration, segment_count, wall_time)

    except asyncio.CancelledError:
        logger.info("Session %s: cancelled", session_id)
        ctx.info.state = SessionState.STOPPED
    except Exception as e:
        logger.error("Session %s: %s", session_id, e)
        ctx.info.state = SessionState.ERROR
        await ctx.broadcast({"type": "error", "message": str(e)})
    finally:
        await vllm.disconnect()
