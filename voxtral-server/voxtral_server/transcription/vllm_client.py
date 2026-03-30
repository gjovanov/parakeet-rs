"""vLLM Realtime API WebSocket client — sends audio, receives transcription deltas."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from typing import AsyncIterator

import numpy as np
import websockets
from websockets.asyncio.client import ClientConnection

from ..config import settings

logger = logging.getLogger(__name__)


def _needs_space(left: str, right: str) -> bool:
    """Check if a space is needed between two token strings.

    Fixes missing word boundaries in vLLM transcription deltas.
    Examples:
        "erst" + "13"   → True  (letter-digit)
        "13"  + "und"   → True  (digit-letter)
        "hello" + "."   → False (no space before punctuation)
        " the" + "cat"  → False (right already has leading space)
        "ab"  + " cd"   → False (right already has leading space)
    """
    if right[0] in ' \t\n':
        return False
    lc = left[-1]
    rc = right[0]
    # Letter followed by digit or digit followed by letter
    if lc.isalpha() and rc.isdigit():
        return True
    if lc.isdigit() and rc.isalpha():
        return True
    return False


class VLLMClient:
    """
    WebSocket client for vLLM's /v1/realtime endpoint (OpenAI Realtime API format).

    Uses a background reader task to continuously consume deltas from vLLM,
    avoiding the problem of missing tokens due to short polling timeouts.

    For long-running streams (SRT), automatically rotates the vLLM session
    before the context window fills up, preventing EngineCore crashes.
    """

    # Each commit adds ~50-80 audio tokens. At 16384 max_model_len,
    # rotate well before the limit. 0.5s batches × 200 commits ≈ 100s of audio.
    MAX_COMMITS_BEFORE_ROTATE = 200

    def __init__(self, language: str = "de") -> None:
        self._url = settings.vllm_url
        self._language = language
        self._ws: ClientConnection | None = None
        self._connected = False
        # Background reader pushes deltas here
        self._delta_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4096)
        self._reader_task: asyncio.Task | None = None
        self._commit_count = 0

    async def connect(self) -> None:
        """Connect to vLLM WebSocket and start background reader."""
        try:
            self._ws = await websockets.connect(
                self._url,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10MB
            )
            self._connected = True
            logger.info("vLLM connected to %s", self._url)

            # Send session.update to validate the model (required by vLLM Realtime API)
            # vLLM expects "model" at the top level of the event
            session_update = json.dumps({
                "type": "session.update",
                "model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
                "input_audio_format": "pcm16",
                "language": self._language,
                "turn_detection": None,
            })
            await self._ws.send(session_update)
            logger.info("vLLM session update sent (language=%s)", self._language)

            # Wait for session.updated confirmation
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                if msg.get("type") == "session.updated":
                    logger.info("vLLM session validated")
                else:
                    logger.warning("vLLM unexpected response: %s", msg.get('type', 'unknown'))
            except asyncio.TimeoutError:
                logger.warning("vLLM no session.updated response, continuing")

            # Start background reader
            self._reader_task = asyncio.create_task(self._background_reader())

        except Exception as e:
            self._connected = False
            logger.error("vLLM connection failed: %s", e)
            raise

    async def _background_reader(self) -> None:
        """Continuously read from vLLM WebSocket and enqueue deltas."""
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        delta_count = 0
        other_count = 0
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type", "")
                if msg_type in ("response.audio_transcript.delta", "transcription.delta"):
                    delta = msg.get("delta", "")
                    if delta:
                        delta_count += 1
                        try:
                            self._delta_queue.put_nowait(delta)
                        except asyncio.QueueFull:
                            try:
                                self._delta_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                pass
                            self._delta_queue.put_nowait(delta)
                elif msg_type == "error":
                    logger.error("vLLM error: %s", msg.get('error', msg))
                else:
                    other_count += 1
                    if other_count <= 5:
                        logger.debug("vLLM reader got: %s", msg_type)

        except websockets.exceptions.ConnectionClosed as e:
            logger.info("vLLM connection closed: %s (%d deltas)", e, delta_count)
        except asyncio.CancelledError:
            logger.info("vLLM reader cancelled (%d deltas)", delta_count)
        except Exception as e:
            logger.error("vLLM reader error: %s (%d deltas)", e, delta_count)
        finally:
            self._connected = False
            logger.info("vLLM reader stopped (%d deltas, %d other)", delta_count, other_count)

    async def disconnect(self) -> None:
        """Close vLLM WebSocket and stop reader."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    async def send_audio(self, samples_f32: list[float]) -> None:
        """
        Send audio samples to vLLM.

        Converts f32 samples to s16le bytes, base64-encodes, and sends as
        input_audio_buffer.append message.
        """
        if not self.is_connected:
            logger.warning("send_audio skipped — not connected (queue=%d)", self._delta_queue.qsize())
            return

        # Convert f32 to s16le bytes using numpy (fast)
        arr = np.array(samples_f32, dtype=np.float32)
        raw = np.clip(arr * 32768, -32768, 32767).astype(np.int16).tobytes()

        # Base64 encode
        audio_b64 = base64.b64encode(raw).decode("ascii")

        # Send append message
        msg = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })
        try:
            await self._ws.send(msg)
        except Exception as e:
            self._connected = False
            logger.error("vLLM send error: %s", e)

    async def commit(self) -> None:
        """Send commit message to trigger transcription."""
        if not self.is_connected:
            return
        try:
            await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            self._commit_count += 1

            # Rotate context before hitting max_model_len
            if self._commit_count >= self.MAX_COMMITS_BEFORE_ROTATE:
                await self._rotate_session()
        except Exception as e:
            self._connected = False
            logger.error("vLLM commit error: %s", e)

    async def _rotate_session(self) -> None:
        """Disconnect and reconnect to reset the vLLM context window.

        This prevents EngineCore crashes when accumulated audio tokens
        exceed max_model_len on long-running streams.
        """
        logger.info("vLLM rotating session (%d commits)", self._commit_count)
        await self.disconnect()
        self._commit_count = 0
        try:
            await self.connect()
            logger.info("vLLM session rotated successfully")
        except Exception as e:
            logger.error("vLLM rotation failed: %s", e)

    def drain_deltas(self) -> str:
        """
        Drain all available deltas from the queue (non-blocking).

        Returns concatenated delta text. Called from the session runner
        after each audio batch. Fixes word boundaries between tokens
        (e.g. "erst" + "13" → "erst 13", not "erst13").
        """
        text = ""
        while True:
            try:
                delta = self._delta_queue.get_nowait()
                if text and delta and _needs_space(text, delta):
                    text += " "
                text += delta
            except asyncio.QueueEmpty:
                break
        return text
