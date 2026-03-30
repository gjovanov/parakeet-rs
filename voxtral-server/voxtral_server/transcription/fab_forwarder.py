"""FAB Live Transcription Forwarder.

Subscribes to a session's subtitle broadcast queue and forwards
transcription text to a FAB endpoint via HTTP GET.

Uses LiveSubtitler to process both partial and final messages:
- Partials: commits stable word prefixes (words stable for 0.5s, chunk >= 15 chars)
- Finals: flushes all remaining uncommitted text

This produces lower-latency output than waiting for finals, while
preventing flicker from unstable trailing words.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from collections import deque
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from ..state import SessionContext
from .segmenter import LiveSubtitler

logger = logging.getLogger(__name__)


# --- Text field selection ---


def select_text_field(msg: dict[str, Any], send_type: str) -> str:
    """Select which text field to send based on send_type.

    - "confirmed": use only the `text` field (finalized segment text)
    - "growing" (default): use `growing_text` (cumulative), fall back to `text`
    """
    if send_type == "confirmed":
        text = msg.get("text", "")
        return text if text else ""
    else:
        growing = msg.get("growing_text", "")
        if growing:
            return growing
        return msg.get("text", "") or ""


# --- Legacy utilities (kept for external callers / tests) ---


def is_dot_in_number(text: str, pos: int) -> bool:
    """Check if character at pos (which should be '.') is between two digits."""
    if pos <= 0 or pos >= len(text) - 1:
        return False
    return text[pos - 1].isdigit() and text[pos + 1].isdigit()


def split_for_teletext(text: str, max_chars: int = 84) -> list[str]:
    """Split text into chunks each <= max_chars characters.

    Prefers splitting at sentence boundaries (. ! ?), then clause
    separators (, ; : em-dash), then word boundaries (spaces).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    result: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= max_chars:
            trimmed = remaining.strip()
            if trimmed:
                result.append(trimmed)
            break

        search_region = remaining[:max_chars]

        best_sentence: int | None = None
        best_clause: int | None = None
        best_word: int | None = None

        for i, ch in enumerate(search_region):
            after_pos = i + 1
            if ch in '.':
                if not is_dot_in_number(remaining, i):
                    best_sentence = after_pos
            elif ch in '!?':
                best_sentence = after_pos
            elif ch in ',;:\u2013\u2014':
                best_clause = after_pos
            elif ch == ' ':
                best_word = i

        split_pos = best_sentence or best_clause or best_word or max_chars

        piece = remaining[:split_pos].strip()
        if piece:
            result.append(piece)

        remaining = remaining[split_pos:].lstrip()

    return result


def is_duplicate_of_history(candidate: str, history: deque[str]) -> bool:
    """Check if candidate text is a near-duplicate of any recently sent text."""
    if not candidate or not candidate.strip():
        return True

    candidate_words = set(candidate.split())
    if not candidate_words:
        return True

    for prev in reversed(history):
        if candidate == prev:
            return True
        if candidate in prev:
            return True
        prev_words = set(prev.split())
        if not prev_words:
            continue
        intersection = len(candidate_words & prev_words)
        if intersection >= 3:
            min_size = min(len(candidate_words), len(prev_words))
            if min_size > 0 and intersection / min_size >= 0.75:
                return True

    return False


# --- Validation ---


def validate_fab_url(url: str) -> bool:
    """Reject FAB URLs pointing to private/loopback addresses (SSRF protection)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = parsed.hostname or ""
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                return False
        except ValueError:
            if host in ("localhost", "metadata.google.internal"):
                return False
        return True
    except Exception:
        return False


# --- Async forwarder ---


async def _send_to_fab(
    client: httpx.AsyncClient,
    fab_url: str,
    session_id: str,
    text: str,
) -> bool:
    """Send text to the FAB endpoint via HTTP GET."""
    params = urlencode({"language": "Default", "text": text})
    url = f"{fab_url}?{params}"
    char_count = len(text)
    display = text[:90]
    logger.info("[FAB] [%s] Sending (%d chars): '%s'", session_id, char_count, display)
    try:
        resp = await client.get(url, timeout=5.0)
        logger.info("[FAB] [%s] Sent, status=%s", session_id, resp.status_code)
        return True
    except Exception as e:
        logger.error("[FAB] [%s] Send error: %s", session_id, e)
        return False


async def run_fab_forwarder(
    session_id: str,
    fab_url: str,
    send_type: str,
    queue: asyncio.Queue,
    cancel_event: asyncio.Event,
    max_chars: int = 42,
) -> None:
    """Main FAB forwarder loop using LiveSubtitler.

    Processes both partial and final subtitle messages. The subtitler
    tracks word-level stability and commits stable prefixes early,
    without waiting for finals.
    """
    subtitler = LiveSubtitler(max_chars=max_chars)
    sent_count = 0

    logger.info("[FAB] Forwarder started for session %s (send_type: %s, max_chars: %d)",
                session_id, send_type, max_chars)

    async with httpx.AsyncClient() as client:
        while not cancel_event.is_set():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Only process subtitle messages
            if msg.get("type") != "subtitle":
                continue

            text = select_text_field(msg, send_type)
            if not text.strip():
                continue

            ts = time.monotonic()
            is_final = msg.get("is_final") is True

            if is_final:
                segments = subtitler.on_final(text, ts)
            else:
                segments = subtitler.on_partial(text, ts)

            for segment in segments:
                if await _send_to_fab(client, fab_url, session_id, segment):
                    sent_count += 1

        # Flush any partial state left on shutdown
        # Drain remaining messages from the queue
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                if msg.get("type") != "subtitle":
                    continue
                text = select_text_field(msg, send_type)
                if text.strip():
                    is_final = msg.get("is_final") is True
                    ts = time.monotonic()
                    if is_final:
                        segments = subtitler.on_final(text, ts)
                    else:
                        segments = subtitler.on_partial(text, ts)
                    for segment in segments:
                        if await _send_to_fab(client, fab_url, session_id, segment):
                            sent_count += 1
            except asyncio.QueueEmpty:
                break

        # Final flush: if session ended mid-partial, force-commit leftovers
        final_segments = subtitler.on_final("", time.monotonic())
        for segment in final_segments:
            if await _send_to_fab(client, fab_url, session_id, segment):
                sent_count += 1

    logger.info("[FAB] Forwarder stopped for session %s (sent: %d)", session_id, sent_count)


def start_fab_forwarder(ctx: SessionContext) -> None:
    """Spawn FAB forwarder task if FAB is enabled for this session."""
    if not ctx.info.fab_enabled or not ctx.info.fab_url:
        return

    # SSRF protection: reject private/loopback FAB URLs from client requests
    if not validate_fab_url(ctx.info.fab_url):
        logger.warning("[FAB] [%s] Rejected FAB URL (private/loopback): %s",
                       ctx.info.id, ctx.info.fab_url)
        return

    ctx.fab_queue = asyncio.Queue(maxsize=256)
    ctx.fab_task = asyncio.create_task(
        run_fab_forwarder(
            session_id=ctx.info.id,
            fab_url=ctx.info.fab_url,
            send_type=ctx.info.fab_send_type,
            queue=ctx.fab_queue,
            cancel_event=ctx.cancel_event,
        ),
        name=f"fab-{ctx.info.id}",
    )
    logger.info("[FAB] [%s] Forwarder spawned (url=%s, type=%s)",
                ctx.info.id, ctx.info.fab_url, ctx.info.fab_send_type)
