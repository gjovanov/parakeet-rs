"""Session CRUD + start endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from fastapi import APIRouter

from ..config import settings
from ..media.manager import get_media_path
from ..models import ApiResponse, CreateSessionRequest, SessionInfo, SessionState
from ..state import app_state

logger = logging.getLogger(__name__)

# --- Shared SRT config (loaded once, not per-request) ---

_srt_config: dict[str, str] | None = None


def _get_srt_config() -> dict[str, str]:
    """Load and cache SRT config from .env files + os.environ."""
    global _srt_config
    if _srt_config is None:
        from dotenv import dotenv_values
        env: dict[str, str | None] = {}
        for path in [Path(".env"), Path("../.env")]:
            if path.is_file():
                env.update(dotenv_values(path))
        env.update(os.environ)
        _srt_config = {k: v for k, v in env.items() if v is not None}
    return _srt_config


def _resolve_srt_url(channel_id: int) -> str | None:
    """Build an SRT URL from channel ID using SRT_ENCODER_IP + SRT_CHANNELS."""
    env = _get_srt_config()

    encoder_ip = env.get("SRT_ENCODER_IP", "")
    channels_json = env.get("SRT_CHANNELS", "")
    if not encoder_ip or not channels_json:
        return None

    try:
        channels = json.loads(channels_json)
    except json.JSONDecodeError:
        return None

    if channel_id < 0 or channel_id >= len(channels):
        return None

    # Validate port is a valid integer
    port_str = channels[channel_id].get("port", "")
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            return None
    except (ValueError, TypeError):
        return None

    latency = env.get("SRT_LATENCY", "200000")
    rcvbuf = env.get("SRT_RCVBUF", "2097152")
    # Validate latency/rcvbuf are positive integers
    try:
        int(latency)
        int(rcvbuf)
    except (ValueError, TypeError):
        latency, rcvbuf = "200000", "2097152"

    return f"srt://{encoder_ip}:{port}?mode=caller&latency={latency}&rcvbuf={rcvbuf}"


router = APIRouter()


@router.get("/api/sessions")
async def list_sessions():
    sessions = app_state.list_sessions()
    return ApiResponse.ok([s.model_dump() for s in sessions])


@router.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    # Resolve media file or SRT channel
    media_path = None
    media_filename = ""
    duration_secs = 0.0
    source_type = "file"
    srt_url = ""

    if req.srt_channel_id is not None:
        srt_url = _resolve_srt_url(req.srt_channel_id) or ""
        if not srt_url:
            return ApiResponse.err(f"SRT channel {req.srt_channel_id} not found or SRT not configured")
        source_type = "srt"
        media_filename = srt_url
    elif req.media_id:
        media_path = get_media_path(req.media_id)
        if media_path is None:
            return ApiResponse.err(f"Media '{req.media_id}' not found")
        media_filename = media_path.name
        from ..media.manager import get_duration
        duration_secs = await get_duration(media_path)
    else:
        return ApiResponse.err("Either media_id or srt_channel_id is required")

    # Resolve FAB config: per-session override > server default
    fab_enabled = req.fab_enabled if req.fab_enabled is not None else settings.fab_enabled
    fab_url = req.fab_url or settings.fab_url
    fab_send_type = req.fab_send_type or settings.fab_send_type

    info = SessionInfo(
        model_id=req.model_id or "voxtral-mini-4b",
        model_name="Voxtral Mini 4B Realtime",
        media_id=req.media_id or srt_url,
        media_filename=media_filename,
        duration_secs=duration_secs,
        mode=req.mode,
        language=req.language,
        noise_cancellation=req.noise_cancellation,
        diarization=req.diarization,
        sentence_completion=req.sentence_completion,
        without_transcription=req.without_transcription,
        source_type=source_type,
        fab_enabled=fab_enabled,
        fab_url=fab_url,
        fab_send_type=fab_send_type,
    )

    app_state.add_session(info)
    fab_label = f", fab={fab_url}" if fab_enabled else ""
    logger.info("Session %s created (model=%s, source=%s, media=%s, lang=%s%s)",
                info.id, info.model_id, source_type, info.media_id, info.language, fab_label)
    return ApiResponse.ok(info.model_dump())


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    ctx = app_state.get_session(session_id)
    if ctx is None:
        return ApiResponse.err(f"Session '{session_id}' not found")
    return ApiResponse.ok(ctx.info.model_dump())


@router.delete("/api/sessions/{session_id}")
async def stop_session(session_id: str):
    ctx = app_state.get_session(session_id)
    if ctx is None:
        return ApiResponse.err(f"Session '{session_id}' not found")

    ctx.info.state = SessionState.STOPPED
    ctx.cancel_event.set()
    if ctx.task and not ctx.task.done():
        ctx.task.cancel()

    logger.info("Session %s stopped", session_id)
    app_state.remove_session(session_id)
    return ApiResponse.ok({"stopped": session_id})


@router.post("/api/sessions/{session_id}/start")
async def start_session(session_id: str):
    ctx = app_state.get_session(session_id)
    if ctx is None:
        return ApiResponse.err(f"Session '{session_id}' not found")

    if ctx.info.state != SessionState.CREATED:
        return ApiResponse.err(f"Session already in state '{ctx.info.state}'")

    ctx.info.state = SessionState.STARTING

    # Spawn FAB forwarder before audio starts (so queue is attached before broadcasts)
    from ..transcription.fab_forwarder import start_fab_forwarder
    start_fab_forwarder(ctx)

    # Resolve audio source: SRT URL or media file path
    if ctx.info.source_type == "srt":
        audio_source: Path | str = ctx.info.media_id
    else:
        media_path = get_media_path(ctx.info.media_id)
        if media_path is None and not ctx.info.without_transcription:
            ctx.info.state = SessionState.ERROR
            return ApiResponse.err(f"Media '{ctx.info.media_id}' not found")
        audio_source = media_path

    from ..transcription.session_runner import run_session

    ctx.task = asyncio.create_task(
        run_session(ctx, audio_source),
        name=f"session-{session_id}",
    )

    # Return STARTING, not RUNNING — let the task set RUNNING when ready
    logger.info("Session %s starting (source=%s)", session_id, audio_source)
    return ApiResponse.ok(ctx.info.model_dump())
