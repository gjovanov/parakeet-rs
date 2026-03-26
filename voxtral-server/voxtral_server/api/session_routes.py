"""Session CRUD + start endpoints."""

from __future__ import annotations

import asyncio
import sys

from fastapi import APIRouter

from ..config import settings
from ..media.manager import get_media_path
from ..models import ApiResponse, CreateSessionRequest, SessionInfo, SessionState
from ..state import app_state

router = APIRouter()


@router.get("/api/sessions")
async def list_sessions():
    sessions = app_state.list_sessions()
    return ApiResponse.ok([s.model_dump() for s in sessions])


@router.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    # Resolve media file
    media_path = None
    media_filename = ""
    duration_secs = 0.0

    if req.media_id:
        media_path = get_media_path(req.media_id)
        if media_path is None:
            return ApiResponse.err(f"Media '{req.media_id}' not found")
        media_filename = media_path.name
        # Duration will be set when we start
        from ..media.manager import get_duration
        duration_secs = await get_duration(media_path)

    info = SessionInfo(
        model_id=req.model_id or "voxtral-mini-4b",
        model_name="Voxtral Mini 4B Realtime",
        media_id=req.media_id or "",
        media_filename=media_filename,
        duration_secs=duration_secs,
        mode=req.mode,
        language=req.language,
        noise_cancellation=req.noise_cancellation,
        diarization=req.diarization,
        sentence_completion=req.sentence_completion,
        without_transcription=req.without_transcription,
        source_type="file" if req.media_id else "srt",
    )

    app_state.add_session(info)
    print(f"[Session {info.id}] Created (model={info.model_id}, media={info.media_id}, lang={info.language})", file=sys.stderr)
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

    print(f"[Session {session_id}] Stopped", file=sys.stderr)
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

    # Resolve media path
    media_path = get_media_path(ctx.info.media_id)
    if media_path is None and not ctx.info.without_transcription:
        ctx.info.state = SessionState.ERROR
        return ApiResponse.err(f"Media '{ctx.info.media_id}' not found")

    # Import here to avoid circular imports
    from ..transcription.session_runner import run_session

    ctx.task = asyncio.create_task(
        run_session(ctx, media_path),
        name=f"session-{session_id}",
    )

    ctx.info.state = SessionState.RUNNING
    print(f"[Session {session_id}] Started (media={media_path})", file=sys.stderr)
    return ApiResponse.ok(ctx.info.model_dump())
