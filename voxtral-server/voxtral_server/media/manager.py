"""Media file management — listing, upload, delete, duration probing."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ..config import settings
from ..models import MediaFile

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}


async def get_duration(filepath: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(filepath),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return float(stdout.decode().strip())
    except Exception as e:
        logger.warning("ffprobe failed for %s: %s", filepath.name, e)
        return 0.0


async def list_media() -> list[MediaFile]:
    """Scan media directory and return list of audio files."""
    media_dir = Path(settings.media_dir).resolve()
    if not media_dir.is_dir():
        return []

    entries = []
    for entry in sorted(media_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
            entries.append(entry)

    # Probe durations concurrently instead of sequentially
    durations = await asyncio.gather(*[get_duration(e) for e in entries])

    files = []
    for entry, duration in zip(entries, durations):
        files.append(MediaFile(
            id=entry.stem,
            filename=entry.name,
            format=entry.suffix.lstrip("."),
            duration_secs=round(duration, 2),
            size_bytes=entry.stat().st_size,
        ))

    return files


async def save_upload(filename: str, data: bytes) -> MediaFile:
    """Save uploaded file and return MediaFile."""
    media_dir = Path(settings.media_dir).resolve()
    media_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename: strip directory components to prevent path traversal
    safe_name = Path(filename).name
    if not safe_name:
        raise ValueError("Invalid filename")

    filepath = media_dir / safe_name
    # Double-check resolved path is inside media_dir
    if not filepath.resolve().is_relative_to(media_dir):
        raise ValueError("Invalid filename")

    filepath.write_bytes(data)

    duration = await get_duration(filepath)
    return MediaFile(
        id=filepath.stem,
        filename=safe_name,
        format=filepath.suffix.lstrip("."),
        duration_secs=round(duration, 2),
        size_bytes=len(data),
    )


def delete_media(media_id: str) -> bool:
    """Delete a media file by ID. Returns True if deleted."""
    media_dir = Path(settings.media_dir).resolve()
    if not media_dir.is_dir():
        return False
    for entry in media_dir.iterdir():
        if entry.stem == media_id and entry.is_file():
            entry.unlink()
            return True
    return False


def get_media_path(media_id: str) -> Path | None:
    """Get the full path to a media file by ID."""
    media_dir = Path(settings.media_dir).resolve()
    if not media_dir.is_dir():
        return None
    for entry in media_dir.iterdir():
        if entry.stem == media_id and entry.is_file():
            if entry.suffix.lower() in AUDIO_EXTENSIONS:
                return entry
    return None
