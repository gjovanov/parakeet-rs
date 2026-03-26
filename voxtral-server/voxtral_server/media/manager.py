"""Media file management — listing, upload, delete, duration probing."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ..config import settings
from ..models import MediaFile

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
    except Exception:
        return 0.0


async def list_media() -> list[MediaFile]:
    """Scan media directory and return list of audio files."""
    media_dir = Path(settings.media_dir).resolve()
    if not media_dir.is_dir():
        return []

    files = []
    for entry in sorted(media_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        media_id = entry.stem
        duration = await get_duration(entry)
        files.append(MediaFile(
            id=media_id,
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

    filepath = media_dir / filename
    filepath.write_bytes(data)

    duration = await get_duration(filepath)
    return MediaFile(
        id=filepath.stem,
        filename=filename,
        format=filepath.suffix.lstrip("."),
        duration_secs=round(duration, 2),
        size_bytes=len(data),
    )


def delete_media(media_id: str) -> bool:
    """Delete a media file by ID. Returns True if deleted."""
    media_dir = Path(settings.media_dir).resolve()
    for entry in media_dir.iterdir():
        if entry.stem == media_id and entry.is_file():
            entry.unlink()
            return True
    return False


def get_media_path(media_id: str) -> Path | None:
    """Get the full path to a media file by ID."""
    media_dir = Path(settings.media_dir).resolve()
    for entry in media_dir.iterdir():
        if entry.stem == media_id and entry.is_file():
            if entry.suffix.lower() in AUDIO_EXTENSIONS:
                return entry
    return None
