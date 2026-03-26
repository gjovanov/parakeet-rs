"""GET /api/config — server configuration for the frontend."""

from fastapi import APIRouter

from ..config import settings

router = APIRouter()


@router.get("/api/config")
async def get_config():
    host = settings.public_ip or "localhost"
    port = settings.port

    ice_servers = [{"urls": "stun:stun.l.google.com:19302"}]
    if settings.turn_server:
        ice_servers.append({
            "urls": settings.turn_server,
            "username": settings.turn_username,
            "credential": settings.turn_password,
        })

    return {
        "wsUrl": f"ws://{host}:{port}/ws",
        "iceServers": ice_servers,
        "audio": {"sampleRate": 16000, "channels": 1, "bufferSize": 4096},
        "subtitles": {"maxSegments": 1000, "autoScroll": True, "showTimestamps": True},
        "speakerColors": [
            "#4A90D9", "#50C878", "#E9967A", "#DDA0DD",
            "#F0E68C", "#87CEEB", "#FFB6C1", "#98FB98",
        ],
        "reconnect": {
            "enabled": True,
            "delay": 2000,
            "maxDelay": 30000,
            "maxAttempts": 10,
        },
    }
