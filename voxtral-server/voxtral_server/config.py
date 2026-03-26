"""Server configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # vLLM server WebSocket URL
    vllm_url: str = "ws://localhost:8001/v1/realtime"

    # Server port
    port: int = 8090

    # Media directory (shared with parakeet-rs)
    media_dir: str = "../media"

    # Frontend directory (shared with parakeet-rs)
    frontend_path: str = "../frontend"

    # Public IP for WebRTC (auto-detected if not set)
    public_ip: str = ""

    # TURN server (optional)
    turn_server: str = ""
    turn_username: str = ""
    turn_password: str = ""

    model_config = {
        "env_prefix": "VOXTRAL_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
