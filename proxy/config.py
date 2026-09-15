import os
from dataclasses import dataclass


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    upstream_url: str = os.getenv("UPSTREAM_URL", "http://127.0.0.1:11434")
    upstream_api_key: str | None = os.getenv("UPSTREAM_API_KEY") or None
    host: str = os.getenv("PROXY_HOST", "127.0.0.1")
    port: int = env_int("PROXY_PORT", 8080)
    # Generation can take minutes, so only the phases that cannot legitimately stall
    # are bounded; a read timeout would kill long completions.
    connect_timeout: float = float(env_int("PROXY_CONNECT_TIMEOUT", 10))
    log_requests: bool = env_bool("PROXY_LOG_REQUESTS")
    compact_tools: bool = env_bool("PROXY_COMPACT_TOOLS", True)
    dynamic_tools: bool = env_bool("PROXY_DYNAMIC_TOOLS", True)


settings = Settings()