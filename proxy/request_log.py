"""Opt-in logging of the traffic between the client, this proxy and the upstream."""

import json
import sys
from collections.abc import Mapping

from fastapi import Request

from . import config

_SECRET_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "openai-api-key",
    "anthropic-api-key",
}


def enabled() -> bool:
    return config.settings.log_requests


def _pretty(body: bytes) -> str:
    if not body:
        return "<empty body>"
    try:
        return json.dumps(json.loads(body), ensure_ascii=False, indent=2)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body.decode("utf-8", errors="replace")


def _redacted(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        key: ("[REDACTED]" if key.lower() in _SECRET_HEADERS else value)
        for key, value in headers.items()
    }


def _emit(title: str, summary: str, headers: Mapping[str, str], body: bytes) -> None:
    print(f"\n=== {title} | {summary} ===", file=sys.stderr)
    print(f"Headers: {_redacted(headers)}", file=sys.stderr)
    print(_pretty(body), file=sys.stderr)
    print(f"=== END {title} ===\n", file=sys.stderr)


def log_client_request(request: Request, body: bytes) -> None:
    """What the editor sent to the proxy, before any optimization."""
    if not enabled():
        return
    _emit(
        "CLIENT REQUEST",
        f"{request.method} {request.url.path} bytes={len(body)}",
        request.headers,
        body,
    )


def log_upstream_request(
    method: str, url: str, headers: Mapping[str, str], body: bytes, client_size: int | None
) -> None:
    """What the proxy actually sends to the model, internal retries included."""
    if not enabled():
        return
    summary = f"{method} {url} bytes={len(body)}"
    if client_size is not None:
        summary += f" client={client_size} saved={client_size - len(body)}"
    _emit("UPSTREAM REQUEST", summary, headers, body)


def log_upstream_response(
    status_code: int, headers: Mapping[str, str], body: bytes, streamed: bool
) -> None:
    """What the model answered, once a streamed body has been fully drained."""
    if not enabled():
        return
    transport = "streamed" if streamed else "buffered"
    _emit("UPSTREAM RESPONSE", f"{status_code} {transport} bytes={len(body)}", headers, body)
