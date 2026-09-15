"""HTTP plumbing between the client, this proxy and the upstream LLM server."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any
import json

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

from . import config
from .request_log import enabled as log_enabled, log_upstream_request, log_upstream_response

_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Bodies are handed to httpx already decoded, so the upstream encoding no longer applies.
_DROPPED_RESPONSE_HEADERS = _HOP_BY_HOP_HEADERS | {"content-encoding"}


def upstream_url(path: str, query: str) -> str:
    base = config.settings.upstream_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}{f'?{query}' if query else ''}"


def forward_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS | {"host"}
    }
    if config.settings.upstream_api_key:
        headers["authorization"] = f"Bearer {config.settings.upstream_api_key}"
    return headers


def encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def is_event_stream(response: httpx.Response) -> bool:
    return "text/event-stream" in response.headers.get("content-type", "")


def _passthrough_headers(response: httpx.Response, dropped: set[str]) -> dict[str, str]:
    return {key: value for key, value in response.headers.items() if key.lower() not in dropped}


@dataclass(frozen=True)
class UpstreamResponse:
    """A fully read upstream response, ready to be inspected and forwarded."""

    body: bytes
    status_code: int
    content_type: str
    headers: dict[str, str]

    def to_response(self) -> Response:
        return Response(
            content=self.body,
            status_code=self.status_code,
            headers=self.headers,
            media_type=self.content_type or None,
        )


async def send_upstream(request: Request, path: str, headers: dict[str, str], body: bytes) -> httpx.Response:
    url = upstream_url(f"/{path}", request.url.query)
    log_upstream_request(request.method, url, headers, body, getattr(request.state, "client_size", None))
    return await request.app.state.client.send(
        request.app.state.client.build_request(
            request.method,
            url,
            headers=headers,
            content=body,
        ),
        stream=True,
    )


async def read_upstream(
    response: httpx.Response,
    prefix: bytes = b"",
    chunks: AsyncIterator[bytes] | None = None,
) -> UpstreamResponse:
    """Buffer a response, optionally resuming from a partially consumed stream."""
    buffered = bytearray(prefix)
    try:
        async for chunk in chunks if chunks is not None else response.aiter_bytes():
            buffered.extend(chunk)
    finally:
        await response.aclose()
    result = UpstreamResponse(
        bytes(buffered),
        response.status_code,
        response.headers.get("content-type", ""),
        _passthrough_headers(response, _DROPPED_RESPONSE_HEADERS),
    )
    log_upstream_response(result.status_code, response.headers, result.body, streamed=False)
    return result


async def peek_stream(
    response: httpx.Response, decided: Callable[[bytes], bool]
) -> tuple[bytes, AsyncIterator[bytes]]:
    """Buffer just enough of the stream for the caller to commit, and hand back the rest."""
    chunks = response.aiter_bytes()
    buffered = bytearray()
    async for chunk in chunks:
        buffered.extend(chunk)
        if decided(bytes(buffered)):
            break
    return bytes(buffered), chunks


async def _drain(
    response: httpx.Response, prefix: bytes, chunks: AsyncIterator[bytes]
) -> AsyncIterator[bytes]:
    recorded = bytearray(prefix) if log_enabled() else None
    if prefix:
        yield prefix
    try:
        async for chunk in chunks:
            if recorded is not None:
                recorded.extend(chunk)
            yield chunk
    finally:
        await response.aclose()
        if recorded is not None:
            log_upstream_response(response.status_code, response.headers, bytes(recorded), streamed=True)


def streaming_response(
    response: httpx.Response,
    prefix: bytes = b"",
    chunks: AsyncIterator[bytes] | None = None,
) -> StreamingResponse:
    return StreamingResponse(
        _drain(response, prefix, chunks if chunks is not None else response.aiter_bytes()),
        status_code=response.status_code,
        headers=_passthrough_headers(response, _DROPPED_RESPONSE_HEADERS),
        media_type="text/event-stream",
    )
