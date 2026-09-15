"""FastAPI application wiring the request optimizations into a transparent proxy."""

from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config
from .dynamic_tools import prepare_dynamic_tools
from .messages import optimize_request
from .request_log import log_client_request
from .resolver import resolve_dynamic_tools
from .responses import loader_intent
from .upstream import (
    UpstreamResponse,
    encode_payload,
    forward_headers,
    is_event_stream,
    peek_stream,
    read_upstream,
    send_upstream,
    streaming_response,
)

_EVENT_STREAM = "text/event-stream"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Completions legitimately take minutes, so only the phases that cannot stall
    # on the model are bounded.
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(None, connect=config.settings.connect_timeout, pool=config.settings.connect_timeout)
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="LLM Request Proxy", lifespan=lifespan)


async def _json_payload(request: Request, body: bytes) -> dict[str, Any] | None:
    if request.method not in {"POST", "PUT", "PATCH"} or not body:
        return None
    if "application/json" not in request.headers.get("content-type", ""):
        return None
    try:
        payload = await request.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _optimized_payloads(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the payload to send, the fallback payload and the dynamic tool registry."""
    base = optimize_request(payload)
    if not config.settings.compact_tools and "tools" in payload:
        base["tools"] = payload["tools"]
    if config.settings.dynamic_tools and payload.get("tools"):
        dynamic, registry = prepare_dynamic_tools(base)
        if registry:
            return dynamic, base, registry
    return base, base, {}


@app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str) -> Any:
    original_body = await request.body()
    body = original_body
    headers = forward_headers(request)
    outgoing_payload: dict[str, Any] = {}
    fallback_payload: dict[str, Any] = {}
    registry: dict[str, Any] = {}

    # Read back by the upstream logger to report how much each send saved.
    request.state.client_size = len(original_body)
    log_client_request(request, original_body)

    payload = await _json_payload(request, body)
    if payload is not None:
        outgoing_payload, fallback_payload, registry = _optimized_payloads(payload)
        body = encode_payload(outgoing_payload)
        headers["content-type"] = "application/json"

    async def send(content: bytes) -> UpstreamResponse:
        return await read_upstream(await send_upstream(request, path, headers, content))

    upstream: httpx.Response | None = None
    try:
        upstream = await send_upstream(request, path, headers, body)
        if not registry:
            if is_event_stream(upstream):
                return streaming_response(upstream)
            return (await read_upstream(upstream)).to_response()

        if not is_event_stream(upstream):
            first = await read_upstream(upstream)
        else:
            # Buffer only until it is clear whether the model wants the loader; anything
            # else has to keep streaming token by token.
            prefix, rest = await peek_stream(
                upstream, lambda buffered: loader_intent(buffered, _EVENT_STREAM) is not None
            )
            if loader_intent(prefix, _EVENT_STREAM) is False:
                return streaming_response(upstream, prefix, rest)
            first = await read_upstream(upstream, prefix, rest)

        resolved = await resolve_dynamic_tools(send, first, outgoing_payload, fallback_payload, registry)
        return resolved.to_response()
    except httpx.HTTPError as exc:
        if upstream is not None:
            await upstream.aclose()
        return JSONResponse({"error": "upstream_unavailable", "detail": str(exc)}, status_code=502)
