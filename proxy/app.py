from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import json
import sys

from .config import settings
from .optimizer import optimize_request, prepare_dynamic_tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=None)
    yield
    await app.state.client.aclose()


app = FastAPI(title="LLM Request Proxy", lifespan=lifespan)


def upstream_url(path: str, query: str) -> str:
    base = settings.upstream_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}{f'?{query}' if query else ''}"


def forward_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }
    if settings.upstream_api_key:
        headers["authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def log_request(request: Request, original: bytes, optimized: bytes) -> None:
    if not settings.log_requests:
        return
    safe_headers = {
        key: ("[REDACTED]" if key.lower() == "authorization" else value)
        for key, value in request.headers.items()
    }
    try:
        original_text = json.dumps(json.loads(original), ensure_ascii=False, indent=2)
    except (UnicodeDecodeError, json.JSONDecodeError):
        original_text = original.decode("utf-8", errors="replace")
    try:
        optimized_text = json.dumps(json.loads(optimized), ensure_ascii=False, indent=2)
    except (UnicodeDecodeError, json.JSONDecodeError):
        optimized_text = optimized.decode("utf-8", errors="replace")
    print(f"\n=== PROXY REQUEST {request.method} {request.url.path} ===", file=sys.stderr)
    print(f"Headers: {safe_headers}", file=sys.stderr)
    print(f"Bytes: original={len(original)} optimized={len(optimized)} saved={len(original) - len(optimized)}", file=sys.stderr)
    print("--- ORIGINAL REQUEST BODY ---", file=sys.stderr)
    print(original_text, file=sys.stderr)
    print("--- OPTIMIZED REQUEST BODY ---", file=sys.stderr)
    print(optimized_text, file=sys.stderr)
    print("=== END PROXY REQUEST ===\n", file=sys.stderr)


async def stream_response(response: httpx.Response) -> AsyncIterator[bytes]:
    async for chunk in response.aiter_raw():
        yield chunk
    await response.aclose()


def extract_tool_call(response_body: bytes, content_type: str) -> tuple[dict[str, Any], list[str]] | None:
    """Extract a get_tools call from JSON or buffered SSE without exposing it to VS Code."""
    chunks = []
    if "text/event-stream" in content_type:
        for line in response_body.decode("utf-8", errors="replace").splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                try:
                    chunks.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    continue
    else:
        try:
            chunks.append(json.loads(response_body))
        except json.JSONDecodeError:
            return None

    calls_by_id: dict[str, dict[str, str]] = {}
    call_order: list[str] = []
    assistant_message: dict[str, Any] = {"role": "assistant", "tool_calls": []}
    for chunk in chunks:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        message = choice.get("message", {})
        delta = choice.get("delta", {})
        calls = message.get("tool_calls", []) or delta.get("tool_calls", [])
        for index, call in enumerate(calls):
            function = call.get("function", {})
            call_id = call.get("id") or (call_order[index] if index < len(call_order) else f"dynamic-tools-call-{index}")
            if call_id not in calls_by_id:
                call_order.append(call_id)
            current = calls_by_id.setdefault(call_id, {"name": "", "arguments": ""})
            current["name"] = function.get("name", current["name"])
            current["arguments"] += function.get("arguments", "")
    get_tools_calls = [call for call in calls_by_id.values() if call["name"] == "get_tools"]
    if len(get_tools_calls) != 1:
        return None
    try:
        parsed_arguments = json.loads(get_tools_calls[0]["arguments"])
    except json.JSONDecodeError:
        return None
    requested_names = parsed_arguments.get("tool_names")
    if not isinstance(requested_names, list) or not all(isinstance(name, str) for name in requested_names):
        return None
    call_id = next(call_id for call_id in call_order if calls_by_id[call_id] is get_tools_calls[0])
    assistant_message["tool_calls"] = [{
        "id": call_id,
        "type": "function",
        "function": {"name": "get_tools", "arguments": json.dumps({"tool_names": requested_names})},
    }]
    return assistant_message, requested_names


async def send_upstream(request: Request, path: str, headers: dict[str, str], body: bytes) -> httpx.Response:
    return await request.app.state.client.send(
        request.app.state.client.build_request(
            request.method,
            upstream_url(f"/{path}", request.url.query),
            headers=headers,
            content=body,
        ),
        stream=True,
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str) -> Any:
    original_body = await request.body()
    body = original_body
    headers = forward_headers(request)
    dynamic_registry = {}

    payload = None
    if request.method in {"POST", "PUT", "PATCH"} and body:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = await request.json()
                if isinstance(payload, dict):
                    optimized_payload = optimize_request(payload)
                    if not settings.compact_tools:
                        optimized_payload["tools"] = payload.get("tools")
                    if settings.dynamic_tools and payload.get("tools"):
                        optimized_payload, dynamic_registry = prepare_dynamic_tools(optimized_payload)
                    body = json.dumps(
                        optimized_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                    headers["content-type"] = "application/json"
            except ValueError:
                pass

    log_request(request, original_body, body)

    try:
        upstream = await send_upstream(request, path, headers, body)
        if dynamic_registry:
            response_body = await upstream.aread()
            response_content_type = upstream.headers.get("content-type", "")
            response_status = upstream.status_code
            response_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}
            }
            await upstream.aclose()
            for _ in range(3):
                extracted = extract_tool_call(response_body, response_content_type)
                if not extracted:
                    break
                assistant_message, requested_names = extracted
                resolved_names = [name for name in requested_names if name in dynamic_registry]
                if not resolved_names:
                    break
                payload_messages = optimized_payload.setdefault("messages", [])
                payload_messages.append(assistant_message)
                payload_messages.append({
                    "role": "tool",
                    "tool_call_id": assistant_message["tool_calls"][0]["id"],
                    "content": json.dumps({"tool_names": resolved_names, "loaded": True}),
                })
                optimized_payload["tools"] = [dynamic_registry[name] for name in resolved_names]
                body = json.dumps(optimized_payload, ensure_ascii=False, separators=(",", ":")).encode()
                upstream = await send_upstream(request, path, headers, body)
                response_body = await upstream.aread()
                response_content_type = upstream.headers.get("content-type", "")
                response_status = upstream.status_code
                response_headers = {
                    key: value
                    for key, value in upstream.headers.items()
                    if key.lower() not in {"content-length", "transfer-encoding", "connection"}
                }
                await upstream.aclose()
            return Response(
                content=response_body,
                status_code=response_status,
                headers=response_headers,
                media_type=response_content_type or None,
            )
    except httpx.HTTPError as exc:
        return JSONResponse({"error": "upstream_unavailable", "detail": str(exc)}, status_code=502)

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "connection"}
    }
    if "text/event-stream" in upstream.headers.get("content-type", ""):
        return StreamingResponse(
            stream_response(upstream),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )

    content = await upstream.aread()
    await upstream.aclose()
    return Response(
        content=content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )