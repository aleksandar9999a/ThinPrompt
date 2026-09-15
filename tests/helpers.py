"""Shared builders for upstream payloads used by the proxy tests."""

from collections.abc import AsyncIterator

import httpx


class _UnreadStream(httpx.AsyncByteStream):
    """Mock body that still has to be streamed, unlike a Response built from bytes."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._data

    async def aclose(self) -> None:
        return None


def sse_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_UnreadStream(text.encode()),
    )


def tool(name: str, description: str = "") -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object"}},
    }


def message_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})


def tool_call_response(name: str, arguments: str, call_id: str = "call-1") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }],
                },
            }],
        },
    )
