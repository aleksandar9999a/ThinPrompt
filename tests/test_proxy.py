import json

import httpx
import pytest

from proxy.app import app
from proxy.config import Settings


@pytest.mark.asyncio
async def test_proxy_optimizes_json_and_preserves_response(monkeypatch):
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(201, json={"id": "upstream", "ok": True})

    monkeypatch.setattr(
        "proxy.app.settings",
        Settings(upstream_url="http://upstream.test"),
    )
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "opaque",
                    "messages": [
                        {"role": "user", "content": "same"},
                        {"role": "user", "content": "same"},
                    ],
                },
            )

        assert response.status_code == 201
        assert response.json() == {"id": "upstream", "ok": True}
        assert httpx.Response(200, content=seen["body"]).json()["messages"] == [
            {"role": "user", "content": "same"}
        ]
    finally:
        await app.state.client.aclose()


@pytest.mark.asyncio
async def test_proxy_resolves_get_tools_internally(monkeypatch):
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "tool_calls": [{
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "get_tools",
                                    "arguments": '{"tool_names":["read_file","other"]}',
                                },
                            }],
                        },
                    }],
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

    monkeypatch.setattr("proxy.app.settings", Settings(upstream_url="http://upstream.test", dynamic_tools=True))
    app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "read it"}],
                    "tools": [
                        {"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object"}}},
                        {"type": "function", "function": {"name": "other", "description": "Other", "parameters": {"type": "object"}}},
                    ],
                },
            )

        assert response.status_code == 200
        assert len(requests) == 2
        assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["get_tools"]
        assert [tool["function"]["name"] for tool in requests[1]["tools"]] == ["read_file", "other"]
    finally:
        await app.state.client.aclose()