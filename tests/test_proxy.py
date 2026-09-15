import json

import httpx

from helpers import message_response, sse_response, tool, tool_call_response


async def test_optimizes_json_and_preserves_the_upstream_response(proxy_client):
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": "upstream", "ok": True})

    async with proxy_client(handler) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "opaque",
                "messages": [
                    {"role": "system", "content": "rules\n\nrules"},
                    {"role": "user", "content": "hi"},
                ],
            },
        )

    assert response.status_code == 201
    assert response.json() == {"id": "upstream", "ok": True}
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]


async def test_resolves_get_tools_internally(proxy_client):
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return tool_call_response("get_tools", '{"tool_names":["read_file","other"]}')
        return message_response("done")

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [tool("read_file", "Read"), tool("other", "Other")],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert len(requests) == 2
    assert [tool_["function"]["name"] for tool_ in requests[0]["tools"]] == ["get_tools"]
    assert [tool_["function"]["name"] for tool_ in requests[1]["tools"]] == ["read_file", "other"]


async def test_resolves_get_tools_with_affinity_co_loading(proxy_client):
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            # Model asks ONLY for read_file
            return tool_call_response("get_tools", '{"tool_names":["read_file"]}')
        return message_response("done")

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "edit it"}],
                "tools": [
                    tool("read_file", "Read"),
                    tool("replace_string_in_file", "Edit"),
                    tool("run_in_terminal", "Run"),
                ],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert len(requests) == 2
    # In round 2, read_file and its affinity partner replace_string_in_file are both loaded,
    # while run_in_terminal remains unloaded behind get_tools.
    tool_names = [tool_["function"]["name"] for tool_ in requests[1]["tools"]]
    assert "get_tools" in tool_names
    assert "read_file" in tool_names
    assert "replace_string_in_file" in tool_names
    assert "run_in_terminal" not in tool_names


async def test_falls_back_to_full_tools_when_the_model_loops_on_get_tools(proxy_client):
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if any(tool_["function"]["name"] == "get_tools" for tool_ in payload["tools"]):
            return tool_call_response("get_tools", '{"tool_names":["hallucinated"]}')
        return message_response("done")

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "read it"}], "tools": [tool("read_file", "Read")]},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "done"
    assert [tool_["function"]["name"] for tool_ in requests[-1]["tools"]] == ["read_file"]


async def test_resolves_get_tools_from_a_streamed_response(proxy_client):
    requests = []
    stream = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":'
        '{"name":"get_tools","arguments":"{\\"tool_names\\":"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"arguments":"[\\"read_file\\"]}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})
        return message_response("done")

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [tool("read_file", "Read"), tool("other", "Other")],
            },
        )

    assert response.status_code == 200
    assert len(requests) == 2
    assert [tool_["function"]["name"] for tool_ in requests[1]["tools"]] == ["get_tools", "read_file"]


async def test_resolves_get_tools_preceded_by_assistant_content_text(proxy_client):
    requests = []
    stream = (
        'data: {"choices":[{"delta":{"content":"I will help you. "}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","function":'
        '{"name":"get_tools","arguments":"{\\"tool_names\\": [\\"read_file\\"]}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, content=stream, headers={"content-type": "text/event-stream"})
        return message_response("done")

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "read it"}],
                "tools": [tool("read_file", "Read"), tool("other", "Other")],
            },
        )

    assert response.status_code == 200
    assert len(requests) == 2
    assert [tool_["function"]["name"] for tool_ in requests[1]["tools"]] == ["get_tools", "read_file"]
    # Check that assistant message content was preserved in the retry round
    assert requests[1]["messages"][-2]["content"] == "I will help you. "


async def test_streams_responses_through_when_dynamic_tools_are_off(proxy_client):
    stream = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(stream)

    async with proxy_client(handler, dynamic_tools=False) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}], "tools": [tool("read_file")]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == stream


async def test_keeps_streaming_a_plain_answer_while_dynamic_tools_are_on(proxy_client):
    stream = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(stream)

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}], "tools": [tool("read_file")]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == stream


async def test_falls_back_when_the_model_mixes_the_loader_with_a_real_call(proxy_client):
    requests = []
    mixed = {
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call-1", "index": 0, "function": {"name": "get_tools", "arguments": '{"tool_names":["read_file"]}'}},
                    {"id": "call-2", "index": 1, "function": {"name": "other", "arguments": "{}"}},
                ],
            },
        }],
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=mixed)
        return message_response("done")

    async with proxy_client(handler, dynamic_tools=True) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "go"}],
                "tools": [tool("read_file", "Read"), tool("other", "Other")],
            },
        )

    assert response.status_code == 200
    assert len(requests) == 2
    assert [tool_["function"]["name"] for tool_ in requests[1]["tools"]] == ["read_file", "other"]


async def test_returns_502_when_the_upstream_is_unreachable(proxy_client):
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with proxy_client(handler) as client:
        response = await client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 502
    assert response.json()["error"] == "upstream_unavailable"


async def test_forwards_non_json_bodies_unchanged(proxy_client):
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(200, content=b"ok")

    async with proxy_client(handler) as client:
        response = await client.post(
            "/v1/embeddings", content=b"raw bytes", headers={"content-type": "application/octet-stream"}
        )

    assert response.status_code == 200
    assert seen["body"] == b"raw bytes"


async def test_logs_both_directions_without_leaking_credentials(proxy_client, capsys):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with proxy_client(handler, log_requests=True) as client:
        await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi\n\n\nhi"}]},
            headers={"authorization": "Bearer super-secret"},
        )

    logged = capsys.readouterr().err

    assert "CLIENT REQUEST | POST /v1/chat/completions" in logged
    assert "UPSTREAM REQUEST | POST http://upstream.test/v1/chat/completions" in logged
    assert "UPSTREAM RESPONSE | 200 buffered" in logged
    assert "saved=" in logged
    assert "super-secret" not in logged


async def test_logs_a_streamed_response_once_it_is_drained(proxy_client, capsys):
    stream = 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    async def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(stream)

    async with proxy_client(handler, log_requests=True) as client:
        response = await client.post("/v1/chat/completions", json={"stream": True, "messages": []})

    assert response.text == stream
    logged = capsys.readouterr().err
    assert "UPSTREAM RESPONSE | 200 streamed" in logged
    assert '"content":"hi"' in logged
