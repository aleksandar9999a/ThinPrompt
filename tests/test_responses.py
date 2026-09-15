import json

from proxy.responses import extract_tool_call, loader_intent


def _json_body(tool_calls):
    return json.dumps({"choices": [{"message": {"role": "assistant", "tool_calls": tool_calls}}]}).encode()


def test_returns_none_without_a_loader_call():
    body = _json_body([{"id": "1", "function": {"name": "read_file", "arguments": "{}"}}])

    assert extract_tool_call(body, "application/json") is None
    assert extract_tool_call(b"not json", "application/json") is None


def test_extracts_requested_names_from_a_json_response():
    body = _json_body([{"id": "call-1", "function": {"name": "get_tools", "arguments": '{"tool_names":["a","b","a"]}'}}])

    call = extract_tool_call(body, "application/json")

    assert call.requested_names == ["a", "b"]
    assert call.assistant_message["tool_calls"][0]["id"] == "call-1"
    assert call.has_other_calls is False


def test_reports_real_tool_calls_emitted_next_to_the_loader():
    body = _json_body([
        {"id": "call-1", "index": 0, "function": {"name": "get_tools", "arguments": '{"tool_names":["a"]}'}},
        {"id": "call-2", "index": 1, "function": {"name": "read_file", "arguments": "{}"}},
    ])

    assert extract_tool_call(body, "application/json").has_other_calls is True


def test_merges_streamed_fragments_of_parallel_tool_calls():
    stream = (
        'data: {"choices":[{"delta":{"tool_calls":['
        '{"index":0,"id":"a","function":{"name":"read_file","arguments":"{\\"p\\":"}},'
        '{"index":1,"id":"b","function":{"name":"get_tools","arguments":"{\\"tool_names\\":"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"[\\"x\\"]}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    call = extract_tool_call(stream, "text/event-stream")

    assert call.requested_names == ["x"]
    assert call.assistant_message["tool_calls"][0]["id"] == "b"


def test_keeps_fragments_together_when_the_id_arrives_late():
    stream = (
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
        '{"name":"get_tools","arguments":"{\\"tool_names\\":"}}]}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"late","function":'
        '{"arguments":"[\\"x\\"]}"}}]}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    call = extract_tool_call(stream, "text/event-stream")

    assert call.requested_names == ["x"]
    assert call.assistant_message["tool_calls"][0]["id"] == "late"


def test_reports_an_empty_request_for_unparsable_arguments():
    body = _json_body([{"id": "call-1", "function": {"name": "get_tools", "arguments": "{broken"}}])

    assert extract_tool_call(body, "application/json").requested_names == []


def test_loader_intent_is_open_until_the_model_commits():
    assert loader_intent(b"data: {\"choices\":[{\"delta\":{}}]}", "text/event-stream") is None
    assert loader_intent(
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read_file"}}]}}]}',
        "text/event-stream",
    ) is False
    assert loader_intent(
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_tools"}}]}}]}',
        "text/event-stream",
    ) is True
    # If the model emits content text before committing to tool calls or finishing, it's still open
    assert loader_intent(b'data: {"choices":[{"delta":{"content":"hi"}}]}', "text/event-stream") is None
    # Once finish_reason is provided or stream ends with [DONE], it commits to False
    assert loader_intent(b'data: {"choices":[{"finish_reason":"stop","delta":{"content":"hi"}}]}', "text/event-stream") is False
    assert loader_intent(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]', "text/event-stream") is False


def test_loader_intent_detects_get_tools_after_introductory_content():
    stream = (
        'data: {"choices":[{"delta":{"content":"I will help you. "}}]}\n\n'
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"get_tools"}}]}}]}\n\n'
    ).encode()
    assert loader_intent(stream, "text/event-stream") is True
