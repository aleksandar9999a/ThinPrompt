from proxy.messages import optimize_request


def test_removes_repeated_instructions_and_repeated_blocks():
    payload = {
        "model": "opaque-model-name",
        "messages": [
            {"role": "system", "content": "rules\n\nrules\n\nmore rules"},
            {"role": "system", "content": "rules\n\nrules\n\nmore rules"},
            {"role": "user", "content": "question"},
        ],
        "temperature": 0,
    }

    result = optimize_request(payload)

    assert result["model"] == payload["model"]
    assert result["temperature"] == 0
    assert result["messages"] == [
        {"role": "system", "content": "rules\n\nmore rules"},
        {"role": "user", "content": "question"},
    ]


def test_keeps_repeated_blocks_inside_user_content():
    payload = {
        "messages": [
            {"role": "user", "content": "def f():\n    pass\n\ndef f():\n    pass"},
            {"role": "user", "content": "go"},
        ]
    }

    result = optimize_request(payload)

    assert result["messages"] == payload["messages"]


def test_keeps_repeated_user_turns_and_never_drops_the_last_message():
    payload = {
        "messages": [
            {"role": "user", "content": "run the tests"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "run the tests"},
        ]
    }

    result = optimize_request(payload)

    assert result["messages"] == payload["messages"]


def test_optimizes_input_and_normalizes_text_content_blocks():
    payload = {
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "  hello  \n\n\n world  "}]},
            {"role": "user", "content": [{"type": "input_text", "text": "hello\n\nworld"}]},
        ],
        "max_output_tokens": 100,
    }

    result = optimize_request(payload)

    assert result["max_output_tokens"] == 100
    assert result["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello\n\n world"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "hello\n\nworld"}]},
    ]


def test_preserves_tool_call_messages_with_empty_content():
    payload = {
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "first"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
            {"role": "tool", "tool_call_id": "2", "content": "second"},
        ]
    }

    result = optimize_request(payload)

    assert result["messages"] == payload["messages"]


def test_compacts_tools_even_without_a_message_list():
    payload = {
        "prompt": "hi",
        "tools": [{"type": "function", "function": {"name": "a", "parameters": {"type": "object", "title": "drop me"}}}],
    }

    result = optimize_request(payload)

    assert result["tools"] == [{"type": "function", "function": {"name": "a", "parameters": {"type": "object"}}}]
