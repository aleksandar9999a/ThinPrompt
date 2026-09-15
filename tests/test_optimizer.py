from proxy.optimizer import compact_tools, optimize_request, prepare_dynamic_tools


def test_removes_duplicate_messages_and_repeated_blocks():
    payload = {
        "model": "opaque-model-name",
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "file.py\n\nfile.py\n\nquestion"},
            {"role": "user", "content": "question"},
        ],
        "temperature": 0,
    }

    result = optimize_request(payload)

    assert result["model"] == payload["model"]
    assert result["temperature"] == 0
    assert result["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "file.py\n\nquestion"},
    ]


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
    ]


def test_compacts_tool_descriptions_without_changing_contract():
    tools = [{
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a file. " + "This is verbose implementation guidance. " * 30,
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "The absolute path to the file to create.",
                    }
                },
                "required": ["filePath"],
                "$comment": "Non-contractual guidance",
            },
        },
    }]

    result = compact_tools(tools)

    assert len(str(result)) < len(str(tools))
    assert result[0]["function"]["name"] == "create_file"
    assert result[0]["function"]["parameters"]["required"] == ["filePath"]
    assert "$comment" not in result[0]["function"]["parameters"]


def test_dynamic_tools_sends_catalog_and_requested_schema_only():
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "run_terminal", "description": "Run a command.", "parameters": {"type": "object"}}},
    ]
    payload = {"messages": [{"role": "user", "content": "inspect"}], "tools": tools}

    result, registry = prepare_dynamic_tools(payload)

    assert set(registry) == {"read_file", "run_terminal"}
    assert [tool["function"]["name"] for tool in result["tools"]] == ["get_tools"]
    assert "read_file: Read a file." in result["messages"][0]["content"]


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