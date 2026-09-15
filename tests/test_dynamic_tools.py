from proxy.dynamic_tools import dynamic_tools_list, expand_tool_affinity, prepare_dynamic_tools

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "run_terminal", "description": "Run a command.", "parameters": {"type": "object"}}},
]


def test_expand_tool_affinity_groups():
    registry = {
        "read_file": {},
        "replace_string_in_file": {},
        "create_file": {},
        "run_in_terminal": {},
        "send_to_terminal": {},
    }
    # When requesting read_file, file-related tools are co-loaded, but terminal tools are not
    expanded = expand_tool_affinity(["read_file"], registry)
    assert "read_file" in expanded
    assert "replace_string_in_file" in expanded
    assert "create_file" in expanded
    assert "run_in_terminal" not in expanded


def test_preloads_related_affinity_tools_from_history():
    tools = [
        {"type": "function", "function": {"name": "read_file", "description": "Read.", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "replace_string_in_file", "description": "Edit.", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "run_in_terminal", "description": "Run.", "parameters": {"type": "object"}}},
    ]
    payload = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "..."},
        ],
        "tools": tools,
    }

    result, _ = prepare_dynamic_tools(payload)

    names = [tool["function"]["name"] for tool in result["tools"]]
    assert "get_tools" in names
    assert "read_file" in names
    assert "replace_string_in_file" in names
    assert "run_in_terminal" not in names


def test_sends_a_single_get_tools_loader_with_a_catalog():
    payload = {"messages": [{"role": "system", "content": "base system"}, {"role": "user", "content": "inspect"}], "tools": TOOLS}

    result, registry = prepare_dynamic_tools(payload)

    assert set(registry) == {"read_file", "run_terminal"}
    assert [tool["function"]["name"] for tool in result["tools"]] == ["get_tools"]
    description = result["tools"][0]["function"]["description"]
    assert "Available tools:" in description
    assert "read_file: Read a file." in description
    assert "run_terminal: Run a command." in description
    assert "DYNAMIC TOOLS:" in result["messages"][0]["content"]
    assert "base system" in result["messages"][0]["content"]
    assert result["messages"][1] == {"role": "user", "content": "inspect"}


def test_prepends_system_instruction_when_no_system_message_exists():
    payload = {"messages": [{"role": "user", "content": "inspect"}], "tools": TOOLS}

    result, registry = prepare_dynamic_tools(payload)

    assert result["messages"][0]["role"] == "system"
    assert "DYNAMIC TOOLS:" in result["messages"][0]["content"]
    assert result["messages"][1] == {"role": "user", "content": "inspect"}


def test_constrains_requested_names_to_the_remaining_tools():
    registry = {"read_file": TOOLS[0], "run_terminal": TOOLS[1]}

    loader = dynamic_tools_list(registry, [])[0]
    names = loader["function"]["parameters"]["properties"]["tool_names"]

    assert names["items"] == {"type": "string", "enum": ["read_file", "run_terminal"]}
    assert dynamic_tools_list(registry, ["read_file"])[0]["function"]["parameters"]["properties"][
        "tool_names"
    ]["items"]["enum"] == ["run_terminal"]


def test_preloads_tools_the_model_already_used():
    payload = {
        "messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "read_file"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "..."},
        ],
        "tools": TOOLS,
    }

    result, _ = prepare_dynamic_tools(payload)

    assert [tool["function"]["name"] for tool in result["tools"]] == ["get_tools", "read_file"]
    assert "read_file:" not in result["tools"][0]["function"]["description"]


def test_keeps_unrecognized_tool_shapes_untouched():
    tools = [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}]
    payload = {"input": [{"role": "user", "content": "inspect"}], "tools": tools}

    result, registry = prepare_dynamic_tools(payload)

    assert registry == {}
    assert result["tools"] == tools


def test_drops_the_loader_once_everything_is_loaded():
    registry = {"read_file": TOOLS[0], "run_terminal": TOOLS[1]}

    assert dynamic_tools_list(registry, ["read_file", "run_terminal", "unknown"]) == TOOLS
