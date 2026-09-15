from proxy.tool_schemas import compact_tools


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


def test_keeps_usage_constraints_beyond_the_first_sentence():
    tools = [{
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": "Run a command. Do NOT use this for editing files.",
            "parameters": {"type": "object"},
        },
    }]

    result = compact_tools(tools)

    assert "Do NOT use this for editing files." in result[0]["function"]["description"]


def test_keeps_enum_descriptions_and_non_list_input():
    tools = [{
        "type": "function",
        "function": {
            "name": "pick",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["a", "b"], "enumDescriptions": ["first", "second"]},
                },
            },
        },
    }]

    result = compact_tools(tools)

    assert result[0]["function"]["parameters"]["properties"]["mode"]["enumDescriptions"] == ["first", "second"]
    assert compact_tools("not-a-list") == "not-a-list"


def test_keeps_properties_named_like_schema_keywords():
    tools = [{
        "type": "function",
        "function": {
            "name": "create_issue",
            "parameters": {
                "type": "object",
                "title": "drop me",
                "properties": {
                    "title": {"type": "string", "description": "Issue title.", "title": "drop me too"},
                    "examples": {"type": "array"},
                },
                "required": ["title"],
            },
        },
    }]

    parameters = compact_tools(tools)[0]["function"]["parameters"]

    assert "title" not in parameters
    assert set(parameters["properties"]) == {"title", "examples"}
    assert parameters["properties"]["title"] == {"type": "string", "description": "Issue title."}
    assert parameters["required"] == ["title"]
