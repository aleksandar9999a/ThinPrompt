"""Compaction of tool definitions that preserves the tool contract."""

from typing import Any

from .text import compact_description

# Tool descriptions carry usage constraints ("do NOT use for ..."), so they get a much
# larger budget than parameter descriptions.
FUNCTION_DESCRIPTION_LIMIT = 600
PARAMETER_DESCRIPTION_LIMIT = 180

_DROPPED_SCHEMA_KEYS = {"$comment", "examples", "title"}

# Keywords whose values are maps of caller-chosen names to schemas. Their keys belong to
# the tool contract, so a parameter called "title" must never be mistaken for a keyword.
_SCHEMA_MAP_KEYS = {"properties", "patternProperties", "dependentSchemas", "$defs", "definitions"}


def tool_name(tool: Any) -> str | None:
    """Extract the name of an OpenAI-style function tool."""
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    return function.get("name") if isinstance(function, dict) else None


def compact_schema(value: Any, description_limit: int = PARAMETER_DESCRIPTION_LIMIT) -> Any:
    """Recursively compact a schema by removing verbose non-semantic text."""
    if isinstance(value, list):
        return [compact_schema(item, description_limit) for item in value]
    if not isinstance(value, dict):
        return value

    compacted: dict[Any, Any] = {}
    for key, item in value.items():
        if key in _DROPPED_SCHEMA_KEYS:
            continue
        if key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
            compacted[key] = {name: compact_schema(sub, description_limit) for name, sub in item.items()}
        elif key == "description" and isinstance(item, str):
            compacted[key] = compact_description(item, description_limit)
        else:
            compacted[key] = compact_schema(item, description_limit)
    return compacted


def _compact_tool(tool: Any) -> Any:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return compact_schema(tool)

    compacted_function: dict[str, Any] = {}
    for key, item in function.items():
        if key == "description" and isinstance(item, str):
            compacted_function[key] = compact_description(item, FUNCTION_DESCRIPTION_LIMIT)
        else:
            compacted_function[key] = compact_schema(item)
    return {
        **{key: item for key, item in tool.items() if key != "function"},
        "function": compacted_function,
    }


def compact_tools(tools: Any) -> Any:
    """Keep tool contracts while removing verbose, non-semantic schema text."""
    if not isinstance(tools, list):
        return tools
    return [_compact_tool(tool) for tool in tools]
