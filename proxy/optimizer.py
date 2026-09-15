from collections.abc import Mapping
from copy import deepcopy
from typing import Any
import re

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

def _normalize_text(text: str) -> str:
    """Normalize text by removing excess blank lines and trailing whitespace."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()

def _canonical_text(text: str) -> str:
    """Convert text to canonical form by stripping each line."""
    return "\n".join(line.strip() for line in _normalize_text(text).split("\n"))

def _compact_description(value: str, limit: int) -> str:
    """Compact a description to a maximum length, preserving the first sentence if possible."""
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    first_sentence = _SENTENCE_END.split(text, maxsplit=1)[0]
    if len(first_sentence) <= limit:
        return first_sentence
    return f"{text[:limit - 3].rstrip()}..."

def _compact_schema(value: Any, description_limit: int = 180) -> Any:
    """Recursively compact schema by removing verbose non-semantic text."""
    if isinstance(value, list):
        return [_compact_schema(item, description_limit) for item in value]
    if not isinstance(value, dict):
        return value

    compacted = {}
    for key, item in value.items():
        if key in {"$comment", "examples", "enumDescriptions", "title"}:
            continue
        if key == "description" and isinstance(item, str):
            compacted[key] = _compact_description(item, description_limit)
        else:
            compacted[key] = _compact_schema(item, description_limit)
    return compacted

def compact_tools(tools: Any) -> Any:
    """Keep tool contracts while removing verbose, non-semantic schema text."""
    if not isinstance(tools, list):
        return tools
    return [_compact_schema(tool) for tool in tools]

def _tool_name(tool: Any) -> str | None:
    """Extract tool name from a tool definition."""
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    return function.get("name") if isinstance(function, dict) else None

def dynamic_tool_definition() -> dict[str, Any]:
    """Create a dynamic tool definition for loading tool schemas."""
    return {
        "type": "function",
        "function": {
            "name": "get_tools",
            "description": "Load the full schemas for one or more available tools by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exact names of the tools to load.",
                    }
                },
                "required": ["tool_names"],
                "additionalProperties": False,
            },
        },
    }

def prepare_dynamic_tools(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace the full tool list with a catalog and schemas requested in history."""
    optimized = deepcopy(dict(payload))
    original_tools = optimized.get("tools")
    if not isinstance(original_tools, list):
        return optimized, {}

    registry = {
        name: tool
        for tool in original_tools
        if (name := _tool_name(tool)) and name != "get_tools"
    }
    requested = set()
    messages = optimized.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls", []):
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = function.get("name") if isinstance(function, dict) else None
                if name in registry:
                    requested.add(name)

    catalog = [
        f"{name}: {_compact_description(tool.get('function', {}).get('description', ''), 80)}"
        for name, tool in registry.items()
        if isinstance(tool.get("function"), dict)
    ]
    catalog_text = "Available tools:\n" + "\n".join(catalog)
    optimized["tools"] = [dynamic_tool_definition()]
    if catalog_text:
        optimized.setdefault("messages", []).insert(0, {"role": "system", "content": catalog_text})
    optimized["tools"].extend(registry[name] for name in requested)
    return optimized, registry

def _content_key(content: Any) -> str | None:
    """Extract canonical text content from various content formats."""
    if isinstance(content, str):
        return _canonical_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(_canonical_text(text))
        return "\n".join(parts) or None
    return None

def optimize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply deterministic, model-agnostic compression to request text."""
    optimized = deepcopy(dict(payload))
    items = optimized.get("messages")
    field_name = "messages"
    if not isinstance(items, list):
        items = optimized.get("input")
        field_name = "input"
    if not isinstance(items, list):
        return optimized

    seen_messages: set[tuple[str, str]] = set()
    seen_blocks: set[str] = set()
    result = []
    for message in items:
        if not isinstance(message, dict):
            result.append(message)
            continue

        role = message.get("role")
        content = message.get("content")
        if role == "tool" or message.get("tool_calls"):
            result.append(message)
            continue
        message_key = (str(role), _content_key(content) or repr(content))
        if message_key in seen_messages:
            continue

        if isinstance(content, str) and content.strip():
            normalized_content = _normalize_text(content)
            blocks = [block.strip() for block in normalized_content.split("\n\n") if block.strip()]
            unique_blocks = []
            for block in blocks:
                block_key = _canonical_text(block)
                if block_key not in seen_blocks:
                    seen_blocks.add(block_key)
                    unique_blocks.append(block)
            if not unique_blocks:
                continue
            message = {**message, "content": "\n\n".join(unique_blocks)}
        elif isinstance(content, list):
            normalized_content = []
            for block in content:
                if not isinstance(block, dict):
                    normalized_content.append(block)
                    continue
                block = {**block}
                if isinstance(block.get("text"), str):
                    block["text"] = _normalize_text(block["text"])
                normalized_content.append(block)
            message = {**message, "content": normalized_content}

        seen_messages.add(message_key)
        result.append(message)

    optimized[field_name] = result
    if "tools" in optimized:
        optimized["tools"] = compact_tools(optimized["tools"])
    return optimized