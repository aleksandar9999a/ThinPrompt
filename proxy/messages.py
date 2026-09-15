"""Conversation-level compression that never restructures the agent loop."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .text import canonical_text, dedupe_blocks, normalize_text
from .tool_schemas import compact_tools

# Dropping a message from history makes the model repeat its previous turn, so only
# instruction roles are deduplicated, and never the final message.
_DEDUPABLE_ROLES = {"system", "developer"}


def _content_key(content: Any) -> str | None:
    if isinstance(content, str):
        return canonical_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(canonical_text(text))
        return "\n".join(parts) or None
    return None


def _message_key(message: Mapping[str, Any]) -> tuple[str, str]:
    content = message.get("content")
    return str(message.get("role")), _content_key(content) or repr(content)


def _normalize_content_blocks(content: list[Any]) -> list[Any]:
    normalized = []
    for block in content:
        if not isinstance(block, dict):
            normalized.append(block)
            continue
        block = {**block}
        if isinstance(block.get("text"), str):
            block["text"] = normalize_text(block["text"])
        normalized.append(block)
    return normalized


def optimize_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply deterministic, model-agnostic compression to request text."""
    optimized = deepcopy(dict(payload))
    items = optimized.get("messages")
    field_name = "messages"
    if not isinstance(items, list):
        items = optimized.get("input")
        field_name = "input"
    if not isinstance(items, list):
        if "tools" in optimized:
            optimized["tools"] = compact_tools(optimized["tools"])
        return optimized

    seen_instructions: set[tuple[str, str]] = set()
    result: list[Any] = []
    last_index = len(items) - 1
    for index, message in enumerate(items):
        if not isinstance(message, dict):
            result.append(message)
            continue

        role = message.get("role")
        content = message.get("content")
        # Tool traffic carries the agent loop state and must survive untouched.
        if role == "tool" or message.get("tool_calls"):
            result.append(message)
            continue

        dedupable = role in _DEDUPABLE_ROLES
        if dedupable and index != last_index:
            key = _message_key(message)
            previous = result[-1] if result and isinstance(result[-1], dict) else None
            if key in seen_instructions or (previous is not None and _message_key(previous) == key):
                continue
            seen_instructions.add(key)

        if isinstance(content, str) and content.strip():
            # A repeated block inside user content is usually pasted source, not
            # boilerplate, so only instructions are deduplicated.
            message = {**message, "content": dedupe_blocks(content) if dedupable else normalize_text(content)}
        elif isinstance(content, list):
            message = {**message, "content": _normalize_content_blocks(content)}

        result.append(message)

    optimized[field_name] = result
    if "tools" in optimized:
        optimized["tools"] = compact_tools(optimized["tools"])
    return optimized
