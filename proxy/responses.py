"""Reading tool calls back out of JSON or buffered SSE upstream responses."""

import json
from typing import Any, NamedTuple

from .dynamic_tools import DYNAMIC_TOOL_NAME

_SSE_DATA_PREFIX = "data: "


class ToolCall(NamedTuple):
    id: str
    name: str
    arguments: str


class LoaderCall(NamedTuple):
    """A get_tools call the proxy has to answer itself."""

    assistant_message: dict[str, Any]
    requested_names: list[str]
    # Real calls emitted next to the loader cannot be answered locally, and dropping
    # them would throw away the model's progress.
    has_other_calls: bool


def response_chunks(body: bytes, content_type: str) -> list[dict[str, Any]]:
    """Parse an upstream body into completion chunks, streamed or not."""
    if "text/event-stream" not in content_type:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return []
        return [parsed] if isinstance(parsed, dict) else []

    chunks = []
    for line in body.decode("utf-8", errors="replace").splitlines():
        if not line.startswith(_SSE_DATA_PREFIX):
            continue
        data = line[len(_SSE_DATA_PREFIX):].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            chunks.append(parsed)
    return chunks


def _delta(choice: Any) -> dict[str, Any]:
    if not isinstance(choice, dict):
        return {}
    source = choice.get("message") or choice.get("delta") or {}
    return source if isinstance(source, dict) else {}


def _call_name(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    function = call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    return name or ""


def merge_tool_calls(chunks: list[dict[str, Any]]) -> list[ToolCall]:
    """Reassemble streamed tool calls, correlating fragments by their delta index."""
    merged: dict[Any, dict[str, str]] = {}
    for chunk in chunks:
        for choice_position, choice in enumerate(chunk.get("choices") or []):
            source = _delta(choice)
            if not source:
                continue
            choice_index = choice.get("index", choice_position)
            for position, call in enumerate(source.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                # The id usually arrives only with the first fragment, so the index is
                # the one key that stays stable across chunks.
                key = (choice_index, call.get("index", position))
                entry = merged.setdefault(key, {"id": "", "name": "", "arguments": ""})
                entry["id"] = entry["id"] or call.get("id") or ""
                entry["name"] = _call_name(call) or entry["name"]
                function = call.get("function") or {}
                entry["arguments"] += function.get("arguments") or ""
    return [
        ToolCall(entry["id"] or f"call_{DYNAMIC_TOOL_NAME}_{position}", entry["name"], entry["arguments"])
        for position, entry in enumerate(merged.values())
    ]


def extract_assistant_content(chunks: list[dict[str, Any]]) -> str | None:
    """Extract assistant content text from parsed chunks if present."""
    content_parts = []
    for chunk in chunks:
        for choice in chunk.get("choices") or []:
            source = _delta(choice)
            content = source.get("content")
            if isinstance(content, str):
                content_parts.append(content)
    if not content_parts:
        return None
    full = "".join(content_parts)
    return full if full else None


def loader_intent(body: bytes, content_type: str) -> bool | None:
    """Whether the response calls get_tools, or None while the answer is still open."""
    for chunk in response_chunks(body, content_type):
        for choice in chunk.get("choices") or []:
            source = _delta(choice)
            for call in source.get("tool_calls") or []:
                name = _call_name(call)
                if name == DYNAMIC_TOOL_NAME:
                    return True
                if name:
                    return False
            if isinstance(choice, dict) and choice.get("finish_reason"):
                return False
    if "text/event-stream" in content_type and b"[DONE]" in body:
        return False
    return None


def extract_tool_call(body: bytes, content_type: str) -> LoaderCall | None:
    """Return the assistant message and requested names of a get_tools call, if any."""
    chunks = response_chunks(body, content_type)
    calls = merge_tool_calls(chunks)
    loader_calls = [call for call in calls if call.name == DYNAMIC_TOOL_NAME]
    if not loader_calls:
        return None

    requested_names: list[str] = []
    for call in loader_calls:
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            continue
        names = arguments.get("tool_names") if isinstance(arguments, dict) else None
        if isinstance(names, str):
            names = [names]
        if isinstance(names, list):
            requested_names.extend(name for name in names if isinstance(name, str))

    requested_names = list(dict.fromkeys(requested_names))
    content = extract_assistant_content(chunks)
    assistant_message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [{
            "id": loader_calls[0].id,
            "type": "function",
            "function": {
                "name": DYNAMIC_TOOL_NAME,
                "arguments": json.dumps({"tool_names": requested_names}),
            },
        }],
    }
    return LoaderCall(assistant_message, requested_names, len(loader_calls) != len(calls))
