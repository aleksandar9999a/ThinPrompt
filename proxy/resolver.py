"""Answering the internal get_tools calls without ever exposing them to the client."""

from collections.abc import Awaitable, Callable
from typing import Any
import json

from .dynamic_tools import dynamic_tools_list, expand_tool_affinity
from .responses import extract_tool_call
from .upstream import UpstreamResponse, encode_payload

MAX_TOOL_RESOLUTION_ROUNDS = 3

SendPayload = Callable[[bytes], Awaitable[UpstreamResponse]]


def _loader_result_message(call_id: str, loaded: list[str], unknown: list[str]) -> dict[str, Any]:
    content: dict[str, Any] = {"loaded": loaded, "unknown": unknown}
    if loaded:
        content["message"] = (
            f"Tool schemas for {loaded} are now loaded and available. "
            "Please call the required tool directly now."
        )
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(content),
    }


async def resolve_dynamic_tools(
    send: SendPayload,
    first: UpstreamResponse,
    payload: dict[str, Any],
    fallback_payload: dict[str, Any],
    registry: dict[str, Any],
) -> UpstreamResponse:
    """Load requested schemas locally and retry until the model answers with real tools."""
    current = first
    loaded: list[str] = []
    for _ in range(MAX_TOOL_RESOLUTION_ROUNDS):
        call = extract_tool_call(current.body, current.content_type)
        if call is None:
            return current
        if call.has_other_calls:
            break

        expanded_names = expand_tool_affinity(call.requested_names, registry)
        resolved = [name for name in expanded_names if name in registry and name not in loaded]
        if not resolved:
            break
        loaded.extend(resolved)

        messages = payload.setdefault("messages", [])
        messages.append(call.assistant_message)
        messages.append(_loader_result_message(
            call.assistant_message["tool_calls"][0]["id"],
            resolved,
            [name for name in call.requested_names if name not in registry],
        ))
        payload["tools"] = dynamic_tools_list(registry, loaded)
        current = await send(encode_payload(payload))

    if extract_tool_call(current.body, current.content_type) is None:
        return current

    # The model keeps asking for schemas it cannot use, or mixed the loader with real
    # calls; retry once with every tool so it can make progress instead of looping.
    return await send(encode_payload(fallback_payload))
