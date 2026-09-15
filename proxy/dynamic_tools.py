"""The internal get_tools loader that keeps tool schemas out of the prompt."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from .text import compact_description
from .tool_schemas import tool_name

DYNAMIC_TOOL_NAME = "get_tools"
CATALOG_DESCRIPTION_LIMIT = 160

DYNAMIC_TOOL_INSTRUCTION = (
    "DYNAMIC TOOLS: Workspace tools are loaded on-demand. "
    "To use any tool (such as read_file, replace_string_in_file, run_in_terminal, etc.), "
    "you MUST first call 'get_tools' with the required tool names in 'tool_names'. "
    "Never state that you lack file access, editing capabilities, or tools without first calling 'get_tools'."
)

# Explicit affinity map: tools frequently used together in developer workflows.
# When any tool in a group is requested or previously used, all available tools in that group are co-loaded.
TOOL_AFFINITY_GROUPS: tuple[frozenset[str], ...] = (
    # File reading, editing, searching and directory navigation
    frozenset({
        "read_file",
        "replace_string_in_file",
        "insert_edit_into_file",
        "create_file",
        "create_directory",
        "list_dir",
        "file_search",
        "grep_search",
        "semantic_search",
        "get_errors",
    }),
    # Terminal execution and task management
    frozenset({
        "run_in_terminal",
        "send_to_terminal",
        "get_terminal_output",
        "kill_terminal",
        "terminal_last_command",
        "terminal_selection",
        "create_and_run_task",
        "get_task_output",
    }),
    # Jupyter notebooks
    frozenset({
        "run_notebook_cell",
        "edit_notebook_file",
        "copilot_getNotebookSummary",
        "read_notebook_cell_output",
        "create_new_jupyter_notebook",
    }),
    # Browser / Web automation
    frozenset({
        "open_browser_page",
        "read_page",
        "click_element",
        "type_in_page",
        "hover_element",
        "drag_element",
        "handle_dialog",
        "navigate_page",
        "screenshot_page",
        "run_playwright_code",
        "fetch_webpage",
    }),
    # VS Code API & code intelligence
    frozenset({
        "vscode_listCodeUsages",
        "vscode_renameSymbol",
        "vscode_askQuestions",
        "vscode_searchExtensions_internal",
        "get_vscode_api",
        "run_vscode_command",
        "install_extension",
        "testFailure",
    }),
    # Memory & session tracking
    frozenset({
        "memory",
        "resolve_memory_file_uri",
        "session_store_sql",
        "manage_todo_list",
    }),
    # GitHub repository search and inspection
    frozenset({
        "github_repo",
        "github_text_search",
    }),
)


def expand_tool_affinity(names: Iterable[str], registry: Mapping[str, Any] | None = None) -> list[str]:
    """Expand tool names with their affinity groups, retaining order and filtering by registry if provided."""
    expanded: list[str] = []
    seen: set[str] = set()

    for name in names:
        if name not in seen:
            seen.add(name)
            if registry is None or name in registry:
                expanded.append(name)
        for group in TOOL_AFFINITY_GROUPS:
            if name in group:
                for related in sorted(group):
                    if related not in seen and (registry is None or related in registry):
                        seen.add(related)
                        expanded.append(related)

    return expanded


def _catalog_text(registry: Mapping[str, Any]) -> str:
    return "\n".join(
        f"- {name}: {compact_description(tool.get('function', {}).get('description', ''), CATALOG_DESCRIPTION_LIMIT)}"
        for name, tool in registry.items()
    )


def dynamic_tool_definition(available: Mapping[str, Any]) -> dict[str, Any]:
    """Build the loader tool the model calls to request full schemas."""
    catalog = _catalog_text(available)
    catalog_section = f"Available tools:\n{catalog}\n\n" if catalog else ""
    description = (
        f"{catalog_section}"
        "Load the full schemas and parameters for one or more available tools by name before using them. "
        "CRITICAL: If you need to read or edit files, search code, run terminal commands, or use any workspace tool, "
        "you MUST call this tool ('get_tools') first with the required tool names in 'tool_names'. "
        "Do NOT refuse, apologize, or claim you lack tools or workspace access without calling get_tools first. "
        "Once a schema is loaded, call that tool directly."
    )
    return {
        "type": "function",
        "function": {
            "name": DYNAMIC_TOOL_NAME,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_names": {
                        "type": "array",
                        "minItems": 1,
                        # The enum lets grammar-constrained backends rule out invented names.
                        "items": {"type": "string", "enum": list(available)},
                        "description": "Names of the tools to load (e.g. ['read_file', 'replace_string_in_file']).",
                    }
                },
                "required": ["tool_names"],
                "additionalProperties": False,
            },
        },
    }


def dynamic_tools_list(registry: Mapping[str, Any], loaded: Iterable[str]) -> list[dict[str, Any]]:
    """Expose the already loaded schemas plus a loader for whatever is left."""
    loaded_names = [name for name in dict.fromkeys(loaded) if name in registry]
    remaining = {name: tool for name, tool in registry.items() if name not in set(loaded_names)}
    loaded_tools = [registry[name] for name in loaded_names]
    if not remaining:
        return loaded_tools
    return [dynamic_tool_definition(remaining), *loaded_tools]


def _previously_called_tools(messages: Any, registry: Mapping[str, Any]) -> list[str]:
    """Tools the model already used in this conversation have to stay available."""
    called: list[str] = []
    if not isinstance(messages, list):
        return called
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            name = function.get("name") if isinstance(function, dict) else None
            if name in registry and name not in called:
                called.append(name)
    return expand_tool_affinity(called, registry)


def _inject_dynamic_instruction(messages: list[Any]) -> list[Any]:
    if not messages:
        return [{"role": "system", "content": DYNAMIC_TOOL_INSTRUCTION}]

    updated = list(messages)
    for msg in updated:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and DYNAMIC_TOOL_INSTRUCTION in content:
                return updated

    for i, msg in enumerate(updated):
        if isinstance(msg, dict) and msg.get("role") in {"system", "developer"}:
            content = msg.get("content")
            if isinstance(content, str):
                updated[i] = {**msg, "content": f"{content}\n\n{DYNAMIC_TOOL_INSTRUCTION}"}
                return updated
            if isinstance(content, list):
                updated[i] = {**msg, "content": [*content, {"type": "text", "text": f"\n\n{DYNAMIC_TOOL_INSTRUCTION}"}]}
                return updated

    return [{"role": "system", "content": DYNAMIC_TOOL_INSTRUCTION}, *updated]


def prepare_dynamic_tools(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace the full tool list with a get_tools loader and a compact catalog."""
    optimized = deepcopy(dict(payload))
    original_tools = optimized.get("tools")
    if not isinstance(original_tools, list) or not original_tools:
        return optimized, {}
    if not isinstance(optimized.get("messages"), list):
        # Only the chat-completions shape can carry the synthetic loader exchange.
        return optimized, {}

    registry: dict[str, Any] = {}
    for tool in original_tools:
        name = tool_name(tool)
        if name is None:
            # Unknown tool shape (e.g. the Responses API): leave the payload untouched
            # instead of silently dropping tools the model needs.
            return optimized, {}
        if name != DYNAMIC_TOOL_NAME:
            registry[name] = tool
    if not registry:
        return optimized, {}

    loaded = _previously_called_tools(optimized.get("messages"), registry)
    optimized["tools"] = dynamic_tools_list(registry, loaded)
    if not any(tool_name(tool) == DYNAMIC_TOOL_NAME for tool in optimized["tools"]):
        # Everything is already loaded, so there is no loader call left to intercept.
        return optimized, {}
    optimized["messages"] = _inject_dynamic_instruction(optimized.get("messages", []))
    return optimized, registry
