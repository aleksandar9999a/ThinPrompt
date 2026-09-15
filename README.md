# ThinPrompt

ThinPrompt is a model-agnostic proxy between VS Code and a local LLM server. It reduces request size while preserving the original API format and tool-calling workflow.

```text
VS Code -> ThinPrompt -> Local LLM server
```

## What it does

- compacts verbose tool descriptions and schemas;
- loads tool schemas dynamically, only when the model requests them;
- removes repeated instruction messages and blocks repeated inside a message;
- normalizes redundant whitespace and line endings;
- forwards responses, including streaming responses, to VS Code;
- does not select, manage, or depend on a specific model provider.

ThinPrompt does not summarize unique code or system instructions. Its optimizations are deterministic and designed to preserve request meaning. Conversation history is never restructured: tool calls, tool results, and the final message are always forwarded as-is, because dropping them makes agents repeat their previous turn.

## Quick start

After cloning the repository, start ThinPrompt with one command:

```bash
./start.sh
```

On the first run, the script creates the virtual environment, installs dependencies, and creates `.env` from `.env.example`. Edit `.env` if you need a different upstream URL or proxy port, then run `./start.sh` again.

The local LLM server should be running before ThinPrompt starts. By default, the example configuration expects it at `http://127.0.0.1:2020`.

## VS Code configuration

Configure the VS Code AI extension to use ThinPrompt as an OpenAI-compatible endpoint:

```text
Base URL: http://127.0.0.1:8080/v1
```

ThinPrompt then forwards requests to the upstream URL configured in `.env`.

## Configuration

```env
UPSTREAM_URL=http://127.0.0.1:2020
UPSTREAM_API_KEY=
PROXY_HOST=127.0.0.1
PROXY_PORT=8080
PROXY_CONNECT_TIMEOUT=10
PROXY_LOG_REQUESTS=false
PROXY_COMPACT_TOOLS=true
PROXY_DYNAMIC_TOOLS=true
```

Set `PROXY_LOG_REQUESTS=true` temporarily to print the full traffic to stderr: `CLIENT REQUEST` (what the editor sent), `UPSTREAM REQUEST` (what ThinPrompt sends to the model, including the internal retries of the dynamic tool loader) and `UPSTREAM RESPONSE` (what the model answered, streamed bodies logged once fully drained). Credential headers are redacted, but bodies may contain source code or other sensitive content, so logging should normally remain disabled.

## Dynamic tools

When `PROXY_DYNAMIC_TOOLS=true`, ThinPrompt receives the complete tool list from VS Code but initially sends the model only:

- a short catalog of available tools;
- an internal `get_tools` function whose `tool_names` argument is an enum of exactly those tool names, so backends with constrained decoding cannot produce an invented name.

When the model requests tool schemas, ThinPrompt resolves them locally and retries the request with those full schemas, keeping `get_tools` available for whatever is still unloaded. The internal `get_tools` call is never exposed to VS Code: if the model keeps requesting unknown tools, ThinPrompt retries once with the complete tool list instead. This can significantly reduce the prompt size while keeping real tool calls compatible with VS Code.

## Development

Run the test suite with:

```bash
.venv/bin/pytest -q
```

The proxy lives in `proxy/`:

| Module | Responsibility |
| --- | --- |
| `config.py` | environment-backed settings |
| `text.py` | text normalization primitives |
| `tool_schemas.py` | tool description and schema compaction |
| `messages.py` | conversation-level compression (`optimize_request`) |
| `dynamic_tools.py` | the `get_tools` loader, catalog and registry |
| `responses.py` | parsing tool calls out of JSON or SSE responses |
| `resolver.py` | the `get_tools` resolution loop and its fallback |
| `upstream.py` | HTTP forwarding, headers and streaming |
| `app.py` | FastAPI routing that composes the above |