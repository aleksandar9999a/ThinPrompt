# ThinPrompt

ThinPrompt is a model-agnostic proxy between VS Code and a local LLM server. It reduces request size while preserving the original API format and tool-calling workflow.

```text
VS Code -> ThinPrompt -> Local LLM server
```

## What it does

- compacts verbose tool descriptions and schemas;
- loads tool schemas dynamically, only when the model requests them;
- removes duplicate messages and repeated context blocks;
- normalizes redundant whitespace and line endings;
- forwards responses, including streaming responses, to VS Code;
- does not select, manage, or depend on a specific model provider.

ThinPrompt does not summarize unique code or system instructions. Its optimizations are deterministic and designed to preserve request meaning.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
uvicorn proxy.app:app --host 127.0.0.1 --port 8080 --env-file .env
```

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
PROXY_LOG_REQUESTS=false
PROXY_COMPACT_TOOLS=true
PROXY_DYNAMIC_TOOLS=true
```

Set `PROXY_LOG_REQUESTS=true` temporarily to print the original and optimized request sizes and bodies. Request bodies may contain source code or other sensitive content, so logging should normally remain disabled.

## Dynamic tools

When `PROXY_DYNAMIC_TOOLS=true`, ThinPrompt receives the complete tool list from VS Code but initially sends the model only:

- a short catalog of available tools;
- an internal `get_tool` function.

When the model requests a tool schema, ThinPrompt resolves it locally and retries the request with only that tool's full schema. The internal `get_tool` call is never exposed to VS Code. This can significantly reduce the prompt size while keeping real tool calls compatible with VS Code.

## Development

Run the test suite with:

```bash
.venv/bin/pytest -q
```

The proxy is implemented in `proxy/`, with optimization logic in `proxy/optimizer.py` and HTTP forwarding in `proxy/app.py`.