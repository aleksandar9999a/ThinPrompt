# LLM Request Proxy

Model-agnostic proxy between VS Code and an upstream HTTP API. It performs only deterministic request optimization, then forwards the request and response without selecting or understanding models.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
uvicorn proxy.app:app --host 127.0.0.1 --port 8080 --env-file .env
```

Set `UPSTREAM_URL` to the base URL of the upstream server. The proxy accepts `/v1/*` routes and forwards them to the same path. An optional `UPSTREAM_API_KEY` replaces the incoming `Authorization` header.

To print the original and optimized JSON request in the terminal, set `PROXY_LOG_REQUESTS=true`. The local `.env` enables this temporarily. The output includes byte counts and redacts the `Authorization` header, but request bodies may still contain source code or other sensitive content.

## Supported behavior

- preserves the request format and all fields;
- normalizes redundant line endings and trailing whitespace;
- removes duplicate messages and repeated content blocks, including whitespace-only variations;
- optimizes both `messages` and `input` request shapes;
- forwards non-chat routes transparently;
- supports normal and SSE streaming responses;
- does not know model names, context windows, token budgets, or providers.

When `PROXY_DYNAMIC_TOOLS=true`, the proxy replaces the full incoming tool list with a short catalog and one internal `get_tool` function. If the upstream requests a tool schema, the proxy resolves it internally and retries the request with only that schema. Dynamic-tool requests are buffered while this decision is made, so their final response is returned after the upstream response is complete.

The optimizer is intentionally deterministic and model-agnostic. It cannot safely remove unique system instructions or source code. If almost all tokens are unique VS Code context, a large reduction requires a context-selection policy or summarization step, which would be a separate, non-transparent feature.

## Example

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"hello"}]}'
```